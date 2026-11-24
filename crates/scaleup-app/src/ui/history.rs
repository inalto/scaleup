//! The history strip under the image area: Back and Forward, a thumbnail per step of this session
//! (the current one outlined in the accent), the actions on the current step and where it is in the
//! list. It only draws and reports what was asked; the app moves the cursor and talks to the
//! worker. The thumbnails live here, keyed by the step's output file, so the state stays free of
//! textures.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use eframe::egui::{self, Color32, Key, Modifiers, Rect, Stroke, TextureHandle, Vec2, pos2, vec2};

use crate::state::{History, Step};
use crate::ui::theme;

/// Height of the strip, separator included.
pub const STRIP_HEIGHT: f32 = 76.0;
/// Height of a thumbnail on screen; the width follows the picture up to [`THUMB_MAX_WIDTH`].
const THUMB_HEIGHT: f32 = 56.0;
const THUMB_MIN_WIDTH: f32 = 28.0;
const THUMB_MAX_WIDTH: f32 = 112.0;
/// Long side, in pixels, of a thumbnail texture.
pub const THUMB_SIDE: usize = 128;

/// A move through the history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Back,
    Forward,
}

/// What the user asked of the strip this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryAction {
    Back,
    Forward,
    /// Go to step `index`.
    Go(usize),
    /// The current step's output becomes the selected image.
    UseAsInput,
    /// The current step's settings come back into the sidebar.
    RestoreSettings,
    /// Open the folder of the current step's output.
    OpenFolder,
    /// Take the current step out of the history; its files stay.
    Remove,
}

impl From<Nav> for HistoryAction {
    fn from(nav: Nav) -> Self {
        match nav {
            Nav::Back => HistoryAction::Back,
            Nav::Forward => HistoryAction::Forward,
        }
    }
}

/// The history move a key press asks for (D10): `Alt+Left` and `Ctrl+Z` go back, `Alt+Right`,
/// `Ctrl+Shift+Z` and `Ctrl+Y` go forward (`Cmd` for `Ctrl` on macOS). Other modifiers held with
/// them make no move, so `Ctrl+Alt+Z` is left to whoever wants it.
pub fn nav_for(modifiers: Modifiers, key: Key) -> Option<Nav> {
    let is = |pattern: Modifiers| modifiers.matches_exact(pattern);
    match key {
        Key::ArrowLeft if is(Modifiers::ALT) => Some(Nav::Back),
        Key::ArrowRight if is(Modifiers::ALT) => Some(Nav::Forward),
        Key::Z if is(Modifiers::COMMAND) => Some(Nav::Back),
        Key::Z if is(Modifiers::COMMAND | Modifiers::SHIFT) => Some(Nav::Forward),
        Key::Y if is(Modifiers::COMMAND) => Some(Nav::Forward),
        _ => None,
    }
}

/// The hover text of the Back or Forward button, with its shortcuts as this system names the keys.
pub fn nav_hint(nav: Nav, mac: bool) -> String {
    let (alt, ctrl) = if mac {
        ("Option", "Cmd")
    } else {
        ("Alt", "Ctrl")
    };
    match nav {
        Nav::Back => format!("Previous step ({alt}+Left or {ctrl}+Z)"),
        Nav::Forward => {
            format!("Next step ({alt}+Right, {ctrl}+Shift+Z or {ctrl}+Y)")
        }
    }
}

/// The name a screen reader says for step `index`: `Step 3: Extend +128 px · LaMa (512) · 4×`.
pub fn step_name(index: usize, step: &Step) -> String {
    format!("Step {}: {}", index + 1, step.label)
}

/// The hover text of a step: what was done, the file it wrote and the size of that file.
pub fn step_hover(step: &Step) -> String {
    let file = step.output.file_name().map_or_else(
        || step.output.display().to_string(),
        |f| f.to_string_lossy().into_owned(),
    );
    let [w, h] = step.result_size;
    format!("{}\n{file}\n{w}×{h}", step.label)
}

/// Where the cursor is: `3 of 7`.
pub fn count_text(history: &History) -> String {
    match history.at() {
        Some(at) => format!("{} of {}", at + 1, history.steps().len()),
        None => String::new(),
    }
}

/// The size of the thumbnail of a `result`-sized picture: [`THUMB_HEIGHT`] high, as wide as the
/// picture's shape asks between the limits (a very wide or tall result is fitted inside).
pub fn thumb_size(result: [u32; 2]) -> Vec2 {
    let [w, h] = result;
    let aspect = w.max(1) as f32 / h.max(1) as f32;
    vec2(
        (THUMB_HEIGHT * aspect).clamp(THUMB_MIN_WIDTH, THUMB_MAX_WIDTH),
        THUMB_HEIGHT,
    )
}

/// `image` shrunk so its long side is at most `max` pixels (never enlarged). Each pixel averages a
/// grid of up to 4×4 samples of the source area it covers: fast enough for the UI thread on a
/// 4096 px preview, and smooth enough for a thumbnail.
pub fn thumbnail(image: &egui::ColorImage, max: usize) -> egui::ColorImage {
    let [w, h] = image.size;
    let long = w.max(h).max(1);
    if long <= max || w == 0 || h == 0 {
        return image.clone();
    }
    let side = |s: usize| ((s * max + long / 2) / long).max(1);
    let (tw, th) = (side(w), side(h));
    let mut pixels = Vec::with_capacity(tw * th);
    for ty in 0..th {
        let (y0, y1) = (ty * h / th, ((ty + 1) * h / th).max(ty * h / th + 1));
        for tx in 0..tw {
            let (x0, x1) = (tx * w / tw, ((tx + 1) * w / tw).max(tx * w / tw + 1));
            let mut sum = [0u32; 4];
            let mut n = 0;
            for sy in 0..4 {
                let y = y0 + (y1 - y0) * sy / 4;
                for sx in 0..4 {
                    let x = x0 + (x1 - x0) * sx / 4;
                    let c = image.pixels[y * w + x];
                    for (s, v) in sum.iter_mut().zip(c.to_array()) {
                        *s += u32::from(v);
                    }
                    n += 1;
                }
            }
            let [r, g, b, a] = sum.map(|s| (s / n) as u8);
            pixels.push(Color32::from_rgba_premultiplied(r, g, b, a));
        }
    }
    egui::ColorImage::new([tw, th], pixels)
}

/// Thumbnails by the output file of their step. A step's output path is unique to it (each
/// single-image run writes its own file, unless Overwrite replaces it, and then the newer picture
/// is the right one for both).
pub struct Thumbs<T> {
    map: HashMap<PathBuf, T>,
}

impl<T> Default for Thumbs<T> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

impl<T> Thumbs<T> {
    pub fn get(&self, output: &Path) -> Option<&T> {
        self.map.get(output)
    }

    /// True when a step of `steps` wrote `output`, so its picture is worth a thumbnail.
    pub fn wants(steps: &[Step], output: &Path) -> bool {
        steps.iter().any(|s| s.output == output)
    }

    /// Keeps `thumb` for `output`, replacing an older one.
    pub fn insert(&mut self, output: PathBuf, thumb: T) {
        self.map.insert(output, thumb);
    }

    /// Drops the thumbnails of steps no longer in `steps` (cut by the cap, or removed); a texture
    /// is freed when it is dropped.
    pub fn retain_steps(&mut self, steps: &[Step]) {
        self.map.retain(|k, _| Self::wants(steps, k));
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// The strip's own state between frames.
#[derive(Default)]
pub struct HistoryStrip {
    thumbs: Thumbs<TextureHandle>,
    /// The cursor and the number of steps when the current thumbnail was last scrolled into view.
    scrolled: Option<(usize, usize)>,
}

impl HistoryStrip {
    /// A result preview arrived for `output`: when a step wrote that file, its thumbnail is made
    /// from `image`.
    pub fn note_result(
        &mut self,
        ctx: &egui::Context,
        history: &History,
        output: &Path,
        image: &egui::ColorImage,
    ) {
        if !Thumbs::<TextureHandle>::wants(history.steps(), output) {
            return;
        }
        let small = thumbnail(image, THUMB_SIDE);
        let name = format!("history:{}", output.display());
        let texture = ctx.load_texture(name, small, egui::TextureOptions::LINEAR);
        self.thumbs.insert(output.to_path_buf(), texture);
    }

    /// Frees the thumbnails of steps no longer in `history`. Called every frame, the strip shown
    /// or not, so the last step removed frees its texture too.
    pub fn forget_dropped(&mut self, history: &History) {
        self.thumbs.retain_steps(history.steps());
    }

    /// How many thumbnails are kept (tests check that removed steps free theirs).
    #[cfg(test)]
    pub fn thumb_count(&self) -> usize {
        self.thumbs.len()
    }

    /// Draws the strip; returns what was asked. Everything is disabled while `running`.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        history: &History,
        running: bool,
    ) -> Option<HistoryAction> {
        let mac = ui.ctx().os() == egui::os::OperatingSystem::Mac;
        let mut action = None;
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = GAP;
                let back = chevron_button(ui, Nav::Back, history.can_back(), mac);
                if back.clicked() {
                    action = Some(HistoryAction::Back);
                }
                let forward = chevron_button(ui, Nav::Forward, history.can_forward(), mac);
                if forward.clicked() {
                    action = Some(HistoryAction::Forward);
                }
                // The actions and the count keep their width; the thumbnails get the rest.
                let actions = ActionsLayout::measure(ui);
                let count_font = egui::TextStyle::Body.resolve(ui.style());
                let count_w = text_width(ui, "30 of 30", count_font);
                // Before the actions and before the count: item spacing, a space, item spacing.
                let right = actions.width() + count_w + 6.0 * GAP;
                let thumbs_w = (ui.available_width() - right).max(THUMB_MIN_WIDTH);
                ui.add_space(GAP);
                ui.allocate_ui(vec2(thumbs_w, THUMB_HEIGHT), |ui| {
                    if let Some(i) = self.thumbnails(ui, history, thumbs_w) {
                        action = Some(HistoryAction::Go(i));
                    }
                });
                ui.add_space(GAP);
                if let Some(a) = step_actions(ui, &actions) {
                    action = Some(a);
                }
                ui.add_space(GAP);
                ui.allocate_ui_with_layout(
                    vec2(count_w, THUMB_HEIGHT),
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        let text = egui::RichText::new(count_text(history)).weak();
                        ui.add(egui::Label::new(text).selectable(false))
                            .on_hover_text("The step shown, and how many steps this session has");
                    },
                );
            });
        });
        action
    }

    /// The scrollable row of thumbnails, oldest on the left. Returns the step clicked.
    fn thumbnails(&mut self, ui: &mut egui::Ui, history: &History, width: f32) -> Option<usize> {
        let mut clicked = None;
        let now = history.at().map(|at| (at, history.steps().len()));
        let scroll = now.is_some() && now != self.scrolled;
        egui::ScrollArea::horizontal()
            .id_salt("history_steps")
            .max_width(width)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    for (i, step) in history.steps().iter().enumerate() {
                        let current = history.at() == Some(i);
                        let texture = self.thumbs.get(&step.output);
                        let resp = step_thumb(ui, i, step, texture, current);
                        if current && scroll {
                            resp.scroll_to_me_animation(
                                Some(egui::Align::Center),
                                egui::style::ScrollAnimation::none(),
                            );
                        }
                        if resp.clicked() {
                            clicked = Some(i);
                        }
                    }
                });
            });
        if scroll {
            self.scrolled = now;
        }
        clicked
    }
}

/// Space between the parts of the strip.
const GAP: f32 = 6.0;
/// Height of an action button; two of them and the gap fill a thumbnail's height.
const ACTION_HEIGHT: f32 = 24.0;

/// The actions on the current step, in two columns of two: label, hover text, action.
const ACTIONS: [[(&str, &str, HistoryAction); 2]; 2] = [
    [
        (
            "Use as input",
            "Select this result as the image, to extend or upscale it further",
            HistoryAction::UseAsInput,
        ),
        (
            "Open folder",
            "Open the folder this result was saved in",
            HistoryAction::OpenFolder,
        ),
    ],
    [
        (
            "Restore settings",
            "Put the settings of this step back into the sidebar",
            HistoryAction::RestoreSettings,
        ),
        (
            "Remove",
            "Take this step out of the history. The file is kept.",
            HistoryAction::Remove,
        ),
    ],
];

/// The width of each column of action buttons, from their labels in the button font.
struct ActionsLayout {
    columns: [f32; 2],
}

impl ActionsLayout {
    fn measure(ui: &egui::Ui) -> Self {
        let font = egui::TextStyle::Button.resolve(ui.style());
        let pad = ui.spacing().button_padding.x * 2.0;
        let column = |c: &[(&str, &str, HistoryAction); 2]| {
            c.iter()
                .map(|(text, ..)| text_width(ui, text, font.clone()) + pad)
                .fold(0.0, f32::max)
                .ceil()
        };
        Self {
            columns: [column(&ACTIONS[0]), column(&ACTIONS[1])],
        }
    }

    fn width(&self) -> f32 {
        self.columns[0] + GAP + self.columns[1]
    }
}

/// The width of `text` set in `font`, on one line.
fn text_width(ui: &egui::Ui, text: &str, font: egui::FontId) -> f32 {
    let color = ui.visuals().text_color();
    ui.painter()
        .layout_no_wrap(text.to_owned(), font, color)
        .size()
        .x
}

/// The actions on the current step, two by two, in the panel's plain button style.
fn step_actions(ui: &mut egui::Ui, layout: &ActionsLayout) -> Option<HistoryAction> {
    let mut action = None;
    let gap_y = THUMB_HEIGHT - 2.0 * ACTION_HEIGHT;
    for (column, width) in ACTIONS.iter().zip(layout.columns) {
        ui.allocate_ui_with_layout(
            vec2(width, THUMB_HEIGHT),
            egui::Layout::top_down_justified(egui::Align::Min),
            |ui| {
                ui.spacing_mut().item_spacing.y = gap_y;
                for (text, hover, a) in column {
                    let button = egui::Button::new(*text).min_size(vec2(width, ACTION_HEIGHT));
                    if ui.add(button).on_hover_text(*hover).clicked() {
                        action = Some(*a);
                    }
                }
            },
        );
    }
    action
}

/// Back or Forward: a button with a chevron drawn on it (the fonts have no arrow glyphs).
fn chevron_button(ui: &mut egui::Ui, nav: Nav, enabled: bool, mac: bool) -> egui::Response {
    let name = match nav {
        Nav::Back => "Back",
        Nav::Forward => "Forward",
    };
    // The whole strip is disabled while a run goes.
    let enabled_around = ui.is_enabled();
    let resp = ui
        .add_enabled_ui(enabled, |ui| {
            let (rect, resp) =
                ui.allocate_exact_size(vec2(28.0, THUMB_HEIGHT), egui::Sense::click());
            let on = ui.is_enabled();
            resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, on, name));
            if ui.is_rect_visible(rect) {
                let v = ui.style().interact(&resp);
                ui.painter().rect(
                    rect,
                    v.corner_radius,
                    v.weak_bg_fill,
                    v.bg_stroke,
                    egui::StrokeKind::Inside,
                );
                chevron(
                    ui.painter(),
                    rect.center(),
                    nav,
                    Stroke::new(2.0, v.fg_stroke.color),
                );
            }
            resp
        })
        .inner;
    let why = match nav {
        _ if !enabled_around => "Not while a run is in progress",
        Nav::Back => "This is the first step",
        Nav::Forward => "This is the last step",
    };
    resp.on_hover_text(nav_hint(nav, mac))
        .on_disabled_hover_text(why)
}

/// A chevron pointing left for Back, right for Forward, centred on `at`.
fn chevron(painter: &egui::Painter, at: egui::Pos2, nav: Nav, stroke: Stroke) {
    let dir = match nav {
        Nav::Back => -1.0,
        Nav::Forward => 1.0,
    };
    let points = vec![
        at + vec2(-3.0 * dir, -6.0),
        at + vec2(3.0 * dir, 0.0),
        at + vec2(-3.0 * dir, 6.0),
    ];
    painter.add(egui::Shape::line(points, stroke));
}

/// One step's thumbnail: a button showing the result, a 1 px outline, the accent on the current one.
fn step_thumb(
    ui: &mut egui::Ui,
    index: usize,
    step: &Step,
    texture: Option<&TextureHandle>,
    current: bool,
) -> egui::Response {
    let size = thumb_size(step.result_size);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let on = ui.is_enabled();
    resp.widget_info(|| {
        let mut info = egui::WidgetInfo::labeled(egui::Role::Button, on, step_name(index, step));
        info.selected = Some(current);
        info
    });
    if ui.is_rect_visible(rect) {
        let visuals = ui.visuals();
        let painter = ui.painter();
        painter.rect_filled(rect, 3.0, theme::stage_fill(visuals));
        match texture {
            Some(t) => {
                let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
                painter.image(t.id(), fit_inside(t.size_vec2(), rect), uv, Color32::WHITE);
            }
            None => {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("{}", index + 1),
                    egui::FontId::proportional(13.0),
                    visuals.weak_text_color(),
                );
            }
        }
        let stroke = if current {
            Stroke::new(2.0, theme::ACCENT)
        } else if resp.hovered() || resp.has_focus() {
            Stroke::new(1.0, visuals.widgets.hovered.fg_stroke.color)
        } else {
            Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color)
        };
        painter.rect_stroke(rect, 3.0, stroke, egui::StrokeKind::Inside);
    }
    resp.on_hover_text(step_hover(step))
}

/// `image`-sized content fitted and centred inside `rect`.
fn fit_inside(image: Vec2, rect: Rect) -> Rect {
    let scale = (rect.width() / image.x.max(1.0)).min(rect.height() / image.y.max(1.0));
    Rect::from_center_size(rect.center(), image * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Settings;
    use scaleup_core::Margins;

    fn step(n: usize) -> Step {
        Step {
            input: PathBuf::from(format!("in{n}.png")),
            output: PathBuf::from(format!(r"C:\out\out{n}.png")),
            settings: Settings::default(),
            source_size: [256, 192],
            result_size: [1024, 768],
            margins: Margins::default(),
            label: format!("Extend +128 px · LaMa (512) · {n}×"),
        }
    }

    fn fonts_have(text: &str) -> bool {
        text.chars().all(|c| c.is_ascii() || "×·…".contains(c))
    }

    #[test]
    fn the_shortcuts_of_d10_move_back_and_forward() {
        let cmd = Modifiers::COMMAND;
        let ctrl_win = Modifiers::CTRL | Modifiers::COMMAND;
        let mac_cmd = Modifiers::MAC_CMD | Modifiers::COMMAND;
        assert_eq!(nav_for(Modifiers::ALT, Key::ArrowLeft), Some(Nav::Back));
        assert_eq!(nav_for(Modifiers::ALT, Key::ArrowRight), Some(Nav::Forward));
        for m in [cmd, ctrl_win, mac_cmd] {
            assert_eq!(nav_for(m, Key::Z), Some(Nav::Back), "{m:?}");
            assert_eq!(nav_for(m | Modifiers::SHIFT, Key::Z), Some(Nav::Forward));
            assert_eq!(nav_for(m, Key::Y), Some(Nav::Forward));
        }
    }

    #[test]
    fn other_keys_and_modifiers_make_no_move() {
        for (m, k) in [
            (Modifiers::NONE, Key::ArrowLeft),
            (Modifiers::SHIFT, Key::ArrowLeft),
            (Modifiers::ALT | Modifiers::SHIFT, Key::ArrowLeft),
            (Modifiers::COMMAND, Key::ArrowLeft),
            (Modifiers::NONE, Key::Z),
            (Modifiers::ALT, Key::Z),
            (Modifiers::COMMAND | Modifiers::ALT, Key::Z),
            (Modifiers::COMMAND | Modifiers::SHIFT, Key::Y),
            (Modifiers::COMMAND, Key::X),
            (Modifiers::ALT, Key::ArrowUp),
        ] {
            assert_eq!(nav_for(m, k), None, "{m:?} {k:?}");
        }
    }

    #[test]
    fn hints_name_the_shortcuts_of_the_system() {
        assert_eq!(
            nav_hint(Nav::Back, false),
            "Previous step (Alt+Left or Ctrl+Z)"
        );
        assert_eq!(
            nav_hint(Nav::Forward, false),
            "Next step (Alt+Right, Ctrl+Shift+Z or Ctrl+Y)"
        );
        assert!(nav_hint(Nav::Back, true).contains("Option+Left or Cmd+Z"));
        assert!(nav_hint(Nav::Forward, true).contains("Cmd+Y"));
    }

    #[test]
    fn names_and_hover_text_of_a_step() {
        let s = step(4);
        assert_eq!(step_name(2, &s), "Step 3: Extend +128 px · LaMa (512) · 4×");
        assert_eq!(
            step_hover(&s),
            "Extend +128 px · LaMa (512) · 4×\nout4.png\n1024×768"
        );
        let mut h = History::default();
        assert_eq!(count_text(&h), "");
        for n in 0..7 {
            h.push(step(n));
        }
        h.go(2);
        assert_eq!(count_text(&h), "3 of 7");
        for text in [
            step_name(2, &s),
            step_hover(&s),
            count_text(&h),
            nav_hint(Nav::Back, false),
            nav_hint(Nav::Forward, true),
        ] {
            assert!(fonts_have(&text), "{text}");
        }
    }

    #[test]
    fn thumbnails_keep_the_shape_between_limits() {
        let s = thumb_size([1024, 768]);
        assert!(
            (s.x - 56.0 * 4.0 / 3.0).abs() < 1e-3 && s.y == 56.0,
            "{s:?}"
        );
        assert_eq!(thumb_size([100, 100]), vec2(56.0, 56.0));
        // Very wide and very tall results are capped, then fitted inside.
        assert_eq!(thumb_size([4000, 500]), vec2(112.0, 56.0));
        assert_eq!(thumb_size([100, 1000]), vec2(28.0, 56.0));
        assert_eq!(thumb_size([0, 0]), vec2(56.0, 56.0));
        let r = Rect::from_min_size(pos2(0.0, 0.0), vec2(112.0, 56.0));
        assert_eq!(
            fit_inside(vec2(128.0, 16.0), r),
            Rect::from_center_size(pos2(56.0, 28.0), vec2(112.0, 14.0))
        );
    }

    #[test]
    fn a_thumbnail_has_at_most_128_pixels_on_its_long_side() {
        let big = egui::ColorImage::filled([4096, 1024], Color32::from_rgb(200, 100, 50));
        let t = thumbnail(&big, THUMB_SIDE);
        assert_eq!(t.size, [128, 32]);
        assert!(
            t.pixels
                .iter()
                .all(|&c| c == Color32::from_rgb(200, 100, 50))
        );
        let tall = egui::ColorImage::filled([300, 3000], Color32::WHITE);
        assert_eq!(thumbnail(&tall, THUMB_SIDE).size, [13, 128]);
        // A small picture is kept as it is.
        let small = egui::ColorImage::filled([20, 10], Color32::BLACK);
        assert_eq!(thumbnail(&small, THUMB_SIDE).size, [20, 10]);
        // A 1-pixel line survives.
        let thin = egui::ColorImage::filled([1000, 1], Color32::WHITE);
        assert_eq!(thumbnail(&thin, THUMB_SIDE).size, [128, 1]);
    }

    #[test]
    fn a_thumbnail_averages_the_area_it_covers() {
        // Left half black, right half white: the thumbnail keeps the halves.
        let mut img = egui::ColorImage::filled([256, 4], Color32::BLACK);
        for y in 0..4 {
            for x in 128..256 {
                img.pixels[y * 256 + x] = Color32::WHITE;
            }
        }
        let t = thumbnail(&img, 128);
        assert_eq!(t.size, [128, 2]);
        assert_eq!(t.pixels[0], Color32::BLACK);
        assert_eq!(t.pixels[127], Color32::WHITE);
    }

    #[test]
    fn thumbs_are_kept_for_steps_and_dropped_with_them() {
        let mut h = History::default();
        for n in 0..3 {
            h.push(step(n));
        }
        let mut thumbs = Thumbs::<u8>::default();
        assert!(Thumbs::<u8>::wants(h.steps(), &step(1).output));
        assert!(!Thumbs::<u8>::wants(h.steps(), Path::new("elsewhere.png")));
        for n in 0..3 {
            thumbs.insert(step(n).output, n as u8);
        }
        // A newer picture of the same file replaces the older thumbnail.
        thumbs.insert(step(2).output, 9);
        assert_eq!(thumbs.get(&step(2).output), Some(&9));
        assert_eq!(thumbs.len(), 3);
        // Removed: its thumbnail goes, the others stay.
        h.remove(1);
        thumbs.retain_steps(h.steps());
        assert_eq!(thumbs.len(), 2);
        assert_eq!(thumbs.get(&step(1).output), None);
        assert_eq!(thumbs.get(&step(0).output), Some(&0));
    }

    #[test]
    fn the_cap_drops_the_oldest_thumbnail() {
        let mut h = History::default();
        let mut thumbs = Thumbs::<usize>::default();
        for n in 0..crate::state::HISTORY_LIMIT + 1 {
            h.push(step(n));
            thumbs.insert(step(n).output, n);
            thumbs.retain_steps(h.steps());
        }
        assert_eq!(thumbs.len(), crate::state::HISTORY_LIMIT);
        assert_eq!(thumbs.get(&step(0).output), None);
        assert_eq!(thumbs.get(&step(1).output), Some(&1));
    }
}
