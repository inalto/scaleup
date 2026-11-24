//! The edit stage of the image area: the planned canvas around the picture, a preview of what will
//! fill the new canvas, and eight handles that drag its border. The geometry is in
//! [`canvas_edit`]; this module decides when the stage shows, draws it, and turns the pointer and
//! the keys into settings.
//!
//! Layout during a drag: when a handle is grabbed, the source rectangle and the scale (points per
//! source pixel) are frozen. While the drag lasts the canvas grows or shrinks around the frozen
//! source, so the dragged edge stays under the pointer 1:1 (to the nearest source pixel), even when
//! the canvas runs past the view. On release the view fits the new canvas again at once, keeping
//! the zoom and pan. Re-fitting under the pointer would make the edge run away from it: a wider
//! canvas fits smaller, which moves the edge back, which shrinks the drag.

use eframe::egui::{self, Color32, Key, Rect, Stroke, TextureHandle, Vec2, pos2, vec2};
use scaleup_core::{Fill, Margins, Registry};

use crate::state::{ExtendMode, Settings, fill_title};
use crate::ui::canvas_edit::{self, CanvasLayout, DragMods, HIT_SIZE, Handle, PreviewKind};
use crate::ui::compare::{around, wheel_zoom};
use crate::ui::theme;

// -------------------------------------------------------------------------------------- stage

/// Which of its two stages the image area shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The planned canvas with its handles.
    Edit,
    /// The before/after view.
    Result,
}

/// The facts the stage is decided from.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageFacts {
    /// "Extend canvas" is on.
    pub extend_on: bool,
    /// A single image is selected and its "before" preview is decoded.
    pub image: bool,
    /// A run or a download is in progress.
    pub running: bool,
    /// A result ("after" preview) is shown.
    pub has_result: bool,
    /// An extend setting changed since the result arrived, or "Adjust canvas" was pressed.
    pub editing: bool,
}

/// Edit needs Extend on, a single image and no run; it shows until there is a result, and again
/// once the extension is changed or "Adjust canvas" is pressed. Everything else is Result.
pub fn stage(f: StageFacts) -> Stage {
    if f.extend_on && f.image && !f.running && (!f.has_result || f.editing) {
        Stage::Edit
    } else {
        Stage::Result
    }
}

/// The extend settings whose change, after a result, sends the image area back to Edit.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendKey {
    enabled: bool,
    mode: ExtendMode,
    sides: [f32; 4],
    linked: bool,
    aspect: (u32, u32),
    fill: String,
    model: String,
}

impl ExtendKey {
    pub fn of(s: &Settings) -> Self {
        Self {
            enabled: s.extend_enabled,
            mode: s.extend_mode,
            sides: [s.extend_left, s.extend_top, s.extend_right, s.extend_bottom],
            linked: s.extend_linked,
            aspect: s.extend_aspect,
            fill: s.extend_fill.clone(),
            model: s.extend_model.clone(),
        }
    }
}

/// The size chip under the canvas: `256×192 to 512×448`, or why the settings cannot be used for
/// this image.
pub fn size_text(s: &Settings, source: [u32; 2]) -> Result<String, String> {
    let [w, h] = source;
    match canvas_edit::current_margins(s, source) {
        Some(m) => Ok(format!(
            "{w}×{h} to {}×{}",
            w + m.left + m.right,
            h + m.top + m.bottom
        )),
        None => Err(match s.extend_summary(Some(source)) {
            Some(Err(e)) => e,
            _ => "These margins cannot be used".into(),
        }),
    }
}

// ----------------------------------------------------------------------------------- keyboard

/// Steps in source pixels, in screen directions (right and down positive), asked for by arrow-key
/// presses: one pixel each, ten with Shift.
pub fn arrow_steps(right: usize, left: usize, down: usize, up: usize, shift: bool) -> Vec2 {
    let per = if shift { 10.0 } else { 1.0 };
    vec2(
        (right as f32 - left as f32) * per,
        (down as f32 - up as f32) * per,
    )
}

/// The screen direction that moves `handle`'s sides outward, one pixel per moving side.
pub fn outward(handle: Handle) -> Vec2 {
    let s = handle.sides();
    let axis = |low: bool, high: bool| f32::from(u8::from(high)) - f32::from(u8::from(low));
    vec2(axis(s.left, s.right), axis(s.top, s.bottom))
}

/// Margins after the keyboard moved `handle` by `steps` source pixels in screen directions. A key
/// across the handle's axis (Up on the right edge) changes nothing; limits are those of a drag.
pub fn step_margins(handle: Handle, current: Margins, steps: Vec2, source: [u32; 2]) -> Margins {
    canvas_edit::drag(handle, current, steps, 1.0, DragMods::default(), source)
}

// --------------------------------------------------------------------------------------- view

/// A handle drag in progress: what was frozen when it began.
#[derive(Clone, Copy, Debug)]
struct HandleDrag {
    handle: Handle,
    start: Margins,
    /// Screen points per source pixel at the start.
    scale: f32,
    /// The source on screen at the start; it stays there while the drag lasts.
    source: Rect,
    /// Where the pointer was pressed.
    origin: egui::Pos2,
}

/// The canvas of `margins` around a source frozen at `source` on screen, at `scale` points per
/// source pixel: the layout while a handle is dragged.
pub fn frozen_layout(source: Rect, scale: f32, m: Margins) -> CanvasLayout {
    let canvas = Rect::from_min_max(
        source.min - vec2(m.left as f32, m.top as f32) * scale,
        source.max + vec2(m.right as f32, m.bottom as f32) * scale,
    );
    CanvasLayout {
        canvas,
        source,
        scale,
    }
}

/// State of the edit stage between frames.
#[derive(Default)]
pub struct EditView {
    drag: Option<HandleDrag>,
    /// The layout drawn last, in screen coordinates.
    last: Option<CanvasLayout>,
}

/// What the edit stage works on.
pub struct EditInput<'a> {
    pub texture: &'a TextureHandle,
    pub source: [u32; 2],
    pub settings: &'a mut Settings,
    pub registry: &'a Registry,
}

/// The id of a handle's focusable widget.
fn handle_id(handle: Handle) -> egui::Id {
    egui::Id::unique(("canvas_handle", handle))
}

/// True while one of the handles has the keyboard focus: its keys belong to it then.
pub fn handle_focused(ctx: &egui::Context) -> bool {
    ctx.memory(|m| m.focused())
        .is_some_and(|id| Handle::ALL.iter().any(|&h| handle_id(h) == id))
}

impl EditView {
    /// Forgets a drag in progress.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The canvas size in source pixels the view fits, for the zoom chips.
    pub fn canvas_pixels(settings: &Settings, source: [u32; 2]) -> Vec2 {
        let m = canvas_edit::planned_margins(settings, source);
        vec2(
            (source[0] + m.left + m.right) as f32,
            (source[1] + m.top + m.bottom) as f32,
        )
    }

    /// The layout drawn last, in screen coordinates (tests drive the pointer with it).
    #[cfg(test)]
    pub fn last_layout(&self) -> Option<CanvasLayout> {
        self.last
    }

    /// Where the canvas is fitted inside the view `rect`: room on every side keeps the handles and
    /// the outline in view, and the chips get a row of their own under the canvas.
    pub fn fit_area(rect: Rect) -> Rect {
        let min = rect.min + vec2(24.0, 24.0);
        let max = (rect.max - vec2(24.0, 56.0)).max(min);
        Rect::from_min_max(min, max)
    }

    /// The layout in the view `rect`: fitted, or frozen while a handle is dragged.
    fn layout(
        &self,
        rect: Rect,
        source: [u32; 2],
        m: Margins,
        zoom: f32,
        pan: Vec2,
    ) -> CanvasLayout {
        if let Some(d) = &self.drag {
            return frozen_layout(d.source, d.scale, m);
        }
        let fit = Self::fit_area(rect);
        let l = canvas_edit::canvas_rect(source, m, fit.size(), zoom, pan);
        let at = fit.min.to_vec2();
        CanvasLayout {
            canvas: l.canvas.translate(at),
            source: l.source.translate(at),
            scale: l.scale,
        }
    }

    /// Draws the stage in `rect` and handles its input. Returns true when the settings changed.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        zoom: &mut f32,
        pan: &mut Vec2,
        input: EditInput<'_>,
    ) -> bool {
        let EditInput {
            texture,
            source,
            settings,
            registry,
        } = input;
        // Drag only: a press on a handle starts moving it at once, with no click to wait for.
        let resp = ui.allocate_rect(rect, egui::Sense::DRAG);
        let before = canvas_edit::planned_margins(settings, source);
        let layout = self.layout(rect, source, before, *zoom, *pan);
        let mut changed = false;

        if self.drag.is_none() {
            wheel_zoom(ui, &resp, Self::fit_area(rect), zoom, pan);
        }
        if resp.dragged_by(egui::PointerButton::Middle)
            || resp.dragged_by(egui::PointerButton::Secondary)
        {
            *pan += resp.drag_delta();
        }
        if resp.drag_started_by(egui::PointerButton::Primary)
            && let Some(origin) = ui.input(|i| i.pointer.press_origin())
            && let Some(handle) = canvas_edit::hit(origin, layout.canvas)
        {
            self.drag = Some(HandleDrag {
                handle,
                start: before,
                scale: layout.scale,
                source: layout.source,
                origin,
            });
            ui.memory_mut(|m| m.request_focus(handle_id(handle)));
        }
        if let Some(d) = self.drag {
            let primary = resp.dragged_by(egui::PointerButton::Primary);
            if (primary || resp.drag_stopped_by(egui::PointerButton::Primary))
                && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
            {
                let mods = DragMods::from_egui(ui.input(|i| i.modifiers));
                let m = canvas_edit::drag(d.handle, d.start, pos - d.origin, d.scale, mods, source);
                if m != before {
                    canvas_edit::write_margins(settings, m, source);
                    changed = true;
                }
            }
            if !primary {
                // Released: the next layout fits the new canvas again.
                self.drag = None;
            }
        }

        // Keyboard: a focused handle moves with the arrow keys and screen-reader steps.
        let margins = canvas_edit::planned_margins(settings, source);
        let mut layout = self.layout(rect, source, margins, *zoom, *pan);
        let mut focused = None;
        for handle in Handle::ALL {
            let at = handle.position(layout.canvas);
            let r = ui.interact(
                Rect::from_center_size(at, Vec2::splat(HIT_SIZE)),
                handle_id(handle),
                egui::Sense::FOCUSABLE,
            );
            r.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Splitter, true, handle.name()));
            if !r.has_focus() {
                continue;
            }
            focused = Some(handle);
            ui.memory_mut(|m| {
                m.set_focus_lock_filter(
                    r.id,
                    egui::EventFilter {
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        ..Default::default()
                    },
                )
            });
            let steps = ui.input(|i| {
                let keys = arrow_steps(
                    i.num_presses(Key::ArrowRight),
                    i.num_presses(Key::ArrowLeft),
                    i.num_presses(Key::ArrowDown),
                    i.num_presses(Key::ArrowUp),
                    i.modifiers.shift,
                );
                use egui::accesskit::Action;
                let reader = i.num_accesskit_action_requests(r.id, Action::Increment) as f32
                    - i.num_accesskit_action_requests(r.id, Action::Decrement) as f32;
                keys + outward(handle) * reader
            });
            if steps != Vec2::ZERO {
                let m = step_margins(handle, margins, steps, source);
                if m != margins {
                    canvas_edit::write_margins(settings, m, source);
                    changed = true;
                    let m = canvas_edit::planned_margins(settings, source);
                    layout = self.layout(rect, source, m, *zoom, *pan);
                }
            }
        }

        let hovered = match &self.drag {
            Some(d) => Some(d.handle),
            None => resp
                .hover_pos()
                .and_then(|p| canvas_edit::hit(p, layout.canvas)),
        };
        if let Some(h) = hovered {
            ui.ctx().set_cursor_icon(h.cursor());
        }

        let painter = ui.painter_at(rect);
        paint_fill(&painter, ui.visuals(), layout, texture, settings, registry);
        let full = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        painter.image(texture.id(), layout.source, full, Color32::WHITE);
        outline(&painter, layout.canvas);
        if layout.source != layout.canvas {
            outline(&painter, layout.source);
        }
        for (handle, r) in canvas_edit::handles(layout.canvas) {
            let active = hovered == Some(handle) || focused == Some(handle);
            paint_handle(&painter, r, active);
        }
        self.last = Some(layout);
        changed
    }
}

// ------------------------------------------------------------------------------------ drawing

/// A 1 px light line with a dark edge on each side: the border reads on light and dark pictures.
fn outline(painter: &egui::Painter, r: Rect) {
    painter.rect_stroke(
        r,
        0.0,
        Stroke::new(3.0, theme::ON_IMAGE_EDGE),
        egui::StrokeKind::Middle,
    );
    painter.rect_stroke(
        r,
        0.0,
        Stroke::new(1.0, Color32::from_white_alpha(220)),
        egui::StrokeKind::Middle,
    );
}

/// A handle square in the split knob's style: white on a dark edge, the accent while it is the one
/// being acted on.
fn paint_handle(painter: &egui::Painter, r: Rect, active: bool) {
    painter.rect_filled(r.expand(1.5), 2.5, theme::ON_IMAGE_EDGE);
    if active {
        painter.rect_filled(r.expand(1.0), 2.0, Color32::WHITE);
        painter.rect_filled(r.shrink(1.0), 1.0, theme::ACCENT);
    } else {
        painter.rect_filled(r, 1.5, Color32::WHITE);
    }
}

/// The new canvas around the source, as the fill will make it.
fn paint_fill(
    painter: &egui::Painter,
    visuals: &egui::Visuals,
    layout: CanvasLayout,
    texture: &TextureHandle,
    settings: &Settings,
    registry: &Registry,
) {
    let CanvasLayout { canvas, source, .. } = layout;
    if canvas == source {
        return;
    }
    let visible = painter.clip_rect();
    let bands = around(canvas, source).map(|b| b.intersect(visible));
    let bands = bands.iter().filter(|b| b.is_positive());
    match Fill::parse(&settings.extend_fill) {
        Ok(Fill::Color([r, g, b])) => {
            for band in bands {
                painter.rect_filled(*band, 0.0, Color32::from_rgb(r, g, b));
            }
        }
        Ok(Fill::Transparent) => {
            for band in bands {
                checkerboard(&painter.with_clip_rect(*band), canvas.min, visuals);
            }
        }
        Ok(kind @ (Fill::Mirror | Fill::Edge)) => {
            let kind = if kind == Fill::Mirror {
                PreviewKind::Mirror
            } else {
                PreviewKind::Edge
            };
            for (screen, uv) in canvas_edit::preview_bands(source, canvas, kind) {
                painter.image(texture.id(), screen, uv, Color32::WHITE);
            }
        }
        Ok(Fill::Ai) => {
            let (tint, line) = theme::hatch_colors(visuals);
            for band in bands.clone() {
                painter.rect_filled(*band, 0.0, tint);
                hatch(&painter.with_clip_rect(*band), canvas.min, line);
            }
            let text = format!("AI fill · {}", fill_title(settings, registry));
            ai_label(painter, bands.copied(), &text);
        }
        // Not a fill the sidebar offers: the tint alone, no promise of what it will be.
        Err(_) => {
            let (tint, _) = theme::hatch_colors(visuals);
            for band in bands {
                painter.rect_filled(*band, 0.0, tint);
            }
        }
    }
}

/// Side of a checkerboard square, in points.
const CHECKER: f32 = 8.0;
/// Distance between hatch lines, in points.
const HATCH: f32 = 10.0;

/// The checkerboard of transparency over the painter's clip rect, its squares anchored at `origin`
/// so they move with the canvas.
fn checkerboard(painter: &egui::Painter, origin: egui::Pos2, visuals: &egui::Visuals) {
    let area = painter.clip_rect();
    let [dark, light] = theme::checker_colors(visuals);
    painter.rect_filled(area, 0.0, dark);
    let first = |from: f32, at: f32| ((from - at) / CHECKER).floor() as i64;
    let (x0, y0) = (first(area.min.x, origin.x), first(area.min.y, origin.y));
    let (x1, y1) = (first(area.max.x, origin.x), first(area.max.y, origin.y));
    let mut mesh = egui::Mesh::default();
    for y in y0..=y1 {
        for x in x0..=x1 {
            if (x + y).rem_euclid(2) == 0 {
                continue;
            }
            let min = origin + vec2(x as f32, y as f32) * CHECKER;
            mesh.add_colored_rect(Rect::from_min_size(min, Vec2::splat(CHECKER)), light);
        }
    }
    painter.add(mesh);
}

/// Diagonal lines (rising to the right) over the painter's clip rect, anchored at `origin`.
fn hatch(painter: &egui::Painter, origin: egui::Pos2, color: Color32) {
    let area = painter.clip_rect();
    // Each line is x + y = c, with c on a grid of HATCH from the origin.
    let c0 = origin.x + origin.y;
    let k0 = ((area.min.x + area.min.y - c0) / HATCH).floor() as i64;
    let k1 = ((area.max.x + area.max.y - c0) / HATCH).ceil() as i64;
    let stroke = Stroke::new(1.0, color);
    for k in k0..=k1 {
        let c = c0 + k as f32 * HATCH;
        painter.line_segment(
            [
                pos2(c - area.min.y, area.min.y),
                pos2(c - area.max.y, area.max.y),
            ],
            stroke,
        );
    }
}

/// The name of the AI fill, centred in the largest band it fits in (the first of equal ones: top,
/// bottom, left, right); nowhere if none is big enough.
fn ai_label(painter: &egui::Painter, bands: impl Iterator<Item = Rect>, text: &str) {
    let galley = painter.layout_no_wrap(
        text.into(),
        egui::FontId::proportional(13.0),
        Color32::WHITE,
    );
    let chip = galley.size() + vec2(16.0, 8.0);
    let room = chip + vec2(16.0, 16.0);
    let Some(band) = bands
        .filter(|b| b.width() >= room.x && b.height() >= room.y)
        .reduce(|best, b| if b.area() > best.area() { b } else { best })
    else {
        return;
    };
    let r = Rect::from_center_size(band.center(), chip);
    painter.rect_filled(r, 4.0, theme::OVERLAY);
    painter.galley(r.min + vec2(8.0, 4.0), galley, Color32::WHITE);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> StageFacts {
        StageFacts {
            extend_on: true,
            image: true,
            running: false,
            has_result: false,
            editing: false,
        }
    }

    #[test]
    fn edit_shows_for_an_extended_image_until_there_is_a_result() {
        assert_eq!(stage(facts()), Stage::Edit);
        let done = StageFacts {
            has_result: true,
            ..facts()
        };
        assert_eq!(stage(done), Stage::Result);
        // A change of the extension (or "Adjust canvas") after the result goes back to Edit.
        assert_eq!(
            stage(StageFacts {
                editing: true,
                ..done
            }),
            Stage::Edit
        );
    }

    #[test]
    fn result_shows_without_extend_an_image_or_while_running() {
        for f in [
            StageFacts {
                extend_on: false,
                ..facts()
            },
            StageFacts {
                image: false,
                ..facts()
            },
            StageFacts {
                running: true,
                ..facts()
            },
            StageFacts {
                running: true,
                has_result: true,
                editing: true,
                ..facts()
            },
            StageFacts {
                extend_on: false,
                has_result: true,
                editing: true,
                ..facts()
            },
        ] {
            assert_eq!(stage(f), Stage::Result, "{f:?}");
        }
    }

    #[test]
    fn every_extend_setting_is_in_the_key() {
        let base = Settings::default();
        let key = ExtendKey::of(&base);
        let changes: [fn(&mut Settings); 9] = [
            |s| s.extend_enabled = !s.extend_enabled,
            |s| s.extend_mode = ExtendMode::Aspect,
            |s| s.extend_left += 1.0,
            |s| s.extend_bottom += 1.0,
            |s| s.extend_linked = !s.extend_linked,
            |s| s.extend_aspect = (1, 1),
            |s| s.extend_fill = "mirror".into(),
            |s| s.extend_model = "migan-512-places2".into(),
            |s| s.extend_top += 1.0,
        ];
        for (i, change) in changes.iter().enumerate() {
            let mut s = base.clone();
            change(&mut s);
            assert_ne!(ExtendKey::of(&s), key, "change {i}");
        }
        // Settings outside the extension do not count.
        let other = Settings {
            scale: Some(2),
            format: "jpg".into(),
            ..base
        };
        assert_eq!(ExtendKey::of(&other), key);
    }

    fn extending() -> Settings {
        Settings {
            extend_enabled: true,
            extend_fill: "mirror".into(),
            extend_linked: false,
            extend_left: 128.0,
            extend_top: 0.0,
            extend_right: 128.0,
            extend_bottom: 256.0,
            ..Settings::default()
        }
    }

    #[test]
    fn size_text_names_source_and_canvas() {
        assert_eq!(
            size_text(&extending(), [256, 192]),
            Ok("256×192 to 512×448".into())
        );
        let mut s = extending();
        s.extend_left = 2000.0;
        let e = size_text(&s, [256, 192]).unwrap_err();
        assert!(e.contains("left margin 2000 px"), "{e}");
        // Every character is one the app's fonts have.
        for text in [size_text(&extending(), [256, 192]).unwrap(), e] {
            assert!(
                text.chars().all(|c| c.is_ascii() || "×·…".contains(c)),
                "{text}"
            );
        }
    }

    #[test]
    fn arrows_step_one_pixel_or_ten_with_shift() {
        assert_eq!(arrow_steps(1, 0, 0, 0, false), vec2(1.0, 0.0));
        assert_eq!(arrow_steps(0, 1, 0, 0, false), vec2(-1.0, 0.0));
        assert_eq!(arrow_steps(0, 0, 1, 0, false), vec2(0.0, 1.0));
        assert_eq!(arrow_steps(0, 0, 0, 1, true), vec2(0.0, -10.0));
        assert_eq!(arrow_steps(2, 0, 0, 0, true), vec2(20.0, 0.0));
        assert_eq!(arrow_steps(1, 1, 1, 1, true), Vec2::ZERO);
    }

    fn m(left: u32, top: u32, right: u32, bottom: u32) -> Margins {
        Margins {
            left,
            top,
            right,
            bottom,
        }
    }

    #[test]
    fn arrow_keys_move_the_focused_handle_out_and_in() {
        let src = [256, 192];
        let start = m(10, 10, 10, 10);
        let key = |h, right, left, down, up, shift| {
            step_margins(h, start, arrow_steps(right, left, down, up, shift), src)
        };
        // The way the key points: outward for the right edge, inward for the left edge.
        assert_eq!(key(Handle::Right, 1, 0, 0, 0, false), m(10, 10, 11, 10));
        assert_eq!(key(Handle::Right, 1, 0, 0, 0, true), m(10, 10, 20, 10));
        assert_eq!(key(Handle::Left, 1, 0, 0, 0, false), m(9, 10, 10, 10));
        assert_eq!(key(Handle::Left, 0, 1, 0, 0, true), m(20, 10, 10, 10));
        assert_eq!(key(Handle::Top, 0, 0, 0, 1, false), m(10, 11, 10, 10));
        assert_eq!(key(Handle::Bottom, 0, 0, 0, 1, true), m(10, 10, 10, 0));
        // Across the axis: nothing.
        assert_eq!(key(Handle::Right, 0, 0, 1, 0, false), start);
        assert_eq!(key(Handle::Top, 1, 0, 0, 0, true), start);
        // A corner takes both axes.
        assert_eq!(key(Handle::TopLeft, 0, 1, 0, 1, false), m(11, 11, 10, 10));
        assert_eq!(
            key(Handle::BottomRight, 1, 0, 0, 1, false),
            m(10, 10, 11, 9)
        );
    }

    #[test]
    fn a_screen_reader_increment_moves_outward() {
        let src = [256, 192];
        let z = Margins::default();
        for h in Handle::ALL {
            let out = step_margins(h, z, outward(h), src);
            let s = h.sides();
            assert_eq!(
                out,
                m(
                    u32::from(s.left),
                    u32::from(s.top),
                    u32::from(s.right),
                    u32::from(s.bottom)
                ),
                "{h:?}"
            );
        }
    }

    #[test]
    fn with_the_scale_frozen_the_dragged_edge_tracks_the_pointer() {
        let src = [256, 192];
        let start = m(64, 64, 64, 64);
        let area = vec2(880.0, 640.0);
        let at_start = canvas_edit::canvas_rect(src, start, area, 1.0, Vec2::ZERO);
        for (handle, delta) in [
            (Handle::Right, vec2(40.0, 0.0)),
            (Handle::Right, vec2(-25.0, 3.0)),
            (Handle::Left, vec2(-61.0, 0.0)),
            (Handle::Bottom, vec2(0.0, 33.0)),
            (Handle::TopLeft, vec2(-17.0, -29.0)),
        ] {
            let after = canvas_edit::drag(
                handle,
                start,
                delta,
                at_start.scale,
                DragMods::default(),
                src,
            );
            let l = frozen_layout(at_start.source, at_start.scale, after);
            assert_eq!(l.source, at_start.source);
            let before_at = handle.position(at_start.canvas);
            let now_at = handle.position(l.canvas);
            // The handle moved with the pointer, to within half a source pixel.
            let moved = now_at - before_at;
            let s = handle.sides();
            let want = vec2(
                if s.left || s.right { delta.x } else { 0.0 },
                if s.top || s.bottom { delta.y } else { 0.0 },
            );
            assert!(
                (moved - want).abs().max_elem() <= at_start.scale / 2.0 + 1e-3,
                "{handle:?} {delta:?}: moved {moved:?}"
            );
        }
        // Re-fitting instead would pull the edge back: the wider canvas fits at a smaller scale.
        let wider = canvas_edit::drag(
            Handle::Right,
            start,
            vec2(40.0, 0.0),
            at_start.scale,
            DragMods::default(),
            src,
        );
        let refit = canvas_edit::canvas_rect(src, wider, area, 1.0, Vec2::ZERO);
        assert!(refit.canvas.max.x - at_start.canvas.max.x < 39.0);
    }
}
