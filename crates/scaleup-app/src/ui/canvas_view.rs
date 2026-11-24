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

use eframe::egui::{self, Color32, Key, Pos2, Rect, Stroke, TextureHandle, Vec2, pos2, vec2};
use scaleup_core::{Fill, MarginSpec, Margins, Registry};

use crate::state::{Settings, fill_title};
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

/// What of the extend settings changes the output, after a result: a change sends the image area
/// back to Edit. Only what a run would do differently counts, so ticking "Same on every side" over
/// equal values, typing side values while in Aspect mode, or picking an inpainting model under a
/// plain fill does not.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendKey {
    enabled: bool,
    /// The margins in pixels for the image; the fields as the run reads them when they do not
    /// resolve (or the image size is not known).
    margins: Result<Margins, MarginSpec>,
    /// The fill; its text when it cannot be read.
    fill: Result<Fill, String>,
    /// The inpainting model, for the AI fill only.
    model: Option<String>,
}

impl ExtendKey {
    /// The key of `s` for an image of size `source`.
    pub fn of(s: &Settings, source: Option<[u32; 2]>) -> Self {
        let spec = s.margin_spec();
        let margins = match source.and_then(|[w, h]| spec.resolve(w, h).ok()) {
            Some(m) => Ok(m),
            None => Err(spec),
        };
        let fill = Fill::parse(&s.extend_fill).map_err(|_| s.extend_fill.clone());
        let model = (fill == Ok(Fill::Ai)).then(|| s.extend_model.clone());
        Self {
            enabled: s.extend_enabled,
            margins,
            fill,
            model,
        }
    }
}

/// The size chip under the canvas: `256×192 to 512×448`, a hint while no canvas is added yet, or
/// why the settings cannot be used for this image.
pub fn size_text(s: &Settings, source: [u32; 2]) -> Result<String, String> {
    let [w, h] = source;
    match canvas_edit::current_margins(s, source) {
        Some(m) if m == Margins::default() => Ok("Drag a handle to add canvas".into()),
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

/// What a screen reader says as a handle's value: the margin it moves, `96 px`, or both of a
/// corner, `left 96 px, top 24 px`.
pub fn handle_value(handle: Handle, m: Margins) -> String {
    let s = handle.sides();
    let horizontal = if s.left {
        Some(("left", m.left))
    } else if s.right {
        Some(("right", m.right))
    } else {
        None
    };
    let vertical = if s.top {
        Some(("top", m.top))
    } else if s.bottom {
        Some(("bottom", m.bottom))
    } else {
        None
    };
    match (horizontal, vertical) {
        (Some((h, x)), Some((v, y))) => format!("{h} {x} px, {v} {y} px"),
        (Some((_, n)), None) | (None, Some((_, n))) => format!("{n} px"),
        (None, None) => String::new(),
    }
}

/// The part of a textured quad (`screen`, drawn with the texture rectangle `uv`) that lies inside
/// `visible`, with its texture rectangle cut to match; `None` when none of it shows. A flipped or
/// degenerate `uv` stays so. Keeps the vertices of a zoomed-in preview near the view.
pub fn clip_quad(screen: Rect, uv: Rect, visible: Rect) -> Option<(Rect, Rect)> {
    let r = screen.intersect(visible);
    if !r.is_positive() {
        return None;
    }
    let at = |p: Pos2| {
        let t = (p - screen.min) / screen.size();
        pos2(
            uv.min.x + (uv.max.x - uv.min.x) * t.x,
            uv.min.y + (uv.max.y - uv.min.y) * t.y,
        )
    };
    Some((r, Rect::from_min_max(at(r.min), at(r.max))))
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
    /// The layout drawn last, in screen coordinates (tests aim the pointer with it).
    #[cfg(test)]
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

    /// The layout drawn last, in screen coordinates.
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
        if self.drag.is_none() {
            wheel_zoom(ui, &resp, Self::fit_area(rect), zoom, pan);
        }
        if resp.dragged_by(egui::PointerButton::Middle)
            || resp.dragged_by(egui::PointerButton::Secondary)
        {
            *pan += resp.drag_delta();
        }
        let mut changed = self.pointer(ui, &resp, layout, before, source, settings);

        let margins = canvas_edit::planned_margins(settings, source);
        let layout = self.layout(rect, source, margins, *zoom, *pan);
        let focused = self.keyboard(ui, layout, margins, source, settings);
        changed |= focused.1;
        let layout = if focused.1 {
            let m = canvas_edit::planned_margins(settings, source);
            self.layout(rect, source, m, *zoom, *pan)
        } else {
            layout
        };

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
        let active = |h| hovered == Some(h) || focused.0 == Some(h);
        paint(
            &painter,
            ui.visuals(),
            layout,
            texture,
            settings,
            registry,
            active,
        );
        #[cfg(test)]
        {
            self.last = Some(layout);
        }
        changed
    }

    /// A primary press on a handle starts a drag (and focuses the handle); while it lasts the
    /// margins follow the pointer from the values frozen at the start. Returns true when it wrote
    /// the settings.
    fn pointer(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        layout: CanvasLayout,
        before: Margins,
        source: [u32; 2],
        settings: &mut Settings,
    ) -> bool {
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
        let Some(d) = self.drag else {
            return false;
        };
        let mut changed = false;
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
        changed
    }

    /// The eight handles as focusable widgets with a name, a value and the screen-reader steps they
    /// take; the focused one moves with the arrow keys. Returns the focused handle and whether the
    /// settings were written.
    fn keyboard(
        &self,
        ui: &egui::Ui,
        layout: CanvasLayout,
        margins: Margins,
        source: [u32; 2],
        settings: &mut Settings,
    ) -> (Option<Handle>, bool) {
        let mut focused = None;
        let mut changed = false;
        for handle in Handle::ALL {
            let at = handle.position(layout.canvas);
            let r = ui.interact(
                Rect::from_center_size(at, Vec2::splat(HIT_SIZE)),
                handle_id(handle),
                egui::Sense::FOCUSABLE,
            );
            r.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Splitter, true, handle.name()));
            let grows = step_margins(handle, margins, outward(handle), source) != margins;
            let shrinks = step_margins(handle, margins, -outward(handle), source) != margins;
            ui.ctx().accesskit_node_builder(r.id, |node| {
                use egui::accesskit::Action;
                node.set_value(handle_value(handle, margins));
                if grows {
                    node.add_action(Action::Increment);
                }
                if shrinks {
                    node.add_action(Action::Decrement);
                }
            });
            let steps = ui.input(|i| {
                use egui::accesskit::Action;
                let reader = i.num_accesskit_action_requests(r.id, Action::Increment) as f32
                    - i.num_accesskit_action_requests(r.id, Action::Decrement) as f32;
                outward(handle) * reader
            });
            let steps = if r.has_focus() {
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
                steps
                    + ui.input(|i| {
                        arrow_steps(
                            i.num_presses(Key::ArrowRight),
                            i.num_presses(Key::ArrowLeft),
                            i.num_presses(Key::ArrowDown),
                            i.num_presses(Key::ArrowUp),
                            i.modifiers.shift,
                        )
                    })
            } else {
                steps
            };
            if steps != Vec2::ZERO && !changed {
                let m = step_margins(handle, margins, steps, source);
                if m != margins {
                    canvas_edit::write_margins(settings, m, source);
                    changed = true;
                }
            }
        }
        (focused, changed)
    }
}

// ------------------------------------------------------------------------------------ drawing

/// Everything of the stage but the input: the fill preview, the source, the outlines and the
/// handles (`active` says which one is being acted on).
fn paint(
    painter: &egui::Painter,
    visuals: &egui::Visuals,
    layout: CanvasLayout,
    texture: &TextureHandle,
    settings: &Settings,
    registry: &Registry,
    active: impl Fn(Handle) -> bool,
) {
    paint_fill(painter, visuals, layout, texture, settings, registry);
    let full = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    if let Some((screen, uv)) = clip_quad(layout.source, full, painter.clip_rect()) {
        painter.image(texture.id(), screen, uv, Color32::WHITE);
    }
    outline(painter, layout.canvas);
    if layout.source != layout.canvas {
        outline(painter, layout.source);
    }
    for (handle, r) in canvas_edit::handles(layout.canvas) {
        paint_handle(painter, r, active(handle));
    }
}

/// A 1 px white line on a 3 px dark one: the border reads on any picture, light or dark, as the
/// split line does.
fn outline(painter: &egui::Painter, r: Rect) {
    painter.rect_stroke(
        r,
        0.0,
        Stroke::new(3.0, theme::OUTLINE_EDGE),
        egui::StrokeKind::Middle,
    );
    painter.rect_stroke(
        r,
        0.0,
        Stroke::new(1.0, Color32::WHITE),
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
                // Cut to the view: at a high zoom the bands reach far past it.
                if let Some((screen, uv)) = clip_quad(screen, uv, visible) {
                    painter.image(texture.id(), screen, uv, Color32::WHITE);
                }
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
    use crate::state::ExtendMode;

    #[test]
    fn handle_values_name_the_margins_they_move() {
        let m = Margins {
            left: 96,
            top: 24,
            right: 7,
            bottom: 0,
        };
        assert_eq!(handle_value(Handle::Left, m), "96 px");
        assert_eq!(handle_value(Handle::Bottom, m), "0 px");
        assert_eq!(handle_value(Handle::TopLeft, m), "left 96 px, top 24 px");
        assert_eq!(
            handle_value(Handle::BottomRight, m),
            "right 7 px, bottom 0 px"
        );
    }

    #[test]
    fn clipped_quads_keep_their_texture_in_place() {
        let r = |x0: f32, y0: f32, x1: f32, y1: f32| Rect::from_min_max(pos2(x0, y0), pos2(x1, y1));
        let view = r(0.0, 0.0, 100.0, 100.0);
        // Half outside on the left: the right half of the texture.
        assert_eq!(
            clip_quad(r(-100.0, 0.0, 100.0, 100.0), r(0.0, 0.0, 1.0, 1.0), view),
            Some((view, r(0.5, 0.0, 1.0, 1.0)))
        );
        // A flipped texture stays flipped; a degenerate one stays a line.
        assert_eq!(
            clip_quad(r(-100.0, 0.0, 100.0, 100.0), r(1.0, 0.0, 0.0, 1.0), view),
            Some((view, r(0.5, 0.0, 0.0, 1.0)))
        );
        assert_eq!(
            clip_quad(r(50.0, -300.0, 150.0, 100.0), r(0.0, 1.0, 0.0, 0.0), view),
            Some((r(50.0, 0.0, 100.0, 100.0), r(0.0, 0.25, 0.0, 0.0)))
        );
        // Outside or empty: nothing.
        assert_eq!(clip_quad(r(200.0, 0.0, 300.0, 100.0), view, view), None);
        assert_eq!(clip_quad(r(10.0, 10.0, 10.0, 90.0), view, view), None);
        // Far past the view at a high zoom: the vertices land on the view's edge.
        let huge = r(-1.0e7, -1.0e7, 1.0e7, 1.0e7);
        let (screen, uv) = clip_quad(huge, r(0.0, 0.0, 1.0, 1.0), view).unwrap();
        assert_eq!(screen, view);
        assert!((uv.min.x - 0.5).abs() < 1e-4 && (uv.max.x - 0.5).abs() < 1e-4);
    }

    #[test]
    fn with_no_canvas_added_the_chip_says_what_to_do() {
        let s = Settings {
            extend_enabled: true,
            extend_left: 0.0,
            ..Settings::default()
        };
        assert_eq!(
            size_text(&s, [256, 192]),
            Ok("Drag a handle to add canvas".into())
        );
    }

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

    const SRC: Option<[u32; 2]> = Some([256, 192]);

    #[test]
    fn what_changes_the_output_is_in_the_key() {
        let base = Settings {
            extend_enabled: true,
            ..Settings::default()
        };
        let key = ExtendKey::of(&base, SRC);
        let changes: [fn(&mut Settings); 7] = [
            |s| s.extend_enabled = false,
            |s| s.extend_mode = ExtendMode::Aspect,
            |s| s.extend_left += 1.0,
            // Unlinked with a different side: the bottom margin changes.
            |s| {
                s.extend_linked = false;
                s.extend_bottom = 10.0;
            },
            |s| s.extend_mode = ExtendMode::Percent,
            |s| s.extend_fill = "mirror".into(),
            // The model of the AI fill paints the canvas.
            |s| s.extend_model = "migan-512-places2".into(),
        ];
        for (i, change) in changes.iter().enumerate() {
            let mut s = base.clone();
            change(&mut s);
            assert_ne!(ExtendKey::of(&s, SRC), key, "change {i}");
        }
        // Settings outside the extension do not count.
        let other = Settings {
            scale: Some(2),
            format: "jpg".into(),
            ..base
        };
        assert_eq!(ExtendKey::of(&other, SRC), key);
    }

    #[test]
    fn what_does_not_change_the_output_is_not_in_the_key() {
        let base = Settings {
            extend_enabled: true,
            extend_fill: "mirror".into(),
            ..Settings::default()
        };
        let key = ExtendKey::of(&base, SRC);
        let same: [fn(&mut Settings); 4] = [
            // "Same on every side" over four equal values.
            |s| s.extend_linked = false,
            // Hidden side fields while linked.
            |s| s.extend_top = 5.0,
            // The inpainting model under a plain fill.
            |s| s.extend_model = "migan-512-places2".into(),
            // The same fill written another way.
            |s| s.extend_fill = "Mirror".into(),
        ];
        for (i, change) in same.iter().enumerate() {
            let mut s = base.clone();
            change(&mut s);
            assert_eq!(ExtendKey::of(&s, SRC), key, "change {i}");
        }
        // Side values typed while in Aspect mode do not move an aspect canvas.
        let aspect = Settings {
            extend_mode: ExtendMode::Aspect,
            ..base.clone()
        };
        let mut typed = aspect.clone();
        typed.extend_linked = false;
        typed.extend_right = 300.0;
        assert_eq!(ExtendKey::of(&typed, SRC), ExtendKey::of(&aspect, SRC));
        // Percent that gives the same pixels on this image is the same canvas.
        let pixels = Settings {
            extend_linked: false,
            extend_left: 64.0,
            extend_top: 48.0,
            extend_right: 64.0,
            extend_bottom: 48.0,
            ..base.clone()
        };
        let percent = Settings {
            extend_mode: ExtendMode::Percent,
            extend_linked: true,
            extend_left: 25.0,
            ..base.clone()
        };
        assert_eq!(ExtendKey::of(&pixels, SRC), ExtendKey::of(&percent, SRC));
        // Without a size, the fields count as they are.
        assert_ne!(ExtendKey::of(&pixels, None), ExtendKey::of(&percent, None));
        // Over the limits: the fields count, so typing on in a bad value is still a change.
        let mut bad = base.clone();
        bad.extend_left = 5000.0;
        let bad_key = ExtendKey::of(&bad, SRC);
        bad.extend_left = 6000.0;
        assert_ne!(ExtendKey::of(&bad, SRC), bad_key);
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
