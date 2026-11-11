//! The before/after view: the result fills the area, the original shows left of a draggable split.

use eframe::egui::{self, Rect, TextureHandle, TextureOptions, Vec2, vec2};
use scaleup_core::Margins;

use crate::worker::Which;

const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 64.0;

/// Where an image of size `image` lands inside an `area`-sized view. Zoom 1 fits and centres it.
pub fn fit_rect(image: Vec2, area: Vec2, zoom: f32, pan: Vec2) -> Rect {
    let size = image * fit_scale(image, area) * zoom;
    Rect::from_center_size((area / 2.0 + pan).to_pos2(), size)
}

fn fit_scale(image: Vec2, area: Vec2) -> f32 {
    (area.x / image.x.max(1.0)).min(area.y / image.y.max(1.0))
}

/// The zoom and pan that scale the view by `factor` while the point under `cursor` stays put.
pub fn zoom_about(zoom: f32, pan: Vec2, area: Vec2, cursor: Vec2, factor: f32) -> (f32, Vec2) {
    let centre = area / 2.0 + pan;
    let new_centre = cursor + (centre - cursor) * factor;
    (zoom * factor, new_centre - area / 2.0)
}

/// Where a `before`-sized source lands inside `frame`, the on-screen rectangle of the result of
/// extending it by `m`: at `(left, top) × scale`, `scale` being the frame size over the canvas size.
pub fn before_rect(frame: Rect, before: Vec2, m: Margins) -> Rect {
    let canvas = before + vec2((m.left + m.right) as f32, (m.top + m.bottom) as f32);
    let scale = frame.size() / canvas.max(Vec2::splat(1.0));
    Rect::from_min_size(
        frame.min + vec2(m.left as f32, m.top as f32) * scale,
        before * scale,
    )
}

/// The four bands of `outer` around `inner` (top, bottom, left, right); `inner` lies inside `outer`.
pub fn around(outer: Rect, inner: Rect) -> [Rect; 4] {
    let (o, i) = (outer, inner);
    [
        Rect::from_min_max(o.min, egui::pos2(o.max.x, i.min.y)),
        Rect::from_min_max(egui::pos2(o.min.x, i.max.y), o.max),
        Rect::from_min_max(egui::pos2(o.min.x, i.min.y), egui::pos2(i.min.x, i.max.y)),
        Rect::from_min_max(egui::pos2(i.max.x, i.min.y), egui::pos2(o.max.x, i.max.y)),
    ]
}

/// The zoom at which one image pixel covers one physical pixel.
pub fn one_to_one_zoom(image: Vec2, area: Vec2, pixels_per_point: f32) -> f32 {
    1.0 / (pixels_per_point * fit_scale(image, area))
}

struct Side {
    texture: TextureHandle,
    /// Size of the file on disk; the texture may be a downscaled copy.
    size: Vec2,
}

pub struct CompareView {
    before: Option<Side>,
    after: Option<Side>,
    /// Canvas the result added around the "before" image; zero when it was not extended.
    margins: Margins,
    /// Split position across the view, 0..1.
    split: f32,
    zoom: f32,
    pan: Vec2,
}

impl Default for CompareView {
    fn default() -> Self {
        Self {
            before: None,
            after: None,
            margins: Margins::default(),
            split: 0.5,
            zoom: 1.0,
            pan: Vec2::ZERO,
        }
    }
}

impl CompareView {
    /// Forgets both images and the view position.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Uploads a decoded preview. A new "before" also retires the old result. `margins` is the
    /// canvas a result added around the "before" image (zero for a "before").
    pub fn set(
        &mut self,
        ctx: &egui::Context,
        which: Which,
        image: egui::ColorImage,
        size: [u32; 2],
        margins: Margins,
    ) {
        // Sharp pixels when magnified (a 20×20 icon stays readable), smooth when shrunk.
        let options = TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            ..TextureOptions::LINEAR
        };
        let name = match which {
            Which::Before => "before",
            Which::After => "after",
        };
        let side = Side {
            texture: ctx.load_texture(name, image, options),
            size: vec2(size[0] as f32, size[1] as f32),
        };
        match which {
            Which::Before => {
                self.before = Some(side);
                self.after = None;
                self.margins = Margins::default();
            }
            Which::After => {
                self.after = Some(side);
                self.margins = margins;
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.before.is_none() && self.after.is_none()
    }

    fn handle_input(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: Rect) {
        if resp.hovered() {
            let delta = ui.input(|i| i.smooth_scroll_delta.y);
            if delta != 0.0
                && let Some(pos) = resp.hover_pos()
            {
                let target = (self.zoom * (delta / 200.0).exp()).clamp(MIN_ZOOM, MAX_ZOOM);
                let (zoom, pan) = zoom_about(
                    self.zoom,
                    self.pan,
                    rect.size(),
                    pos - rect.min,
                    target / self.zoom,
                );
                self.zoom = zoom;
                self.pan = pan;
            }
        }
        if resp.dragged_by(egui::PointerButton::Middle)
            || resp.dragged_by(egui::PointerButton::Secondary)
        {
            self.pan += resp.drag_delta();
        }
        if resp.dragged_by(egui::PointerButton::Primary)
            && let Some(pos) = resp.interact_pointer_pos()
        {
            self.split = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let Some(native) = self.after.as_ref().or(self.before.as_ref()).map(|s| s.size) else {
            return;
        };
        self.handle_input(ui, &resp, rect);

        let painter = ui.painter_at(rect);
        let image_rect =
            fit_rect(native, rect.size(), self.zoom, self.pan).translate(rect.min.to_vec2());
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        let white = egui::Color32::WHITE;
        let split_x = rect.left() + rect.width() * self.split;

        if let Some(after) = &self.after {
            painter.image(after.texture.id(), image_rect, uv, white);
        }
        if let Some(before) = &self.before {
            let (clip, before_r) = if self.after.is_some() {
                let mut r = rect;
                r.set_right(split_x);
                (r, before_rect(image_rect, before.size, self.margins))
            } else {
                (rect, image_rect)
            };
            let painter = painter.with_clip_rect(clip);
            painter.image(before.texture.id(), before_r, uv, white);
            // On the "before" side, the canvas the extension added shows the result, dimmed.
            if self.after.is_some() && self.margins != Margins::default() {
                for band in around(image_rect, before_r) {
                    painter.rect_filled(band, 0.0, egui::Color32::from_black_alpha(150));
                }
            }
        }

        let label = |pos: egui::Pos2, align: egui::Align2, text: &str| {
            let galley_rect = painter.text(
                pos,
                align,
                text,
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
            painter.rect_filled(
                galley_rect.expand(4.0),
                4.0,
                egui::Color32::from_black_alpha(140),
            );
            painter.text(
                pos,
                align,
                text,
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
        };
        if self.after.is_some() && self.before.is_some() {
            let stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
            painter.line_segment(
                [
                    egui::pos2(split_x, rect.top()),
                    egui::pos2(split_x, rect.bottom()),
                ],
                stroke,
            );
            painter.circle_filled(
                egui::pos2(split_x, rect.center().y),
                7.0,
                egui::Color32::WHITE,
            );
            if resp
                .hover_pos()
                .is_some_and(|p| (p.x - split_x).abs() < 8.0)
            {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            label(
                rect.left_top() + vec2(12.0, 12.0),
                egui::Align2::LEFT_TOP,
                "Before",
            );
            label(
                rect.right_top() + vec2(-12.0, 12.0),
                egui::Align2::RIGHT_TOP,
                "After",
            );
        } else if self.before.is_some() {
            label(
                rect.left_top() + vec2(12.0, 12.0),
                egui::Align2::LEFT_TOP,
                "Before",
            );
        }

        let toolbar = Rect::from_min_size(rect.left_bottom() + vec2(8.0, -32.0), vec2(90.0, 24.0));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(toolbar));
        child.horizontal(|ui| {
            if ui.button("Fit").clicked() {
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
            }
            if ui.button("1:1").clicked() {
                self.zoom = one_to_one_zoom(native, rect.size(), ui.ctx().pixels_per_point());
                self.pan = Vec2::ZERO;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2};

    fn margins(left: u32, top: u32, right: u32, bottom: u32) -> Margins {
        Margins {
            left,
            top,
            right,
            bottom,
        }
    }

    #[test]
    fn before_rect_without_margins_is_the_frame() {
        let frame = Rect::from_min_size(pos2(10.0, 20.0), vec2(400.0, 300.0));
        assert_eq!(
            before_rect(frame, vec2(100.0, 75.0), Margins::default()),
            frame
        );
    }

    #[test]
    fn before_rect_sits_at_its_offset_inside_the_extended_frame() {
        // 1200×800 extended by 200 left and right is a 1600×800 canvas, shown 800×400 at (10, 20):
        // half size, so the source starts 100 points in and is 600×400.
        let frame = Rect::from_min_size(pos2(10.0, 20.0), vec2(800.0, 400.0));
        assert_eq!(
            before_rect(frame, vec2(1200.0, 800.0), margins(200, 0, 200, 0)),
            Rect::from_min_size(pos2(110.0, 20.0), vec2(600.0, 400.0))
        );
        // Uneven margins on both axes: canvas 100+10+30 × 50+5+15 = 140×70, shown at 2×.
        let frame = Rect::from_min_size(pos2(0.0, 0.0), vec2(280.0, 140.0));
        assert_eq!(
            before_rect(frame, vec2(100.0, 50.0), margins(10, 5, 30, 15)),
            Rect::from_min_size(pos2(20.0, 10.0), vec2(200.0, 100.0))
        );
    }

    #[test]
    fn around_covers_the_frame_outside_the_inner_rect() {
        let outer = Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 100.0));
        let inner = Rect::from_min_max(pos2(10.0, 20.0), pos2(90.0, 80.0));
        assert_eq!(
            around(outer, inner),
            [
                Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 20.0)),
                Rect::from_min_max(pos2(0.0, 80.0), pos2(100.0, 100.0)),
                Rect::from_min_max(pos2(0.0, 20.0), pos2(10.0, 80.0)),
                Rect::from_min_max(pos2(90.0, 20.0), pos2(100.0, 80.0)),
            ]
        );
        // Nothing around a rect that fills the frame.
        assert!(around(outer, outer).iter().all(|r| r.area() == 0.0));
    }

    #[test]
    fn fit_rect_letterboxes() {
        let r = fit_rect(vec2(4000.0, 3000.0), vec2(800.0, 800.0), 1.0, Vec2::ZERO);
        assert_eq!(r, Rect::from_min_max(pos2(0.0, 100.0), pos2(800.0, 700.0)));
    }

    #[test]
    fn fit_rect_zoom_2x() {
        let r = fit_rect(vec2(4000.0, 3000.0), vec2(800.0, 800.0), 2.0, Vec2::ZERO);
        assert_eq!(
            r,
            Rect::from_min_max(pos2(-400.0, -200.0), pos2(1200.0, 1000.0))
        );
    }

    #[test]
    fn fit_rect_applies_pan() {
        let r = fit_rect(
            vec2(100.0, 100.0),
            vec2(100.0, 100.0),
            1.0,
            vec2(10.0, -5.0),
        );
        assert_eq!(r.min, pos2(10.0, -5.0));
    }

    #[test]
    fn zoom_about_keeps_the_point_under_the_cursor() {
        let (image, area) = (vec2(400.0, 300.0), vec2(800.0, 600.0));
        let (zoom, pan) = (1.5, vec2(30.0, -20.0));
        let cursor = vec2(100.0, 200.0);
        let before = fit_rect(image, area, zoom, pan);
        let u = (cursor - before.min.to_vec2()) / before.size();
        let (z2, p2) = zoom_about(zoom, pan, area, cursor, 2.0);
        assert_eq!(z2, 3.0);
        let after = fit_rect(image, area, z2, p2);
        let u2 = (cursor - after.min.to_vec2()) / after.size();
        assert!((u - u2).length() < 1e-5, "{u:?} {u2:?}");
    }

    #[test]
    fn one_to_one_zoom_maps_image_pixels_to_physical_pixels() {
        // 1600 px wide fits an 800 pt area at zoom 1; with 2 px per point that is already 1:1.
        assert_eq!(
            one_to_one_zoom(vec2(1600.0, 100.0), vec2(800.0, 800.0), 2.0),
            1.0
        );
        assert_eq!(
            one_to_one_zoom(vec2(1600.0, 100.0), vec2(800.0, 800.0), 1.0),
            2.0
        );
    }
}
