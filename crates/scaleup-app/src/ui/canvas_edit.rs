//! Geometry of dragging the image border to add canvas: layout, handles, hit test, drag to margins,
//! writing margins back to the settings, and the texture rectangles of the mirror/edge previews.
//! No window is needed: everything here is plain arithmetic under test.

use eframe::egui::{CursorIcon, Modifiers, Pos2, Rect, Vec2, pos2, vec2};
use scaleup_core::{MarginSpec, Margins};

use crate::state::{ExtendMode, Settings};
use crate::ui::compare::fit_rect;

/// Side of the square drawn for a handle, in points.
pub const HANDLE_SIZE: f32 = 10.0;
/// Side of the square that grabs a handle, in points: larger than the drawing, easy to hit.
pub const HIT_SIZE: f32 = 24.0;

/// Same limits as `MarginSpec::resolve` in the core (they are private there): each margin at most
/// this many times the side it extends, and the canvas at most `MAX_CANVAS_SIDE` per side.
const MAX_MARGIN_FACTOR: i64 = 4;
const MAX_CANVAS_SIDE: i64 = 16384;
/// A drag is bounded to this many source pixels, far beyond any limit, before it is used.
const PIXEL_BOUND: f64 = 2_147_483_648.0;

// ------------------------------------------------------------------------------------- layout

/// Where the planned canvas and the source image are on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasLayout {
    /// The whole planned canvas (source plus margins). Its edge is the one the handles sit on.
    pub canvas: Rect,
    /// The source image inside the canvas.
    pub source: Rect,
    /// Screen points per source pixel.
    pub scale: f32,
}

/// Lays out the planned canvas of a `source`-sized image extended by `margins` in a view of size
/// `area`. The canvas is fitted like an image in [`fit_rect`] (zoom 1 fits and centres it), so the
/// rectangles are relative to the view's top-left corner: the caller adds the view's origin.
pub fn canvas_rect(
    source: [u32; 2],
    margins: Margins,
    area: Vec2,
    zoom: f32,
    pan: Vec2,
) -> CanvasLayout {
    let canvas_px = vec2(
        source[0] as f32 + (margins.left + margins.right) as f32,
        source[1] as f32 + (margins.top + margins.bottom) as f32,
    );
    let canvas = fit_rect(canvas_px, area, zoom, pan);
    // `fit_rect` treats an empty side as 1 px; do the same so the scale never divides by zero.
    let scale = canvas.width() / canvas_px.x.max(1.0);
    let source_rect = Rect::from_min_size(
        canvas.min + vec2(margins.left as f32, margins.top as f32) * scale,
        vec2(source[0] as f32, source[1] as f32) * scale,
    );
    CanvasLayout {
        canvas,
        source: source_rect,
        scale,
    }
}

// ------------------------------------------------------------------------------------ handles

/// The sides a handle moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sides {
    pub left: bool,
    pub top: bool,
    pub right: bool,
    pub bottom: bool,
}

/// One of the eight grips on the border: four sides and four corners.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handle {
    Left,
    Top,
    Right,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Handle {
    pub const ALL: [Handle; 8] = [
        Handle::Left,
        Handle::Top,
        Handle::Right,
        Handle::Bottom,
        Handle::TopLeft,
        Handle::TopRight,
        Handle::BottomLeft,
        Handle::BottomRight,
    ];

    /// Accessible name, as a screen reader says it.
    pub fn name(self) -> &'static str {
        match self {
            Handle::Left => "Left edge",
            Handle::Top => "Top edge",
            Handle::Right => "Right edge",
            Handle::Bottom => "Bottom edge",
            Handle::TopLeft => "Top-left corner",
            Handle::TopRight => "Top-right corner",
            Handle::BottomLeft => "Bottom-left corner",
            Handle::BottomRight => "Bottom-right corner",
        }
    }

    /// The pointer shape that shows which way the handle drags.
    pub fn cursor(self) -> CursorIcon {
        match self {
            Handle::Left | Handle::Right => CursorIcon::ResizeHorizontal,
            Handle::Top | Handle::Bottom => CursorIcon::ResizeVertical,
            Handle::TopLeft | Handle::BottomRight => CursorIcon::ResizeNwSe,
            Handle::TopRight | Handle::BottomLeft => CursorIcon::ResizeNeSw,
        }
    }

    /// The margins this handle moves: one for a side, two for a corner.
    pub fn sides(self) -> Sides {
        let (left, top, right, bottom) = match self {
            Handle::Left => (true, false, false, false),
            Handle::Top => (false, true, false, false),
            Handle::Right => (false, false, true, false),
            Handle::Bottom => (false, false, false, true),
            Handle::TopLeft => (true, true, false, false),
            Handle::TopRight => (false, true, true, false),
            Handle::BottomLeft => (true, false, false, true),
            Handle::BottomRight => (false, false, true, true),
        };
        Sides {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Where the handle sits on the border of `rect`: a side at the middle of its edge, a corner at
    /// the corner. The handles you drag are on the outer edge of the planned canvas, so pass the
    /// canvas rectangle ([`CanvasLayout::canvas`]), not the source rectangle. With no margins the
    /// two coincide and the handles sit on the picture's own border.
    pub fn position(self, rect: Rect) -> Pos2 {
        let c = rect.center();
        match self {
            Handle::Left => pos2(rect.min.x, c.y),
            Handle::Top => pos2(c.x, rect.min.y),
            Handle::Right => pos2(rect.max.x, c.y),
            Handle::Bottom => pos2(c.x, rect.max.y),
            Handle::TopLeft => rect.min,
            Handle::TopRight => pos2(rect.max.x, rect.min.y),
            Handle::BottomLeft => pos2(rect.min.x, rect.max.y),
            Handle::BottomRight => rect.max,
        }
    }

    fn is_corner(self) -> bool {
        matches!(
            self,
            Handle::TopLeft | Handle::TopRight | Handle::BottomLeft | Handle::BottomRight
        )
    }
}

/// The squares to draw, one per handle on the border of `canvas` ([`HANDLE_SIZE`] points each).
pub fn handles(canvas: Rect) -> [(Handle, Rect); 8] {
    Handle::ALL.map(|h| {
        (
            h,
            Rect::from_center_size(h.position(canvas), Vec2::splat(HANDLE_SIZE)),
        )
    })
}

/// The handle under `pos`, grabbed within a [`HIT_SIZE`] square around it. Where areas overlap a
/// corner wins over a side; between two of a kind the nearer one wins (small canvases).
///
/// Along an edge a handle's area reaches at most a third of the edge's length each way, so on a
/// canvas smaller than about 72 points the middle third of every side stays reachable: the
/// corners' areas cannot swallow it.
pub fn hit(pos: Pos2, canvas: Rect) -> Option<Handle> {
    let half = HIT_SIZE / 2.0;
    let along_x = half.min(canvas.width() / 3.0);
    let along_y = half.min(canvas.height() / 3.0);
    Handle::ALL
        .into_iter()
        .filter_map(|h| {
            let at = h.position(canvas);
            let reach = match h {
                Handle::Left | Handle::Right => vec2(half, along_y),
                Handle::Top | Handle::Bottom => vec2(along_x, half),
                _ => vec2(along_x, along_y),
            };
            Rect::from_center_size(at, reach * 2.0)
                .contains(pos)
                .then(|| (!h.is_corner(), (pos - at).length_sq(), h))
        })
        .min_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)))
        .map(|(_, _, h)| h)
}

// --------------------------------------------------------------------------------------- drag

/// Modifier keys held during a drag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DragMods {
    /// Shift: the dragged amount goes to all four sides.
    pub all_sides: bool,
    /// Alt: the opposite side (for a corner, the opposite sides) moves by the same amount.
    pub mirror: bool,
}

impl DragMods {
    pub fn from_egui(m: Modifiers) -> Self {
        Self {
            all_sides: m.shift,
            mirror: m.alt,
        }
    }
}

/// How far, in whole pixels, one axis grows and which of its two margins move.
struct Axis {
    low: bool,
    high: bool,
    grow: i64,
}

/// The largest growth that keeps one axis within the limits of `MarginSpec::resolve`, for the
/// margins that move (`low`, `high`) of an axis whose image side is `side` pixels.
fn axis_max(start_low: i64, start_high: i64, low: bool, high: bool, side: i64) -> i64 {
    let each = MAX_MARGIN_FACTOR * side;
    let room = (MAX_CANVAS_SIDE - side).max(0) - start_low - start_high;
    let mut max = i64::MAX;
    if low {
        max = max.min(each - start_low);
    }
    if high {
        max = max.min(each - start_high);
    }
    match (low, high) {
        (true, true) => max.min(room.div_euclid(2)),
        (false, false) => max,
        _ => max.min(room),
    }
}

/// Margins after dragging `handle` by `delta_screen` points, from the margins `start` it began with.
///
/// `scale` is the screen points per source pixel ([`CanvasLayout::scale`]); the drag is rounded to
/// whole source pixels. Outward grows a margin, inward shrinks it, never below 0, and never past
/// what `MarginSpec::resolve` accepts for a `source`-sized image (4x the side, 16384 px canvas),
/// so the result always resolves.
///
/// With `all_sides` the dragged amount goes to all four margins; a corner has two amounts, and the
/// larger one counts. With `mirror` the opposite margin moves by the same amount. `all_sides`
/// wins when both are held. A margin that cannot shrink further stops at 0 on its own while the
/// others go on; the growth limit is shared, so the sides stay in step.
///
/// `start` is expected to resolve already; if it does not, the result is cut back until it does.
///
/// `start` and `scale` must be the values frozen when the drag began, with `delta_screen` the
/// pointer's total movement since then. The layout re-fits as the margins change, so the live
/// margins and scale would feed back into the next frame's result.
pub fn drag(
    handle: Handle,
    start: Margins,
    delta_screen: Vec2,
    scale: f32,
    mods: DragMods,
    source: [u32; 2],
) -> Margins {
    let to_pixels = |v: f32| {
        if scale.is_finite() && scale > 0.0 && v.is_finite() {
            // Bounded so that negating it below can never overflow.
            (f64::from(v) / f64::from(scale))
                .round()
                .clamp(-PIXEL_BOUND, PIXEL_BOUND) as i64
        } else {
            0
        }
    };
    let sides = handle.sides();
    // Growth is outward: left and top grow when the pointer goes left and up.
    let dx = if sides.left {
        Some(-to_pixels(delta_screen.x))
    } else if sides.right {
        Some(to_pixels(delta_screen.x))
    } else {
        None
    };
    let dy = if sides.top {
        Some(-to_pixels(delta_screen.y))
    } else if sides.bottom {
        Some(to_pixels(delta_screen.y))
    } else {
        None
    };
    let [w, h] = source.map(i64::from);
    let s = [start.left, start.top, start.right, start.bottom].map(i64::from);

    let (horizontal, vertical) = if mods.all_sides {
        let grow = match (dx, dy) {
            (Some(x), Some(y)) if y.abs() > x.abs() => y,
            (Some(x), _) => x,
            (None, y) => y.unwrap_or(0),
        };
        let all = |grow| Axis {
            low: true,
            high: true,
            grow,
        };
        (Some(all(grow)), Some(all(grow)))
    } else {
        let axis = |grow: Option<i64>, low: bool, high: bool| {
            grow.map(|grow| Axis {
                low: low || mods.mirror,
                high: high || mods.mirror,
                grow,
            })
        };
        (
            axis(dx, sides.left, sides.right),
            axis(dy, sides.top, sides.bottom),
        )
    };

    // One growth limit for everything that moves together (all four sides share it).
    let mut limit = i64::MAX;
    if let Some(a) = &horizontal {
        limit = limit.min(axis_max(s[0], s[2], a.low, a.high, w));
    }
    if let Some(a) = &vertical {
        limit = limit.min(axis_max(s[1], s[3], a.low, a.high, h));
    }
    let shared = mods.all_sides;
    let apply = |a: &Option<Axis>, low: i64, high: i64, side: i64| -> [i64; 2] {
        let Some(a) = a else { return [low, high] };
        let max = if shared {
            limit
        } else {
            axis_max(low, high, a.low, a.high, side)
        };
        let grow = a.grow.min(max);
        [
            if a.low { (low + grow).max(0) } else { low },
            if a.high { (high + grow).max(0) } else { high },
        ]
    };
    let [left, right] = apply(&horizontal, s[0], s[2], w);
    let [top, bottom] = apply(&vertical, s[1], s[3], h);
    fit_limits(
        Margins {
            left: left as u32,
            top: top as u32,
            right: right as u32,
            bottom: bottom as u32,
        },
        source,
    )
}

/// Cuts margins back to what `MarginSpec::resolve` accepts: a safety net for a start that was
/// already outside the limits. Valid margins pass through unchanged.
fn fit_limits(m: Margins, source: [u32; 2]) -> Margins {
    let fit = |low: u32, high: u32, side: u32| -> [u32; 2] {
        let side = i64::from(side);
        let each = MAX_MARGIN_FACTOR * side;
        let (mut low, mut high) = (i64::from(low).min(each), i64::from(high).min(each));
        let room = (MAX_CANVAS_SIDE - side).max(0);
        let mut excess = low + high - room;
        for m in if low >= high {
            [&mut low, &mut high]
        } else {
            [&mut high, &mut low]
        } {
            let cut = excess.clamp(0, *m);
            *m -= cut;
            excess -= cut;
        }
        [low as u32, high as u32]
    };
    let [left, right] = fit(m.left, m.right, source[0]);
    let [top, bottom] = fit(m.top, m.bottom, source[1]);
    Margins {
        left,
        top,
        right,
        bottom,
    }
}

// ---------------------------------------------------------------------------------- settings

/// The margins the settings ask for on a `source`-sized image, in pixels: `None` when extension
/// is off or the settings do not resolve (bad fill, over the limits, empty image).
pub fn current_margins(settings: &Settings, source: [u32; 2]) -> Option<Margins> {
    let options = settings.to_extend_options().ok()??;
    options.margins.resolve(source[0], source[1]).ok()
}

/// The margins the edit view draws and a drag starts from: what [`current_margins`] gives when the
/// settings resolve, whether or not extension is on and the fill can be read. Pixel and percent
/// values over the limits are cut back to the limits, so the canvas still shows about what was
/// asked (the size chip says why it cannot be used); an aspect that cannot be reached shows no
/// margins.
pub fn planned_margins(settings: &Settings, source: [u32; 2]) -> Margins {
    let spec = settings.margin_spec();
    if let Ok(m) = spec.resolve(source[0], source[1]) {
        return m;
    }
    let px = |v: f32, side: u32| (f64::from(v.max(0.0)) * f64::from(side) / 100.0).round() as u32;
    match spec {
        MarginSpec::Pixels(m) => fit_limits(m, source),
        MarginSpec::Percent {
            left,
            top,
            right,
            bottom,
        } => fit_limits(
            Margins {
                left: px(left, source[0]),
                top: px(top, source[1]),
                right: px(right, source[0]),
                bottom: px(bottom, source[1]),
            },
            source,
        ),
        MarginSpec::Aspect { .. } => Margins::default(),
    }
}

/// Stores dragged pixel margins in the settings, the same fields the sidebar edits.
///
/// Pixels mode stores pixels. Percent mode stores percent of the side each margin extends (left
/// and right of the width, top and bottom of the height), so `to_extend_options` resolves back to
/// the dragged pixels. Aspect mode cannot hold free margins and switches to Pixels. All four
/// values are always written; `extend_linked` becomes true only when the four stored values are
/// equal (then the app uses `extend_left` for every side, which agrees with them). It does not
/// turn extension on.
pub fn write_margins(settings: &mut Settings, margins: Margins, source: [u32; 2]) {
    let px = [margins.left, margins.top, margins.right, margins.bottom];
    let sides = [source[0], source[1], source[0], source[1]];
    let percent = settings.extend_mode == ExtendMode::Percent && source.iter().all(|&s| s > 0);
    let values: [f32; 4] = if percent {
        std::array::from_fn(|i| (f64::from(px[i]) * 100.0 / f64::from(sides[i])) as f32)
    } else {
        settings.extend_mode = ExtendMode::Pixels;
        px.map(|v| v as f32)
    };
    settings.extend_left = values[0];
    settings.extend_top = values[1];
    settings.extend_right = values[2];
    settings.extend_bottom = values[3];
    settings.extend_linked = values.iter().all(|v| *v == values[0]);
}

/// The pixels the settings resolve to (test helper: the round trip through the core).
#[cfg(test)]
fn resolved(settings: &Settings, source: [u32; 2]) -> Margins {
    let opts = settings.to_extend_options().unwrap().unwrap();
    opts.margins.resolve(source[0], source[1]).unwrap()
}

// ----------------------------------------------------------------------------------- preview

/// What the new canvas shows in the preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewKind {
    /// The source reflected about its border.
    Mirror,
    /// The border pixels stretched outward.
    Edge,
}

/// The screen rectangles around `source` inside `canvas` and the texture coordinates (a rectangle
/// in 0..1 of the source texture) to draw the source into each, in the order left, right, top,
/// bottom, then the corners top-left, top-right, bottom-left, bottom-right. Rectangles without
/// area are left out. A flipped texture rectangle (min above max) mirrors the picture.
///
/// Mirror reflects about the shared edge, which is what the core's mirror fill does up to the one
/// edge pixel it does not repeat; at preview scale that is invisible. A margin deeper than the
/// source, where the core reflects over and over, is approximated by squeezing one reflection of
/// the whole source into it. Edge samples the border row or column alone, so a corner shows its
/// corner pixel.
pub fn preview_bands(source: Rect, canvas: Rect, kind: PreviewKind) -> Vec<(Rect, Rect)> {
    if source.width() <= 0.0 || source.height() <= 0.0 {
        return Vec::new();
    }
    let (s, c) = (source, canvas);
    // Margins as a fraction of the source side, at most one reflection deep.
    let depth = |m: f32, side: f32| (m / side).clamp(0.0, 1.0);
    let (ml, mr) = (
        depth(s.min.x - c.min.x, s.width()),
        depth(c.max.x - s.max.x, s.width()),
    );
    let (mt, mb) = (
        depth(s.min.y - c.min.y, s.height()),
        depth(c.max.y - s.max.y, s.height()),
    );
    // Texture span of the low (left/top) and high (right/bottom) band of an axis: `(from, to)` runs
    // across the band in screen direction. Mirror counts back from the edge, edge repeats one line.
    let low = |m: f32| match kind {
        PreviewKind::Mirror => (m, 0.0),
        PreviewKind::Edge => (0.0, 0.0),
    };
    let high = |m: f32| match kind {
        PreviewKind::Mirror => (1.0, 1.0 - m),
        PreviewKind::Edge => (1.0, 1.0),
    };
    let whole = (0.0, 1.0);
    let (xl, xr, xm) = (low(ml), high(mr), whole);
    let (yt, yb, ym) = (low(mt), high(mb), whole);
    let (x0, x1, x2, x3) = (c.min.x, s.min.x, s.max.x, c.max.x);
    let (y0, y1, y2, y3) = (c.min.y, s.min.y, s.max.y, c.max.y);
    let band = |x: (f32, f32), y: (f32, f32), ux: (f32, f32), uy: (f32, f32)| {
        (
            Rect::from_min_max(pos2(x.0, y.0), pos2(x.1, y.1)),
            Rect::from_min_max(pos2(ux.0, uy.0), pos2(ux.1, uy.1)),
        )
    };
    [
        band((x0, x1), (y1, y2), xl, ym),
        band((x2, x3), (y1, y2), xr, ym),
        band((x1, x2), (y0, y1), xm, yt),
        band((x1, x2), (y2, y3), xm, yb),
        band((x0, x1), (y0, y1), xl, yt),
        band((x2, x3), (y0, y1), xr, yt),
        band((x0, x1), (y2, y3), xl, yb),
        band((x2, x3), (y2, y3), xr, yb),
    ]
    .into_iter()
    .filter(|(screen, _)| screen.width() > 0.0 && screen.height() > 0.0)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(left: u32, top: u32, right: u32, bottom: u32) -> Margins {
        Margins {
            left,
            top,
            right,
            bottom,
        }
    }

    fn resolves(margins: Margins, source: [u32; 2]) -> bool {
        MarginSpec::Pixels(margins)
            .resolve(source[0], source[1])
            .is_ok()
    }

    fn no_mods() -> DragMods {
        DragMods::default()
    }

    fn approx(a: Rect, b: Rect) -> bool {
        [a.min.x, a.min.y, a.max.x, a.max.y]
            .iter()
            .zip([b.min.x, b.min.y, b.max.x, b.max.y])
            .all(|(x, y)| (x - y).abs() < 1e-4)
    }

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect::from_min_max(pos2(x0, y0), pos2(x1, y1))
    }

    // ----------------------------------------------------------------------------- layout

    #[test]
    fn layout_without_margins_is_the_fitted_image() {
        let l = canvas_rect(
            [100, 50],
            Margins::default(),
            vec2(400.0, 400.0),
            1.0,
            Vec2::ZERO,
        );
        assert_eq!(l.canvas, rect(0.0, 100.0, 400.0, 300.0));
        assert_eq!(l.source, l.canvas);
        assert_eq!(l.scale, 4.0);
    }

    #[test]
    fn layout_places_the_source_inside_the_canvas() {
        // 100x50 plus 50 left and right is 200x50: fitted into 400x400 at 2 points per pixel.
        let l = canvas_rect(
            [100, 50],
            m(50, 0, 50, 0),
            vec2(400.0, 400.0),
            1.0,
            Vec2::ZERO,
        );
        assert_eq!(l.scale, 2.0);
        assert_eq!(l.canvas, rect(0.0, 150.0, 400.0, 250.0));
        assert_eq!(l.source, rect(100.0, 150.0, 300.0, 250.0));
        // Uneven margins on both axes: canvas 100+10+30 by 50+5+15 is 140x70 at 2 points per pixel.
        let l = canvas_rect(
            [100, 50],
            m(10, 5, 30, 15),
            vec2(280.0, 140.0),
            1.0,
            Vec2::ZERO,
        );
        assert_eq!(l.canvas, rect(0.0, 0.0, 280.0, 140.0));
        assert_eq!(l.source, rect(20.0, 10.0, 220.0, 110.0));
    }

    #[test]
    fn layout_follows_zoom_and_pan() {
        let l = canvas_rect(
            [100, 50],
            m(50, 0, 50, 0),
            vec2(400.0, 400.0),
            2.0,
            vec2(10.0, 0.0),
        );
        assert_eq!(l.scale, 4.0);
        // Centre (210, 200), size 800x200.
        assert_eq!(l.canvas, rect(-190.0, 100.0, 610.0, 300.0));
        assert_eq!(l.source, rect(10.0, 100.0, 410.0, 300.0));
    }

    #[test]
    fn layout_survives_an_empty_source() {
        let l = canvas_rect(
            [0, 0],
            Margins::default(),
            vec2(100.0, 100.0),
            1.0,
            Vec2::ZERO,
        );
        assert!(l.scale.is_finite());
        assert!(l.canvas.width().is_finite() && l.source.width().is_finite());
    }

    // ---------------------------------------------------------------------------- handles

    #[test]
    fn handle_names_cursors_and_sides() {
        let names: Vec<_> = Handle::ALL.iter().map(|h| h.name()).collect();
        assert_eq!(
            names,
            [
                "Left edge",
                "Top edge",
                "Right edge",
                "Bottom edge",
                "Top-left corner",
                "Top-right corner",
                "Bottom-left corner",
                "Bottom-right corner"
            ]
        );
        assert_eq!(Handle::Left.cursor(), CursorIcon::ResizeHorizontal);
        assert_eq!(Handle::Right.cursor(), CursorIcon::ResizeHorizontal);
        assert_eq!(Handle::Top.cursor(), CursorIcon::ResizeVertical);
        assert_eq!(Handle::Bottom.cursor(), CursorIcon::ResizeVertical);
        assert_eq!(Handle::TopLeft.cursor(), CursorIcon::ResizeNwSe);
        assert_eq!(Handle::BottomRight.cursor(), CursorIcon::ResizeNwSe);
        assert_eq!(Handle::TopRight.cursor(), CursorIcon::ResizeNeSw);
        assert_eq!(Handle::BottomLeft.cursor(), CursorIcon::ResizeNeSw);
        for h in Handle::ALL {
            let s = h.sides();
            let count = [s.left, s.top, s.right, s.bottom]
                .iter()
                .filter(|b| **b)
                .count();
            assert_eq!(count, if h.is_corner() { 2 } else { 1 }, "{h:?}");
        }
        let s = Handle::TopRight.sides();
        assert!(s.top && s.right && !s.left && !s.bottom);
    }

    #[test]
    fn handle_positions_are_midpoints_and_corners() {
        let r = rect(100.0, 200.0, 300.0, 280.0);
        assert_eq!(Handle::Left.position(r), pos2(100.0, 240.0));
        assert_eq!(Handle::Right.position(r), pos2(300.0, 240.0));
        assert_eq!(Handle::Top.position(r), pos2(200.0, 200.0));
        assert_eq!(Handle::Bottom.position(r), pos2(200.0, 280.0));
        assert_eq!(Handle::TopLeft.position(r), pos2(100.0, 200.0));
        assert_eq!(Handle::TopRight.position(r), pos2(300.0, 200.0));
        assert_eq!(Handle::BottomLeft.position(r), pos2(100.0, 280.0));
        assert_eq!(Handle::BottomRight.position(r), pos2(300.0, 280.0));
    }

    #[test]
    fn handles_are_ten_point_squares_on_the_canvas_edge() {
        let canvas = rect(100.0, 200.0, 300.0, 280.0);
        let all = handles(canvas);
        assert_eq!(all.map(|(h, _)| h), Handle::ALL);
        for (h, r) in all {
            assert_eq!(r.size(), Vec2::splat(10.0), "{h:?}");
            assert_eq!(r.center(), h.position(canvas), "{h:?}");
        }
    }

    #[test]
    fn hit_finds_the_handle_within_twenty_four_points() {
        let canvas = rect(100.0, 200.0, 300.0, 400.0);
        assert_eq!(hit(pos2(100.0, 300.0), canvas), Some(Handle::Left));
        assert_eq!(hit(pos2(111.9, 311.9), canvas), Some(Handle::Left));
        assert_eq!(hit(pos2(112.5, 300.0), canvas), None);
        assert_eq!(hit(pos2(200.0, 190.0), canvas), Some(Handle::Top));
        assert_eq!(hit(pos2(310.0, 300.0), canvas), Some(Handle::Right));
        assert_eq!(hit(pos2(200.0, 410.0), canvas), Some(Handle::Bottom));
        assert_eq!(hit(pos2(200.0, 300.0), canvas), None);
        assert_eq!(hit(pos2(0.0, 0.0), canvas), None);
    }

    #[test]
    fn hit_prefers_corners_over_sides() {
        // 30 points high: the left side handle (y 215) and the corners (y 200, 230) overlap near y 208.
        let canvas = rect(100.0, 200.0, 300.0, 230.0);
        assert_eq!(hit(pos2(100.0, 205.0), canvas), Some(Handle::TopLeft));
        assert_eq!(hit(pos2(100.0, 208.0), canvas), Some(Handle::TopLeft));
        assert_eq!(hit(pos2(102.0, 228.0), canvas), Some(Handle::BottomLeft));
        assert_eq!(hit(pos2(300.0, 208.0), canvas), Some(Handle::TopRight));
        // On a big canvas the corner's own area is a corner.
        let big = rect(0.0, 0.0, 400.0, 400.0);
        assert_eq!(hit(pos2(5.0, 5.0), big), Some(Handle::TopLeft));
    }

    #[test]
    fn hit_keeps_every_side_reachable_on_small_canvases() {
        for (w, h) in [
            (200.0, 30.0),
            (200.0, 20.0),
            (30.0, 200.0),
            (20.0, 200.0),
            (30.0, 30.0),
        ] {
            let canvas = rect(100.0, 200.0, 100.0 + w, 200.0 + h);
            for handle in Handle::ALL {
                let at = handle.position(canvas);
                assert_eq!(hit(at, canvas), Some(handle), "{w}x{h} {handle:?}");
            }
        }
    }

    #[test]
    fn hit_on_a_tiny_canvas_takes_the_nearest_of_a_kind() {
        // 8x8: every handle's area covers the whole canvas; corners win, nearest corner first.
        let tiny = rect(0.0, 0.0, 8.0, 8.0);
        assert_eq!(hit(pos2(1.0, 1.0), tiny), Some(Handle::TopLeft));
        assert_eq!(hit(pos2(7.0, 1.0), tiny), Some(Handle::TopRight));
        assert_eq!(hit(pos2(1.0, 7.0), tiny), Some(Handle::BottomLeft));
        assert_eq!(hit(pos2(7.0, 7.0), tiny), Some(Handle::BottomRight));
    }

    // ------------------------------------------------------------------------------ drag

    const SRC: [u32; 2] = [100, 50];

    fn go(handle: Handle, start: Margins, dx: f32, dy: f32, scale: f32, mods: DragMods) -> Margins {
        drag(handle, start, vec2(dx, dy), scale, mods, SRC)
    }

    #[test]
    fn drag_side_outward_grows_in_source_pixels() {
        let z = Margins::default();
        assert_eq!(
            go(Handle::Right, z, 20.0, 99.0, 2.0, no_mods()),
            m(0, 0, 10, 0)
        );
        assert_eq!(
            go(Handle::Left, z, -20.0, 0.0, 2.0, no_mods()),
            m(10, 0, 0, 0)
        );
        assert_eq!(
            go(Handle::Bottom, z, 0.0, 12.0, 2.0, no_mods()),
            m(0, 0, 0, 6)
        );
        assert_eq!(
            go(Handle::Top, z, 5.0, -12.0, 2.0, no_mods()),
            m(0, 6, 0, 0)
        );
    }

    #[test]
    fn drag_inward_shrinks_and_stops_at_zero() {
        let start = m(5, 8, 5, 8);
        assert_eq!(
            go(Handle::Left, start, 4.0, 0.0, 1.0, no_mods()),
            m(1, 8, 5, 8)
        );
        assert_eq!(
            go(Handle::Left, start, 400.0, 0.0, 1.0, no_mods()),
            m(0, 8, 5, 8)
        );
        assert_eq!(
            go(Handle::Bottom, start, 0.0, -400.0, 1.0, no_mods()),
            m(5, 8, 5, 0)
        );
        // Dragging an empty side inward does nothing.
        assert_eq!(
            go(
                Handle::Right,
                Margins::default(),
                -30.0,
                0.0,
                1.0,
                no_mods()
            ),
            Margins::default()
        );
    }

    #[test]
    fn drag_rounds_to_whole_pixels() {
        let z = Margins::default();
        assert_eq!(go(Handle::Right, z, 0.4, 0.0, 1.0, no_mods()), z);
        assert_eq!(
            go(Handle::Right, z, 0.6, 0.0, 1.0, no_mods()),
            m(0, 0, 1, 0)
        );
        assert_eq!(
            go(Handle::Left, z, -0.6, 0.0, 1.0, no_mods()),
            m(1, 0, 0, 0)
        );
        // Zero, negative and non-finite scales or deltas change nothing.
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                go(Handle::Right, z, 50.0, 0.0, scale, no_mods()),
                z,
                "{scale}"
            );
        }
        assert_eq!(go(Handle::Right, z, f32::NAN, 0.0, 1.0, no_mods()), z);
        let start = m(1, 2, 3, 4);
        assert_eq!(go(Handle::TopLeft, start, 0.0, 0.0, 3.0, no_mods()), start);
    }

    #[test]
    fn drag_stops_at_four_times_the_side() {
        let z = Margins::default();
        // 100 wide: 400 at most; 50 high: 200 at most.
        assert_eq!(
            go(Handle::Right, z, 9999.0, 0.0, 1.0, no_mods()),
            m(0, 0, 400, 0)
        );
        assert_eq!(
            go(Handle::Left, z, -9999.0, 0.0, 1.0, no_mods()),
            m(400, 0, 0, 0)
        );
        assert_eq!(
            go(Handle::Bottom, z, 0.0, 9999.0, 1.0, no_mods()),
            m(0, 0, 0, 200)
        );
        assert_eq!(
            go(Handle::Top, z, 0.0, -9999.0, 1.0, no_mods()),
            m(0, 200, 0, 0)
        );
        assert_eq!(
            go(Handle::BottomRight, z, 9999.0, 9999.0, 1.0, no_mods()),
            m(0, 0, 400, 200)
        );
    }

    #[test]
    fn drag_stops_at_the_canvas_limit() {
        let big = [12000, 12000];
        let d = |start, handle| drag(handle, start, vec2(1e6, 1e6), 1.0, no_mods(), big);
        // 16384 - 12000 leaves 4384 for both sides of an axis together.
        assert_eq!(
            d(Margins::default(), Handle::BottomRight),
            m(0, 0, 4384, 4384)
        );
        assert_eq!(
            d(m(2000, 0, 0, 0), Handle::BottomRight),
            m(2000, 0, 2384, 4384)
        );
        let out = drag(
            Handle::Left,
            m(0, 0, 4384, 0),
            vec2(-1e6, 0.0),
            1.0,
            no_mods(),
            big,
        );
        assert_eq!(out, m(0, 0, 4384, 0));
        assert!(resolves(d(m(2000, 0, 0, 0), Handle::BottomRight), big));
    }

    #[test]
    fn drag_corner_moves_two_sides() {
        let z = Margins::default();
        assert_eq!(
            go(Handle::TopLeft, z, -10.0, -6.0, 1.0, no_mods()),
            m(10, 6, 0, 0)
        );
        assert_eq!(
            go(Handle::TopRight, z, 10.0, -6.0, 1.0, no_mods()),
            m(0, 6, 10, 0)
        );
        assert_eq!(
            go(Handle::BottomLeft, z, -10.0, 6.0, 1.0, no_mods()),
            m(10, 0, 0, 6)
        );
        assert_eq!(
            go(Handle::BottomRight, z, 10.0, 6.0, 1.0, no_mods()),
            m(0, 0, 10, 6)
        );
        // One axis in, one out.
        let start = m(20, 20, 20, 20);
        assert_eq!(
            go(Handle::BottomRight, start, -5.0, 5.0, 1.0, no_mods()),
            m(20, 20, 15, 25)
        );
        // Each axis stops on its own limit: top is at 200 and stays, right keeps growing.
        assert_eq!(
            go(
                Handle::TopRight,
                m(0, 200, 0, 0),
                30.0,
                -30.0,
                1.0,
                no_mods()
            ),
            m(0, 200, 30, 0)
        );
    }

    #[test]
    fn shift_applies_the_amount_to_all_four_sides() {
        let all = DragMods {
            all_sides: true,
            mirror: false,
        };
        assert_eq!(
            go(Handle::Right, m(1, 2, 3, 4), 10.0, 0.0, 1.0, all),
            m(11, 12, 13, 14)
        );
        assert_eq!(
            go(Handle::Left, Margins::default(), -7.0, 0.0, 1.0, all),
            m(7, 7, 7, 7)
        );
        assert_eq!(
            go(Handle::Bottom, Margins::default(), 0.0, 9.0, 1.0, all),
            m(9, 9, 9, 9)
        );
        // Shrinking: each side stops at 0 on its own.
        assert_eq!(
            go(Handle::Right, m(5, 10, 5, 10), -7.0, 0.0, 1.0, all),
            m(0, 3, 0, 3)
        );
        // A corner has two amounts; the larger one counts.
        assert_eq!(
            go(Handle::TopLeft, Margins::default(), -3.0, -8.0, 1.0, all),
            m(8, 8, 8, 8)
        );
        assert_eq!(
            go(Handle::BottomRight, Margins::default(), 8.0, 3.0, 1.0, all),
            m(8, 8, 8, 8)
        );
        // The shared limit is the tightest one: the height allows 4 * 50 = 200.
        assert_eq!(
            go(Handle::Right, Margins::default(), 5000.0, 0.0, 1.0, all),
            m(200, 200, 200, 200)
        );
    }

    #[test]
    fn alt_moves_the_opposite_side_too() {
        let mirror = DragMods {
            all_sides: false,
            mirror: true,
        };
        assert_eq!(
            go(Handle::Right, Margins::default(), 10.0, 0.0, 1.0, mirror),
            m(10, 0, 10, 0)
        );
        assert_eq!(
            go(Handle::Top, m(0, 3, 0, 5), 0.0, -4.0, 1.0, mirror),
            m(0, 7, 0, 9)
        );
        // A corner mirrors across both axes, each by its own amount.
        assert_eq!(
            go(
                Handle::BottomRight,
                Margins::default(),
                10.0,
                6.0,
                1.0,
                mirror
            ),
            m(10, 6, 10, 6)
        );
        // Shrinking: one side hits 0 and the other keeps going.
        assert_eq!(
            go(Handle::Left, m(2, 0, 9, 0), 5.0, 0.0, 1.0, mirror),
            m(0, 0, 4, 0)
        );
        // Each side stops at 4x the width.
        assert_eq!(
            go(Handle::Right, Margins::default(), 5000.0, 0.0, 1.0, mirror),
            m(400, 0, 400, 0)
        );
    }

    #[test]
    fn both_modifiers_give_all_sides() {
        let both = DragMods {
            all_sides: true,
            mirror: true,
        };
        let all = DragMods {
            all_sides: true,
            mirror: false,
        };
        let start = m(1, 2, 3, 4);
        assert_eq!(
            go(Handle::TopLeft, start, -9.0, -2.0, 1.0, both),
            go(Handle::TopLeft, start, -9.0, -2.0, 1.0, all)
        );
    }

    #[test]
    fn drag_survives_absurd_deltas() {
        let z = Margins::default();
        let huge = [-3e38, 3e38, f32::MIN, f32::MAX];
        let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let all = DragMods {
            all_sides: true,
            mirror: true,
        };
        for handle in Handle::ALL {
            for mods in [no_mods(), all] {
                for dx in huge.into_iter().chain(bad) {
                    for dy in [dx, 0.0, 1e30] {
                        for scale in [1.0, 1e-30, f32::MIN_POSITIVE] {
                            let out = drag(handle, z, vec2(dx, dy), scale, mods, SRC);
                            assert!(resolves(out, SRC), "{handle:?} {dx} {dy} {scale}: {out:?}");
                        }
                    }
                }
            }
        }
        assert_eq!(
            go(Handle::Left, z, -3e38, 0.0, 1.0, no_mods()),
            m(400, 0, 0, 0)
        );
        assert_eq!(go(Handle::Right, z, f32::NAN, 0.0, 1.0, no_mods()), z);
    }

    #[test]
    fn shift_stops_everything_when_one_axis_is_already_at_its_limit() {
        // Top and bottom are at 4x the height (200): the shared growth limit is 0, so a Shift drag
        // that would widen the sides leaves all four margins as they were.
        let all = DragMods {
            all_sides: true,
            mirror: false,
        };
        let start = m(0, 200, 0, 200);
        assert_eq!(go(Handle::Right, start, 30.0, 0.0, 1.0, all), start);
        // Shrinking still works: each side then gives way by the same amount.
        assert_eq!(
            go(Handle::Right, start, -30.0, 0.0, 1.0, all),
            m(0, 170, 0, 170)
        );
    }

    #[test]
    fn drag_on_a_tiny_source_always_resolves() {
        let tiny = [20, 20];
        let all = DragMods {
            all_sides: true,
            mirror: false,
        };
        let out = drag(
            Handle::Right,
            Margins::default(),
            vec2(9999.0, 0.0),
            1.0,
            no_mods(),
            tiny,
        );
        assert_eq!(out, m(0, 0, 80, 0));
        let out = drag(
            Handle::BottomRight,
            Margins::default(),
            vec2(9999.0, 9999.0),
            1.0,
            all,
            tiny,
        );
        assert_eq!(out, m(80, 80, 80, 80));
        assert!(resolves(out, tiny));
    }

    #[test]
    fn an_invalid_start_is_cut_back_to_valid() {
        let out = go(Handle::Right, m(0, 0, 9999, 0), 0.0, 0.0, 1.0, no_mods());
        assert_eq!(out, m(0, 0, 400, 0));
        let big = [12000, 12000];
        let out = drag(
            Handle::Top,
            m(9000, 0, 9000, 0),
            Vec2::ZERO,
            1.0,
            no_mods(),
            big,
        );
        assert!(resolves(out, big));
        assert_eq!(out.left + out.right, 4384);
    }

    /// Deterministic pseudo-random numbers in `0..n` (LCG).
    fn lcg(s: &mut u32, n: u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*s >> 8) % n
    }

    #[test]
    fn every_drag_resolves() {
        let mut s = 99;
        for _ in 0..4000 {
            let source = [1 + lcg(&mut s, 600), 1 + lcg(&mut s, 600)];
            let source = match lcg(&mut s, 4) {
                0 => [1 + lcg(&mut s, 20_000).min(16_000), source[1]],
                1 => [20, 20],
                _ => source,
            };
            let start = fit_limits(
                m(
                    lcg(&mut s, 5000),
                    lcg(&mut s, 5000),
                    lcg(&mut s, 5000),
                    lcg(&mut s, 5000),
                ),
                source,
            );
            let handle = Handle::ALL[lcg(&mut s, 8) as usize];
            let delta = vec2(
                lcg(&mut s, 40_000) as f32 - 20_000.0,
                lcg(&mut s, 40_000) as f32 - 20_000.0,
            );
            let mods = DragMods {
                all_sides: lcg(&mut s, 2) == 0,
                mirror: lcg(&mut s, 2) == 0,
            };
            let scale = 0.1 + lcg(&mut s, 100) as f32 / 10.0;
            let out = drag(handle, start, delta, scale, mods, source);
            if !mods.all_sides && !mods.mirror {
                let moved = handle.sides();
                let before = [start.left, start.top, start.right, start.bottom];
                let after = [out.left, out.top, out.right, out.bottom];
                for (i, moves) in [moved.left, moved.top, moved.right, moved.bottom]
                    .into_iter()
                    .enumerate()
                {
                    assert!(
                        moves || before[i] == after[i],
                        "{handle:?} moved side {i}: {start:?} to {out:?}"
                    );
                }
            }
            assert!(
                resolves(out, source),
                "{source:?} {start:?} {handle:?} {delta:?} {mods:?} {scale} gave {out:?}"
            );
        }
    }

    #[test]
    fn drag_mods_come_from_egui_modifiers() {
        let shift = Modifiers {
            shift: true,
            ..Modifiers::NONE
        };
        let alt = Modifiers {
            alt: true,
            ..Modifiers::NONE
        };
        assert_eq!(
            DragMods::from_egui(shift),
            DragMods {
                all_sides: true,
                mirror: false
            }
        );
        assert_eq!(
            DragMods::from_egui(alt),
            DragMods {
                all_sides: false,
                mirror: true
            }
        );
        assert_eq!(DragMods::from_egui(Modifiers::NONE), DragMods::default());
    }

    // ------------------------------------------------------------------------- settings

    fn extending(mode: ExtendMode) -> Settings {
        Settings {
            extend_enabled: true,
            extend_mode: mode,
            extend_fill: "mirror".into(),
            ..Settings::default()
        }
    }

    #[test]
    fn write_pixels_stores_pixels_and_detects_linked() {
        let mut s = extending(ExtendMode::Pixels);
        write_margins(&mut s, m(10, 20, 30, 40), [300, 200]);
        assert_eq!(
            [s.extend_left, s.extend_top, s.extend_right, s.extend_bottom],
            [10.0, 20.0, 30.0, 40.0]
        );
        assert!(!s.extend_linked);
        assert_eq!(s.extend_mode, ExtendMode::Pixels);
        assert_eq!(resolved(&s, [300, 200]), m(10, 20, 30, 40));
        write_margins(&mut s, m(25, 25, 25, 25), [300, 200]);
        assert!(s.extend_linked);
        assert_eq!(s.extend_left, 25.0);
        assert_eq!(resolved(&s, [300, 200]), m(25, 25, 25, 25));
        // Almost equal is not linked, and zero margins are.
        write_margins(&mut s, m(25, 25, 25, 26), [300, 200]);
        assert!(!s.extend_linked);
        write_margins(&mut s, Margins::default(), [300, 200]);
        assert!(s.extend_linked);
        assert_eq!(resolved(&s, [300, 200]), Margins::default());
    }

    #[test]
    fn write_aspect_switches_to_pixels() {
        let mut s = extending(ExtendMode::Aspect);
        write_margins(&mut s, m(1, 2, 3, 4), [100, 100]);
        assert_eq!(s.extend_mode, ExtendMode::Pixels);
        assert_eq!(resolved(&s, [100, 100]), m(1, 2, 3, 4));
    }

    #[test]
    fn write_percent_round_trips_within_a_pixel() {
        let source = [300, 200];
        let mut s = extending(ExtendMode::Percent);
        for margins in [
            m(30, 20, 60, 10),
            m(1, 0, 399, 7),
            m(1200, 800, 1, 1),
            Margins::default(),
            m(37, 37, 37, 37),
        ] {
            write_margins(&mut s, margins, source);
            assert_eq!(s.extend_mode, ExtendMode::Percent);
            let back = resolved(&s, source);
            for (a, b) in [
                (back.left, margins.left),
                (back.top, margins.top),
                (back.right, margins.right),
                (back.bottom, margins.bottom),
            ] {
                assert!(a.abs_diff(b) <= 1, "{margins:?} came back as {back:?}");
            }
        }
        write_margins(&mut s, m(30, 20, 60, 10), source);
        assert_eq!(s.extend_left, 10.0);
        assert_eq!(s.extend_top, 10.0);
        assert_eq!(s.extend_right, 20.0);
        assert_eq!(s.extend_bottom, 5.0);
    }

    #[test]
    fn write_percent_reads_back_the_same_pixels() {
        // A drag writes percent and the next frame reads pixels back: they must agree exactly,
        // else the two would chase each other a pixel apart on every frame.
        let mut s = extending(ExtendMode::Percent);
        for w in (1..=4096u32).step_by(37).chain([16384 / 5, 12000]) {
            let h = 1 + w / 3;
            for px in (0..=4 * w.min(4096)).step_by(11) {
                let px = px.min(16384 - w);
                let m = m(px, px.min(4 * h), 0, 0);
                if !resolves(m, [w, h]) {
                    continue;
                }
                write_margins(&mut s, m, [w, h]);
                assert_eq!(planned_margins(&s, [w, h]), m, "{w}x{h}");
            }
        }
    }

    #[test]
    fn write_percent_links_only_equal_percentages() {
        // Square source: equal pixels are equal percentages, so the sidebar may link them.
        let mut s = extending(ExtendMode::Percent);
        write_margins(&mut s, m(20, 20, 20, 20), [200, 200]);
        assert!(s.extend_linked);
        assert_eq!(s.extend_left, 10.0);
        assert_eq!(resolved(&s, [200, 200]), m(20, 20, 20, 20));
        // 300x200: 30 px is 10% across but 15% down. Linked would give 10% to the top as well.
        write_margins(&mut s, m(30, 30, 30, 30), [300, 200]);
        assert!(!s.extend_linked);
        assert_eq!(resolved(&s, [300, 200]), m(30, 30, 30, 30));
        // Equal percentages on a wide source are linked and give each side its own pixels.
        write_margins(&mut s, m(30, 20, 30, 20), [300, 200]);
        assert!(s.extend_linked);
        assert_eq!(resolved(&s, [300, 200]), m(30, 20, 30, 20));
    }

    #[test]
    fn write_percent_with_an_empty_source_falls_back_to_pixels() {
        let mut s = extending(ExtendMode::Percent);
        write_margins(&mut s, m(5, 5, 5, 5), [0, 10]);
        assert_eq!(s.extend_mode, ExtendMode::Pixels);
        assert_eq!(s.extend_left, 5.0);
    }

    #[test]
    fn write_leaves_the_rest_of_the_settings_alone() {
        let mut s = extending(ExtendMode::Pixels);
        s.extend_enabled = false;
        let before = s.clone();
        write_margins(&mut s, m(5, 5, 5, 5), [100, 100]);
        assert!(!s.extend_enabled);
        assert_eq!(s.extend_fill, before.extend_fill);
        assert_eq!(s.extend_model, before.extend_model);
        assert_eq!(s.scale, before.scale);
    }

    #[test]
    fn current_margins_reads_what_the_settings_resolve_to() {
        let mut s = extending(ExtendMode::Pixels);
        s.extend_linked = false;
        s.extend_left = 10.0;
        s.extend_top = 20.0;
        s.extend_right = 30.0;
        s.extend_bottom = 40.0;
        assert_eq!(current_margins(&s, [300, 200]), Some(m(10, 20, 30, 40)));
        // Linked: every side takes the left value.
        s.extend_linked = true;
        assert_eq!(current_margins(&s, [300, 200]), Some(m(10, 10, 10, 10)));
        // Percent and aspect resolve against the size.
        s.extend_mode = ExtendMode::Percent;
        assert_eq!(current_margins(&s, [300, 200]), Some(m(30, 20, 30, 20)));
        s.extend_mode = ExtendMode::Aspect;
        s.extend_aspect = (2, 1);
        assert_eq!(current_margins(&s, [100, 100]), Some(m(50, 0, 50, 0)));
    }

    #[test]
    fn current_margins_is_none_when_off_or_invalid() {
        let mut s = extending(ExtendMode::Pixels);
        s.extend_left = 500.0;
        s.extend_linked = false;
        // 500 px is more than 4x a 100 px side.
        assert_eq!(current_margins(&s, [100, 100]), None);
        assert!(current_margins(&s, [200, 200]).is_some());
        assert_eq!(current_margins(&s, [0, 100]), None);
        s.extend_fill = "blur".into();
        assert_eq!(current_margins(&s, [200, 200]), None);
        s.extend_fill = "mirror".into();
        s.extend_enabled = false;
        assert_eq!(current_margins(&s, [200, 200]), None);
    }

    #[test]
    fn planned_margins_follow_the_fields_even_when_they_cannot_be_used() {
        let mut s = extending(ExtendMode::Pixels);
        s.extend_linked = false;
        [s.extend_left, s.extend_top, s.extend_right, s.extend_bottom] = [10.0, 20.0, 30.0, 40.0];
        // Valid: the same as what a run would use, whether extension is on or not.
        assert_eq!(planned_margins(&s, [300, 200]), m(10, 20, 30, 40));
        s.extend_enabled = false;
        assert_eq!(planned_margins(&s, [300, 200]), m(10, 20, 30, 40));
        // An unreadable fill does not hide the margins.
        s.extend_fill = "blur".into();
        assert_eq!(planned_margins(&s, [300, 200]), m(10, 20, 30, 40));
        // Over 4x the side: cut back to the limit, the rest untouched.
        s.extend_left = 5000.0;
        assert_eq!(planned_margins(&s, [100, 100]), m(400, 20, 30, 40));
        // Percent over the limit: 1000% of 100 px is cut back to 400 px.
        s.extend_mode = ExtendMode::Percent;
        s.extend_left = 1000.0;
        [s.extend_top, s.extend_right, s.extend_bottom] = [10.0, 0.0, 0.0];
        assert_eq!(planned_margins(&s, [100, 200]), m(400, 20, 0, 0));
        // Aspect resolves as current_margins says.
        s.extend_mode = ExtendMode::Aspect;
        s.extend_aspect = (2, 1);
        s.extend_fill = "mirror".into();
        assert_eq!(planned_margins(&s, [100, 100]), m(50, 0, 50, 0));
        // An empty image cannot be extended at all.
        assert_eq!(planned_margins(&s, [0, 0]), Margins::default());
    }

    // ------------------------------------------------------------------------- preview

    #[test]
    fn mirror_bands_flip_the_texture_about_the_shared_edge() {
        // Source 100x100 at (100, 100), 50 more on the left and right only.
        let source = rect(100.0, 100.0, 200.0, 200.0);
        let canvas = rect(50.0, 100.0, 250.0, 200.0);
        let bands = preview_bands(source, canvas, PreviewKind::Mirror);
        assert_eq!(bands.len(), 2);
        // Left band: 50 points wide, the half of the picture next to the edge, flipped.
        assert!(approx(bands[0].0, rect(50.0, 100.0, 100.0, 200.0)));
        assert!(approx(bands[0].1, rect(0.5, 0.0, 0.0, 1.0)));
        assert!(approx(bands[1].0, rect(200.0, 100.0, 250.0, 200.0)));
        assert!(approx(bands[1].1, rect(1.0, 0.0, 0.5, 1.0)));
    }

    #[test]
    fn mirror_corners_flip_both_ways() {
        let source = rect(100.0, 100.0, 200.0, 200.0);
        let canvas = rect(90.0, 80.0, 220.0, 230.0);
        let bands = preview_bands(source, canvas, PreviewKind::Mirror);
        assert_eq!(bands.len(), 8);
        let find = |screen: Rect| {
            bands
                .iter()
                .find(|(r, _)| approx(*r, screen))
                .unwrap_or_else(|| panic!("no band at {screen:?}"))
                .1
        };
        // Top-left: 10 across, 20 down; reflected about the corner.
        assert!(approx(
            find(rect(90.0, 80.0, 100.0, 100.0)),
            rect(0.1, 0.2, 0.0, 0.0)
        ));
        // Top-right: 20 across, 20 down.
        assert!(approx(
            find(rect(200.0, 80.0, 220.0, 100.0)),
            rect(1.0, 0.2, 0.8, 0.0)
        ));
        // Bottom-left and bottom-right: 30 down.
        assert!(approx(
            find(rect(90.0, 200.0, 100.0, 230.0)),
            rect(0.1, 1.0, 0.0, 0.7)
        ));
        assert!(approx(
            find(rect(200.0, 200.0, 220.0, 230.0)),
            rect(1.0, 1.0, 0.8, 0.7)
        ));
        // Top and bottom bands span the source's columns, unflipped across.
        assert!(approx(
            find(rect(100.0, 80.0, 200.0, 100.0)),
            rect(0.0, 0.2, 1.0, 0.0)
        ));
        assert!(approx(
            find(rect(100.0, 200.0, 200.0, 230.0)),
            rect(0.0, 1.0, 1.0, 0.7)
        ));
        // Left and right bands span its rows.
        assert!(approx(
            find(rect(90.0, 100.0, 100.0, 200.0)),
            rect(0.1, 0.0, 0.0, 1.0)
        ));
        assert!(approx(
            find(rect(200.0, 100.0, 220.0, 200.0)),
            rect(1.0, 0.0, 0.8, 1.0)
        ));
    }

    #[test]
    fn mirror_deeper_than_the_source_squeezes_one_reflection() {
        // 300 points of margin on a 100 point source: the whole source, reflected once.
        let source = rect(300.0, 0.0, 400.0, 100.0);
        let canvas = rect(0.0, 0.0, 400.0, 100.0);
        let bands = preview_bands(source, canvas, PreviewKind::Mirror);
        assert_eq!(bands.len(), 1);
        assert!(approx(bands[0].0, rect(0.0, 0.0, 300.0, 100.0)));
        assert!(approx(bands[0].1, rect(1.0, 0.0, 0.0, 1.0)));
        for (_, uv) in &bands {
            for v in [uv.min.x, uv.min.y, uv.max.x, uv.max.y] {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn edge_bands_sample_the_border_line() {
        let source = rect(100.0, 100.0, 200.0, 200.0);
        let canvas = rect(80.0, 70.0, 230.0, 250.0);
        let bands = preview_bands(source, canvas, PreviewKind::Edge);
        assert_eq!(bands.len(), 8);
        let find = |screen: Rect| {
            bands
                .iter()
                .find(|(r, _)| approx(*r, screen))
                .unwrap_or_else(|| panic!("no band at {screen:?}"))
                .1
        };
        // Left: the first column across the rows; right: the last column.
        assert!(approx(
            find(rect(80.0, 100.0, 100.0, 200.0)),
            rect(0.0, 0.0, 0.0, 1.0)
        ));
        assert!(approx(
            find(rect(200.0, 100.0, 230.0, 200.0)),
            rect(1.0, 0.0, 1.0, 1.0)
        ));
        // Top: the first row across the columns; bottom: the last row.
        assert!(approx(
            find(rect(100.0, 70.0, 200.0, 100.0)),
            rect(0.0, 0.0, 1.0, 0.0)
        ));
        assert!(approx(
            find(rect(100.0, 200.0, 200.0, 250.0)),
            rect(0.0, 1.0, 1.0, 1.0)
        ));
        // Corners: the corner pixel alone.
        assert!(approx(
            find(rect(80.0, 70.0, 100.0, 100.0)),
            rect(0.0, 0.0, 0.0, 0.0)
        ));
        assert!(approx(
            find(rect(200.0, 70.0, 230.0, 100.0)),
            rect(1.0, 0.0, 1.0, 0.0)
        ));
        assert!(approx(
            find(rect(80.0, 200.0, 100.0, 250.0)),
            rect(0.0, 1.0, 0.0, 1.0)
        ));
        assert!(approx(
            find(rect(200.0, 200.0, 230.0, 250.0)),
            rect(1.0, 1.0, 1.0, 1.0)
        ));
    }

    #[test]
    fn bands_without_area_are_left_out() {
        let source = rect(100.0, 100.0, 200.0, 200.0);
        for kind in [PreviewKind::Mirror, PreviewKind::Edge] {
            // No margins at all: nothing to draw.
            assert!(preview_bands(source, source, kind).is_empty());
            // Only the top: one band, no corners, no sides.
            let bands = preview_bands(source, rect(100.0, 60.0, 200.0, 200.0), kind);
            assert_eq!(bands.len(), 1);
            assert!(approx(bands[0].0, rect(100.0, 60.0, 200.0, 100.0)));
            // Left and bottom: two sides and the one corner between them.
            let bands = preview_bands(source, rect(70.0, 100.0, 200.0, 230.0), kind);
            assert_eq!(bands.len(), 3);
            assert!(
                bands
                    .iter()
                    .all(|(r, _)| r.width() > 0.0 && r.height() > 0.0)
            );
        }
    }

    #[test]
    fn bands_of_an_empty_source_are_empty() {
        let flat = rect(100.0, 100.0, 100.0, 200.0);
        let canvas = rect(50.0, 50.0, 150.0, 250.0);
        for kind in [PreviewKind::Mirror, PreviewKind::Edge] {
            assert!(preview_bands(flat, canvas, kind).is_empty());
        }
    }

    #[test]
    fn bands_tile_the_canvas_around_the_source() {
        // The screen rectangles and the source fill the canvas exactly.
        let l = canvas_rect(
            [100, 50],
            m(30, 10, 5, 40),
            vec2(400.0, 300.0),
            1.5,
            vec2(7.0, -3.0),
        );
        for kind in [PreviewKind::Mirror, PreviewKind::Edge] {
            let bands = preview_bands(l.source, l.canvas, kind);
            let area: f32 = bands.iter().map(|(r, _)| r.area()).sum::<f32>() + l.source.area();
            assert!(
                (area - l.canvas.area()).abs() < 1.0,
                "{area} {}",
                l.canvas.area()
            );
        }
    }
}
