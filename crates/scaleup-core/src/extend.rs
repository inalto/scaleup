//! Canvas extension: margin specs, fill modes, the plain (non-AI) padding of an image and the
//! window-by-window AI fill ([`outpaint`]).

use image::{Rgba, RgbaImage};

use crate::backend::Inpainter;
use crate::image32::Image32;
use crate::pipeline::{CancelToken, Phase, Progress};
use crate::resize::resize_rgba;
use crate::{Result, ScaleUpError};

/// Largest extended canvas, in pixels per side.
const MAX_CANVAS_SIDE: u32 = 16384;
/// Largest margin, as a multiple of the side it extends.
const MAX_MARGIN_FACTOR: u64 = 4;

/// Pixels added on each side of an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Margins {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// How much canvas to add, before it is resolved against an image size.
#[derive(Clone, Debug, PartialEq)]
pub enum MarginSpec {
    Pixels(Margins),
    /// Per side, in percent of the side it extends (left/right of width, top/bottom of height).
    Percent {
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    },
    /// Grow, centred, to `w:h`. Never crops.
    Aspect {
        w: u32,
        h: u32,
    },
}

impl MarginSpec {
    /// Parses `64` (all sides), `l,t,r,b` (pixels), `20%` (all sides), `l%,t%,r%,b%` or `16:9`.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let s = s.trim();
        if let Some((a, b)) = s.split_once(':') {
            let a = parse_ratio_part(a, s)?;
            let b = parse_ratio_part(b, s)?;
            return Ok(MarginSpec::Aspect { w: a, h: b });
        }
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 1 && parts.len() != 4 {
            return Err(format!(
                "margins {s:?}: expected one value or four (left,top,right,bottom)"
            ));
        }
        let percent = parts.iter().filter(|p| p.ends_with('%')).count();
        if percent == parts.len() {
            let v = parts
                .iter()
                .map(|p| parse_percent(&p[..p.len() - 1], s))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let (l, t, r, b) = spread(&v);
            return Ok(MarginSpec::Percent {
                left: l,
                top: t,
                right: r,
                bottom: b,
            });
        }
        if percent != 0 {
            return Err(format!(
                "margins {s:?}: use percent on every value or on none"
            ));
        }
        let v = parts
            .iter()
            .map(|p| {
                p.parse::<u32>()
                    .map_err(|_| format!("margins {s:?}: {p:?} is not a whole number of pixels"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let (left, top, right, bottom) = spread(&v);
        Ok(MarginSpec::Pixels(Margins {
            left,
            top,
            right,
            bottom,
        }))
    }

    /// Margins in pixels for a `w`×`h` image. Percentages round to the nearest pixel; aspect growth
    /// is centred with any odd pixel on the right/bottom. Each margin may be at most 4× the side it
    /// extends and the extended canvas at most 16384 px per side, else `InvalidOption`.
    pub fn resolve(&self, w: u32, h: u32) -> Result<Margins> {
        if w == 0 || h == 0 {
            return Err(ScaleUpError::InvalidOption(format!(
                "cannot extend an empty {w}x{h} image"
            )));
        }
        let (left, top, right, bottom): (u64, u64, u64, u64) = match *self {
            MarginSpec::Pixels(m) => (m.left.into(), m.top.into(), m.right.into(), m.bottom.into()),
            MarginSpec::Percent {
                left,
                top,
                right,
                bottom,
            } => (
                percent_of(left, w)?,
                percent_of(top, h)?,
                percent_of(right, w)?,
                percent_of(bottom, h)?,
            ),
            MarginSpec::Aspect { w: aw, h: ah } => {
                if aw == 0 || ah == 0 {
                    return Err(ScaleUpError::InvalidOption(format!(
                        "aspect {aw}:{ah} must have positive sides"
                    )));
                }
                let (w64, h64, aw, ah) = (u64::from(w), u64::from(h), u64::from(aw), u64::from(ah));
                if w64 * ah < h64 * aw {
                    // Narrower than the target: grow the width.
                    let grow = (h64 * aw).div_ceil(ah) - w64;
                    (grow / 2, 0, grow - grow / 2, 0)
                } else {
                    let grow = (w64 * ah).div_ceil(aw) - h64;
                    (0, grow / 2, 0, grow - grow / 2)
                }
            }
        };
        let (w64, h64) = (u64::from(w), u64::from(h));
        for (name, margin, side) in [
            ("left", left, w64),
            ("top", top, h64),
            ("right", right, w64),
            ("bottom", bottom, h64),
        ] {
            if margin > MAX_MARGIN_FACTOR * side {
                return Err(ScaleUpError::InvalidOption(format!(
                    "{name} margin {margin} px exceeds {MAX_MARGIN_FACTOR}x the image side ({side} px)"
                )));
            }
        }
        let (cw, ch) = (w64 + left + right, h64 + top + bottom);
        if cw > u64::from(MAX_CANVAS_SIDE) || ch > u64::from(MAX_CANVAS_SIDE) {
            return Err(ScaleUpError::InvalidOption(format!(
                "extended canvas {cw}x{ch} exceeds {MAX_CANVAS_SIDE} px per side"
            )));
        }
        // Every margin is now below the canvas limit, so it fits in u32.
        Ok(Margins {
            left: left as u32,
            top: top as u32,
            right: right as u32,
            bottom: bottom as u32,
        })
    }
}

fn parse_ratio_part(p: &str, s: &str) -> std::result::Result<u32, String> {
    match p.trim().parse::<u32>() {
        Ok(v) if v > 0 => Ok(v),
        _ => Err(format!(
            "aspect {s:?}: expected two positive whole numbers like 16:9"
        )),
    }
}

fn parse_percent(p: &str, s: &str) -> std::result::Result<f32, String> {
    match p.trim().parse::<f32>() {
        Ok(v) if v.is_finite() && v >= 0.0 => Ok(v),
        _ => Err(format!(
            "margins {s:?}: {p:?}% is not a non-negative percentage"
        )),
    }
}

/// One value for every side, or four in `left,top,right,bottom` order.
fn spread<T: Copy>(v: &[T]) -> (T, T, T, T) {
    match *v {
        [a] => (a, a, a, a),
        [l, t, r, b] => (l, t, r, b),
        _ => unreachable!("parse checks for one or four values"),
    }
}

/// `pct` percent of `side`, rounded to the nearest pixel.
fn percent_of(pct: f32, side: u32) -> Result<u64> {
    if !pct.is_finite() || pct < 0.0 {
        return Err(ScaleUpError::InvalidOption(format!(
            "margin percentage {pct} must be finite and non-negative"
        )));
    }
    // Saturates for absurd values; the limits in `resolve` reject those.
    Ok((f64::from(side) * f64::from(pct) / 100.0).round() as u64)
}

/// What fills the new canvas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fill {
    /// AI outpainting; handled by the outpainting driver, not by [`pad_canvas`].
    Ai,
    /// Reflection about the border that does not repeat the edge pixel.
    Mirror,
    /// The nearest border pixel, repeated.
    Edge,
    /// A solid opaque colour.
    Color([u8; 3]),
    /// Fully transparent black.
    Transparent,
}

impl Fill {
    /// Parses `ai`, `mirror`, `edge`, `transparent` or `#rrggbb` (case-insensitive).
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "ai" => return Ok(Fill::Ai),
            "mirror" => return Ok(Fill::Mirror),
            "edge" => return Ok(Fill::Edge),
            "transparent" => return Ok(Fill::Transparent),
            _ => {}
        }
        let bad = || format!("fill {s:?}: expected ai, mirror, edge, transparent or #rrggbb");
        let hex = lower.strip_prefix('#').ok_or_else(bad)?;
        if hex.len() != 6 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad());
        }
        let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| bad());
        Ok(Fill::Color([channel(0)?, channel(2)?, channel(4)?]))
    }
}

/// A canvas-extension request: where to add canvas and what to fill it with.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendOptions {
    pub margins: MarginSpec,
    pub fill: Fill,
}

/// Canvas of `w+left+right` × `h+top+bottom` with `img` at (left, top). `Fill::Ai` is not valid here.
///
/// The original pixels are copied bit-exact. New pixels are opaque (alpha 255) for every fill except
/// [`Fill::Transparent`], which gives `[0, 0, 0, 0]`.
pub fn pad_canvas(img: &RgbaImage, m: Margins, fill: &Fill) -> Result<RgbaImage> {
    let (w, h) = img.dimensions();
    let too_big =
        || ScaleUpError::InvalidOption(format!("extended canvas for {w}x{h} with {m:?} overflows"));
    let cw = w
        .checked_add(m.left)
        .and_then(|v| v.checked_add(m.right))
        .ok_or_else(too_big)?;
    let ch = h
        .checked_add(m.top)
        .and_then(|v| v.checked_add(m.bottom))
        .ok_or_else(too_big)?;
    if m == Margins::default() && *fill != Fill::Ai {
        return Ok(img.clone());
    }
    if w == 0 || h == 0 {
        return Err(ScaleUpError::InvalidOption(format!(
            "cannot extend an empty {w}x{h} image"
        )));
    }
    // Source coordinate for a canvas coordinate, per axis.
    let source: fn(i64, u32) -> u32 = match fill {
        Fill::Ai => {
            return Err(ScaleUpError::InvalidOption(
                "AI fill is done by the outpainting driver, not by pad_canvas".into(),
            ));
        }
        Fill::Mirror => reflect,
        Fill::Edge => |i, n| i.clamp(0, i64::from(n) - 1) as u32,
        Fill::Color(_) | Fill::Transparent => |_, _| 0,
    };
    let solid = match fill {
        Fill::Color([r, g, b]) => Some(Rgba([*r, *g, *b, 255])),
        Fill::Transparent => Some(Rgba([0, 0, 0, 0])),
        _ => None,
    };
    let (left, top) = (i64::from(m.left), i64::from(m.top));
    Ok(RgbaImage::from_fn(cw, ch, |x, y| {
        let (sx, sy) = (i64::from(x) - left, i64::from(y) - top);
        let inside = (0..i64::from(w)).contains(&sx) && (0..i64::from(h)).contains(&sy);
        if inside {
            return *img.get_pixel(sx as u32, sy as u32);
        }
        if let Some(p) = solid {
            return p;
        }
        let p = img.get_pixel(source(sx, w), source(sy, h));
        Rgba([p[0], p[1], p[2], 255])
    }))
}

/// Reflect padding (period `2·(n−1)`, the edge pixel is not repeated) for any depth.
fn reflect(i: i64, n: u32) -> u32 {
    if n == 1 {
        return 0;
    }
    let n = i64::from(n);
    let period = 2 * (n - 1);
    let r = i.rem_euclid(period);
    (if r < n { r } else { period - r }) as u32
}

/// Margin depth, in source pixels, that one step of the context factor covers.
const CONTEXT_STEP: u32 = 192;

/// Context factor for margins `m`: `clamp(ceil(max_margin / 192), 1, max)`, where `max_margin` is
/// the largest of the four margins and `max` the model's [`Inpainter::max_context_factor`].
///
/// With factor `k` a window covers `window·k` source pixels per side and is downsampled to `window`
/// for the model, so deep margins take fewer rings and see wider context.
pub fn context_factor(m: Margins, max: u32) -> u32 {
    let deepest = m.left.max(m.top).max(m.right).max(m.bottom);
    deepest.div_ceil(CONTEXT_STEP).clamp(1, max.max(1))
}

/// One planned window: canvas position (may be negative or overhang when the canvas is smaller than
/// the window) and the ring it belongs to.
///
/// Positions are in canvas (source-resolution) pixels and the window covers `window·k` of them per
/// side. Rings count from 0; ring `r` fills the margin depth between `r·span/2` and `(r+1)·span/2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub x: i64,
    pub y: i64,
    pub ring: u32,
    pub corner: bool,
}

/// Windows that fill the margins `m` around a `w`×`h` image, in the order they run.
///
/// The margins are filled in rings: each ring adds at most `span / 2` pixels (`span = window·k`) on
/// every side that still has margin left, side windows first (right, left, top, bottom; along a side
/// they overlap by half a window), then the corner windows. Every window is flush with the outer
/// edge of its ring, so in the canvas it is at least half known when the source is at least `span`
/// on both sides (a corner window sees three known quadrants).
///
/// When the known extent along a side is shorter than `span`, no placement can be half known; the
/// single window for that strip is centred on it, so it covers the whole strip and as much known
/// context as there is, and runs anyway. Such a window may overhang the canvas, may cover part of a
/// corner region and may reach past its ring; the driver writes only the pixels of its ring, and the
/// ring's own windows cover that ring, so every canvas pixel is filled by the end of the plan.
pub fn plan_windows(w: u32, h: u32, m: Margins, window: u32, k: u32) -> Vec<Window> {
    let span = i64::from(window) * i64::from(k.max(1));
    let depth = (span / 2).max(1);
    let (mut x0, mut y0) = (i64::from(m.left), i64::from(m.top));
    let (mut x1, mut y1) = (x0 + i64::from(w), y0 + i64::from(h));
    let mut rem = [m.left, m.top, m.right, m.bottom].map(i64::from);
    let mut plan = Vec::new();
    let mut ring = 0;
    while rem.iter().any(|&r| r > 0) {
        let [l, t, r, b] = rem.map(|v| v.min(depth));
        let mut push = |x, y, corner| plan.push(Window { x, y, ring, corner });
        // Outer positions of this ring's windows on each side.
        let (left, right) = (x0 - l, x1 + r - span);
        let (top, bottom) = (y0 - t, y1 + b - span);
        for (d, x) in [(r, right), (l, left)] {
            if d > 0 {
                for y in strip_starts(y0, y1, span) {
                    push(x, y, false);
                }
            }
        }
        for (d, y) in [(t, top), (b, bottom)] {
            if d > 0 {
                for x in strip_starts(x0, x1, span) {
                    push(x, y, false);
                }
            }
        }
        for (dx, dy, x, y) in [
            (r, t, right, top),
            (l, t, left, top),
            (r, b, right, bottom),
            (l, b, left, bottom),
        ] {
            if dx > 0 && dy > 0 {
                push(x, y, true);
            }
        }
        (x0, y0, x1, y1) = (x0 - l, y0 - t, x1 + r, y1 + b);
        rem = [rem[0] - l, rem[1] - t, rem[2] - r, rem[3] - b];
        ring += 1;
    }
    plan
}

/// Window starts that cover the strip `lo..hi` at half overlap; one centred window when the strip is
/// no longer than `span`.
fn strip_starts(lo: i64, hi: i64, span: i64) -> Vec<i64> {
    let n = hi - lo;
    if n <= span {
        return vec![lo + n / 2 - span / 2];
    }
    let step = (span / 2).max(1) as usize;
    let mut starts: Vec<i64> = (lo..hi - span).step_by(step).collect();
    starts.push(hi - span);
    starts
}

/// Extends `img` by `m`, inventing the new pixels with `model` window by window.
///
/// The canvas is first padded by reflection ([`Fill::Mirror`]), which only serves as the starting
/// guess for the unknown pixels. Windows then run in [`plan_windows`] order with the context factor
/// from [`context_factor`]: each is cut from the canvas (pixels past the canvas edge are reflected
/// and passed as unknown), downsampled when `k > 1`, filled, upsampled, and only its still-unknown
/// pixels inside its ring are written and marked known, so one ring adds at most `window·k / 2`
/// pixels of depth on each side. Where windows overlap the earlier one wins.
///
/// The original pixels, alpha included, are copied bit-exact; transparent ones count as known and
/// their stored RGB is used as context. New pixels are opaque. Reports [`Phase::Extending`], then
/// one [`Progress::Tiles`] per planned window; cancellation is checked after every window. An error
/// from `model` is returned as is.
pub fn outpaint(
    model: &dyn Inpainter,
    img: &RgbaImage,
    m: Margins,
    progress: &mut dyn FnMut(Progress),
    cancel: &CancelToken,
) -> Result<RgbaImage> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(ScaleUpError::InvalidOption(format!(
            "cannot extend an empty {w}x{h} image"
        )));
    }
    let window = model.window();
    if window < 2 {
        return Err(ScaleUpError::InvalidOption(format!(
            "inpainting window of {window} px is too small"
        )));
    }
    if cancel.is_cancelled() {
        return Err(ScaleUpError::Cancelled);
    }
    let k = context_factor(m, model.max_context_factor());
    let plan = plan_windows(w, h, m, window, k);
    let mut canvas = pad_canvas(img, m, &Fill::Mirror)?;
    progress(Progress::Phase(Phase::Extending));
    let cw = canvas.width() as usize;
    let mut known = vec![false; cw * canvas.height() as usize];
    for y in m.top..m.top + h {
        let row = y as usize * cw;
        known[row + m.left as usize..row + (m.left + w) as usize].fill(true);
    }
    let total = plan.len() as u32;
    for (i, win) in plan.iter().enumerate() {
        let target = ring_extent(w, h, m, window * k, win.ring);
        fill_window(model, &mut canvas, &mut known, win, target, window, k)?;
        progress(Progress::Tiles {
            done: i as u32 + 1,
            total,
        });
        if cancel.is_cancelled() {
            return Err(ScaleUpError::Cancelled);
        }
    }
    Ok(canvas)
}

/// The canvas rectangle `[x0, y0, x1, y1)` known once `ring` is done: the source grown on each
/// side by the ring's cumulative depth, `min(margin, (ring + 1)·span/2)`.
fn ring_extent(w: u32, h: u32, m: Margins, span: u32, ring: u32) -> [i64; 4] {
    let depth = (i64::from(span) / 2).max(1);
    let grown = |margin: u32| i64::from(margin).min((i64::from(ring) + 1) * depth);
    let (left, top) = (i64::from(m.left), i64::from(m.top));
    [
        left - grown(m.left),
        top - grown(m.top),
        left + i64::from(w) + grown(m.right),
        top + i64::from(h) + grown(m.bottom),
    ]
}

/// Runs one window of `window·k` canvas pixels and writes its still-unknown pixels that lie in
/// `target` (the window's ring, see [`ring_extent`]), so a ring never fills deeper than its depth.
fn fill_window(
    model: &dyn Inpainter,
    canvas: &mut RgbaImage,
    known: &mut [bool],
    win: &Window,
    target: [i64; 4],
    window: u32,
    k: u32,
) -> Result<()> {
    let (cw, ch) = canvas.dimensions();
    let span = window * k;
    let inside =
        |x: i64, y: i64| (0..i64::from(cw)).contains(&x) && (0..i64::from(ch)).contains(&y);
    let index = |x: i64, y: i64| y as usize * cw as usize + x as usize;
    // `target` lies inside the canvas.
    let [tx0, ty0, tx1, ty1] = target;
    let writable = |x: i64, y: i64| (tx0..tx1).contains(&x) && (ty0..ty1).contains(&y);
    // Cut with reflection past the canvas edge. Opaque, so resizing ignores the source alpha.
    let mut cut = RgbaImage::new(span, span);
    let mut cut_known = vec![false; span as usize * span as usize];
    let mut anything_to_fill = false;
    for j in 0..span {
        for i in 0..span {
            let (x, y) = (win.x + i64::from(i), win.y + i64::from(j));
            let p = canvas.get_pixel(reflect(x, cw), reflect(y, ch));
            cut.put_pixel(i, j, Rgba([p[0], p[1], p[2], 255]));
            if inside(x, y) {
                let kn = known[index(x, y)];
                cut_known[(j * span + i) as usize] = kn;
                anything_to_fill |= !kn && writable(x, y);
            }
        }
    }
    if !anything_to_fill {
        return Ok(());
    }
    let (input, mask) = if k == 1 {
        (cut, cut_known)
    } else {
        // A downsampled pixel is known only when its whole k×k block is.
        let mask = (0..window * window)
            .map(|p| {
                let (bx, by) = (p % window * k, p / window * k);
                (by..by + k).all(|y| (bx..bx + k).all(|x| cut_known[(y * span + x) as usize]))
            })
            .collect();
        (resize_rgba(&cut, window, window), mask)
    };
    let (rgb, _) = Image32::from_rgba(&input);
    let out = model.fill(&rgb, &mask)?;
    if (out.width, out.height) != (window, window) {
        return Err(ScaleUpError::Model(format!(
            "inpainter returned {}x{} for a {window}x{window} window",
            out.width, out.height
        )));
    }
    let filled = resize_rgba(&out.to_rgba(None), span, span);
    for j in 0..span {
        for i in 0..span {
            let (x, y) = (win.x + i64::from(i), win.y + i64::from(j));
            if writable(x, y) && !known[index(x, y)] {
                let p = filled.get_pixel(i, j);
                canvas.put_pixel(x as u32, y as u32, Rgba([p[0], p[1], p[2], 255]));
                known[index(x, y)] = true;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(left: u32, top: u32, right: u32, bottom: u32) -> Margins {
        Margins {
            left,
            top,
            right,
            bottom,
        }
    }

    /// A small image with every pixel distinct and a non-opaque alpha, so copies are traceable.
    fn sample(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| {
            Rgba([
                (x as u8).wrapping_mul(10).wrapping_add(1),
                (y as u8).wrapping_mul(10).wrapping_add(2),
                (x + y) as u8,
                (x as u8).wrapping_add(100),
            ])
        })
    }

    fn assert_known_region(out: &RgbaImage, img: &RgbaImage, m: Margins) {
        assert_eq!(
            out.dimensions(),
            (
                img.width() + m.left + m.right,
                img.height() + m.top + m.bottom
            )
        );
        for (x, y, p) in img.enumerate_pixels() {
            assert_eq!(
                out.get_pixel(x + m.left, y + m.top),
                p,
                "known pixel ({x},{y})"
            );
        }
    }

    fn is_new(x: u32, y: u32, img: &RgbaImage, m: Margins) -> bool {
        x < m.left || y < m.top || x >= m.left + img.width() || y >= m.top + img.height()
    }

    fn invalid(r: Result<Margins>) -> bool {
        matches!(r, Err(ScaleUpError::InvalidOption(_)))
    }

    #[test]
    fn parse_accepts_every_form() {
        assert_eq!(
            MarginSpec::parse("64"),
            Ok(MarginSpec::Pixels(px(64, 64, 64, 64)))
        );
        assert_eq!(
            MarginSpec::parse("1,2,3,4"),
            Ok(MarginSpec::Pixels(px(1, 2, 3, 4)))
        );
        assert_eq!(
            MarginSpec::parse(" 0, 10 ,0,5 "),
            Ok(MarginSpec::Pixels(px(0, 10, 0, 5)))
        );
        assert_eq!(
            MarginSpec::parse("20%"),
            Ok(MarginSpec::Percent {
                left: 20.0,
                top: 20.0,
                right: 20.0,
                bottom: 20.0
            })
        );
        assert_eq!(
            MarginSpec::parse("10%,0%,12.5%,5%"),
            Ok(MarginSpec::Percent {
                left: 10.0,
                top: 0.0,
                right: 12.5,
                bottom: 5.0
            })
        );
        assert_eq!(
            MarginSpec::parse("16:9"),
            Ok(MarginSpec::Aspect { w: 16, h: 9 })
        );
    }

    #[test]
    fn parse_rejects_bad_input() {
        for bad in [
            "",
            " ",
            "abc",
            "-5",
            "1.5",
            "1,2",
            "1,2,3",
            "1,2,3,4,5",
            "1,,3,4",
            "10%,5,5,5",
            "-5%",
            "NaN%",
            "inf%",
            "%",
            "1%,2%",
            "16:0",
            "0:9",
            "16:",
            ":9",
            "16:9:1",
            "a:b",
            "64px",
        ] {
            assert!(MarginSpec::parse(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn resolve_pixels_and_percent() {
        let m = MarginSpec::Pixels(px(1, 2, 3, 4)).resolve(100, 50).unwrap();
        assert_eq!(m, px(1, 2, 3, 4));
        // 10% of 105 = 10.5 -> 11 ; 10% of 33 = 3.3 -> 3 ; 25% of 105 = 26.25 -> 26 ; 50% of 33 = 16.5 -> 17
        let spec = MarginSpec::Percent {
            left: 10.0,
            top: 10.0,
            right: 25.0,
            bottom: 50.0,
        };
        assert_eq!(spec.resolve(105, 33).unwrap(), px(11, 3, 26, 17));
        let zero = MarginSpec::Percent {
            left: 0.0,
            top: 0.0,
            right: 0.0,
            bottom: 0.0,
        };
        assert_eq!(zero.resolve(10, 10).unwrap(), Margins::default());
    }

    #[test]
    fn resolve_rejects_bad_percent() {
        for v in [-1.0, f32::NAN, f32::INFINITY] {
            let spec = MarginSpec::Percent {
                left: v,
                top: 0.0,
                right: 0.0,
                bottom: 0.0,
            };
            assert!(invalid(spec.resolve(100, 100)), "accepted {v}");
        }
    }

    #[test]
    fn resolve_aspect() {
        let wide = MarginSpec::Aspect { w: 16, h: 9 };
        // Square 90x90 -> 160x90: grow width by 70, centred.
        assert_eq!(wide.resolve(90, 90).unwrap(), px(35, 0, 35, 0));
        // 100x90 -> 160x90: grow by 60.
        assert_eq!(wide.resolve(100, 90).unwrap(), px(30, 0, 30, 0));
        // Odd growth: 3x3 to 2:1 -> 6x3, grow 3: extra pixel on the right.
        let two_one = MarginSpec::Aspect { w: 2, h: 1 };
        assert_eq!(two_one.resolve(3, 3).unwrap(), px(1, 0, 2, 0));
        // Wider than target: 1000x100 to 16:9 -> height ceil(1000*9/16) = 563, grow 463.
        assert_eq!(wide.resolve(1000, 100).unwrap(), px(0, 231, 0, 232));
        // Taller image to 1:1 grows the width.
        let square = MarginSpec::Aspect { w: 1, h: 1 };
        assert_eq!(square.resolve(10, 30).unwrap(), px(10, 0, 10, 0));
        // Already at the ratio: nothing to add.
        assert_eq!(wide.resolve(160, 90).unwrap(), Margins::default());
        assert_eq!(square.resolve(7, 7).unwrap(), Margins::default());
        // Never crops, and a fractional target rounds up so the ratio is reached.
        let m = wide.resolve(17, 10).unwrap();
        let (cw, ch) = (17 + m.left + m.right, 10 + m.top + m.bottom);
        assert!(cw >= 17 && ch >= 10);
        assert!(cw as u64 * 9 <= ch as u64 * 16 + 16, "{cw}x{ch}");
        // Zero components are rejected.
        assert!(invalid(MarginSpec::Aspect { w: 0, h: 9 }.resolve(10, 10)));
        assert!(invalid(MarginSpec::Aspect { w: 16, h: 0 }.resolve(10, 10)));
    }

    #[test]
    fn resolve_limits() {
        // Exactly 4x each side is allowed.
        assert_eq!(
            MarginSpec::Pixels(px(400, 200, 400, 200))
                .resolve(100, 50)
                .unwrap(),
            px(400, 200, 400, 200)
        );
        // One past 4x on any side is rejected (left/right against width, top/bottom against height).
        for m in [
            px(401, 0, 0, 0),
            px(0, 201, 0, 0),
            px(0, 0, 401, 0),
            px(0, 0, 0, 201),
        ] {
            assert!(invalid(MarginSpec::Pixels(m).resolve(100, 50)), "{m:?}");
        }
        assert!(invalid(
            MarginSpec::Percent {
                left: 400.1,
                top: 0.0,
                right: 0.0,
                bottom: 0.0
            }
            .resolve(1000, 10)
        ));
        // Canvas of exactly 16384 per side is allowed, one more is not.
        assert!(
            MarginSpec::Pixels(px(2192, 2192, 2192, 2192))
                .resolve(12000, 12000)
                .is_ok()
        );
        assert!(invalid(
            MarginSpec::Pixels(px(2193, 0, 2192, 0)).resolve(12000, 12000)
        ));
        assert!(invalid(
            MarginSpec::Pixels(px(0, 2192, 0, 2193)).resolve(12000, 12000)
        ));
        // Aspect growth past the canvas limit is rejected too (12000x4000 to 1:2 needs 24000 tall).
        assert!(invalid(
            MarginSpec::Aspect { w: 1, h: 2 }.resolve(12000, 4000)
        ));
        // Aspect growth past 4x a side is rejected (10x1 to 1:1 needs 9 rows, 5 > 4 at the bottom).
        assert!(invalid(MarginSpec::Aspect { w: 1, h: 1 }.resolve(10, 1)));
        // Huge margins do not overflow.
        assert!(invalid(
            MarginSpec::Pixels(px(u32::MAX, 0, u32::MAX, 0)).resolve(u32::MAX, 1)
        ));
    }

    #[test]
    fn fill_parse() {
        assert_eq!(Fill::parse("ai"), Ok(Fill::Ai));
        assert_eq!(Fill::parse("mirror"), Ok(Fill::Mirror));
        assert_eq!(Fill::parse("Edge"), Ok(Fill::Edge));
        assert_eq!(Fill::parse("transparent"), Ok(Fill::Transparent));
        assert_eq!(Fill::parse("#ff8000"), Ok(Fill::Color([255, 128, 0])));
        assert_eq!(Fill::parse("#A0b1C2"), Ok(Fill::Color([0xa0, 0xb1, 0xc2])));
        for bad in [
            "", "blur", "ff8000", "#ff800", "#ff80000", "#gg0000", "#fff", "# ff800", "#+f8000",
        ] {
            assert!(Fill::parse(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn pad_zero_margins_return_input() {
        let img = sample(3, 2);
        for fill in [
            Fill::Mirror,
            Fill::Edge,
            Fill::Color([1, 2, 3]),
            Fill::Transparent,
        ] {
            assert_eq!(pad_canvas(&img, Margins::default(), &fill).unwrap(), img);
        }
    }

    #[test]
    fn pad_ai_is_rejected() {
        assert!(matches!(
            pad_canvas(&sample(2, 2), px(1, 1, 1, 1), &Fill::Ai),
            Err(ScaleUpError::InvalidOption(_))
        ));
    }

    #[test]
    fn pad_color_and_transparent() {
        let img = sample(3, 2);
        let m = px(2, 1, 3, 4);
        let out = pad_canvas(&img, m, &Fill::Color([9, 8, 7])).unwrap();
        assert_known_region(&out, &img, m);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                assert_eq!(*p, Rgba([9, 8, 7, 255]), "({x},{y})");
            }
        }
        let out = pad_canvas(&img, m, &Fill::Transparent).unwrap();
        assert_known_region(&out, &img, m);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                assert_eq!(*p, Rgba([0, 0, 0, 0]), "({x},{y})");
            }
        }
    }

    #[test]
    fn pad_edge_repeats_border_pixel() {
        let img = sample(3, 2);
        let m = px(2, 3, 4, 1);
        let out = pad_canvas(&img, m, &Fill::Edge).unwrap();
        assert_known_region(&out, &img, m);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                let sx = x.saturating_sub(m.left).min(img.width() - 1);
                let sy = y.saturating_sub(m.top).min(img.height() - 1);
                let s = img.get_pixel(sx, sy);
                assert_eq!(*p, Rgba([s[0], s[1], s[2], 255]), "({x},{y})");
            }
        }
    }

    #[test]
    fn pad_mirror_reflects_without_repeating_edge() {
        // One row 0..4: reflect padding by 3 on the left gives 3 2 1 | 0 1 2 3 | 2 1 0.
        let img = RgbaImage::from_fn(4, 1, |x, _| Rgba([x as u8, 0, 0, 255]));
        let m = px(3, 0, 3, 0);
        let out = pad_canvas(&img, m, &Fill::Mirror).unwrap();
        assert_known_region(&out, &img, m);
        let row: Vec<u8> = (0..out.width()).map(|x| out.get_pixel(x, 0)[0]).collect();
        assert_eq!(row, [3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
    }

    #[test]
    fn pad_mirror_deeper_than_image_wraps() {
        // Width 3, period 4: left margin 7 reads 1 2 1 0 1 2 1 | 0 1 2 | 1 0 1 2 1 0 1 2.
        let img = RgbaImage::from_fn(3, 2, |x, y| Rgba([x as u8, y as u8, 5, 40]));
        let m = px(7, 5, 8, 4);
        let out = pad_canvas(&img, m, &Fill::Mirror).unwrap();
        assert_known_region(&out, &img, m);
        let row: Vec<u8> = (0..out.width()).map(|x| out.get_pixel(x, 0)[0]).collect();
        assert_eq!(row, [1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2]);
        // Height 2, period 2: rows alternate 1 0 1 0 1 | 0 1 | 0 1 0 1.
        let col: Vec<u8> = (0..out.height()).map(|y| out.get_pixel(7, y)[1]).collect();
        assert_eq!(col, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                assert_eq!(p[3], 255, "({x},{y}) not opaque");
                assert_eq!(p[2], 5);
            }
        }
    }

    #[test]
    fn pad_mirror_one_pixel_image() {
        let img = RgbaImage::from_pixel(1, 1, Rgba([10, 20, 30, 77]));
        let m = px(3, 2, 1, 4);
        let out = pad_canvas(&img, m, &Fill::Mirror).unwrap();
        assert_known_region(&out, &img, m);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                assert_eq!(*p, Rgba([10, 20, 30, 255]), "({x},{y})");
            }
        }
        // One pixel wide, taller: columns repeat, rows reflect.
        let img = RgbaImage::from_fn(1, 3, |_, y| Rgba([y as u8, 0, 0, 255]));
        let m = px(2, 2, 2, 2);
        let out = pad_canvas(&img, m, &Fill::Mirror).unwrap();
        assert_known_region(&out, &img, m);
        for x in 0..out.width() {
            let col: Vec<u8> = (0..out.height()).map(|y| out.get_pixel(x, y)[0]).collect();
            assert_eq!(col, [2, 1, 0, 1, 2, 1, 0]);
        }
    }

    // ----------------------------------------------------------------- outpainting driver

    use crate::backend::fake::MeanInpainter;
    use std::sync::Mutex;

    #[test]
    fn context_factor_rule() {
        assert_eq!(context_factor(Margins::default(), 4), 1);
        assert_eq!(context_factor(px(1, 0, 0, 0), 4), 1);
        assert_eq!(context_factor(px(0, 192, 0, 0), 4), 1);
        assert_eq!(context_factor(px(0, 0, 193, 0), 4), 2);
        assert_eq!(context_factor(px(10, 384, 20, 0), 4), 2);
        assert_eq!(context_factor(px(0, 0, 0, 385), 4), 3);
        assert_eq!(context_factor(px(768, 0, 0, 0), 4), 4);
        assert_eq!(context_factor(px(5000, 0, 0, 0), 4), 4);
        // The model's limit caps it; a limit of 0 still gives 1.
        assert_eq!(context_factor(px(5000, 0, 0, 0), 2), 2);
        assert_eq!(context_factor(px(5000, 0, 0, 0), 1), 1);
        assert_eq!(context_factor(px(5000, 0, 0, 0), 0), 1);
    }

    /// Replays `plan` on a known mask the way the driver does (a window writes only inside its
    /// ring's extent). Returns, per window, the number of known canvas pixels it saw and the
    /// bounding box `[x0, y0, x1, y1)` of the known pixels after it ran, and whether every canvas
    /// pixel is known at the end.
    fn replay(
        w: u32,
        h: u32,
        m: Margins,
        span: u32,
        plan: &[Window],
    ) -> (Vec<u64>, Vec<[i64; 4]>, bool) {
        let (cw, ch) = (
            i64::from(w + m.left + m.right),
            i64::from(h + m.top + m.bottom),
        );
        let mut known = vec![false; (cw * ch) as usize];
        for y in m.top..m.top + h {
            for x in m.left..m.left + w {
                known[(i64::from(y) * cw + i64::from(x)) as usize] = true;
            }
        }
        let (left, top) = (i64::from(m.left), i64::from(m.top));
        let mut bbox = [left, top, left + i64::from(w), top + i64::from(h)];
        let mut seen = Vec::new();
        let mut boxes = Vec::new();
        for win in plan {
            let [tx0, ty0, tx1, ty1] = ring_extent(w, h, m, span, win.ring);
            let span = i64::from(span);
            let xs = win.x.max(0)..(win.x + span).min(cw);
            let ys = win.y.max(0)..(win.y + span).min(ch);
            let mut n = 0;
            for y in ys.clone() {
                for x in xs.clone() {
                    n += u64::from(known[(y * cw + x) as usize]);
                }
            }
            seen.push(n);
            let wx = xs.start.max(tx0)..xs.end.min(tx1);
            let wy = ys.start.max(ty0)..ys.end.min(ty1);
            if !wx.is_empty() && !wy.is_empty() {
                bbox = [
                    bbox[0].min(wx.start),
                    bbox[1].min(wy.start),
                    bbox[2].max(wx.end),
                    bbox[3].max(wy.end),
                ];
            }
            for y in wy {
                for x in wx.clone() {
                    known[(y * cw + x) as usize] = true;
                }
            }
            boxes.push(bbox);
        }
        (seen, boxes, known.iter().all(|&k| k))
    }

    /// After each window of ring `r`, no known pixel lies deeper than `(r+1)·span/2` from the
    /// source, checked on each side separately.
    fn assert_ring_depth(
        m: Margins,
        w: u32,
        h: u32,
        span: u32,
        plan: &[Window],
        boxes: &[[i64; 4]],
        case: &str,
    ) {
        let half = (i64::from(span) / 2).max(1);
        let (left, top) = (i64::from(m.left), i64::from(m.top));
        for (win, b) in plan.iter().zip(boxes) {
            let limit = (i64::from(win.ring) + 1) * half;
            let depths = [
                ("left", left - b[0]),
                ("top", top - b[1]),
                ("right", b[2] - left - i64::from(w)),
                ("bottom", b[3] - top - i64::from(h)),
            ];
            for (side, depth) in depths {
                assert!(
                    depth <= limit,
                    "{case}: {win:?} filled {depth} px deep on the {side}, ring limit {limit}"
                );
            }
        }
    }

    /// Rings never go back, and within a ring every side window comes before every corner.
    fn assert_order(plan: &[Window]) {
        for pair in plan.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(a.ring <= b.ring, "ring order: {a:?} then {b:?}");
            if a.ring == b.ring {
                assert!(
                    !a.corner || b.corner,
                    "corner before side: {a:?} then {b:?}"
                );
            }
        }
    }

    #[test]
    fn plan_one_side_only() {
        // Review Focus 1: margin on the right only. Strip of 800 rows, window 512 at half overlap.
        let plan = plan_windows(1000, 800, px(0, 0, 256, 0), 512, 1);
        let w = |y| Window {
            x: 1256 - 512,
            y,
            ring: 0,
            corner: false,
        };
        assert_eq!(plan, vec![w(0), w(256), w(288)]);
        let (seen, _, full) = replay(1000, 800, px(0, 0, 256, 0), 512, &plan);
        assert!(full);
        assert!(seen.iter().all(|&n| n * 2 >= 512 * 512));
        // Each side alone: windows only on that side, never on the three without margin.
        for (m, check) in [
            (
                px(100, 0, 0, 0),
                (|w: &Window| w.x == 0) as fn(&Window) -> bool,
            ),
            (px(0, 100, 0, 0), |w: &Window| w.y == 0),
            (px(0, 0, 0, 100), |w: &Window| w.y == 800 + 100 - 512),
        ] {
            let plan = plan_windows(1000, 800, m, 512, 1);
            assert!(!plan.is_empty());
            for win in &plan {
                assert!(check(win) && !win.corner && win.ring == 0, "{m:?}: {win:?}");
            }
        }
        // Nothing to add, nothing to plan.
        assert!(plan_windows(1000, 800, Margins::default(), 512, 1).is_empty());
    }

    #[test]
    fn plan_sides_before_corners() {
        let m = px(100, 50, 200, 80);
        let plan = plan_windows(1000, 800, m, 512, 1);
        assert_order(&plan);
        let corners: Vec<_> = plan.iter().filter(|w| w.corner).collect();
        assert_eq!(corners.len(), 4);
        // Right-top, left-top, right-bottom, left-bottom, each flush with its canvas corner.
        let (cw, ch) = (1300, 930);
        let at = |x, y| corners.iter().any(|w| (w.x, w.y) == (x, y));
        assert!(at(cw - 512, 0) && at(0, 0) && at(cw - 512, ch - 512) && at(0, ch - 512));
        let (seen, _, full) = replay(1000, 800, m, 512, &plan);
        assert!(full);
        assert!(seen.iter().all(|&n| n * 2 >= 512 * 512));
        // A corner window sees three known quadrants: at least 3/4 known.
        for (win, &n) in plan.iter().zip(&seen) {
            if win.corner {
                assert!(n * 4 >= 3 * 512 * 512, "{win:?} saw {n}");
            }
        }
    }

    #[test]
    fn plan_deep_margin_takes_rings() {
        // Review Focus 3: 600 px with a 512 window: rings of 256, 256 and a partial 88.
        let plan = plan_windows(1200, 1200, px(0, 0, 600, 0), 512, 1);
        assert_order(&plan);
        let rings: Vec<u32> = plan.iter().map(|w| w.ring).collect();
        assert_eq!(rings, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
        let xs: Vec<i64> = plan.iter().map(|w| w.x).step_by(4).collect();
        assert_eq!(xs, [944, 1200, 1288]);
        // The partial ring stops at the canvas edge.
        assert!(
            plan.iter()
                .all(|w| w.x + 512 <= 1800 && w.y >= 0 && w.y + 512 <= 1200)
        );
        let (seen, _, full) = replay(1200, 1200, px(0, 0, 600, 0), 512, &plan);
        assert!(full);
        assert!(seen.iter().all(|&n| n * 2 >= 512 * 512));
        // With context factor 2 the same margin takes two rings of 512 and 88 source pixels.
        let plan = plan_windows(1200, 1200, px(0, 0, 600, 0), 512, 2);
        let xs: Vec<(i64, u32)> = plan.iter().map(|w| (w.x, w.ring)).collect();
        assert_eq!(xs, [(688, 0), (688, 0), (776, 1), (776, 1)]);
        assert!(replay(1200, 1200, px(0, 0, 600, 0), 1024, &plan).2);
    }

    #[test]
    fn plan_source_smaller_than_window() {
        // Review Focus 2: 200x120 extended by 64 on every side; the canvas is smaller than 512.
        let m = px(64, 64, 64, 64);
        let plan = plan_windows(200, 120, m, 512, 1);
        assert_order(&plan);
        assert!(!plan.is_empty());
        let (seen, _, full) = replay(200, 120, m, 512, &plan);
        assert!(full);
        // Every window still runs with real context, and the side windows cover the whole source.
        assert!(seen.iter().all(|&n| n > 0));
        for win in plan.iter().filter(|w| !w.corner) {
            assert!(win.x <= 64 && win.x + 512 >= 264 && win.y <= 64 && win.y + 512 >= 184);
        }
    }

    #[test]
    fn narrow_source_ring_stays_within_its_depth() {
        // 300 px wide, left 1200, right 10, window 512: k = 4, span 2048, ring depth 1024. The
        // ring-0 right window reaches 2038 px left of the source's right edge, deep into the left
        // margin, but may fill only this ring's 1024 px of it.
        let (w, h, m) = (300, 200, px(1200, 0, 10, 0));
        let k = context_factor(m, 4);
        assert_eq!(k, 4);
        let span = 512 * k;
        let plan = plan_windows(w, h, m, 512, k);
        let (_, boxes, full) = replay(w, h, m, span, &plan);
        assert!(full);
        assert_ring_depth(m, w, h, span, &plan, &boxes, "narrow");
        // After the first (right) window only the right margin and ring 0 of the left are filled.
        assert_eq!(boxes[0], [1200 - 1024, 0, 1510, 200]);

        // The driver: the right window fills the right margin and ring 0 of the left; the ring-0 left
        // window then has nothing left and is skipped; the ring-1 left window fills the last 176 px.
        // A driver that wrote the whole window would fill everything in the first call.
        assert_eq!(plan.len(), 3);
        let img = sample(w, h);
        let model = Paint::new(512);
        let out = run(&model, &img, m).unwrap();
        assert_known_region(&out, &img, m);
        assert_eq!(model.calls.lock().unwrap().len(), 2);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                assert_eq!(p[3], 255, "({x},{y})");
            }
        }
    }

    /// Deterministic pseudo-random numbers in `0..n` (LCG).
    fn lcg(s: &mut u32, n: u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*s >> 8) % n
    }

    #[test]
    fn plan_properties_large_source() {
        // Source at least one window span on both sides: every window is at least half known.
        let mut s = 7;
        for _ in 0..300 {
            let (window, k) = (2 * (1 + lcg(&mut s, 8)), 1 + lcg(&mut s, 3));
            let span = window * k;
            let (w, h) = (span + lcg(&mut s, 40), span + lcg(&mut s, 40));
            let m = px(
                lcg(&mut s, 4 * w + 1) * lcg(&mut s, 2),
                lcg(&mut s, 4 * h + 1) * lcg(&mut s, 2),
                lcg(&mut s, 4 * w + 1) * lcg(&mut s, 2),
                lcg(&mut s, 4 * h + 1) * lcg(&mut s, 2),
            );
            let plan = plan_windows(w, h, m, window, k);
            assert_order(&plan);
            let (seen, _, full) = replay(w, h, m, span, &plan);
            let case = format!("{w}x{h} {m:?} window {window} k {k}");
            assert!(full, "{case}: holes");
            let area = u64::from(span) * u64::from(span);
            for (win, &n) in plan.iter().zip(&seen) {
                assert!(n * 2 >= area, "{case}: {win:?} saw {n} of {area}");
            }
            let (cw, ch) = (
                i64::from(w + m.left + m.right),
                i64::from(h + m.top + m.bottom),
            );
            let span = i64::from(span);
            for win in &plan {
                assert!(win.x >= 0 && win.y >= 0, "{case}: {win:?}");
                assert!(win.x + span <= cw && win.y + span <= ch, "{case}: {win:?}");
                // One ring adds at most half a span of depth.
                assert!(
                    i64::from(win.ring) * span / 2
                        < i64::from(m.left.max(m.top).max(m.right).max(m.bottom))
                );
            }
            // A side without margin gets no window; corners only where both sides have one.
            for win in &plan {
                if m.left == 0 && m.right == 0 {
                    assert!(!win.corner && win.x >= 0, "{case}: {win:?}");
                }
                if m.top == 0 && m.bottom == 0 {
                    assert!(!win.corner, "{case}: {win:?}");
                }
            }
        }
    }

    #[test]
    fn plan_properties_any_source() {
        // Any source, including ones shorter than the window: the canvas is always fully covered.
        let mut s = 11;
        for _ in 0..400 {
            let (window, k) = (2 + lcg(&mut s, 15), 1 + lcg(&mut s, 3));
            let span = window * k;
            let (w, h) = (1 + lcg(&mut s, 2 * span), 1 + lcg(&mut s, 2 * span));
            let m = px(
                lcg(&mut s, 4 * w + 1) * lcg(&mut s, 2),
                lcg(&mut s, 4 * h + 1) * lcg(&mut s, 2),
                lcg(&mut s, 4 * w + 1) * lcg(&mut s, 2),
                lcg(&mut s, 4 * h + 1) * lcg(&mut s, 2),
            );
            let plan = plan_windows(w, h, m, window, k);
            assert_order(&plan);
            let (seen, boxes, full) = replay(w, h, m, span, &plan);
            let case = format!("{w}x{h} {m:?} window {window} k {k}");
            assert!(full, "{case}: holes");
            assert!(
                seen.iter().all(|&n| n > 0),
                "{case}: a window without context"
            );
            assert_ring_depth(m, w, h, span, &plan, &boxes, &case);
        }
    }

    /// Paints every unknown pixel pure green, and records the known count of each call.
    struct Paint {
        window: u32,
        calls: Mutex<Vec<usize>>,
    }

    impl Paint {
        fn new(window: u32) -> Self {
            Self {
                window,
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    impl Inpainter for Paint {
        fn window(&self) -> u32 {
            self.window
        }

        fn fill(&self, rgb: &Image32, known: &[bool]) -> Result<Image32> {
            self.calls
                .lock()
                .unwrap()
                .push(known.iter().filter(|&&k| k).count());
            let mut out = rgb.clone();
            for (i, _) in known.iter().enumerate().filter(|(_, k)| !**k) {
                let (x, y) = (i as u32 % rgb.width, i as u32 / rgb.width);
                out.set(0, x, y, 0.0);
                out.set(1, x, y, 1.0);
                out.set(2, x, y, 0.0);
            }
            Ok(out)
        }
    }

    fn run(model: &dyn Inpainter, img: &RgbaImage, m: Margins) -> Result<RgbaImage> {
        outpaint(model, img, m, &mut |_| {}, &CancelToken::new())
    }

    #[test]
    fn outpaint_keeps_known_and_fills_every_new_pixel() {
        // k = 1 here (margins <= 192), so every new pixel comes straight from a window.
        for (w, h, m) in [
            (40, 30, px(0, 0, 9, 0)),
            (40, 30, px(5, 7, 9, 3)),
            (20, 20, px(30, 0, 30, 0)),
            (5, 3, px(20, 12, 2, 1)),
            (1, 1, px(4, 4, 4, 4)),
            (40, 30, px(100, 0, 0, 120)),
        ] {
            let img = sample(w, h);
            let model = Paint::new(16);
            let out = run(&model, &img, m).unwrap();
            assert_known_region(&out, &img, m);
            for (x, y, p) in out.enumerate_pixels() {
                if is_new(x, y, &img, m) {
                    assert_eq!(*p, Rgba([0, 255, 0, 255]), "{w}x{h} {m:?} ({x},{y})");
                }
            }
            assert!(!model.calls.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn outpaint_reflected_pixels_are_unknown() {
        // A 4x3 source on a 6x5 canvas, window 16: the window overhangs the canvas on every side and
        // its reflected pixels must reach the model as unknown. Only the 12 real pixels are known.
        let model = Paint::new(16);
        run(&model, &sample(4, 3), px(1, 1, 1, 1)).unwrap();
        let calls = model.calls.lock().unwrap();
        assert_eq!(calls[0], 12);
    }

    #[test]
    fn outpaint_flat_colour_stays_flat() {
        // Margins up to 400 px give context factors 1 to 3 with a 16 px window.
        let img = RgbaImage::from_pixel(120, 100, Rgba([40, 120, 200, 255]));
        for m in [px(16, 0, 0, 0), px(8, 300, 0, 0), px(0, 0, 400, 200)] {
            let out = run(&MeanInpainter { window: 16 }, &img, m).unwrap();
            assert_known_region(&out, &img, m);
            assert!(
                out.pixels().all(|p| *p == Rgba([40, 120, 200, 255])),
                "{m:?}"
            );
        }
    }

    #[test]
    fn outpaint_large_context_factor_keeps_known() {
        // k = 3: windows of 48 source pixels go through a 16 px model.
        let img = sample(60, 50);
        let m = px(0, 0, 400, 30);
        let out = run(&MeanInpainter { window: 16 }, &img, m).unwrap();
        assert_known_region(&out, &img, m);
        for (x, y, p) in out.enumerate_pixels() {
            if is_new(x, y, &img, m) {
                assert_eq!(p[3], 255, "({x},{y})");
            }
        }
    }

    #[test]
    fn outpaint_zero_margins_return_input() {
        let img = sample(5, 4);
        let model = Paint::new(16);
        assert_eq!(run(&model, &img, Margins::default()).unwrap(), img);
        assert!(model.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn outpaint_model_errors_propagate() {
        struct Broken(bool);
        impl Inpainter for Broken {
            fn window(&self) -> u32 {
                16
            }
            fn fill(&self, rgb: &Image32, _: &[bool]) -> Result<Image32> {
                if self.0 {
                    Err(ScaleUpError::Model("boom".into()))
                } else {
                    Ok(Image32::new(rgb.width - 1, rgb.height))
                }
            }
        }
        let img = sample(20, 20);
        assert!(matches!(
            run(&Broken(true), &img, px(4, 0, 0, 0)),
            Err(ScaleUpError::Model(m)) if m == "boom"
        ));
        assert!(matches!(
            run(&Broken(false), &img, px(4, 0, 0, 0)),
            Err(ScaleUpError::Model(_))
        ));
        assert!(matches!(
            run(&MeanInpainter { window: 1 }, &img, px(4, 0, 0, 0)),
            Err(ScaleUpError::InvalidOption(_))
        ));
        assert!(matches!(
            run(
                &MeanInpainter { window: 16 },
                &RgbaImage::new(0, 3),
                px(4, 0, 0, 0)
            ),
            Err(ScaleUpError::InvalidOption(_))
        ));
    }

    #[test]
    fn outpaint_progress_reports_every_window() {
        let img = sample(40, 30);
        for m in [px(5, 7, 9, 3), px(0, 0, 300, 0)] {
            let model = MeanInpainter { window: 16 };
            let total = plan_windows(40, 30, m, 16, context_factor(m, 4)).len() as u32;
            assert!(total > 1);
            let mut events = Vec::new();
            outpaint(
                &model,
                &img,
                m,
                &mut |p| events.push(p),
                &CancelToken::new(),
            )
            .unwrap();
            let mut expected = vec![Progress::Phase(Phase::Extending)];
            expected.extend((1..=total).map(|done| Progress::Tiles { done, total }));
            assert_eq!(events, expected, "{m:?}");
        }
    }

    #[test]
    fn outpaint_cancel_after_first_window() {
        let img = sample(40, 30);
        let model = Paint::new(16);
        let token = CancelToken::new();
        let res = outpaint(
            &model,
            &img,
            px(5, 7, 9, 3),
            &mut |p| {
                if let Progress::Tiles { done: 1, .. } = p {
                    token.cancel();
                }
            },
            &token,
        );
        assert!(matches!(res, Err(ScaleUpError::Cancelled)));
        assert_eq!(model.calls.lock().unwrap().len(), 1);
        // Already cancelled: no window runs.
        let model = Paint::new(16);
        let res = outpaint(&model, &img, px(5, 7, 9, 3), &mut |_| {}, &token);
        assert!(matches!(res, Err(ScaleUpError::Cancelled)));
        assert!(model.calls.lock().unwrap().is_empty());
    }
}
