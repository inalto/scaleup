//! Canvas extension: margin specs, fill modes and the plain (non-AI) padding of an image.

use image::{Rgba, RgbaImage};

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
                x as u8 * 10 + 1,
                y as u8 * 10 + 2,
                (x + y) as u8,
                100 + x as u8,
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
}
