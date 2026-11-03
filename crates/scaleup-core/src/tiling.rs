//! Tile grid geometry and tiled model execution.

use crate::backend::{DeviceKind, Model};
use crate::image32::Image32;
use crate::{Result, ScaleUpError};

/// Pixel rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// One unit of tiled work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileJob {
    /// Region of the image fed to the model (includes overlap), in image coordinates.
    pub input: TileRect,
    /// Region to paste back, relative to `input`. All `keep` rects partition the image.
    pub keep: TileRect,
}

/// Splits a `width`×`height` image into tiles of at most `tile`×`tile` pixels.
///
/// Keep rects are laid out on a grid with stride `tile - 2*overlap`; the last row/column shrinks to
/// fit. `overlap` is capped at `tile / 4` (see [`effective_overlap`]). Each input rect is its keep rect grown by `overlap` on every side, clamped to the image.
/// An image that fits in one tile yields a single job covering it whole.
pub fn tile_grid(width: u32, height: u32, tile: u32, overlap: u32) -> Result<Vec<TileJob>> {
    if tile < 32 {
        return Err(ScaleUpError::InvalidOption(format!(
            "tile size {tile} is below the minimum of 32"
        )));
    }
    let overlap = effective_overlap(tile, overlap);
    if width <= tile && height <= tile {
        let whole = TileRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        return Ok(vec![TileJob {
            input: whole,
            keep: whole,
        }]);
    }
    let stride = tile - 2 * overlap;
    // (start, len) of keep spans and their input spans along one axis.
    let spans = |len: u32| -> Vec<(u32, u32, u32, u32)> {
        (0..len)
            .step_by(stride as usize)
            .map(|k0| {
                let k1 = (k0 + stride).min(len);
                let i0 = k0.saturating_sub(overlap);
                let i1 = (k1 + overlap).min(len);
                (i0, i1 - i0, k0 - i0, k1 - k0)
            })
            .collect()
    };
    let (xs, ys) = (spans(width), spans(height));
    let mut jobs = Vec::with_capacity(xs.len() * ys.len());
    for &(iy, ih, ky, kh) in &ys {
        for &(ix, iw, kx, kw) in &xs {
            jobs.push(TileJob {
                input: TileRect {
                    x: ix,
                    y: iy,
                    w: iw,
                    h: ih,
                },
                keep: TileRect {
                    x: kx,
                    y: ky,
                    w: kw,
                    h: kh,
                },
            });
        }
    }
    Ok(jobs)
}

/// Overlap actually used for `tile`: capped at `tile / 4` so the stride stays at least half a tile.
pub fn effective_overlap(tile: u32, overlap: u32) -> u32 {
    overlap.min(tile / 4)
}

/// Default tile size (spec "Default tile size"): GPU with ≥ 6 GiB free → 512, other GPU → 256, CPU → 128.
pub fn auto_tile_size(kind: DeviceKind, free_vram_bytes: Option<u64>) -> u32 {
    const SIX_GIB: u64 = 6 << 30;
    if !kind.is_gpu() {
        128
    } else if free_vram_bytes.is_some_and(|b| b >= SIX_GIB) {
        512
    } else {
        256
    }
}

/// Mirror index `i` into `0..n` without repeating the edge pixel (`-1 → 1`, `n → n-2`).
fn reflect_index(i: i64, n: u32) -> u32 {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as i64 - 1);
    let m = i.rem_euclid(period);
    (if m < n as i64 { m } else { period - m }) as u32
}

/// Extracts the `w`×`h` window whose top-left is (`x0`, `y0`) in image coordinates; samples outside
/// the image are reflected back in.
fn window_reflect(img: &Image32, x0: i64, y0: i64, w: u32, h: u32) -> Image32 {
    let mut out = Image32::new(w, h);
    let xs: Vec<u32> = (0..w)
        .map(|x| reflect_index(x0 + x as i64, img.width))
        .collect();
    for c in 0..3 {
        for y in 0..h {
            let sy = reflect_index(y0 + y as i64, img.height);
            for (x, &sx) in xs.iter().enumerate() {
                out.set(c, x as u32, y, img.get(c, sx, sy));
            }
        }
    }
    out
}

/// Pads `img` by reflection (edge pixel not repeated). Pads larger than the image fold back and forth.
pub fn reflect_pad(img: &Image32, left: u32, top: u32, right: u32, bottom: u32) -> Image32 {
    window_reflect(
        img,
        -(left as i64),
        -(top as i64),
        img.width + left + right,
        img.height + top + bottom,
    )
}

/// Runs `model` over `img` tile by tile and stitches the result.
///
/// Every tile is fed as its keep rect plus `overlap` on each side, reflect-padded where it crosses the
/// image border, so the model always sees full context. An image that fits in one tile is processed
/// whole. Every model input is then reflect-padded on the right/bottom to a multiple of
/// [`Model::input_multiple`] (4 by default). `on_tile(done, total)` is called
/// after each tile; returning `false` aborts with [`ScaleUpError::Cancelled`].
pub fn run_tiled(
    model: &dyn Model,
    img: &Image32,
    tile: u32,
    overlap: u32,
    on_tile: &mut dyn FnMut(u32, u32) -> bool,
) -> Result<Image32> {
    let jobs = tile_grid(img.width, img.height, tile, overlap)?;
    let overlap = effective_overlap(tile, overlap);
    let s = model.scale();
    let multiple = model.input_multiple().max(1);
    let total = jobs.len() as u32;
    let mut out = Image32::new(img.width * s, img.height * s);
    let single = total == 1;
    for (n, job) in jobs.iter().enumerate() {
        let kx = job.input.x + job.keep.x;
        let ky = job.input.y + job.keep.y;
        let pad = if single { 0 } else { overlap };
        // Extra pixels needed to reach the model's multiple go on the right/bottom (reflected).
        let w = (job.keep.w + 2 * pad).next_multiple_of(multiple);
        let h = (job.keep.h + 2 * pad).next_multiple_of(multiple);
        let input = window_reflect(img, kx as i64 - pad as i64, ky as i64 - pad as i64, w, h);
        let result = model.run(&input)?;
        if result.width != w * s || result.height != h * s {
            return Err(ScaleUpError::Model(format!(
                "model returned {}x{} for a {w}x{h} tile at scale {s}",
                result.width, result.height
            )));
        }
        let kept = result.crop(pad * s, pad * s, job.keep.w * s, job.keep.h * s);
        out.paste(&kept, kx * s, ky * s);
        if !on_tile(n as u32 + 1, total) {
            return Err(ScaleUpError::Cancelled);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceKind;

    const GIB: u64 = 1 << 30;

    #[test]
    fn grid_partitions_exactly() {
        let sizes = [1u32, 5, 31, 32, 33, 255, 256, 257, 1000, 733];
        for &w in &sizes {
            for &h in &sizes {
                for &tile in &[32u32, 64, 256] {
                    let jobs = tile_grid(w, h, tile, 16).unwrap();
                    let mut cover = vec![0u8; (w * h) as usize];
                    let mut area = 0u64;
                    for j in &jobs {
                        let (i, k) = (j.input, j.keep);
                        assert!(i.x + i.w <= w && i.y + i.h <= h, "input outside image");
                        assert!(i.w <= tile && i.h <= tile, "input larger than tile");
                        assert!(k.x + k.w <= i.w && k.y + k.h <= i.h, "keep outside input");
                        area += (k.w * k.h) as u64;
                        for y in 0..k.h {
                            for x in 0..k.w {
                                cover[((i.y + k.y + y) * w + i.x + k.x + x) as usize] += 1;
                            }
                        }
                    }
                    assert_eq!(area, (w * h) as u64, "{w}x{h} tile {tile}");
                    assert!(cover.iter().all(|&c| c == 1), "{w}x{h} tile {tile}");
                }
            }
        }
    }

    #[test]
    fn small_image_single_tile() {
        let jobs = tile_grid(200, 100, 256, 16).unwrap();
        assert_eq!(jobs.len(), 1);
        let whole = TileRect {
            x: 0,
            y: 0,
            w: 200,
            h: 100,
        };
        assert_eq!(jobs[0].input, whole);
        assert_eq!(jobs[0].keep, whole);
    }

    #[test]
    fn input_includes_overlap_clamped() {
        // tile 64, overlap 16 -> stride 32; 100 px wide -> keep columns 0..32, 32..64, 64..96, 96..100
        let jobs = tile_grid(100, 40, 64, 16).unwrap();
        let first = jobs[0];
        assert_eq!(
            first.input,
            TileRect {
                x: 0,
                y: 0,
                w: 48,
                h: 40
            }
        );
        assert_eq!(
            first.keep,
            TileRect {
                x: 0,
                y: 0,
                w: 32,
                h: 32
            }
        );
        let interior = jobs.iter().find(|j| j.input.x == 16).unwrap();
        assert_eq!(interior.input.w, 64);
        assert_eq!(interior.keep.x, 16);
        assert!(tile_grid(100, 40, 31, 4).is_err());
        // Overlap is capped at tile/4: tile 32 + overlap 16 behaves like overlap 8 (stride 16).
        let capped = tile_grid(100, 40, 32, 16).unwrap();
        assert_eq!(
            capped[0].input,
            TileRect {
                x: 0,
                y: 0,
                w: 24,
                h: 24
            }
        );
    }

    #[test]
    fn auto_tile_size_table() {
        assert_eq!(auto_tile_size(DeviceKind::Cuda, Some(8 * GIB)), 512);
        assert_eq!(auto_tile_size(DeviceKind::Cuda, Some(4 * GIB)), 256);
        assert_eq!(auto_tile_size(DeviceKind::Cuda, None), 256);
        assert_eq!(auto_tile_size(DeviceKind::DirectMl, Some(6 * GIB)), 512);
        assert_eq!(auto_tile_size(DeviceKind::Cpu, Some(64 * GIB)), 128);
        assert_eq!(auto_tile_size(DeviceKind::Cpu, None), 128);
    }
}

#[cfg(test)]
mod run_tests {
    use super::*;
    use crate::backend::Model;
    use crate::backend::fake::NearestModel;
    use crate::image32::Image32;

    /// Nearest 2× that, like pixel-unshuffle / window-attention nets, rejects inputs whose sides are
    /// not a multiple of 16.
    struct StrictModel;

    impl Model for StrictModel {
        fn scale(&self) -> u32 {
            2
        }

        fn input_multiple(&self) -> u32 {
            16
        }

        fn run(&self, tile: &Image32) -> Result<Image32> {
            if !tile.width.is_multiple_of(16) || !tile.height.is_multiple_of(16) {
                return Err(ScaleUpError::Model(format!(
                    "{}x{} not a multiple of 16",
                    tile.width, tile.height
                )));
            }
            NearestModel { scale: 2 }.run(tile)
        }
    }

    #[test]
    fn windows_padded_to_model_multiple() {
        for &(w, h, tile, overlap) in &[
            (5, 3, 64, 8),
            (1000, 733, 256, 16),
            (1, 1000, 32, 16),
            (1001, 1001, 256, 16),
            (37, 300, 100, 10),
        ] {
            let img = ramp(w, h);
            let tiled = run_tiled(&StrictModel, &img, tile, overlap, &mut |_, _| true)
                .unwrap_or_else(|e| panic!("{w}x{h} tile {tile}: {e}"));
            assert_eq!(
                tiled,
                NearestModel { scale: 2 }.run(&img).unwrap(),
                "{w}x{h} tile {tile}"
            );
        }
    }

    fn ramp(w: u32, h: u32) -> Image32 {
        let mut img = Image32::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = ((x * 7 + y * 13 + c as u32 * 31) % 256) as f32 / 255.0;
                    img.set(c, x, y, v);
                }
            }
        }
        img
    }

    #[test]
    fn tiled_equals_untiled_nearest() {
        let model = NearestModel { scale: 2 };
        for &(w, h, tile, overlap) in &[
            (5, 3, 64, 8),
            (33, 31, 64, 8),
            (300, 217, 64, 8),
            (1000, 733, 256, 16),
        ] {
            let img = ramp(w, h);
            let tiled = run_tiled(&model, &img, tile, overlap, &mut |_, _| true).unwrap();
            assert_eq!(tiled, model.run(&img).unwrap(), "{w}x{h} tile {tile}");
        }
    }

    #[test]
    fn reflect_pad_values() {
        let mut row = Image32::new(3, 1);
        for (x, v) in [0.1f32, 0.2, 0.3].into_iter().enumerate() {
            row.set(0, x as u32, 0, v);
        }
        let padded = reflect_pad(&row, 2, 0, 0, 0);
        let got: Vec<f32> = (0..5).map(|x| padded.get(0, x, 0)).collect();
        assert_eq!(got, vec![0.3, 0.2, 0.1, 0.2, 0.3]);
        let both = reflect_pad(&row, 1, 1, 2, 0);
        assert_eq!((both.width, both.height), (6, 2));
        assert_eq!(both.get(0, 5, 1), 0.1);
        // A 1-pixel-wide image reflects onto itself.
        let one = reflect_pad(&Image32::new(1, 1), 3, 3, 3, 3);
        assert_eq!((one.width, one.height), (7, 7));
    }

    #[test]
    fn cancel_stops_early() {
        let model = NearestModel { scale: 2 };
        let img = ramp(300, 217);
        let mut calls = 0;
        let res = run_tiled(&model, &img, 64, 8, &mut |done, total| {
            calls += 1;
            assert!(done < total);
            false
        });
        assert!(matches!(res, Err(ScaleUpError::Cancelled)));
        assert_eq!(calls, 1);
    }

    #[test]
    fn progress_counts_every_tile() {
        let model = NearestModel { scale: 2 };
        let img = ramp(300, 217);
        let total_jobs = tile_grid(300, 217, 64, 8).unwrap().len() as u32;
        let mut seen = Vec::new();
        run_tiled(&model, &img, 64, 8, &mut |d, t| {
            seen.push((d, t));
            true
        })
        .unwrap();
        assert_eq!(seen.len() as u32, total_jobs);
        assert_eq!(*seen.last().unwrap(), (total_jobs, total_jobs));
    }
}
