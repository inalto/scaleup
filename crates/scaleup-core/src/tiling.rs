//! Tile grid geometry and tiled model execution.

use crate::backend::DeviceKind;
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
