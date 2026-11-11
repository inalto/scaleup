//! The backend-agnostic `upscale()` pipeline: tiling, alpha, TTA, resize, progress and cancellation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use image::{Rgba, RgbaImage};

use crate::backend::{DeviceKind, Model};
use crate::image32::Image32;
use crate::resize::{resize_rgba, target_size};
use crate::tiling::{auto_tile_size, room_for_overlap, run_tiled_into, tile_grid};
use crate::tta::TtaModel;
use crate::{Result, ScaleUpError};

/// `tile`, shrunk so that a tile with its overlap, padded to the input multiple, fits the model.
/// Overlap is capped at a quarter tile, so two thirds of the limit always fits.
fn fit_tile(tile: u32, overlap: u32, model: &dyn Model) -> u32 {
    let Some(max) = model.max_input_side() else {
        return tile;
    };
    let room = max
        .saturating_sub(model.input_multiple())
        .saturating_sub(2 * overlap);
    tile.min(room.max(max * 2 / 3 / model.input_multiple().max(1) * model.input_multiple().max(1)))
}

/// Knobs for one upscale.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UpscaleOptions {
    /// Tile edge in pixels; 0 = automatic.
    pub tile_size: u32,
    /// Context added around each tile, in pixels; `None` = what the model asks for.
    pub tile_overlap: Option<u32>,
    /// Average over 8 dihedral variants (8× slower, slightly better).
    pub tta: bool,
    /// Final scale 2, 3 or 4; `None` = the model's scale.
    pub output_scale: Option<u32>,
    /// Final width in pixels (height keeps the aspect ratio); overrides `output_scale`.
    pub output_width: Option<u32>,
}

impl UpscaleOptions {
    /// The overlap to use with `model`.
    pub fn overlap_for(&self, model: &dyn Model) -> u32 {
        self.tile_overlap
            .unwrap_or_else(|| model.preferred_overlap())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Loading,
    /// Inventing new canvas around the image (AI fill), before upscaling.
    Extending,
    Upscaling,
    Resizing,
    Saving,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    Phase(Phase),
    /// Tiles finished so far across all passes (RGB, then alpha if present).
    Tiles {
        done: u32,
        total: u32,
    },
}

/// Shared cancellation flag; clones observe the same flag.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Upscales `input` with `model`.
///
/// The alpha channel is upscaled by a second model pass only when it is not uniform; a constant alpha
/// (e.g. fully opaque or fully transparent) is copied exactly.
/// `tile_size == 0` uses the CPU default here; device-aware defaults are applied by
/// [`crate::facade::Upscaler::effective_tile`]. Cancellation is checked after every tile.
pub fn upscale(
    model: &dyn Model,
    input: &RgbaImage,
    opts: &UpscaleOptions,
    progress: &mut dyn FnMut(Progress),
    cancel: &CancelToken,
) -> Result<RgbaImage> {
    let (w, h) = input.dimensions();
    if w == 0 || h == 0 {
        return Err(ScaleUpError::InvalidOption(
            "image has zero width or height".into(),
        ));
    }
    let target = target_size(w, h, model.scale(), opts.output_scale, opts.output_width)?;
    let overlap = opts.overlap_for(model);
    let tile = if opts.tile_size == 0 {
        room_for_overlap(auto_tile_size(DeviceKind::Cpu, None), overlap)
    } else {
        opts.tile_size
    };
    let tile = fit_tile(tile, overlap, model);
    let per_pass = tile_grid(w, h, tile, overlap)?.len() as u32;
    if cancel.is_cancelled() {
        return Err(ScaleUpError::Cancelled);
    }

    let tta;
    let model: &dyn Model = if opts.tta {
        tta = TtaModel { inner: model };
        &tta
    } else {
        model
    };

    let (rgb, alpha) = Image32::from_rgba(input);
    // A uniform alpha plane (fully opaque, fully transparent, or any constant) is copied rather than
    // inferred: exact, and half the work.
    let uniform_alpha = alpha.iter().all(|&a| a == alpha[0]).then_some(alpha[0]);
    let has_alpha = uniform_alpha.is_none();
    let total = per_pass * if has_alpha { 2 } else { 1 };

    progress(Progress::Phase(Phase::Upscaling));
    // Tiles are written straight into the 8-bit output, so peak memory is the output plus one tile.
    let s = model.scale();
    let mut out =
        RgbaImage::from_pixel(w * s, h * s, Rgba([0, 0, 0, uniform_alpha.unwrap_or(255)]));
    let mut pass = |img: &Image32,
                    offset: u32,
                    channels: std::ops::Range<usize>,
                    progress: &mut dyn FnMut(Progress)| {
        let mut on_tile = |done, _| {
            progress(Progress::Tiles {
                done: offset + done,
                total,
            });
            !cancel.is_cancelled()
        };
        // Output channel `c` comes from tile channel `c` (RGB) or from channel 0 (alpha → byte 3).
        let mut sink = |t: &Image32, x0: u32, y0: u32| {
            for y in 0..t.height {
                for x in 0..t.width {
                    let px = out.get_pixel_mut(x0 + x, y0 + y);
                    for c in channels.clone() {
                        let src = if c == 3 { 0 } else { c };
                        px[c] = to_u8(t.get(src, x, y));
                    }
                }
            }
        };
        run_tiled_into(model, img, tile, overlap, &mut on_tile, &mut sink)
    };
    pass(&rgb, 0, 0..3, progress)?;
    if has_alpha {
        let gray = Image32::from_gray_replicated(&alpha, w, h);
        pass(&gray, per_pass, 3..4, progress)?;
    }

    if out.dimensions() == target {
        return Ok(out);
    }
    progress(Progress::Phase(Phase::Resizing));
    Ok(resize_rgba(&out, target.0, target.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refuses inputs above a side limit, as a GPU model with bounded buffers does.
    struct LimitedModel {
        max: u32,
    }

    impl Model for LimitedModel {
        fn scale(&self) -> u32 {
            2
        }

        fn max_input_side(&self) -> Option<u32> {
            Some(self.max)
        }

        fn run(&self, tile: &Image32) -> Result<Image32> {
            assert!(
                tile.width <= self.max && tile.height <= self.max,
                "{}x{} exceeds {}",
                tile.width,
                tile.height,
                self.max
            );
            crate::backend::fake::NearestModel { scale: 2 }.run(tile)
        }
    }

    #[test]
    fn tile_shrinks_to_fit_the_model_limit() {
        let img = RgbaImage::from_pixel(300, 200, Rgba([9, 8, 7, 255]));
        let opts = UpscaleOptions {
            tile_size: 256,
            ..UpscaleOptions::default()
        };
        let out = upscale(
            &LimitedModel { max: 64 },
            &img,
            &opts,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!(out.dimensions(), (600, 400));
        assert_eq!(out.get_pixel(599, 399).0, [9, 8, 7, 255]);
    }
    use crate::ScaleUpError;
    use crate::backend::fake::NearestModel;
    use crate::tiling::tile_grid;
    use image::{Rgba, RgbaImage};

    fn opaque(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| Rgba([(x * 9) as u8, (y * 5) as u8, 77, 255]))
    }

    fn run(
        model: &NearestModel,
        img: &RgbaImage,
        opts: &UpscaleOptions,
    ) -> (Result<RgbaImage>, Vec<Progress>) {
        let mut events = Vec::new();
        let res = upscale(
            model,
            img,
            opts,
            &mut |p| events.push(p),
            &CancelToken::new(),
        );
        (res, events)
    }

    /// Nearest 2× that asks for a given overlap.
    struct WantsOverlap(u32);

    impl Model for WantsOverlap {
        fn scale(&self) -> u32 {
            2
        }

        fn preferred_overlap(&self) -> u32 {
            self.0
        }

        fn run(&self, tile: &Image32) -> Result<Image32> {
            NearestModel { scale: 2 }.run(tile)
        }
    }

    /// Tiles reported when `model` upscales a 300×200 image.
    fn tiles_used(model: &dyn Model, opts: &UpscaleOptions) -> u32 {
        let mut tiles = 0;
        upscale(
            model,
            &opaque(300, 200),
            opts,
            &mut |p| {
                if let Progress::Tiles { total, .. } = p {
                    tiles = total
                }
            },
            &CancelToken::new(),
        )
        .unwrap();
        tiles
    }

    #[test]
    fn model_chooses_the_overlap_by_default() {
        let opts = UpscaleOptions {
            tile_size: 128,
            ..Default::default()
        };
        let want = tile_grid(300, 200, 128, 32).unwrap().len() as u32;
        assert_eq!(tiles_used(&WantsOverlap(32), &opts), want);
    }

    #[test]
    fn explicit_overlap_overrides_the_model() {
        let opts = UpscaleOptions {
            tile_size: 128,
            tile_overlap: Some(8),
            ..Default::default()
        };
        let want = tile_grid(300, 200, 128, 8).unwrap().len() as u32;
        assert_eq!(tiles_used(&WantsOverlap(32), &opts), want);
    }

    #[test]
    fn automatic_tile_grows_so_the_overlap_is_not_capped() {
        let want = tile_grid(300, 200, 256, 64).unwrap().len() as u32;
        assert_eq!(
            tiles_used(&WantsOverlap(64), &UpscaleOptions::default()),
            want
        );
    }

    #[test]
    fn opaque_image_single_pass() {
        let model = NearestModel { scale: 2 };
        let img = opaque(300, 200);
        let opts = UpscaleOptions {
            tile_size: 64,
            ..Default::default()
        };
        let (res, events) = run(&model, &img, &opts);
        let out = res.unwrap();
        assert_eq!(out.dimensions(), (600, 400));
        assert!(out.pixels().all(|p| p[3] == 255));
        let per_pass = tile_grid(300, 200, 64, 16).unwrap().len() as u32;
        let tiles: Vec<(u32, u32)> = events
            .iter()
            .filter_map(|e| match e {
                Progress::Tiles { done, total } => Some((*done, *total)),
                _ => None,
            })
            .collect();
        assert_eq!(tiles.len() as u32, per_pass);
        assert!(tiles.iter().all(|&(_, t)| t == per_pass));
    }

    #[test]
    fn alpha_roundtrip_exact() {
        let model = NearestModel { scale: 2 };
        let img = RgbaImage::from_fn(70, 45, |x, y| {
            Rgba([10, 20, 30, if (x / 3 + y / 3) % 2 == 0 { 0 } else { 255 }])
        });
        let opts = UpscaleOptions {
            tile_size: 32,
            ..Default::default()
        };
        let (res, events) = run(&model, &img, &opts);
        let out = res.unwrap();
        for (x, y, p) in out.enumerate_pixels() {
            assert_eq!(p[3], img.get_pixel(x / 2, y / 2)[3], "alpha at {x},{y}");
        }
        // Two passes (RGB + alpha) share one tile counter.
        let last = events.iter().rev().find_map(|e| match e {
            Progress::Tiles { done, total } => Some((*done, *total)),
            _ => None,
        });
        let per_pass = tile_grid(70, 45, 32, 16).unwrap().len() as u32;
        assert_eq!(last, Some((2 * per_pass, 2 * per_pass)));
    }

    /// Like a real network: nearest 2× plus a small bias, so values are never reproduced exactly.
    struct BiasedModel;

    impl Model for BiasedModel {
        fn scale(&self) -> u32 {
            2
        }

        fn run(&self, tile: &Image32) -> Result<Image32> {
            let mut out = NearestModel { scale: 2 }.run(tile)?;
            out.data.iter_mut().for_each(|v| *v += 3.0 / 255.0);
            Ok(out)
        }
    }

    #[test]
    fn uniform_alpha_is_copied_not_inferred() {
        for a in [0u8, 128, 255] {
            let img = RgbaImage::from_pixel(9, 7, Rgba([1, 2, 3, a]));
            let mut tiles = 0;
            let out = upscale(
                &BiasedModel,
                &img,
                &UpscaleOptions::default(),
                &mut |p| {
                    if let Progress::Tiles { total, .. } = p {
                        tiles = total
                    }
                },
                &CancelToken::new(),
            )
            .unwrap();
            assert!(out.pixels().all(|p| p[3] == a), "alpha {a}");
            assert_eq!(tiles, 1, "alpha {a}: only the RGB pass runs");
        }
    }

    #[test]
    fn fully_transparent_and_opaque_alpha_exact() {
        let model = NearestModel { scale: 2 };
        for a in [0u8, 255] {
            let img = RgbaImage::from_pixel(9, 7, Rgba([1, 2, 3, a]));
            let out = run(&model, &img, &UpscaleOptions::default()).0.unwrap();
            assert!(out.pixels().all(|p| p[3] == a), "alpha {a}");
        }
    }

    #[test]
    fn progress_phases_in_order() {
        let model = NearestModel { scale: 2 };
        let img = opaque(20, 10);
        let phases = |opts: &UpscaleOptions| -> Vec<Phase> {
            run(&model, &img, opts)
                .1
                .into_iter()
                .filter_map(|e| match e {
                    Progress::Phase(p) => Some(p),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(phases(&UpscaleOptions::default()), vec![Phase::Upscaling]);
        let resized = UpscaleOptions {
            output_scale: Some(3),
            ..Default::default()
        };
        assert_eq!(phases(&resized), vec![Phase::Upscaling, Phase::Resizing]);
    }

    #[test]
    fn cancel_returns_cancelled() {
        let model = NearestModel { scale: 2 };
        let token = CancelToken::new();
        token.cancel();
        let res = upscale(
            &model,
            &opaque(20, 10),
            &UpscaleOptions::default(),
            &mut |_| {},
            &token,
        );
        assert!(matches!(res, Err(ScaleUpError::Cancelled)));
    }

    #[test]
    fn cancel_mid_run_stops_within_one_tile() {
        let model = NearestModel { scale: 2 };
        let token = CancelToken::new();
        let observer = token.clone();
        let mut tiles_seen = 0;
        let opts = UpscaleOptions {
            tile_size: 32,
            ..Default::default()
        };
        let res = upscale(
            &model,
            &opaque(200, 200),
            &opts,
            &mut |p| {
                if let Progress::Tiles { .. } = p {
                    tiles_seen += 1;
                    observer.cancel();
                }
            },
            &token,
        );
        assert!(matches!(res, Err(ScaleUpError::Cancelled)));
        assert_eq!(tiles_seen, 1);
    }

    #[test]
    fn custom_width_applies() {
        let model = NearestModel { scale: 2 };
        let opts = UpscaleOptions {
            output_width: Some(50),
            ..Default::default()
        };
        let out = run(&model, &opaque(20, 10), &opts).0.unwrap();
        assert_eq!(out.dimensions(), (50, 25));
    }

    #[test]
    fn invalid_scale_rejected_before_work() {
        let model = NearestModel { scale: 2 };
        let opts = UpscaleOptions {
            output_scale: Some(7),
            ..Default::default()
        };
        let (res, events) = run(&model, &opaque(20, 10), &opts);
        assert!(matches!(res, Err(ScaleUpError::InvalidOption(_))));
        assert!(events.is_empty());
    }

    #[test]
    fn tta_option_runs() {
        let model = NearestModel { scale: 2 };
        let img = opaque(20, 10);
        let plain = run(&model, &img, &UpscaleOptions::default()).0.unwrap();
        let tta = run(
            &model,
            &img,
            &UpscaleOptions {
                tta: true,
                ..Default::default()
            },
        )
        .0
        .unwrap();
        assert_eq!(plain, tta);
    }
}
