//! The backend-agnostic `upscale()` pipeline: tiling, alpha, TTA, resize, progress and cancellation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use image::RgbaImage;

use crate::backend::{DeviceKind, Model};
use crate::image32::Image32;
use crate::resize::{resize_rgba, target_size};
use crate::tiling::{auto_tile_size, run_tiled, tile_grid};
use crate::tta::TtaModel;
use crate::{Result, ScaleUpError};

/// Knobs for one upscale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpscaleOptions {
    /// Tile edge in pixels; 0 = automatic.
    pub tile_size: u32,
    /// Context added around each tile, in pixels.
    pub tile_overlap: u32,
    /// Average over 8 dihedral variants (8× slower, slightly better).
    pub tta: bool,
    /// Final scale 2, 3 or 4; `None` = the model's scale.
    pub output_scale: Option<u32>,
    /// Final width in pixels (height keeps the aspect ratio); overrides `output_scale`.
    pub output_width: Option<u32>,
}

impl Default for UpscaleOptions {
    fn default() -> Self {
        Self {
            tile_size: 0,
            tile_overlap: 16,
            tta: false,
            output_scale: None,
            output_width: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Loading,
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
    let tile = if opts.tile_size == 0 {
        auto_tile_size(DeviceKind::Cpu, None)
    } else {
        opts.tile_size
    };
    let per_pass = tile_grid(w, h, tile, opts.tile_overlap)?.len() as u32;
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
    let pass = |img: &Image32, offset: u32, progress: &mut dyn FnMut(Progress)| {
        run_tiled(model, img, tile, opts.tile_overlap, &mut |done, _| {
            progress(Progress::Tiles {
                done: offset + done,
                total,
            });
            !cancel.is_cancelled()
        })
    };
    let rgb_up = pass(&rgb, 0, progress)?;
    let alpha_up = match uniform_alpha {
        Some(255) => None,
        Some(a) => Some(vec![a; (rgb_up.width * rgb_up.height) as usize]),
        None => {
            let gray = Image32::from_gray_replicated(&alpha, w, h);
            Some(pass(&gray, per_pass, progress)?.channel0_to_u8())
        }
    };
    let out = rgb_up.to_rgba(alpha_up.as_deref());

    if out.dimensions() == target {
        return Ok(out);
    }
    progress(Progress::Phase(Phase::Resizing));
    Ok(resize_rgba(&out, target.0, target.1))
}

#[cfg(test)]
mod tests {
    use super::*;
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
