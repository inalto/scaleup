//! [`scaleup_core::Model`] on top of the executor.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use scaleup_core::{Image32, Model, Result, ScaleUpError};

use crate::exec::{Executor, Plan, plan};
use crate::ncnn::{Graph, LayerKind, infer_channels, infer_scale};

const SIDE_STEP: u32 = 16;
const SIDE_LIMIT: u32 = 16_384;

/// The largest square input whose buffers all fit `max_buffer` bytes.
fn largest_side(graph: &Graph, max_buffer: u64) -> Result<u32> {
    let (mut fits, mut too_big) = (SIDE_STEP, SIDE_LIMIT + SIDE_STEP);
    while too_big - fits > SIDE_STEP {
        let mid = (fits + too_big) / 2 / SIDE_STEP * SIDE_STEP;
        if plan(graph, 3, mid, mid)?.max_slot_bytes() <= max_buffer {
            fits = mid;
        } else {
            too_big = mid;
        }
    }
    Ok(fits)
}

/// An ncnn network ready to run on one device.
pub struct NcnnModel {
    graph: Graph,
    scale: u32,
    exec: Executor,
    max_side: u32,
    plans: Mutex<HashMap<(u32, u32), Arc<Plan>>>,
}

impl NcnnModel {
    /// `max_buffer` is the largest single buffer the device can bind.
    pub fn new(graph: Graph, exec: Executor, max_buffer: u64) -> Result<Self> {
        let channels = infer_channels(&graph, 3)?[graph.output_blob];
        if channels != 3 {
            return Err(ScaleUpError::Model(format!(
                "the network outputs {channels} channels, expected 3"
            )));
        }
        Ok(Self {
            scale: infer_scale(&graph),
            max_side: largest_side(&graph, max_buffer)?,
            graph,
            exec,
            plans: Mutex::new(HashMap::new()),
        })
    }

    fn plan_for(&self, h: u32, w: u32) -> Result<Arc<Plan>> {
        let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = plans.get(&(h, w)) {
            return Ok(p.clone());
        }
        let p = Arc::new(plan(&self.graph, 3, h, w)?);
        plans.insert((h, w), p.clone());
        Ok(p)
    }
}

impl Model for NcnnModel {
    fn scale(&self) -> u32 {
        self.scale
    }

    /// These networks are fully convolutional: any input size works.
    fn input_multiple(&self) -> u32 {
        1
    }

    fn max_input_side(&self) -> Option<u32> {
        Some(self.max_side)
    }

    /// By depth, from the seams measured in `BENCH.md`: the file does not name its architecture.
    fn preferred_overlap(&self) -> u32 {
        let convs = self
            .graph
            .layers
            .iter()
            .filter(|l| matches!(l.kind, LayerKind::Conv { .. }))
            .count();
        match convs {
            0..=40 => 32,
            41..=100 => 48,
            _ => 64,
        }
    }

    fn run(&self, tile: &Image32) -> Result<Image32> {
        let plan = self.plan_for(tile.height, tile.width)?;
        self.exec.run(&self.graph, &plan, tile)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use scaleup_core::backend::Backend;
    use scaleup_core::tiling::run_tiled;
    use scaleup_core::{DevicePreference, Image32};

    use crate::WgpuBackend;

    fn model_path(name: &str) -> Option<PathBuf> {
        let p = Path::new("F:/Apps/ScaleUP/Upscayl/resources/models").join(format!("{name}.param"));
        p.is_file().then_some(p)
    }

    fn backend() -> Option<WgpuBackend> {
        WgpuBackend::new(DevicePreference::Auto).ok()
    }

    /// A smooth test picture with some texture, in 0..1.
    fn picture(w: u32, h: u32) -> Image32 {
        let mut img = Image32::new(w, h);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let v = 0.5
                        + 0.25 * ((x as f32 / 5.0 + c as f32).sin())
                        + 0.2 * ((y as f32 / 3.0).cos());
                    img.set(c, x, y, v);
                }
            }
        }
        img
    }

    #[test]
    fn overlap_follows_the_depth_of_the_network() {
        let Some(b) = backend() else { return };
        for (name, want) in [
            ("upscayl-lite-4x", 32),
            ("digital-art-4x", 48),
            ("upscayl-standard-4x", 64),
        ] {
            let Some(p) = model_path(name) else { continue };
            let m = b.load_model(&p, None).unwrap();
            assert_eq!(m.preferred_overlap(), want, "{name}");
        }
    }

    #[test]
    fn loads_only_param_files() {
        let Some(b) = backend() else { return };
        let err = b
            .load_model(Path::new("model.onnx"), None)
            .err()
            .expect("onnx refused");
        assert!(err.to_string().contains("ncnn"), "{err}");
    }

    #[test]
    fn upscayl_lite_runs() {
        let (Some(b), Some(p)) = (backend(), model_path("upscayl-lite-4x")) else {
            return;
        };
        let m = b.load_model(&p, None).unwrap();
        assert_eq!(m.scale(), 4);
        let out = m.run(&picture(32, 24)).unwrap();
        assert_eq!((out.width, out.height), (128, 96));
        for &v in &out.data {
            assert!(v.is_finite() && (-0.2..=1.2).contains(&v), "{v}");
        }
    }

    #[test]
    fn digital_art_rrdb_runs() {
        let (Some(b), Some(p)) = (backend(), model_path("digital-art-4x")) else {
            return;
        };
        let m = b.load_model(&p, None).unwrap();
        let out = m.run(&picture(16, 16)).unwrap();
        assert_eq!((out.width, out.height), (64, 64));
        let mean = out.data.iter().sum::<f32>() / out.data.len() as f32;
        let input_mean = picture(16, 16).data.iter().sum::<f32>() / (3.0 * 256.0);
        assert!((mean - input_mean).abs() < 0.1, "{mean} vs {input_mean}");
    }

    /// Away from the image edge (a whole run zero-pads there, tiles see reflected context) tiling
    /// must not change the result once the overlap covers the receptive radius of the network: 34
    /// input pixels for this 34-convolution model, and overlap is capped at a quarter tile.
    #[test]
    fn tiled_equals_whole_within_tolerance() {
        let (Some(b), Some(p)) = (backend(), model_path("upscayl-lite-4x")) else {
            return;
        };
        let m = b.load_model(&p, None).unwrap();
        let img = picture(300, 220);
        let whole = m.run(&img).unwrap();
        let tiled = run_tiled(m.as_ref(), &img, 128, 32, &mut |_, _| true).unwrap();
        let margin = 40 * 4;
        let mut worst = 0.0f32;
        for c in 0..3 {
            for y in margin..whole.height - margin {
                for x in margin..whole.width - margin {
                    worst = worst.max((whole.get(c, x, y) - tiled.get(c, x, y)).abs());
                }
            }
        }
        assert!(worst < 2e-3, "max abs diff {worst}");
    }

    #[test]
    fn real_lite_network_matches_cpu_reference() {
        let (Some(b), Some(p)) = (backend(), model_path("upscayl-lite-4x")) else {
            return;
        };
        let m = b.load_model(&p, None).unwrap();
        let img = picture(8, 6);
        let graph = crate::ncnn::load_graph(&p).unwrap();
        let want = crate::cpu_ref::run_graph(
            &graph,
            &crate::cpu_ref::Tensor {
                c: 3,
                h: 6,
                w: 8,
                data: img.data.clone(),
            },
        );
        let got = m.run(&img).unwrap();
        let worst = got
            .data
            .iter()
            .zip(&want.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "max abs diff {worst}");
    }

    /// Ground truth: the original ncnn engine on the same weights. One tile covers the whole image
    /// and only the interior counts, because the two engines pad the image border differently
    /// (replicate against zero). Run with `--ignored`.
    #[test]
    #[ignore = "needs the Upscayl install and a GPU"]
    fn parity_with_upscayl_bin() {
        let root = Path::new("F:/Apps/ScaleUP/Upscayl/resources");
        let Some(b) = backend() else { return };
        let d = tempfile::tempdir().unwrap();
        let input = d.path().join("in.png");
        let img = picture(200, 160);
        img.to_rgba(None).save(&input).unwrap();
        for name in ["upscayl-lite-4x", "digital-art-4x"] {
            let reference = d.path().join(format!("{name}.png"));
            let status = std::process::Command::new(root.join("bin/upscayl-bin.exe"))
                .arg("-i")
                .arg(&input)
                .arg("-o")
                .arg(&reference)
                .args([
                    "-n", name, "-s", "4", "-f", "png", "-g", "0", "-t", "256", "-m",
                ])
                .arg(root.join("models"))
                .output()
                .unwrap();
            assert!(status.status.success());
            let m = b.load_model(&model_path(name).unwrap(), None).unwrap();
            let opts = scaleup_core::UpscaleOptions {
                tile_size: 256,
                tile_overlap: Some(10),
                ..Default::default()
            };
            let ours = scaleup_core::upscale(
                m.as_ref(),
                &image::open(&input).unwrap().into_rgba8(),
                &opts,
                &mut |_| {},
                &scaleup_core::CancelToken::new(),
            )
            .unwrap();
            let theirs = image::open(&reference).unwrap().into_rgba8();
            let margin = 48 * 4;
            let (mut sum, mut n) = (0f64, 0u64);
            for y in margin..ours.height() - margin {
                for x in margin..ours.width() - margin {
                    for c in 0..3 {
                        let d = ours.get_pixel(x, y).0[c].abs_diff(theirs.get_pixel(x, y).0[c]);
                        sum += f64::from(d).powi(2);
                        n += 1;
                    }
                }
            }
            let mse = sum / n as f64;
            let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
            println!("{name}: {psnr:.2} dB");
            assert!(psnr > 40.0, "{name}: {psnr:.2} dB");
        }
    }

    #[test]
    fn max_input_side_fits_the_device_limits() {
        let (Some(b), Some(p)) = (backend(), model_path("upscayl-standard-4x")) else {
            return;
        };
        let m = b.load_model(&p, None).unwrap();
        let side = m.max_input_side().expect("a limit");
        assert!(side >= 64, "{side}");
    }
}
