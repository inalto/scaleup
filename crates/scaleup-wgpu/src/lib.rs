//! `scaleup_wgpu`: a [`scaleup_core::Backend`] that runs ncnn `.param`/`.bin` models and PyTorch
//! `.pth`/`.safetensors` weights of the architectures in [`native_archs`] on Vulkan, Metal or DX12
//! through `wgpu` compute shaders, with no C or C++ dependency.

pub mod arch;
pub mod build;
pub mod cpu_ref;
pub mod device;
pub mod exec;
pub mod kernels;
pub mod model;
pub mod ncnn;
pub mod statedict;

use scaleup_core::backend::{is_extra_backend_model, is_ncnn_model};
use scaleup_core::{
    Backend, DeviceInfo, DevicePreference, ExtraBackend, Model, Result, ScaleUpError,
    register_extra_backend,
};
use std::path::Path;

pub use device::list_adapters;

/// Makes this backend available to `scaleup_core`. Call once at program start.
pub fn register() {
    register_extra_backend(ExtraBackend {
        create: |pref| Ok(Box::new(WgpuBackend::new(pref)?)),
        devices: || {
            list_adapters()
                .into_iter()
                .filter(|d| d.kind.is_gpu())
                .collect()
        },
        native_archs,
    });
}

/// Architectures (OpenModelDB names) whose `.pth`/`.safetensors` files this backend runs.
pub fn native_archs() -> Vec<String> {
    build::NATIVE_ARCHS.iter().map(|a| a.to_string()).collect()
}

/// The network in `path`, and the multiple its input sides must be.
fn load_graph(path: &Path) -> Result<(ncnn::Graph, u32)> {
    if is_ncnn_model(path) {
        return Ok((ncnn::load_graph(path)?, 1));
    }
    if !is_extra_backend_model(path) {
        return Err(ScaleUpError::Model(format!(
            "the wgpu backend runs ncnn .param and PyTorch .pth/.safetensors models, not {}",
            path.display()
        )));
    }
    let in_file = |e| match e {
        ScaleUpError::Model(m) => ScaleUpError::Model(format!("{}: {m}", path.display())),
        other => other,
    };
    // The state dict is dropped as soon as its weights are in the graph.
    let sd = statedict::StateDict::load(path)?;
    let spec = arch::detect(&sd).map_err(in_file)?;
    let graph = build::build(&spec, &sd).map_err(in_file)?;
    Ok((graph, spec.input_multiple_of()))
}

/// Runs ncnn and PyTorch models on one adapter.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: DeviceInfo,
    limits: wgpu::Limits,
}

impl WgpuBackend {
    pub fn new(pref: DevicePreference) -> Result<Self> {
        let o = device::open(pref)?;
        Ok(Self {
            device: o.device,
            queue: o.queue,
            info: o.info,
            limits: o.limits,
        })
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn device(&self) -> &DeviceInfo {
        &self.info
    }

    fn handles(&self, path: &Path) -> bool {
        is_extra_backend_model(path)
    }

    /// `path` is a `.param` file (the `.bin` next to it holds the weights), or a `.pth` or
    /// `.safetensors` state dict. The scale is read from the network, so the hint is not needed.
    fn load_model(&self, path: &Path, _scale_hint: Option<u32>) -> Result<Box<dyn Model>> {
        let (graph, input_multiple) = load_graph(path)?;
        let exec = exec::Executor::new(self.device.clone(), self.queue.clone(), &graph)?;
        let max_buffer = self
            .limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size);
        Ok(Box::new(model::GraphModel::new(
            graph,
            exec,
            max_buffer,
            input_multiple,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scaleup_core::backend::OrtBackend;
    use scaleup_core::{DevicePreference, Image32};
    use std::path::PathBuf;

    /// ESRGAN+ is covered by `esrgan_plus_matches_rrdb_forward_pointwise`.
    #[test]
    fn native_archs_are_the_architectures_that_build() {
        assert_eq!(
            native_archs(),
            ["esrgan", "esrgan+", "compact", "span", "realplksr"]
        );
        for fixture in [
            "tiny_esrgan.pth",
            "tiny_compact.pth",
            "tiny_span.pth",
            "tiny_realplksr_gn_ea.pth",
            "tiny_realplksr_ln.pth",
        ] {
            let (graph, multiple) = load_graph(&repo("tests/fixtures/sd").join(fixture)).unwrap();
            assert_eq!(multiple, 1);
            let spec = arch::detect(
                &statedict::StateDict::load(&repo("tests/fixtures/sd").join(fixture)).unwrap(),
            )
            .unwrap();
            assert!(native_archs().contains(&spec.name().to_ascii_lowercase()));
            assert_eq!(ncnn::infer_scale(&graph).unwrap(), spec.external_scale());
        }
    }

    #[test]
    fn state_dict_errors_name_the_file() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("odd.safetensors");
        let one = 1.0_f32.to_le_bytes().to_vec();
        let view = ::safetensors::tensor::TensorView::new(
            ::safetensors::tensor::Dtype::F32,
            vec![1],
            &one,
        )
        .unwrap();
        ::safetensors::tensor::serialize_to_file([("head.weight", view)], None, &path).unwrap();
        let err = load_graph(&path).unwrap_err().to_string();
        assert!(
            err.contains("odd.safetensors") && err.contains("unsupported architecture"),
            "{err}"
        );
    }

    fn repo(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel)
    }

    /// The public API: `Upscaler` routes a `.pth` path to this backend and writes the result.
    #[test]
    fn upscaler_runs_a_pth_file_end_to_end() {
        if WgpuBackend::new(DevicePreference::Auto).is_err() {
            return;
        }
        register();
        let d = tempfile::tempdir().unwrap();
        let up = scaleup_core::Upscaler::new(
            DevicePreference::Auto,
            scaleup_core::Registry::with_dir(d.path().join("models")),
        )
        .unwrap();
        let model_path = repo("tests/fixtures/sd/tiny_compact.pth");
        let model = up
            .load(model_path.to_str().unwrap(), &mut |_, _| {})
            .unwrap();
        assert_eq!(model.scale(), 2);
        assert_eq!(up.active_device().kind, scaleup_core::DeviceKind::Vulkan);
        let input = repo("tests/fixtures/images/gradient_64x48.png");
        let output = d.path().join("out.png");
        let job = scaleup_core::FileJob {
            input,
            output: output.clone(),
            format: scaleup_core::OutputFormat::Png,
            quality: 90,
            overwrite: false,
            double: false,
        };
        up.run_file(
            model.as_ref(),
            &job,
            &scaleup_core::UpscaleOptions::default(),
            &mut |_| {},
            &scaleup_core::CancelToken::new(),
        )
        .unwrap();
        let out = image::open(&output).unwrap();
        assert_eq!((out.width(), out.height()), (128, 96));
    }

    /// A 64×64 crop of the photo-like fixture, in 0..1.
    fn crop64() -> Image32 {
        let img = image::open(repo("tests/fixtures/images/photo_like_256x192.jpg"))
            .unwrap()
            .into_rgba8();
        Image32::from_rgba(&img).0.crop(96, 64, 64, 64)
    }

    /// PSNR in dB of `a` against `b`, both clamped to 0..1 as they would be saved.
    fn psnr(a: &Image32, b: &Image32) -> f64 {
        assert_eq!((a.width, a.height), (b.width, b.height));
        let mse = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| f64::from(x.clamp(0.0, 1.0) - y.clamp(0.0, 1.0)).powi(2))
            .sum::<f64>()
            / a.data.len() as f64;
        10.0 * (1.0 / mse.max(f64::MIN_POSITIVE)).log10()
    }

    /// Runs `pth` natively on the GPU and its ONNX export on ONNX Runtime's CPU provider, on the
    /// same 64×64 crop, and checks the PSNR between them. Skips when a file or the GPU is missing.
    fn matches_ort(pth: &str, onnx: &str) {
        let cache = Path::new("F:/Apps/ScaleUP/scaleup/models/cache");
        let (pth, onnx) = (cache.join("src").join(pth), cache.join("onnx").join(onnx));
        if !pth.is_file() || !onnx.is_file() {
            eprintln!("skipped: {} or {} is absent", pth.display(), onnx.display());
            return;
        }
        let Ok(b) = WgpuBackend::new(DevicePreference::Auto) else {
            eprintln!("skipped: no GPU adapter");
            return;
        };
        let native = b.load_model(&pth, None).unwrap();
        let reference = OrtBackend::new(DevicePreference::Cpu)
            .unwrap()
            .load_model(&onnx, None)
            .unwrap();
        assert_eq!(native.scale(), reference.scale());
        let tile = crop64();
        let ours = native.run(&tile).unwrap();
        let theirs = reference.run(&tile).unwrap();
        let worst = ours
            .data
            .iter()
            .zip(&theirs.data)
            .map(|(a, b): (&f32, &f32)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let db = psnr(&ours, &theirs);
        println!(
            "{}: {db:.2} dB, max abs diff {worst:.2e}",
            pth.file_name().unwrap().to_string_lossy()
        );
        assert!(db > 45.0, "{db:.2} dB");
    }

    #[test]
    #[ignore = "needs the real weights in models/cache and a GPU"]
    fn real_esrgan_x4plus_matches_ort() {
        matches_ort("RealESRGAN_x4plus.pth", "realesrgan-x4plus.onnx");
    }

    #[test]
    #[ignore = "needs the real weights in models/cache and a GPU"]
    fn real_esrgan_x2plus_matches_ort() {
        matches_ort("RealESRGAN_x2plus.pth", "realesrgan-x2plus.onnx");
    }

    #[test]
    #[ignore = "needs the real weights in models/cache and a GPU"]
    fn real_compact_general_x4v3_matches_ort() {
        matches_ort("realesr-general-x4v3.pth", "realesr-general-x4v3.onnx");
    }

    #[test]
    #[ignore = "needs the real weights in models/cache and a GPU"]
    fn real_legacy_ultrasharp_matches_ort() {
        matches_ort("4x-UltraSharp.pth", "4x-UltraSharp.onnx");
    }

    #[test]
    #[ignore = "needs the real weights in models/cache and a GPU"]
    fn span_nomosuni_matches_ort() {
        matches_ort(
            "4xNomosUni_span_multijpg.safetensors",
            "4xNomosUni_span_multijpg.onnx",
        );
    }
}
