//! ONNX Runtime backend: the only FFI boundary in the crate.
//!
//! Execution providers are tried in the spec's "GPU policy" order and the first one that registers
//! wins: Windows CUDA → DirectML → CPU, Linux CUDA → CPU, macOS CoreML → CPU.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ::ort::environment::Environment;
use ::ort::session::Session;
use ::ort::session::builder::SessionBuilder;
use ::ort::value::TensorRef;

use crate::backend::{Backend, DeviceInfo, DeviceKind, DevicePreference, Model};
use crate::image32::Image32;
use crate::{Result, ScaleUpError};

fn backend_err(e: impl std::fmt::Display) -> ScaleUpError {
    ScaleUpError::Backend(e.to_string())
}

fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ::ort::init()
        .with_name("scaleup")
        .build()
        .map_err(backend_err)?;
    Ok(ENV.get_or_init(|| env))
}

/// GPU execution providers compiled in for this target, in registration order.
fn gpu_candidates() -> &'static [DeviceKind] {
    if cfg!(target_os = "windows") {
        &[DeviceKind::Cuda, DeviceKind::DirectMl]
    } else if cfg!(target_os = "linux") {
        &[DeviceKind::Cuda]
    } else if cfg!(target_os = "macos") {
        &[DeviceKind::CoreMl]
    } else {
        &[]
    }
}

fn cpu_device() -> DeviceInfo {
    DeviceInfo {
        kind: DeviceKind::Cpu,
        index: 0,
        name: "CPU".into(),
    }
}

/// Registers the execution provider for `kind`/`index` on `builder`. CPU registers nothing.
#[allow(unused_variables)]
fn register(builder: &mut SessionBuilder, kind: DeviceKind, index: u32) -> Result<()> {
    #[allow(unused_imports)]
    use ::ort::ep::ExecutionProvider;
    match kind {
        DeviceKind::Cpu => Ok(()),
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        DeviceKind::Cuda => ::ort::ep::CUDA::default()
            .with_device_id(index as i32)
            .register(builder)
            .map_err(backend_err),
        #[cfg(target_os = "windows")]
        DeviceKind::DirectMl => ::ort::ep::DirectML::default()
            .with_device_id(index as i32)
            .register(builder)
            .map_err(backend_err),
        #[cfg(target_os = "macos")]
        DeviceKind::CoreMl => ::ort::ep::CoreML::default()
            .register(builder)
            .map_err(backend_err),
        other => Err(ScaleUpError::Backend(format!(
            "{other:?} is not supported by the ONNX Runtime backend"
        ))),
    }
}

fn device_name(kind: DeviceKind, index: u32) -> String {
    match kind {
        DeviceKind::Cuda => format!("CUDA GPU {index}"),
        DeviceKind::DirectMl => format!("DirectML GPU {index}"),
        DeviceKind::CoreMl => "CoreML".into(),
        DeviceKind::Vulkan => format!("Vulkan GPU {index}"),
        DeviceKind::Cpu => "CPU".into(),
    }
}

/// Fresh session builder with the provider for `device` registered and its required options set.
fn builder_for(device: &DeviceInfo) -> Result<SessionBuilder> {
    let mut builder = Session::builder(environment()?).map_err(backend_err)?;
    if device.kind == DeviceKind::DirectMl {
        // DirectML does not support memory patterns or parallel execution.
        builder = builder
            .with_memory_pattern(false)
            .map_err(backend_err)?
            .with_parallel_execution(false)
            .map_err(backend_err)?;
    }
    register(&mut builder, device.kind, device.index)?;
    Ok(builder)
}

/// The first GPU provider that registers for device `index`, if any.
fn probe_gpu(index: u32) -> Option<DeviceInfo> {
    gpu_candidates().iter().find_map(|&kind| {
        let device = DeviceInfo {
            kind,
            index,
            name: device_name(kind, index),
        };
        match builder_for(&device) {
            Ok(_) => Some(device),
            Err(e) => {
                tracing::debug!("{kind:?} execution provider unavailable: {e}");
                None
            }
        }
    })
}

pub(crate) fn available_devices() -> Vec<DeviceInfo> {
    probe_gpu(0).into_iter().chain([cpu_device()]).collect()
}

/// ONNX Runtime backend bound to one device.
pub struct OrtBackend {
    device: DeviceInfo,
}

impl OrtBackend {
    /// Picks the device for `pref`. A GPU that fails to initialise falls back to the CPU with a warning.
    pub fn new(pref: DevicePreference) -> Result<Self> {
        environment()?;
        let device = match pref {
            DevicePreference::Cpu => cpu_device(),
            DevicePreference::Auto | DevicePreference::Gpu(_) => {
                let index = if let DevicePreference::Gpu(i) = pref {
                    i
                } else {
                    0
                };
                probe_gpu(index).unwrap_or_else(|| {
                    tracing::warn!("no GPU execution provider could be initialised for device {index}; using CPU");
                    cpu_device()
                })
            }
        };
        tracing::info!("ONNX Runtime device: {}", device.name);
        Ok(Self { device })
    }
}

impl Backend for OrtBackend {
    fn name(&self) -> &'static str {
        "onnxruntime"
    }

    fn device(&self) -> &DeviceInfo {
        &self.device
    }

    fn load_model(&self, path: &Path, scale_hint: Option<u32>) -> Result<Box<dyn Model>> {
        if !path.is_file() {
            return Err(ScaleUpError::Model(format!(
                "model file not found: {}",
                path.display()
            )));
        }
        let session = builder_for(&self.device)?
            .commit_from_file(path)
            .map_err(|e| ScaleUpError::Model(format!("{}: {e}", path.display())))?;
        let Some(input_name) = session.inputs().first().map(|o| o.name().to_string()) else {
            return Err(ScaleUpError::Model(format!(
                "{}: model has no inputs",
                path.display()
            )));
        };
        if session.outputs().is_empty() {
            return Err(ScaleUpError::Model(format!(
                "{}: model has no outputs",
                path.display()
            )));
        }
        let mut model = OrtModel {
            session: Mutex::new(session),
            input_name,
            scale: scale_hint.unwrap_or(0),
        };
        if scale_hint.is_none() {
            model.scale = model.probe_scale()?;
        }
        Ok(Box::new(model))
    }
}

struct OrtModel {
    session: Mutex<Session>,
    input_name: String,
    scale: u32,
}

impl OrtModel {
    /// Raw inference on `[1,3,H,W]`: returns (width, height, CHW data) of the first output.
    fn infer(&self, tile: &Image32) -> Result<(u32, u32, Vec<f32>)> {
        let shape = [1usize, 3, tile.height as usize, tile.width as usize];
        let input =
            TensorRef::from_array_view((shape, tile.data.as_slice())).map_err(backend_err)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| ScaleUpError::Backend("session mutex poisoned".into()))?;
        let outputs = session
            .run(::ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| ScaleUpError::Model(e.to_string()))?;
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ScaleUpError::Model(e.to_string()))?;
        match **out_shape {
            [1, 3, h, w] if h > 0 && w > 0 => Ok((w as u32, h as u32, data.to_vec())),
            ref other => Err(ScaleUpError::Model(format!(
                "expected output shape [1,3,H,W], got {other:?}"
            ))),
        }
    }

    /// Runs a 16×16 tile and derives the integer scale from the output size.
    fn probe_scale(&self) -> Result<u32> {
        let (w, h, _) = self.infer(&Image32::new(16, 16))?;
        if w == 0 || w % 16 != 0 || w != h {
            return Err(ScaleUpError::Model(format!(
                "cannot infer scale: 16x16 input gave {w}x{h} output"
            )));
        }
        Ok(w / 16)
    }
}

impl Model for OrtModel {
    fn scale(&self) -> u32 {
        self.scale
    }

    fn run(&self, tile: &Image32) -> Result<Image32> {
        let (w, h, data) = self.infer(tile)?;
        if (w, h) != (tile.width * self.scale, tile.height * self.scale) {
            return Err(ScaleUpError::Model(format!(
                "model output {w}x{h} does not match {}x{} at scale {}",
                tile.width, tile.height, self.scale
            )));
        }
        Ok(Image32 {
            width: w,
            height: h,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fake::NearestModel;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/nearest2x.onnx")
    }

    fn ramp(w: u32, h: u32) -> Image32 {
        let mut img = Image32::new(w, h);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = i as f32 / (3 * w * h) as f32;
        }
        img
    }

    #[test]
    fn cpu_backend_runs_nearest2x_fixture() {
        let backend = OrtBackend::new(DevicePreference::Cpu).unwrap();
        assert_eq!(backend.device().kind, DeviceKind::Cpu);
        let model = backend.load_model(&fixture(), Some(2)).unwrap();
        assert_eq!(model.scale(), 2);
        let img = ramp(4, 3);
        let out = model.run(&img).unwrap();
        assert_eq!((out.width, out.height), (8, 6));
        assert_eq!(out, NearestModel { scale: 2 }.run(&img).unwrap());
    }

    #[test]
    fn probe_detects_scale_without_hint() {
        let backend = OrtBackend::new(DevicePreference::Cpu).unwrap();
        let model = backend.load_model(&fixture(), None).unwrap();
        assert_eq!(model.scale(), 2);
    }

    #[test]
    fn wrong_scale_hint_is_caught_at_run() {
        let backend = OrtBackend::new(DevicePreference::Cpu).unwrap();
        let model = backend.load_model(&fixture(), Some(4)).unwrap();
        assert!(matches!(
            model.run(&ramp(4, 4)),
            Err(ScaleUpError::Model(_))
        ));
    }

    #[test]
    fn missing_model_file_is_error() {
        let backend = OrtBackend::new(DevicePreference::Cpu).unwrap();
        assert!(
            backend
                .load_model(std::path::Path::new("nope.onnx"), Some(2))
                .is_err()
        );
    }

    #[test]
    fn list_devices_ends_with_cpu() {
        let devices = crate::backend::list_devices();
        assert_eq!(devices.last().unwrap().kind, DeviceKind::Cpu);
    }

    #[test]
    #[ignore = "needs a GPU; run with: cargo test -p scaleup-core -- --ignored"]
    fn gpu_backend_reports_non_cpu() {
        let backend = OrtBackend::new(DevicePreference::Auto).unwrap();
        println!("selected device: {:?}", backend.device());
        assert_ne!(backend.device().kind, DeviceKind::Cpu);
        let model = backend.load_model(&fixture(), None).unwrap();
        let img = ramp(37, 21);
        assert_eq!(
            model.run(&img).unwrap(),
            NearestModel { scale: 2 }.run(&img).unwrap()
        );
    }
}
