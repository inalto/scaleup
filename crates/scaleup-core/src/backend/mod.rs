//! Inference backends. A [`Backend`] creates [`Model`]s; everything above this layer only calls
//! [`Model::run`] on RGB tiles.

use std::path::Path;
use std::sync::OnceLock;

use crate::image32::Image32;
use crate::{Result, ScaleUpError};

#[cfg(any(test, feature = "testing"))]
pub mod fake;

/// Hardware family a backend executes on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cuda,
    DirectMl,
    CoreMl,
    Vulkan,
    Cpu,
}

impl DeviceKind {
    pub fn is_gpu(self) -> bool {
        self != DeviceKind::Cpu
    }
}

/// The device a backend actually ended up on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub index: u32,
    pub name: String,
}

/// What the caller asked for: best available, CPU only, or a specific GPU index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DevicePreference {
    #[default]
    Auto,
    Cpu,
    Gpu(u32),
    /// A specific adapter of the registered Vulkan/Metal/DX12 backend.
    Vulkan(u32),
}

/// A backend this crate cannot depend on directly (it depends on this crate), plugged in at startup.
#[derive(Clone, Copy)]
pub struct ExtraBackend {
    pub create: fn(DevicePreference) -> Result<Box<dyn Backend>>,
    pub devices: fn() -> Vec<DeviceInfo>,
    /// Architectures (lowercase OpenModelDB names) whose `.pth`/`.safetensors` files this backend
    /// runs without conversion.
    pub native_archs: fn() -> Vec<String>,
}

static EXTRA: OnceLock<ExtraBackend> = OnceLock::new();

/// Makes `extra` available to [`create_backend`] and [`list_devices`]. The first call wins.
pub fn register_extra_backend(extra: ExtraBackend) {
    let _ = EXTRA.set(extra);
}

/// The registered backend, if any.
pub fn extra_backend() -> Option<ExtraBackend> {
    EXTRA.get().copied()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackendChoice {
    Ort,
    Extra,
}

/// Which backend serves `pref`. ONNX Runtime first, for the GPU providers it has; the extra backend
/// covers GPUs it cannot reach; a CPU run stays on ONNX Runtime. `forced` is `SCALEUP_BACKEND`.
pub(crate) fn choose_backend(
    pref: DevicePreference,
    forced: Option<&str>,
    ort_has_gpu: bool,
    extra_has_gpu: bool,
) -> BackendChoice {
    use BackendChoice::{Extra, Ort};
    match (pref, forced) {
        (DevicePreference::Vulkan(_), _) | (_, Some("wgpu")) => Extra,
        (_, Some("ort")) => Ort,
        (DevicePreference::Auto, _) if !ort_has_gpu && extra_has_gpu => Extra,
        _ => Ort,
    }
}

impl DevicePreference {
    /// `auto`, `cpu` or a single digit GPU index.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            n if n.starts_with("vk:") => n[3..]
                .parse()
                .map(Self::Vulkan)
                .map_err(|_| format!("invalid device {n:?} (expected vk:<index>)")),
            n if n.len() == 1 && n.as_bytes()[0].is_ascii_digit() => {
                Ok(Self::Gpu(u32::from(n.as_bytes()[0] - b"0"[0])))
            }
            other => Err(format!(
                "invalid device {other:?} (expected auto, cpu or 0..9)"
            )),
        }
    }
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn device(&self) -> &DeviceInfo;
    fn load_model(&self, path: &Path, scale_hint: Option<u32>) -> Result<Box<dyn Model>>;
    /// Whether this backend can load the model file at `path`.
    fn handles(&self, path: &Path) -> bool {
        !is_extra_backend_model(path)
    }
}

pub trait Model: Send + Sync {
    fn scale(&self) -> u32;
    /// Input sides must be a multiple of this; tiles are reflect-padded up to it. Default 4.
    fn input_multiple(&self) -> u32 {
        4
    }
    /// The largest input side this model can process, when something limits it. Tiles shrink to fit.
    fn max_input_side(&self) -> Option<u32> {
        None
    }
    /// Context this network wants around each tile for the stitched result to show no seams, in
    /// input pixels. A network that does not say gets the most any measured network needed.
    fn preferred_overlap(&self) -> u32 {
        64
    }
    /// Output dimensions are input dimensions × [`Model::scale`].
    fn run(&self, tile: &Image32) -> Result<Image32>;
}

pub mod ort;

pub use self::ort::OrtBackend;

/// The preference to give the backend of the other model format: an explicit choice carries over.
pub(crate) fn cross_preference(pref: DevicePreference, to_extra: bool) -> DevicePreference {
    match (pref, to_extra) {
        (DevicePreference::Gpu(i), true) => DevicePreference::Vulkan(i),
        (DevicePreference::Vulkan(i), false) => DevicePreference::Gpu(i),
        (other, _) => other,
    }
}

/// An ncnn network, named by its `.param` file.
pub fn is_ncnn_model(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("param"))
}

/// A model file only the registered extra backend runs: an ncnn `.param`, or PyTorch weights
/// (`.pth`, `.safetensors`).
pub fn is_extra_backend_model(path: &Path) -> bool {
    path.extension().is_some_and(|e| {
        ["param", "pth", "safetensors"]
            .iter()
            .any(|x| e.eq_ignore_ascii_case(x))
    })
}

/// Devices usable on this machine: the GPU the ORT backend would pick (if any), then the CPU.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut devices = self::ort::available_devices();
    if let Some(extra) = extra_backend() {
        devices.extend((extra.devices)());
    }
    devices
}

/// The default backend (ONNX Runtime) for `pref`. A GPU that fails to initialise falls back to the
/// CPU with a warning rather than an error.
pub fn create_backend(pref: DevicePreference) -> Result<Box<dyn Backend>> {
    let extra = extra_backend();
    let extra_gpu = extra.is_some_and(|e| (e.devices)().iter().any(|d| d.kind.is_gpu()));
    let forced = std::env::var("SCALEUP_BACKEND").ok();
    let forced = forced.as_deref();
    if choose_backend(pref, forced, false, true) == BackendChoice::Extra
        && choose_backend(pref, forced, true, true) == BackendChoice::Extra
    {
        // Asked for by name: failing is better than quietly running somewhere else.
        let extra = extra.ok_or_else(|| {
            ScaleUpError::Backend("this build has no Vulkan/Metal/DX12 backend".into())
        })?;
        return (extra.create)(pref);
    }
    let ort = OrtBackend::new(pref)?;
    if choose_backend(pref, forced, ort.device().kind.is_gpu(), extra_gpu) == BackendChoice::Extra
        && let Some(extra) = extra
    {
        match (extra.create)(pref) {
            Ok(backend) => return Ok(backend),
            Err(e) => tracing::warn!("wgpu backend unavailable: {e}"),
        }
    }
    Ok(Box::new(ort))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_backend_prefers_ort_gpu_then_wgpu_then_cpu() {
        use BackendChoice::{Extra, Ort};
        use DevicePreference::{Auto, Cpu, Gpu, Vulkan};
        // Auto: ONNX Runtime GPU, else the extra backend's GPU, else ONNX Runtime on the CPU.
        assert_eq!(choose_backend(Auto, None, true, true), Ort);
        assert_eq!(choose_backend(Auto, None, false, true), Extra);
        assert_eq!(choose_backend(Auto, None, false, false), Ort);
        // Explicit choices are honoured.
        assert_eq!(choose_backend(Cpu, None, false, true), Ort);
        assert_eq!(choose_backend(Gpu(0), None, false, true), Ort);
        assert_eq!(choose_backend(Vulkan(0), None, true, true), Extra);
        // The environment override, for deterministic runs.
        assert_eq!(choose_backend(Auto, Some("wgpu"), true, true), Extra);
        assert_eq!(choose_backend(Auto, Some("ort"), false, true), Ort);
        assert_eq!(choose_backend(Vulkan(0), Some("ort"), true, true), Extra);
    }

    #[test]
    fn explicit_device_choice_carries_across_formats() {
        use DevicePreference::{Auto, Cpu, Gpu, Vulkan};
        assert_eq!(cross_preference(Auto, true), Auto);
        assert_eq!(cross_preference(Auto, false), Auto);
        assert_eq!(cross_preference(Cpu, true), Cpu);
        assert_eq!(cross_preference(Cpu, false), Cpu);
        assert_eq!(cross_preference(Gpu(1), true), Vulkan(1));
        assert_eq!(cross_preference(Vulkan(1), false), Gpu(1));
    }

    #[test]
    fn ncnn_models_are_recognised_by_extension() {
        assert!(is_ncnn_model(Path::new("a/b.PARAM")));
        assert!(!is_ncnn_model(Path::new("a/b.onnx")));
        assert!(!is_ncnn_model(Path::new("a/b.pth")));
    }

    #[test]
    fn extra_backend_models_are_recognised_by_extension() {
        for runs in [
            "a/b.param",
            "a/b.PTH",
            "b.pth",
            "c.SafeTensors",
            "d.safetensors",
        ] {
            assert!(is_extra_backend_model(Path::new(runs)), "{runs}");
        }
        for not in ["a/b.onnx", "b.bin", "pth", "c.pt", "d"] {
            assert!(!is_extra_backend_model(Path::new(not)), "{not}");
        }
    }

    #[test]
    fn parse_preference_cases() {
        assert_eq!(DevicePreference::parse("auto"), Ok(DevicePreference::Auto));
        assert_eq!(DevicePreference::parse("CPU"), Ok(DevicePreference::Cpu));
        assert_eq!(DevicePreference::parse(" 3 "), Ok(DevicePreference::Gpu(3)));
        assert_eq!(
            DevicePreference::parse("vk:2"),
            Ok(DevicePreference::Vulkan(2))
        );
        assert!(DevicePreference::parse("vk:x").is_err());
        assert!(DevicePreference::parse("10").is_err());
        assert!(DevicePreference::parse("").is_err());
    }
}
