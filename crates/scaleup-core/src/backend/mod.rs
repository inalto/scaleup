//! Inference backends. A [`Backend`] creates [`Model`]s; everything above this layer only calls
//! [`Model::run`] on RGB tiles.

use std::path::Path;

use crate::Result;
use crate::image32::Image32;

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
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn device(&self) -> &DeviceInfo;
    fn load_model(&self, path: &Path, scale_hint: Option<u32>) -> Result<Box<dyn Model>>;
}

pub trait Model: Send + Sync {
    fn scale(&self) -> u32;
    /// Output dimensions are input dimensions × [`Model::scale`].
    fn run(&self, tile: &Image32) -> Result<Image32>;
}

pub mod ort;

pub use self::ort::OrtBackend;

/// Devices usable on this machine: the GPU the ORT backend would pick (if any), then the CPU.
pub fn list_devices() -> Vec<DeviceInfo> {
    self::ort::available_devices()
}

/// The default backend (ONNX Runtime) for `pref`. A GPU that fails to initialise falls back to the
/// CPU with a warning rather than an error.
pub fn create_backend(pref: DevicePreference) -> Result<Box<dyn Backend>> {
    Ok(Box::new(OrtBackend::new(pref)?))
}
