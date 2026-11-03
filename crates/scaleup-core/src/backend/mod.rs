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

impl DevicePreference {
    /// `auto`, `cpu` or a single digit GPU index.
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
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
}

pub trait Model: Send + Sync {
    fn scale(&self) -> u32;
    /// Input sides must be a multiple of this; tiles are reflect-padded up to it. Default 4.
    fn input_multiple(&self) -> u32 {
        4
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_preference_cases() {
        assert_eq!(DevicePreference::parse("auto"), Ok(DevicePreference::Auto));
        assert_eq!(DevicePreference::parse("CPU"), Ok(DevicePreference::Cpu));
        assert_eq!(DevicePreference::parse(" 3 "), Ok(DevicePreference::Gpu(3)));
        assert!(DevicePreference::parse("10").is_err());
        assert!(DevicePreference::parse("").is_err());
    }
}
