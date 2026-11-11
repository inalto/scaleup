//! `scaleup_core`: upscale RGBA images with ONNX super-resolution models on GPU (CUDA, DirectML,
//! CoreML) or CPU, with tiling, alpha, TTA, post-resize, progress and cancellation, plus a model
//! registry with verified downloads.
//!
//! Start with [`Upscaler`] for file-to-file work, or [`upscale`] for in-memory images.

pub mod backend;
pub mod error;
pub mod extend;
pub mod facade;
pub mod hub;
pub mod image32;
pub mod io;
pub mod pipeline;
pub mod registry;
pub mod resize;
pub mod tiling;
pub mod tta;

pub use backend::{
    Backend, DeviceInfo, DeviceKind, DevicePreference, ExtraBackend, Inpainter, Model,
    create_backend, list_devices, register_extra_backend,
};
pub use error::{Result, ScaleUpError};
pub use extend::{ExtendOptions, Fill, MarginSpec, Margins, outpaint, pad_canvas};
pub use facade::{FileJob, Upscaler};
pub use image32::Image32;
pub use io::{OutputFormat, load_image, save_image};
pub use pipeline::{CancelToken, Phase, Progress, UpscaleOptions, upscale};
pub use registry::{ModelEntry, ModelKind, Registry, ResolvedModel, scale_from_name};

/// Compiles the README example as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
