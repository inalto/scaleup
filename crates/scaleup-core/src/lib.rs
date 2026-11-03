//! `scaleup_core`: upscale RGBA images with ONNX super-resolution models.

pub mod backend;
pub mod error;
pub mod image32;
pub mod io;
pub mod pipeline;
pub mod registry;
pub mod resize;
pub mod tiling;
pub mod tta;

pub use error::{Result, ScaleUpError};
