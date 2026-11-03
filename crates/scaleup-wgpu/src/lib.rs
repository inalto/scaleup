//! `scaleup_wgpu`: a [`scaleup_core::Backend`] that runs ncnn `.param`/`.bin` models on Vulkan, Metal
//! or DX12 through `wgpu` compute shaders, with no C or C++ dependency.

pub mod device;
pub mod ncnn;

pub use device::list_adapters;
