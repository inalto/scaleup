//! `scaleup_wgpu`: a [`scaleup_core::Backend`] that runs ncnn `.param`/`.bin` models on Vulkan, Metal
//! or DX12 through `wgpu` compute shaders, with no C or C++ dependency.

pub mod cpu_ref;
pub mod device;
pub mod exec;
pub mod kernels;
pub mod model;
pub mod ncnn;
pub mod statedict;

use scaleup_core::backend::is_ncnn_model;
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
    });
}

/// Runs ncnn models on one adapter.
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
        is_ncnn_model(path)
    }

    /// `path` is the `.param` file; the `.bin` next to it holds the weights. The scale is read from
    /// the network, so the hint is not needed.
    fn load_model(&self, path: &Path, _scale_hint: Option<u32>) -> Result<Box<dyn Model>> {
        if !is_ncnn_model(path) {
            return Err(ScaleUpError::Model(format!(
                "the wgpu backend runs ncnn .param models, not {}",
                path.display()
            )));
        }
        let graph = ncnn::load_graph(path)?;
        let exec = exec::Executor::new(self.device.clone(), self.queue.clone(), &graph)?;
        let max_buffer = self
            .limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size);
        Ok(Box::new(model::NcnnModel::new(graph, exec, max_buffer)?))
    }
}
