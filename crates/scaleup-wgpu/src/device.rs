//! Adapter discovery and device creation.

use scaleup_core::{DeviceInfo, DeviceKind, DevicePreference, Result, ScaleUpError};

/// The compute-capable adapters of this machine.
pub fn list_adapters() -> Vec<DeviceInfo> {
    usable_adapters()
        .iter()
        .enumerate()
        .map(|(i, a)| device_info(i as u32, &a.get_info()))
        .collect()
}

fn device_info(index: u32, info: &wgpu::AdapterInfo) -> DeviceInfo {
    let kind = if is_software(info) {
        DeviceKind::Cpu
    } else {
        DeviceKind::Vulkan
    };
    DeviceInfo {
        kind,
        index,
        name: format!("{} ({:?})", info.name, info.backend),
    }
}

fn is_software(info: &wgpu::AdapterInfo) -> bool {
    info.device_type == wgpu::DeviceType::Cpu
}

/// Preference among the APIs that expose the same card.
fn backend_rank(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        _ => 3,
    }
}

/// Hardware adapters first, fastest kind first; each card once, through its best API. OpenGL is left
/// out: its compute support is too limited for these shaders.
fn usable_adapters() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    adapters.retain(|a| {
        a.get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    });
    let type_rank = |info: &wgpu::AdapterInfo| match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    adapters.sort_by_key(|a| {
        let info = a.get_info();
        (type_rank(&info), backend_rank(info.backend))
    });
    let mut seen = Vec::new();
    adapters.retain(|a| {
        let info = a.get_info();
        let id = (info.vendor, info.device, info.name);
        if seen.contains(&id) {
            false
        } else {
            seen.push(id);
            true
        }
    });
    adapters
}

pub(crate) struct Opened {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: DeviceInfo,
    pub limits: wgpu::Limits,
}

/// Opens the adapter for `pref` with the largest buffers it allows.
pub(crate) fn open(pref: DevicePreference) -> Result<Opened> {
    open_impl(pref, None)
}

/// Opens the adapter for `pref` restricted to `limits`.
#[cfg(test)]
pub(crate) fn open_with_limits(pref: DevicePreference, limits: wgpu::Limits) -> Result<Opened> {
    open_impl(pref, Some(limits))
}

fn open_impl(pref: DevicePreference, limits: Option<wgpu::Limits>) -> Result<Opened> {
    let adapters = usable_adapters();
    let hardware = |a: &&wgpu::Adapter| !is_software(&a.get_info());
    let chosen = match pref {
        DevicePreference::Auto => adapters.iter().find(hardware).or(adapters.first()),
        DevicePreference::Gpu(i) => adapters.get(i as usize),
        DevicePreference::Cpu => adapters.iter().find(|a| is_software(&a.get_info())),
    };
    let Some(adapter) = chosen else {
        return Err(ScaleUpError::Backend(match pref {
            DevicePreference::Cpu => "no software GPU adapter on this machine".into(),
            DevicePreference::Gpu(i) => format!("no GPU adapter with index {i}"),
            DevicePreference::Auto => "no GPU adapter on this machine".into(),
        }));
    };
    let index = adapters
        .iter()
        .position(|a| std::ptr::eq(a, adapter))
        .expect("chosen from this list") as u32;
    let info = device_info(index, &adapter.get_info());
    let limits = limits.unwrap_or_else(|| adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("scaleup"),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|e| ScaleUpError::Backend(format!("{}: {e}", info.name)))?;
    tracing::info!("wgpu device: {}", info.name);
    Ok(Opened {
        device,
        queue,
        info,
        limits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_adapters_does_not_panic_without_gpu() {
        // On a GPU-less machine the list is empty; on any machine indices are 0..n in order.
        for (i, d) in list_adapters().iter().enumerate() {
            assert_eq!(d.index as usize, i);
            assert!(!d.name.is_empty());
        }
    }

    #[test]
    fn this_machine_opens_its_first_adapter_when_it_has_one() {
        if list_adapters().is_empty() {
            eprintln!("skipped: no adapter");
            return;
        }
        let opened = open(DevicePreference::Gpu(0)).unwrap();
        assert_eq!(opened.info, list_adapters()[0]);
    }

    #[test]
    fn out_of_range_gpu_index_is_an_error() {
        assert!(open(DevicePreference::Gpu(99)).is_err());
    }

    #[test]
    fn new_cpu_pref_errors_or_software() {
        match open(DevicePreference::Cpu) {
            Ok(o) => assert_eq!(o.info.kind, scaleup_core::DeviceKind::Cpu),
            Err(e) => assert!(e.to_string().contains("software"), "{e}"),
        }
    }
}
