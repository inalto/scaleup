//! `scaleup devices`.

use scaleup_core::{DeviceInfo, DevicePreference, ScaleUpError, create_backend, list_devices};

fn kind(d: &DeviceInfo) -> String {
    format!("{:?}", d.kind).to_lowercase()
}

pub fn run_devices() -> Result<i32, ScaleUpError> {
    for d in list_devices() {
        println!("{}  {}  {}", d.index, kind(&d), d.name);
    }
    let backend = create_backend(DevicePreference::Auto)?;
    let d = backend.device();
    println!("selected by -g auto: {}  {}", kind(d), d.name);
    Ok(0)
}
