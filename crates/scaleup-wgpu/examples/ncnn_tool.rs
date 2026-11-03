//! Manual checks against other engines.
//!
//! ```text
//! ncnn_tool run <model.param> <in.png> <out.png> <tile> <overlap>   upscale, print the seconds
//! ncnn_tool psnr <a.png> <b.png>                                    PSNR and worst channel difference
//! ```

use std::path::Path;
use std::time::Instant;

use scaleup_core::backend::Backend;
use scaleup_core::{CancelToken, DevicePreference, UpscaleOptions, upscale};
use scaleup_wgpu::WgpuBackend;

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let backend = WgpuBackend::new(DevicePreference::Auto)?;
    println!("device: {}", backend.device().name);
    let model = backend.load_model(Path::new(&args[0]), None)?;
    let input = image::open(&args[1])?.into_rgba8();
    let opts = UpscaleOptions {
        tile_size: args[3].parse()?,
        tile_overlap: args[4].parse()?,
        ..UpscaleOptions::default()
    };
    let started = Instant::now();
    let out = upscale(
        model.as_ref(),
        &input,
        &opts,
        &mut |_| {},
        &CancelToken::new(),
    )?;
    println!("seconds: {:.2}", started.elapsed().as_secs_f32());
    out.save(&args[2])?;
    Ok(())
}

fn psnr(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let a = image::open(&args[0])?.into_rgb8();
    let b = image::open(&args[1])?.into_rgb8();
    assert_eq!(a.dimensions(), b.dimensions(), "sizes differ");
    let (mut sum, mut worst) = (0f64, 0u8);
    for (x, y) in a.as_raw().iter().zip(b.as_raw()) {
        let d = x.abs_diff(*y);
        worst = worst.max(d);
        sum += f64::from(d) * f64::from(d);
    }
    let mse = sum / a.as_raw().len() as f64;
    let db = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    println!("psnr {db:.2} dB, worst difference {worst}/255");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("psnr") => psnr(&args[1..]),
        _ => Err("usage: ncnn_tool run|psnr ...".into()),
    }
}
