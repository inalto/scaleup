//! Manual checks against other engines.
//!
//! ```text
//! ncnn_tool run <model.param> <in.png> <out.png> <tile> <overlap>   upscale, print the seconds
//! ncnn_tool psnr <a.png> <b.png>                                    PSNR and worst channel difference
//! ncnn_tool seams <whole.png> <tiled.png> <scale> <tile> <overlap>  difference along tile boundaries
//! ```

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use image::RgbImage;
use scaleup_core::backend::Backend;
use scaleup_core::tiling::tile_grid;
use scaleup_core::{CancelToken, DevicePreference, UpscaleOptions, upscale};
use scaleup_wgpu::WgpuBackend;

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let backend = WgpuBackend::new(DevicePreference::Auto)?;
    println!("device: {}", backend.device().name);
    let model = backend.load_model(Path::new(&args[0]), None)?;
    let input = image::open(&args[1])?.into_rgba8();
    let opts = UpscaleOptions {
        tile_size: args[3].parse()?,
        tile_overlap: Some(args[4].parse()?),
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

/// Tiled output against a whole-image run of the same model, in 1/255 steps. A *step* is how much the
/// difference between the two changes from one side of a tile boundary to the other: the part of the
/// difference an eye can see as a line.
#[derive(Debug)]
struct SeamReport {
    boundaries: usize,
    /// Largest channel difference anywhere.
    worst: u32,
    /// Mean difference within 8 output pixels of a boundary, and further away.
    near_mean: f64,
    far_mean: f64,
    /// Largest and mean step of single pixels.
    step_max: u32,
    step_mean: f64,
    /// Largest step averaged along a 32-pixel segment of boundary, which is what shows as a line.
    segment_max: f64,
    /// The same, only where the whole-image run is flat, because a line shows most there.
    flat_segment_max: f64,
    /// Share of segments, in percent, whose mean step is at least 1.
    segments_over_1: f64,
}

/// Compares two `scale`× outputs, ignoring 64 input pixels around the image: a whole run zero-pads
/// there, whereas tiles see reflected context.
fn seam_report(
    whole: &RgbImage,
    tiled: &RgbImage,
    scale: u32,
    tile: u32,
    overlap: u32,
) -> SeamReport {
    const SEGMENT: u32 = 32;
    const FLAT_RANGE: i32 = 12;
    assert_eq!(whole.dimensions(), tiled.dimensions(), "sizes differ");
    let (w, h) = whole.dimensions();
    let margin = 64 * scale;
    let jobs = tile_grid(w / scale, h / scale, tile, overlap).expect("valid tile grid");
    let lines = |pick: &dyn Fn(&scaleup_core::tiling::TileJob) -> u32, len: u32| -> Vec<u32> {
        jobs.iter()
            .map(|j| pick(j) * scale)
            .filter(|&v| v > margin && v < len - margin)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    let columns = lines(&|j| j.input.x + j.keep.x, w);
    let rows = lines(&|j| j.input.y + j.keep.y, h);

    let diff = |x: u32, y: u32, c: usize| {
        i32::from(tiled.get_pixel(x, y).0[c]) - i32::from(whole.get_pixel(x, y).0[c])
    };
    let (mut worst, mut near, mut near_n, mut far, mut far_n) = (0, 0u64, 0u64, 0u64, 0u64);
    for y in margin..h - margin {
        for x in margin..w - margin {
            let close = columns.iter().any(|&b| b.abs_diff(x) < 8)
                || rows.iter().any(|&b| b.abs_diff(y) < 8);
            for c in 0..3 {
                let d = diff(x, y, c).unsigned_abs();
                worst = worst.max(d);
                if close {
                    near += u64::from(d);
                    near_n += 1;
                } else {
                    far += u64::from(d);
                    far_n += 1;
                }
            }
        }
    }

    let (mut step_max, mut step_sum, mut step_n) = (0, 0u64, 0u64);
    let (mut segment_max, mut flat_segment_max, mut over_1, mut segments) =
        (0f64, 0f64, 0u64, 0u64);
    // Each boundary as (line, length along it, pixel at (line, along)).
    let mut walk = |line: u32, along: u32, at: &dyn Fn(u32, u32) -> (u32, u32)| {
        let mut start = margin;
        while start + SEGMENT <= along - margin {
            for c in 0..3 {
                let (mut sum, mut lo, mut hi) = (0i32, i32::MAX, i32::MIN);
                for i in start..start + SEGMENT {
                    let (x1, y1) = at(line, i);
                    let (x0, y0) = at(line - 1, i);
                    let step = diff(x1, y1, c) - diff(x0, y0, c);
                    sum += step;
                    step_max = step_max.max(step.unsigned_abs());
                    step_sum += u64::from(step.unsigned_abs());
                    step_n += 1;
                    for (x, y) in [(x0, y0), (x1, y1)] {
                        let v = i32::from(whole.get_pixel(x, y).0[c]);
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                let mean = (f64::from(sum) / f64::from(SEGMENT)).abs();
                segment_max = segment_max.max(mean);
                if hi - lo <= FLAT_RANGE {
                    flat_segment_max = flat_segment_max.max(mean);
                }
                segments += 1;
                over_1 += u64::from(mean >= 1.0);
            }
            start += SEGMENT;
        }
    };
    for &x in &columns {
        walk(x, h, &|line, i| (line, i));
    }
    for &y in &rows {
        walk(y, w, &|line, i| (i, line));
    }
    SeamReport {
        boundaries: columns.len() + rows.len(),
        worst,
        near_mean: near as f64 / near_n.max(1) as f64,
        far_mean: far as f64 / far_n.max(1) as f64,
        step_max,
        step_mean: step_sum as f64 / step_n.max(1) as f64,
        segment_max,
        flat_segment_max,
        segments_over_1: 100.0 * over_1 as f64 / segments.max(1) as f64,
    }
}

fn seams(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let whole = image::open(&args[0])?.into_rgb8();
    let tiled = image::open(&args[1])?.into_rgb8();
    let r = seam_report(
        &whole,
        &tiled,
        args[2].parse()?,
        args[3].parse()?,
        args[4].parse()?,
    );
    println!(
        "lines={} worst={} near={:.3} far={:.3} step_max={} step_mean={:.3} seg_max={:.2} flat_max={:.2} seg>=1={:.2}%",
        r.boundaries,
        r.worst,
        r.near_mean,
        r.far_mean,
        r.step_max,
        r.step_mean,
        r.segment_max,
        r.flat_segment_max,
        r.segments_over_1
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("psnr") => psnr(&args[1..]),
        Some("seams") => seams(&args[1..]),
        _ => Err("usage: ncnn_tool run|psnr|seams ...".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth image whose tiled twin is `delta` brighter right of the tile boundary at `BOUNDARY`.
    fn with_step(delta: u8) -> (RgbImage, RgbImage) {
        let whole = RgbImage::from_fn(800, 800, |_, y| image::Rgb([100, (y / 16) as u8, 50]));
        let mut tiled = whole.clone();
        for (x, _, p) in tiled.enumerate_pixels_mut() {
            if x >= 4 * BOUNDARY {
                p.0[0] += delta;
            }
        }
        (whole, tiled)
    }

    /// Tile 128 overlap 32 has a stride of 64, so the grid has boundaries at multiples of 64.
    const BOUNDARY: u32 = 128;

    #[test]
    fn a_step_on_a_boundary_is_reported() {
        let (whole, tiled) = with_step(3);
        let r = seam_report(&whole, &tiled, 4, 128, 32);
        assert_eq!(r.step_max, 3);
        assert_eq!(r.segment_max, 3.0);
        assert_eq!(r.flat_segment_max, 3.0);
        assert!(r.segments_over_1 > 0.0);
    }

    #[test]
    fn identical_images_report_nothing() {
        let (whole, _) = with_step(0);
        let r = seam_report(&whole, &whole, 4, 128, 32);
        assert_eq!((r.worst, r.step_max), (0, 0));
        assert_eq!(r.segment_max, 0.0);
    }
}
