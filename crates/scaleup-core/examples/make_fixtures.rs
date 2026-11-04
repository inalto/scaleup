//! Generates the deterministic test images in `tests/fixtures/images/`.
//!
//! `cargo run -p scaleup-core --example make_fixtures` rewrites them; the output is byte-identical
//! on every run and platform (no `rand`: a fixed-seed LCG supplies the noise).

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder, Rgb, RgbImage, Rgba, RgbaImage};

/// Fixed-seed linear congruential generator (Numerical Recipes constants): the only noise source,
/// so the images are identical on every platform.
struct Lcg(u32);

impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}

fn gradient_64x48() -> RgbImage {
    RgbImage::from_fn(64, 48, |x, y| {
        Rgb([(x * 255 / 63) as u8, (y * 255 / 47) as u8, 96])
    })
}

fn alpha_checker_32x32() -> RgbaImage {
    RgbaImage::from_fn(32, 32, |x, y| {
        let on = ((x / 4) + (y / 4)) % 2 == 0;
        let a = if on { 255 } else { 0 };
        Rgba([220, 40, 40, a])
    })
}

fn tiny_3x2() -> RgbImage {
    let px = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ];
    RgbImage::from_fn(3, 2, |x, y| Rgb(px[(y * 3 + x) as usize]))
}

fn photo_like_256x192() -> RgbImage {
    let mut lcg = Lcg(0x5EED_1234);
    let mut img = RgbImage::new(256, 192);
    for y in 0..192u32 {
        for x in 0..256u32 {
            let base_r = (x * 200 / 255) as i32 + 20;
            let base_g = (y * 200 / 191) as i32 + 20;
            let base_b = ((x + y) * 120 / 447) as i32 + 60;
            let n = |l: &mut Lcg| (l.next_u8() as i32 % 33) - 16;
            let (nr, ng, nb) = (n(&mut lcg), n(&mut lcg), n(&mut lcg));
            img.put_pixel(
                x,
                y,
                Rgb([
                    (base_r + nr).clamp(0, 255) as u8,
                    (base_g + ng).clamp(0, 255) as u8,
                    (base_b + nb).clamp(0, 255) as u8,
                ]),
            );
        }
    }
    img
}

fn odd_1001x733() -> RgbImage {
    RgbImage::from_fn(1001, 733, |x, y| {
        // A one-pixel diagonal from the top-left to the bottom-right corner.
        let on_line = (x as u64 * 732).abs_diff(y as u64 * 1000) < 1000;
        if on_line {
            Rgb([250, 250, 250])
        } else {
            Rgb([30, 90, 160])
        }
    })
}

fn png(bytes: &[u8], w: u32, h: u32, color: ExtendedColorType) -> Vec<u8> {
    let mut out = Vec::new();
    PngEncoder::new_with_quality(&mut out, CompressionType::Best, FilterType::Adaptive)
        .write_image(bytes, w, h, color)
        .expect("png encode");
    out
}

fn jpeg(img: &RgbImage, quality: u8) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut out, quality)
        .encode(
            img.as_raw(),
            img.width(),
            img.height(),
            ExtendedColorType::Rgb8,
        )
        .expect("jpeg encode");
    out.into_inner()
}

/// Every fixture as `(file name, encoded bytes)`, in a fixed order.
fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let gradient = gradient_64x48();
    let checker = alpha_checker_32x32();
    let tiny = tiny_3x2();
    let odd = odd_1001x733();
    vec![
        (
            "gradient_64x48.png",
            png(gradient.as_raw(), 64, 48, ExtendedColorType::Rgb8),
        ),
        (
            "alpha_checker_32x32.png",
            png(checker.as_raw(), 32, 32, ExtendedColorType::Rgba8),
        ),
        (
            "tiny_3x2.png",
            png(tiny.as_raw(), 3, 2, ExtendedColorType::Rgb8),
        ),
        ("photo_like_256x192.jpg", jpeg(&photo_like_256x192(), 85)),
        (
            "odd_1001x733.png",
            png(odd.as_raw(), 1001, 733, ExtendedColorType::Rgb8),
        ),
    ]
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/images");
    fs::create_dir_all(&dir).expect("create fixtures dir");
    for (name, bytes) in fixtures() {
        let path = dir.join(name);
        fs::write(&path, &bytes).expect("write fixture");
        println!("{} ({} bytes)", path.display(), bytes.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    fn decode(name: &str) -> image::DynamicImage {
        let bytes = fixtures()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, b)| b)
            .unwrap_or_else(|| panic!("fixture {name} missing"));
        image::load_from_memory(&bytes).expect("decodes")
    }

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(fixtures(), fixtures());
    }

    #[test]
    fn fixture_set_and_dimensions() {
        let names: Vec<&str> = fixtures().iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            [
                "gradient_64x48.png",
                "alpha_checker_32x32.png",
                "tiny_3x2.png",
                "photo_like_256x192.jpg",
                "odd_1001x733.png"
            ]
        );
        assert_eq!(decode("gradient_64x48.png").dimensions(), (64, 48));
        assert_eq!(decode("alpha_checker_32x32.png").dimensions(), (32, 32));
        assert_eq!(decode("tiny_3x2.png").dimensions(), (3, 2));
        assert_eq!(decode("photo_like_256x192.jpg").dimensions(), (256, 192));
        assert_eq!(decode("odd_1001x733.png").dimensions(), (1001, 733));
    }

    #[test]
    fn alpha_checker_is_binary_rgba() {
        let img = decode("alpha_checker_32x32.png");
        assert_eq!(img.color(), image::ColorType::Rgba8);
        let rgba = img.to_rgba8();
        let alphas: std::collections::BTreeSet<u8> = rgba.pixels().map(|p| p.0[3]).collect();
        assert_eq!(alphas.into_iter().collect::<Vec<_>>(), [0, 255]);
    }

    #[test]
    fn gradient_and_odd_are_rgb() {
        assert_eq!(decode("gradient_64x48.png").color(), image::ColorType::Rgb8);
        assert_eq!(decode("odd_1001x733.png").color(), image::ColorType::Rgb8);
    }

    /// The committed files are what the generator produces (compared as decoded pixels, so an
    /// encoder update that changes bytes but not content does not fail the build).
    #[test]
    fn committed_fixtures_match_generator() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/images");
        for (name, bytes) in fixtures() {
            let committed = fs::read(dir.join(name))
                .unwrap_or_else(|e| panic!("tests/fixtures/images/{name}: {e}"));
            let want = image::load_from_memory(&bytes)
                .expect("generated decodes")
                .to_rgba8();
            let got = image::load_from_memory(&committed)
                .expect("committed decodes")
                .to_rgba8();
            assert_eq!(got.dimensions(), want.dimensions(), "{name}: dimensions");
            assert!(
                got.as_raw() == want.as_raw(),
                "{name}: pixels differ; rerun make_fixtures"
            );
        }
    }

    #[test]
    fn total_size_under_100_kib() {
        let total: usize = fixtures().iter().map(|(_, b)| b.len()).sum();
        assert!(total < 100 * 1024, "fixtures total {total} bytes");
    }
}
