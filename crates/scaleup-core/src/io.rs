//! Image file loading and saving.
//!
//! WebP output is always lossless: the `image` crate's pure-Rust WebP encoder has no lossy mode, and
//! the `webp` crate would add a C library (libwebp) to an otherwise pure-Rust build. The quality
//! value is therefore ignored for WebP.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::codecs::webp::WebPEncoder;
use image::{
    DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageError, ImageReader, RgbaImage,
};

use crate::{Result, ScaleUpError};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    Png,
    Jpg,
    Webp,
}

impl OutputFormat {
    /// Case-insensitive file extension lookup (`jpeg` is accepted for JPEG).
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpg),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }

    pub fn ext(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpg => "jpg",
            Self::Webp => "webp",
        }
    }

    /// Like [`OutputFormat::from_ext`] but with an [`ScaleUpError::InvalidOption`] for unknown names.
    pub fn parse(s: &str) -> Result<Self> {
        Self::from_ext(s)
            .ok_or_else(|| ScaleUpError::InvalidOption(format!("unsupported output format '{s}'")))
    }
}

const INPUT_EXTS: [&str; 7] = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"];

/// True when the extension is one of jpg/jpeg/png/webp/bmp/tif/tiff (case-insensitive).
pub fn is_supported_input(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| INPUT_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

fn decode_err(e: ImageError) -> ScaleUpError {
    match e {
        ImageError::IoError(io) => ScaleUpError::Io(io),
        other => ScaleUpError::Decode(other.to_string()),
    }
}

fn encode_err(e: ImageError) -> ScaleUpError {
    match e {
        ImageError::IoError(io) => ScaleUpError::Io(io),
        other => ScaleUpError::Encode(other.to_string()),
    }
}

/// Loads any supported image as RGBA8, with EXIF orientation applied.
pub fn load_image(path: &Path) -> Result<RgbaImage> {
    let reader = ImageReader::open(path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder().map_err(decode_err)?;
    let orientation = decoder.orientation().map_err(decode_err)?;
    let mut img = DynamicImage::from_decoder(decoder).map_err(decode_err)?;
    img.apply_orientation(orientation);
    Ok(img.into_rgba8())
}

/// Composites onto white and drops alpha (for formats without transparency).
fn flatten_on_white(img: &RgbaImage) -> Vec<u8> {
    img.pixels()
        .flat_map(|p| {
            let a = p[3] as u32;
            let mix = move |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
            [mix(p[0]), mix(p[1]), mix(p[2])]
        })
        .collect()
}

fn encode(
    w: impl Write,
    img: &RgbaImage,
    fmt: OutputFormat,
    quality: u8,
) -> std::result::Result<(), ImageError> {
    let (width, height) = img.dimensions();
    match fmt {
        OutputFormat::Png => {
            let compression = match quality {
                0..34 => CompressionType::Fast,
                34..=66 => CompressionType::Default,
                _ => CompressionType::Best,
            };
            PngEncoder::new_with_quality(w, compression, FilterType::Adaptive).write_image(
                img.as_raw(),
                width,
                height,
                ExtendedColorType::Rgba8,
            )
        }
        OutputFormat::Jpg => JpegEncoder::new_with_quality(w, quality.clamp(1, 100)).write_image(
            &flatten_on_white(img),
            width,
            height,
            ExtendedColorType::Rgb8,
        ),
        OutputFormat::Webp => WebPEncoder::new_lossless(w).write_image(
            img.as_raw(),
            width,
            height,
            ExtendedColorType::Rgba8,
        ),
    }
}

/// Encodes `img` and writes it atomically: the data goes to `<path>.tmp`, which is renamed over
/// `path` only after a successful encode, and removed on failure.
///
/// `quality` is 0..=100: JPEG quality; PNG compression level (<34 fast, ≤66 default, else best);
/// ignored for WebP (always lossless).
pub fn save_image(path: &Path, img: &RgbaImage, fmt: OutputFormat, quality: u8) -> Result<()> {
    let mut tmp = PathBuf::from(path).into_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let result = (|| -> Result<()> {
        let mut writer = BufWriter::new(File::create(&tmp)?);
        encode(&mut writer, img, fmt, quality).map_err(encode_err)?;
        writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScaleUpError;
    use image::{Rgba, RgbaImage};

    fn sample() -> RgbaImage {
        RgbaImage::from_fn(13, 9, |x, y| {
            Rgba([
                (x * 19) as u8,
                (y * 27) as u8,
                ((x + y) * 7) as u8,
                (x * y * 3) as u8,
            ])
        })
    }

    #[test]
    fn png_roundtrip_exact() {
        let dir = tempfile::tempdir().unwrap();
        for q in [0u8, 50, 100] {
            let path = dir.path().join(format!("out{q}.png"));
            save_image(&path, &sample(), OutputFormat::Png, q).unwrap();
            assert_eq!(load_image(&path).unwrap(), sample());
        }
    }

    #[test]
    fn webp_roundtrip_exact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.webp");
        save_image(&path, &sample(), OutputFormat::Webp, 100).unwrap();
        assert_eq!(load_image(&path).unwrap(), sample());
    }

    #[test]
    fn jpg_drops_alpha_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.jpg");
        let transparent = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 0]));
        save_image(&path, &transparent, OutputFormat::Jpg, 90).unwrap();
        let back = load_image(&path).unwrap();
        assert_eq!(back.dimensions(), (8, 8));
        // Composited on white, and opaque.
        assert!(back.pixels().all(|p| p[3] == 255 && p[0] > 240));
        save_image(&path, &sample(), OutputFormat::Jpg, 0).unwrap();
    }

    #[test]
    fn save_is_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atomic.png");
        std::fs::write(&path, b"old").unwrap();
        save_image(&path, &sample(), OutputFormat::Png, 90).unwrap();
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("atomic.png")]);
        assert_eq!(load_image(&path).unwrap(), sample());
        // Failing save (missing directory) leaves nothing behind and is an I/O error.
        let bad = dir.path().join("missing").join("x.png");
        assert!(matches!(
            save_image(&bad, &sample(), OutputFormat::Png, 90),
            Err(ScaleUpError::Io(_))
        ));
    }

    #[test]
    fn from_ext_cases() {
        assert_eq!(OutputFormat::from_ext("JPEG"), Some(OutputFormat::Jpg));
        assert_eq!(OutputFormat::from_ext("jpg"), Some(OutputFormat::Jpg));
        assert_eq!(OutputFormat::from_ext("webp"), Some(OutputFormat::Webp));
        assert_eq!(OutputFormat::from_ext("Png"), Some(OutputFormat::Png));
        assert_eq!(OutputFormat::from_ext("gif"), None);
        assert_eq!(OutputFormat::Jpg.ext(), "jpg");
        assert_eq!(OutputFormat::parse("webp").unwrap(), OutputFormat::Webp);
        assert!(matches!(
            OutputFormat::parse("gif"),
            Err(ScaleUpError::InvalidOption(_))
        ));
    }

    #[test]
    fn supported_inputs() {
        for ok in [
            "a.jpg", "b.JPEG", "c.png", "d.webp", "e.bmp", "f.tif", "g.TIFF",
        ] {
            assert!(is_supported_input(Path::new(ok)), "{ok}");
        }
        for no in ["a.gif", "b", "c.onnx", "d.png.txt"] {
            assert!(!is_supported_input(Path::new(no)), "{no}");
        }
    }

    #[test]
    fn load_missing_file_is_io_error() {
        let res = load_image(Path::new("definitely/not/here.png"));
        assert!(matches!(res, Err(ScaleUpError::Io(_))));
    }

    #[test]
    fn load_garbage_is_decode_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.png");
        std::fs::write(&path, b"not an image at all").unwrap();
        assert!(matches!(load_image(&path), Err(ScaleUpError::Decode(_))));
    }
}
