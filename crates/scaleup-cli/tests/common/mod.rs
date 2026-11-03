#![allow(dead_code)]
use std::path::{Path, PathBuf};

use assert_cmd::Command;

pub fn scaleup() -> Command {
    Command::cargo_bin("scaleup").unwrap()
}

pub fn fixture_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/nearest2x.onnx")
}

/// Writes a 4x4 RGBA png.
pub fn write_png(path: &Path) {
    let img = image::RgbaImage::from_fn(4, 4, |x, y| {
        image::Rgba([x as u8 * 60, y as u8 * 60, 128, 255])
    });
    img.save(path).unwrap();
}
