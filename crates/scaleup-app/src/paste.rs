//! Clipboard images become ordinary input files.

use std::path::{Path, PathBuf};

use eframe::egui;

/// egui swallows the Ctrl+V press to deliver text, so an image paste shows only as this release.
pub fn is_paste_key_release(ev: &egui::Event) -> bool {
    matches!(
        ev,
        egui::Event::Key {
            key: egui::Key::V,
            pressed: false,
            modifiers,
            ..
        } if modifiers.command
    )
}

/// Writes clipboard pixels to `<dir>/scaleup-paste-<stamp>.png`.
pub fn write_clipboard_image(
    data: &arboard::ImageData<'_>,
    dir: &Path,
    stamp: u128,
) -> Result<PathBuf, String> {
    let image =
        image::RgbaImage::from_raw(data.width as u32, data.height as u32, data.bytes.to_vec())
            .ok_or("the clipboard image has an unexpected size")?;
    let path = dir.join(format!("scaleup-paste-{stamp}.png"));
    image.save(&path).map_err(|e| e.to_string())?;
    Ok(path)
}

/// The clipboard's image saved to the temp folder; `None` when it holds no image.
pub fn paste_from_clipboard() -> Option<Result<PathBuf, String>> {
    let image = arboard::Clipboard::new().ok()?.get_image().ok()?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    Some(write_clipboard_image(&image, &std::env::temp_dir(), stamp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Mode, mode_from_path};
    use std::borrow::Cow;

    #[test]
    fn paste_writes_png_and_selects_single() {
        let d = tempfile::tempdir().unwrap();
        let data = arboard::ImageData {
            width: 2,
            height: 1,
            bytes: Cow::Borrowed(&[255, 0, 0, 255, 0, 255, 0, 128]),
        };
        let path = write_clipboard_image(&data, d.path(), 42).unwrap();
        assert_eq!(path, d.path().join("scaleup-paste-42.png"));
        let img = image::open(&path).unwrap().into_rgba8();
        assert_eq!(img.dimensions(), (2, 1));
        assert_eq!(img.get_pixel(1, 0).0, [0, 255, 0, 128]);
        assert_eq!(mode_from_path(&path), Some(Mode::Single(path)));
    }

    #[test]
    fn paste_rejects_mismatched_buffer() {
        let d = tempfile::tempdir().unwrap();
        let data = arboard::ImageData {
            width: 2,
            height: 2,
            bytes: Cow::Borrowed(&[0, 0, 0, 0]),
        };
        assert!(write_clipboard_image(&data, d.path(), 1).is_err());
    }

    #[test]
    fn paste_key_release_needs_the_command_modifier() {
        let key = |pressed, modifiers| egui::Event::Key {
            key: egui::Key::V,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers,
        };
        assert!(is_paste_key_release(&key(false, egui::Modifiers::COMMAND)));
        assert!(!is_paste_key_release(&key(true, egui::Modifiers::COMMAND)));
        assert!(!is_paste_key_release(&key(false, egui::Modifiers::NONE)));
    }
}
