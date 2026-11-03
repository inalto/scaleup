//! Output size rules and the final Lanczos3 resize.

use image::RgbaImage;
use image::imageops::{self, FilterType};

use crate::{Result, ScaleUpError};

/// Final output size. `output_width` wins (height keeps the aspect ratio, rounded, at least 1);
/// otherwise `output_scale` (must be 2, 3 or 4), otherwise the model's own scale.
pub fn target_size(
    in_w: u32,
    in_h: u32,
    model_scale: u32,
    output_scale: Option<u32>,
    output_width: Option<u32>,
) -> Result<(u32, u32)> {
    if let Some(s) = output_scale
        && !(2..=4).contains(&s)
    {
        return Err(ScaleUpError::InvalidOption(format!(
            "output scale must be 2, 3 or 4, got {s}"
        )));
    }
    if let Some(w) = output_width {
        if w == 0 {
            return Err(ScaleUpError::InvalidOption(
                "output width must be positive".into(),
            ));
        }
        let h = (in_h as f64 * w as f64 / in_w as f64).round().max(1.0) as u32;
        return Ok((w, h));
    }
    let s = output_scale.unwrap_or(model_scale);
    Ok((in_w * s, in_h * s))
}

/// Lanczos3 resize; returns a copy when the size already matches.
pub fn resize_rgba(img: &RgbaImage, w: u32, h: u32) -> RgbaImage {
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    imageops::resize(img, w, h, FilterType::Lanczos3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScaleUpError;

    #[test]
    fn target_size_table() {
        assert_eq!(target_size(100, 50, 4, None, None).unwrap(), (400, 200));
        assert_eq!(target_size(100, 50, 4, Some(2), None).unwrap(), (200, 100));
        assert_eq!(target_size(100, 50, 2, Some(3), None).unwrap(), (300, 150));
        assert_eq!(
            target_size(100, 50, 4, Some(2), Some(1920)).unwrap(),
            (1920, 960)
        );
        // 7 * 2 / 3 = 4.67 -> 5 ; 7 * 4 / 3 = 9.33 -> 9
        assert_eq!(target_size(3, 7, 4, None, Some(2)).unwrap(), (2, 5));
        assert_eq!(target_size(3, 7, 4, None, Some(4)).unwrap(), (4, 9));
        // Height never rounds to zero.
        assert_eq!(target_size(1000, 1, 4, None, Some(10)).unwrap(), (10, 1));
        for bad in [0, 1, 5, 8] {
            assert!(matches!(
                target_size(100, 50, 4, Some(bad), None),
                Err(ScaleUpError::InvalidOption(_))
            ));
        }
        assert!(matches!(
            target_size(100, 50, 4, None, Some(0)),
            Err(ScaleUpError::InvalidOption(_))
        ));
    }

    #[test]
    fn resize_identity_when_same_size() {
        let img = image::RgbaImage::from_fn(5, 4, |x, y| {
            image::Rgba([x as u8 * 40, y as u8 * 50, 7, 200])
        });
        assert_eq!(resize_rgba(&img, 5, 4), img);
        let half = resize_rgba(&img, 3, 2);
        assert_eq!(half.dimensions(), (3, 2));
    }
}
