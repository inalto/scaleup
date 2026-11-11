//! Deterministic fakes for tests: a nearest-neighbour upscaler and a mean-colour inpainter.

use crate::backend::{Inpainter, Model};
use crate::image32::Image32;
use crate::{Result, ScaleUpError};

/// Paints every unknown pixel with the mean of the known ones, per channel; known pixels are
/// returned unchanged. A window with no known pixel is returned as given.
#[derive(Clone, Copy, Debug)]
pub struct MeanInpainter {
    pub window: u32,
}

impl Inpainter for MeanInpainter {
    fn window(&self) -> u32 {
        self.window
    }

    fn fill(&self, rgb: &Image32, known: &[bool]) -> Result<Image32> {
        let n = self.window as usize * self.window as usize;
        if (rgb.width, rgb.height) != (self.window, self.window) || known.len() != n {
            return Err(ScaleUpError::Model(format!(
                "MeanInpainter expects a {0}x{0} window and mask, got {1}x{2} and {3}",
                self.window,
                rgb.width,
                rgb.height,
                known.len()
            )));
        }
        let count = known.iter().filter(|&&k| k).count();
        if count == 0 {
            return Ok(rgb.clone());
        }
        let mut out = rgb.clone();
        for c in 0..3 {
            let plane = &mut out.data[c * n..(c + 1) * n];
            let sum: f64 = plane
                .iter()
                .zip(known)
                .filter(|(_, k)| **k)
                .map(|(v, _)| f64::from(*v))
                .sum();
            let mean = (sum / count as f64) as f32;
            for (v, _) in plane.iter_mut().zip(known).filter(|(_, k)| !**k) {
                *v = mean;
            }
        }
        Ok(out)
    }
}

/// Upscales by repeating every pixel `scale`×`scale` times. Pointwise, so tiled and untiled
/// output must match exactly.
#[derive(Clone, Copy, Debug)]
pub struct NearestModel {
    pub scale: u32,
}

impl Model for NearestModel {
    fn scale(&self) -> u32 {
        self.scale
    }

    fn run(&self, tile: &Image32) -> Result<Image32> {
        let s = self.scale;
        let mut out = Image32::new(tile.width * s, tile.height * s);
        for c in 0..3 {
            for y in 0..out.height {
                for x in 0..out.width {
                    out.set(c, x, y, tile.get(c, x / s, y / s));
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_replicates_pixels() {
        let mut img = Image32::new(2, 1);
        img.set(0, 1, 0, 0.5);
        let out = NearestModel { scale: 3 }.run(&img).unwrap();
        assert_eq!((out.width, out.height), (6, 3));
        assert_eq!(out.get(0, 2, 2), 0.0);
        assert_eq!(out.get(0, 3, 0), 0.5);
        assert_eq!(out.get(0, 5, 2), 0.5);
    }

    #[test]
    fn mean_inpainter_fills_unknown_with_known_mean() {
        let model = MeanInpainter { window: 2 };
        assert_eq!(model.window(), 2);
        assert_eq!(model.max_context_factor(), 4);
        let mut img = Image32::new(2, 2);
        for (i, v) in [0.2, 0.6, 0.9, 0.1].into_iter().enumerate() {
            img.set(0, i as u32 % 2, i as u32 / 2, v);
            img.set(2, i as u32 % 2, i as u32 / 2, 1.0);
        }
        let known = [true, true, false, false];
        let out = model.fill(&img, &known).unwrap();
        // Known pixels unchanged.
        assert_eq!(out.get(0, 0, 0), 0.2);
        assert_eq!(out.get(0, 1, 0), 0.6);
        // Unknown pixels get the mean of the known ones, per channel.
        for x in 0..2 {
            assert!((out.get(0, x, 1) - 0.4).abs() < 1e-6);
            assert_eq!(out.get(1, x, 1), 0.0);
            assert_eq!(out.get(2, x, 1), 1.0);
        }
        // No known pixel: returned as given.
        assert_eq!(model.fill(&img, &[false; 4]).unwrap(), img);
        // Wrong sizes are errors.
        assert!(model.fill(&img, &[true; 3]).is_err());
        assert!(model.fill(&Image32::new(3, 2), &[true; 6]).is_err());
    }
}
