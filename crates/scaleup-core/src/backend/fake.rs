//! Deterministic fake model for tests: nearest-neighbour replication.

use crate::Result;
use crate::backend::Model;
use crate::image32::Image32;

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
}
