//! Test-time augmentation: average the model over the 8 dihedral transforms of each tile.

use crate::Result;
use crate::backend::Model;
use crate::image32::Image32;

/// Rotates 90° clockwise.
fn rot90(img: &Image32) -> Image32 {
    let (w, h) = (img.width, img.height);
    let mut out = Image32::new(h, w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                out.set(c, h - 1 - y, x, img.get(c, x, y));
            }
        }
    }
    out
}

/// Mirrors left-right.
fn flip_h(img: &Image32) -> Image32 {
    let mut out = Image32::new(img.width, img.height);
    for c in 0..3 {
        for y in 0..img.height {
            for x in 0..img.width {
                out.set(c, img.width - 1 - x, y, img.get(c, x, y));
            }
        }
    }
    out
}

fn rotate(img: &Image32, quarter_turns: u8) -> Image32 {
    let mut out = img.clone();
    for _ in 0..quarter_turns % 4 {
        out = rot90(&out);
    }
    out
}

/// Dihedral transform `k` in `0..8`: rotate 90° clockwise `k % 4` times, then flip left-right if `k >= 4`.
pub fn dihedral(img: &Image32, k: u8) -> Image32 {
    let rotated = rotate(img, k % 4);
    if k >= 4 { flip_h(&rotated) } else { rotated }
}

/// Undoes [`dihedral`] with the same `k`.
pub fn dihedral_inverse(img: &Image32, k: u8) -> Image32 {
    let unflipped = if k >= 4 { flip_h(img) } else { img.clone() };
    rotate(&unflipped, (4 - k % 4) % 4)
}

/// Wraps a model so that each run averages the outputs of all 8 dihedral variants (8× slower).
pub struct TtaModel<'a> {
    pub inner: &'a dyn Model,
}

impl Model for TtaModel<'_> {
    fn scale(&self) -> u32 {
        self.inner.scale()
    }

    fn run(&self, tile: &Image32) -> Result<Image32> {
        let mut sum: Option<Image32> = None;
        for k in 0..8 {
            let out = dihedral_inverse(&self.inner.run(&dihedral(tile, k))?, k);
            match &mut sum {
                None => sum = Some(out),
                Some(acc) => acc
                    .data
                    .iter_mut()
                    .zip(&out.data)
                    .for_each(|(a, b)| *a += b),
            }
        }
        let mut avg = sum.expect("8 variants");
        avg.data.iter_mut().for_each(|v| *v /= 8.0);
        Ok(avg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fake::NearestModel;

    fn numbered(w: u32, h: u32) -> Image32 {
        let mut img = Image32::new(w, h);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = i as f32 / 1000.0;
        }
        img
    }

    #[test]
    fn dihedral_inverse_roundtrip() {
        let img = numbered(4, 6);
        for k in 0..8 {
            let t = dihedral(&img, k);
            assert_eq!(dihedral_inverse(&t, k), img, "k={k}");
        }
    }

    #[test]
    fn dihedral_variants_are_distinct_transforms() {
        let img = numbered(4, 6);
        let r1 = dihedral(&img, 1);
        assert_eq!((r1.width, r1.height), (6, 4));
        // Clockwise rotation: top-left of the source becomes top-right.
        assert_eq!(r1.get(0, 5, 0), img.get(0, 0, 0));
        let f = dihedral(&img, 4);
        assert_eq!(f.get(2, 3, 1), img.get(2, 0, 1));
        let all: Vec<Image32> = (0..8).map(|k| dihedral(&img, k)).collect();
        for a in 0..8 {
            for b in a + 1..8 {
                assert_ne!(all[a], all[b], "k={a} and k={b} coincide");
            }
        }
    }

    #[test]
    fn tta_nearest_is_identity_to_plain() {
        let inner = NearestModel { scale: 2 };
        let img = numbered(5, 3);
        let plain = inner.run(&img).unwrap();
        let tta = TtaModel { inner: &inner }.run(&img).unwrap();
        assert_eq!((tta.width, tta.height), (plain.width, plain.height));
        for (a, b) in tta.data.iter().zip(&plain.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn tta_scale_passthrough() {
        let inner = NearestModel { scale: 3 };
        assert_eq!(TtaModel { inner: &inner }.scale(), 3);
    }
}
