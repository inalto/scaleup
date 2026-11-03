//! Planar float RGB image used for model input/output.

use image::{Rgba, RgbaImage};

/// Planar CHW image: exactly 3 channels of `f32` in `0.0..=1.0`, `data.len() == 3 * width * height`.
#[derive(Clone, Debug, PartialEq)]
pub struct Image32 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl Image32 {
    /// All-zero image.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; 3 * width as usize * height as usize],
        }
    }

    fn plane(&self) -> usize {
        self.width as usize * self.height as usize
    }

    fn index(&self, c: usize, x: u32, y: u32) -> usize {
        debug_assert!(c < 3 && x < self.width && y < self.height);
        c * self.plane() + y as usize * self.width as usize + x as usize
    }

    pub fn get(&self, c: usize, x: u32, y: u32) -> f32 {
        self.data[self.index(c, x, y)]
    }

    pub fn set(&mut self, c: usize, x: u32, y: u32, v: f32) {
        let i = self.index(c, x, y);
        self.data[i] = v;
    }

    /// Splits an RGBA image into a float RGB image and a row-major alpha plane.
    pub fn from_rgba(img: &RgbaImage) -> (Image32, Vec<u8>) {
        let (w, h) = img.dimensions();
        let mut out = Image32::new(w, h);
        let plane = out.plane();
        let mut alpha = Vec::with_capacity(plane);
        for (i, p) in img.pixels().enumerate() {
            for c in 0..3 {
                out.data[c * plane + i] = p[c] as f32 / 255.0;
            }
            alpha.push(p[3]);
        }
        (out, alpha)
    }

    /// Builds a 3-channel image with `a / 255` in every channel.
    pub fn from_gray_replicated(alpha: &[u8], width: u32, height: u32) -> Image32 {
        let mut out = Image32::new(width, height);
        let plane = out.plane();
        assert_eq!(alpha.len(), plane, "alpha plane size mismatch");
        for (i, &a) in alpha.iter().enumerate() {
            let v = a as f32 / 255.0;
            for c in 0..3 {
                out.data[c * plane + i] = v;
            }
        }
        out
    }

    /// Converts back to RGBA8 (clamped, rounded). `alpha == None` means fully opaque.
    pub fn to_rgba(&self, alpha: Option<&[u8]>) -> RgbaImage {
        let plane = self.plane();
        if let Some(a) = alpha {
            assert_eq!(a.len(), plane, "alpha plane size mismatch");
        }
        let mut out = RgbaImage::new(self.width, self.height);
        for (i, p) in out.pixels_mut().enumerate() {
            *p = Rgba([
                to_u8(self.data[i]),
                to_u8(self.data[plane + i]),
                to_u8(self.data[2 * plane + i]),
                alpha.map_or(255, |a| a[i]),
            ]);
        }
        out
    }

    /// Channel 0 as bytes, row-major.
    pub fn channel0_to_u8(&self) -> Vec<u8> {
        self.data[..self.plane()]
            .iter()
            .copied()
            .map(to_u8)
            .collect()
    }

    /// Copies the `w`×`h` region at (`x`, `y`). The region must lie inside the image.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Image32 {
        assert!(
            x + w <= self.width && y + h <= self.height,
            "crop out of bounds"
        );
        let mut out = Image32::new(w, h);
        for c in 0..3 {
            for row in 0..h {
                let src = self.index(c, x, y + row);
                let dst = out.index(c, 0, row);
                out.data[dst..dst + w as usize].copy_from_slice(&self.data[src..src + w as usize]);
            }
        }
        out
    }

    /// Writes `src` with its top-left corner at (`x`, `y`). `src` must fit inside `self`.
    pub fn paste(&mut self, src: &Image32, x: u32, y: u32) {
        assert!(
            x + src.width <= self.width && y + src.height <= self.height,
            "paste out of bounds"
        );
        let w = src.width as usize;
        for c in 0..3 {
            for row in 0..src.height {
                let s = src.index(c, 0, row);
                let d = self.index(c, x, y + row);
                self.data[d..d + w].copy_from_slice(&src.data[s..s + w]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    /// Deterministic pseudo-random RGBA image (LCG), so tests need no `rand` crate.
    pub(crate) fn random_rgba(w: u32, h: u32, seed: u32) -> RgbaImage {
        let mut s = seed;
        RgbaImage::from_fn(w, h, |_, _| {
            let mut px = [0u8; 4];
            for v in &mut px {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = (s >> 24) as u8;
            }
            Rgba(px)
        })
    }

    #[test]
    fn rgba_roundtrip_exact() {
        let img = random_rgba(7, 5, 42);
        let (rgb, alpha) = Image32::from_rgba(&img);
        assert_eq!(rgb.to_rgba(Some(&alpha)), img);
    }

    #[test]
    fn alpha_plane_extracted() {
        let img = random_rgba(7, 5, 7);
        let (rgb, alpha) = Image32::from_rgba(&img);
        assert_eq!(rgb.data.len(), 3 * 7 * 5);
        let expected: Vec<u8> = img.pixels().map(|p| p[3]).collect();
        assert_eq!(alpha, expected);
        let gray = Image32::from_gray_replicated(&alpha, 7, 5);
        assert_eq!(gray.channel0_to_u8(), expected);
        assert_eq!(gray.get(1, 3, 2), gray.get(0, 3, 2));
        assert_eq!(gray.get(2, 3, 2), gray.get(0, 3, 2));
    }

    #[test]
    fn crop_paste_identity() {
        let (rgb, _) = Image32::from_rgba(&random_rgba(9, 6, 3));
        let piece = rgb.crop(2, 1, 4, 3);
        assert_eq!((piece.width, piece.height), (4, 3));
        assert_eq!(piece.get(1, 0, 0), rgb.get(1, 2, 1));
        assert_eq!(piece.get(2, 3, 2), rgb.get(2, 5, 3));
        let mut copy = rgb.clone();
        for c in 0..3 {
            for y in 1..4 {
                for x in 2..6 {
                    copy.set(c, x, y, 0.0);
                }
            }
        }
        copy.paste(&piece, 2, 1);
        assert_eq!(copy, rgb);
    }

    #[test]
    fn to_rgba_clamps() {
        let mut img = Image32::new(2, 1);
        img.set(0, 0, 0, 1.7);
        img.set(1, 0, 0, -0.2);
        img.set(2, 1, 0, 0.5);
        let out = img.to_rgba(None);
        assert_eq!(out.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(out.get_pixel(1, 0).0, [0, 0, 128, 255]);
    }
}
