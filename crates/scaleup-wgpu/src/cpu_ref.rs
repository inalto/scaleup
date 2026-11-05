//! Reference implementations of every operation, in plain Rust. Each GPU kernel is tested against
//! these; they favour being obviously correct over being fast.

use crate::ncnn::Activation;

/// Planar CHW tensor: `data[(c * h + y) * w + x]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(c: u32, h: u32, w: u32) -> Self {
        Self {
            c,
            h,
            w,
            data: vec![0.0; (c * h * w) as usize],
        }
    }

    fn index(&self, c: u32, y: u32, x: u32) -> usize {
        ((c * self.h + y) * self.w + x) as usize
    }

    pub fn get(&self, c: u32, y: u32, x: u32) -> f32 {
        self.data[self.index(c, y, x)]
    }

    pub fn set(&mut self, c: u32, y: u32, x: u32, v: f32) {
        let i = self.index(c, y, x);
        self.data[i] = v;
    }
}

pub fn activate(v: f32, act: Activation) -> f32 {
    match act {
        Activation::None => v,
        Activation::Relu => v.max(0.0),
        Activation::Leaky(slope) => {
            if v < 0.0 {
                v * slope
            } else {
                v
            }
        }
    }
}

/// `weights` are `[out][in][k][k]`; pixels outside the input read as zero. Empty `biases` means none.
pub fn conv(
    src: &Tensor,
    out: u32,
    k: u32,
    weights: &[f32],
    biases: &[f32],
    act: Activation,
) -> Tensor {
    let mut dst = Tensor::zeros(out, src.h, src.w);
    let half = (k / 2) as i64;
    for co in 0..out {
        for y in 0..src.h {
            for x in 0..src.w {
                let mut sum = biases.get(co as usize).copied().unwrap_or(0.0);
                for ci in 0..src.c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let sy = y as i64 + ky as i64 - half;
                            let sx = x as i64 + kx as i64 - half;
                            if sy < 0 || sx < 0 || sy >= src.h as i64 || sx >= src.w as i64 {
                                continue;
                            }
                            let wi = ((co * src.c + ci) * k * k + ky * k + kx) as usize;
                            sum += weights[wi] * src.get(ci, sy as u32, sx as u32);
                        }
                    }
                }
                dst.set(co, y, x, activate(sum, act));
            }
        }
    }
    dst
}

pub fn leaky(src: &Tensor, slope: f32) -> Tensor {
    let mut dst = src.clone();
    for v in &mut dst.data {
        *v = activate(*v, Activation::Leaky(slope));
    }
    dst
}

/// One slope per channel, or a single slope for all.
pub fn prelu(src: &Tensor, slopes: &[f32]) -> Tensor {
    let mut dst = src.clone();
    for c in 0..src.c {
        let slope = if slopes.len() == 1 {
            slopes[0]
        } else {
            slopes[c as usize]
        };
        for y in 0..src.h {
            for x in 0..src.w {
                dst.set(
                    c,
                    y,
                    x,
                    activate(src.get(c, y, x), Activation::Leaky(slope)),
                );
            }
        }
    }
    dst
}

/// Along the channel axis.
pub fn concat(inputs: &[&Tensor]) -> Tensor {
    let first = inputs[0];
    let c = inputs.iter().map(|t| t.c).sum();
    let mut data = Vec::with_capacity((c * first.h * first.w) as usize);
    for t in inputs {
        data.extend_from_slice(&t.data);
    }
    Tensor {
        c,
        h: first.h,
        w: first.w,
        data,
    }
}

/// Weighted sum; empty `coeffs` means all ones.
pub fn eltwise(inputs: &[&Tensor], coeffs: &[f32]) -> Tensor {
    let mut dst = Tensor::zeros(inputs[0].c, inputs[0].h, inputs[0].w);
    for (i, t) in inputs.iter().enumerate() {
        let coeff = coeffs.get(i).copied().unwrap_or(1.0);
        for (d, s) in dst.data.iter_mut().zip(&t.data) {
            *d += coeff * s;
        }
    }
    dst
}

pub fn binary_mul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut dst = a.clone();
    for (d, s) in dst.data.iter_mut().zip(&b.data) {
        *d *= s;
    }
    dst
}

pub fn scalar_add(a: &Tensor, s: f32) -> Tensor {
    let mut dst = a.clone();
    dst.data.iter_mut().for_each(|d| *d += s);
    dst
}

pub fn scalar_mul(a: &Tensor, s: f32) -> Tensor {
    let mut dst = a.clone();
    dst.data.iter_mut().for_each(|d| *d *= s);
    dst
}

pub fn interp_nearest(src: &Tensor, scale: u32) -> Tensor {
    let mut dst = Tensor::zeros(src.c, src.h * scale, src.w * scale);
    for c in 0..dst.c {
        for y in 0..dst.h {
            for x in 0..dst.w {
                dst.set(c, y, x, src.get(c, y / scale, x / scale));
            }
        }
    }
    dst
}

/// `out[c][y*r+dy][x*r+dx] = in[c*r*r + dy*r + dx][y][x]`, ncnn's layout.
pub fn pixel_shuffle(src: &Tensor, r: u32) -> Tensor {
    let mut dst = Tensor::zeros(src.c / (r * r), src.h * r, src.w * r);
    for c in 0..dst.c {
        for y in 0..src.h {
            for x in 0..src.w {
                for dy in 0..r {
                    for dx in 0..r {
                        let v = src.get(c * r * r + dy * r + dx, y, x);
                        dst.set(c, y * r + dy, x * r + dx, v);
                    }
                }
            }
        }
    }
    dst
}

/// `out[c*r*r + dy*r + dx][y][x] = in[c][y*r+dy][x*r+dx]`, torch's layout; sides must be multiples
/// of `r`.
pub fn pixel_unshuffle(src: &Tensor, r: u32) -> Tensor {
    assert!(src.h.is_multiple_of(r) && src.w.is_multiple_of(r));
    let mut dst = Tensor::zeros(src.c * r * r, src.h / r, src.w / r);
    for c in 0..src.c {
        for y in 0..dst.h {
            for x in 0..dst.w {
                for dy in 0..r {
                    for dx in 0..r {
                        let v = src.get(c, y * r + dy, x * r + dx);
                        dst.set(c * r * r + dy * r + dx, y, x, v);
                    }
                }
            }
        }
    }
    dst
}

fn map(src: &Tensor, f: impl Fn(f64) -> f64) -> Tensor {
    let mut dst = src.clone();
    for v in &mut dst.data {
        *v = f(f64::from(*v)) as f32;
    }
    dst
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid(src: &Tensor) -> Tensor {
    map(src, sigmoid64)
}

pub fn silu(src: &Tensor) -> Tensor {
    map(src, |x| x * sigmoid64(x))
}

/// `(x - mean[c]) · range`.
pub fn normalize(src: &Tensor, mean: &[f32], range: f32) -> Tensor {
    let mut dst = src.clone();
    let plane = (src.h * src.w) as usize;
    for (i, v) in dst.data.iter_mut().enumerate() {
        *v = (*v - mean[i / plane]) * range;
    }
    dst
}

/// `x · tanh(ln(1 + eˣ))`.
pub fn mish(src: &Tensor) -> Tensor {
    map(src, |x| x * x.exp().ln_1p().tanh())
}

/// Each pixel normalised over its channels (biased variance), then `gamma[c] · x + beta[c]`.
pub fn layer_norm_c(src: &Tensor, eps: f32, gamma: &[f32], beta: &[f32]) -> Tensor {
    let mut dst = src.clone();
    let plane = (src.h * src.w) as usize;
    let n = f64::from(src.c);
    for px in 0..plane {
        let at = |c: usize| f64::from(src.data[c * plane + px]);
        let mean = (0..src.c as usize).map(at).sum::<f64>() / n;
        let var = (0..src.c as usize)
            .map(|c| (at(c) - mean).powi(2))
            .sum::<f64>()
            / n;
        let r = 1.0 / (var + f64::from(eps)).sqrt();
        for c in 0..src.c as usize {
            let v = f64::from(gamma[c]) * ((at(c) - mean) * r) + f64::from(beta[c]);
            dst.data[c * plane + px] = v as f32;
        }
    }
    dst
}

/// `groups` runs of consecutive channels, each normalised over all its values (biased variance),
/// then `gamma[c] · x + beta[c]`.
pub fn group_norm(src: &Tensor, groups: u32, eps: f32, gamma: &[f32], beta: &[f32]) -> Tensor {
    let mut dst = src.clone();
    let plane = (src.h * src.w) as usize;
    let per_group = (src.c / groups) as usize;
    let len = per_group * plane;
    for (g, values) in dst.data.chunks_mut(len).enumerate() {
        let n = len as f64;
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let r = 1.0 / (var + f64::from(eps)).sqrt();
        for (i, v) in values.iter_mut().enumerate() {
            let c = g * per_group + i / plane;
            *v = (f64::from(gamma[c]) * ((f64::from(*v) - mean) * r) + f64::from(beta[c])) as f32;
        }
    }
    dst
}

/// Channels `from..to`.
pub fn slice(src: &Tensor, from: u32, to: u32) -> Tensor {
    let plane = (src.h * src.w) as usize;
    Tensor {
        c: to - from,
        h: src.h,
        w: src.w,
        data: src.data[from as usize * plane..to as usize * plane].to_vec(),
    }
}

/// Runs a whole graph on the CPU.
pub fn run_graph(graph: &crate::ncnn::Graph, input: &Tensor) -> Tensor {
    use crate::ncnn::{BinOp, LayerKind};
    let mut blobs: Vec<Option<Tensor>> = vec![None; graph.blob_count];
    for layer in &graph.layers {
        let inputs: Vec<&Tensor> = layer
            .inputs
            .iter()
            .map(|&b| blobs[b].as_ref().expect("topological order"))
            .collect();
        let first = inputs.first().copied();
        let output = match &layer.kind {
            LayerKind::Input => input.clone(),
            LayerKind::Split => first.unwrap().clone(),
            LayerKind::Conv {
                out,
                k,
                act,
                weights,
                biases,
                ..
            } => conv(first.unwrap(), *out, *k, weights, biases, *act),
            LayerKind::Concat => concat(&inputs),
            LayerKind::Eltwise { coeffs } => eltwise(&inputs, coeffs),
            LayerKind::BinaryOp { op, scalar } => match (op, scalar) {
                (BinOp::Add, Some(s)) => scalar_add(first.unwrap(), *s),
                (BinOp::Mul, Some(s)) => scalar_mul(first.unwrap(), *s),
                (BinOp::Add, None) => eltwise(&inputs, &[]),
                (BinOp::Mul, None) => binary_mul(inputs[0], inputs[1]),
            },
            LayerKind::LeakyRelu { slope } => leaky(first.unwrap(), *slope),
            LayerKind::PRelu { slopes } => prelu(first.unwrap(), slopes),
            LayerKind::InterpNearest { scale } => interp_nearest(first.unwrap(), *scale),
            LayerKind::PixelShuffle { factor } => pixel_shuffle(first.unwrap(), *factor),
            LayerKind::PixelUnshuffle { factor } => pixel_unshuffle(first.unwrap(), *factor),
            LayerKind::Sigmoid => sigmoid(first.unwrap()),
            LayerKind::Silu => silu(first.unwrap()),
            LayerKind::Normalize { mean, range } => normalize(first.unwrap(), mean, *range),
            LayerKind::Mish => mish(first.unwrap()),
            LayerKind::LayerNormC { eps, gamma, beta } => {
                layer_norm_c(first.unwrap(), *eps, gamma, beta)
            }
            LayerKind::GroupNorm {
                groups,
                eps,
                gamma,
                beta,
            } => group_norm(first.unwrap(), *groups, *eps, gamma, beta),
            LayerKind::Slice { from, to } => slice(first.unwrap(), *from, *to),
        };
        for &o in &layer.outputs {
            blobs[o] = Some(output.clone());
        }
    }
    blobs[graph.output_blob]
        .take()
        .expect("output was computed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_identity_kernel_copies_and_zero_pads() {
        let src = Tensor {
            c: 1,
            h: 1,
            w: 2,
            data: vec![1.0, 2.0],
        };
        // Reads the pixel to the left: out(x) = in(x - 1), zero at the border.
        let weights = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = conv(&src, 1, 3, &weights, &[], Activation::None);
        assert_eq!(out.data, [0.0, 1.0]);
    }

    #[test]
    fn pixel_shuffle_layout_matches_ncnn() {
        let src = Tensor {
            c: 4,
            h: 1,
            w: 1,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let out = pixel_shuffle(&src, 2);
        assert_eq!((out.c, out.h, out.w), (1, 2, 2));
        assert_eq!(out.data, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pixel_unshuffle_layout_matches_torch() {
        // torch.pixel_unshuffle(arange(4).reshape(1, 1, 2, 2), 2) is [0, 1, 2, 3] over 4 channels.
        let src = Tensor {
            c: 1,
            h: 2,
            w: 2,
            data: vec![0.0, 1.0, 2.0, 3.0],
        };
        let out = pixel_unshuffle(&src, 2);
        assert_eq!((out.c, out.h, out.w), (4, 1, 1));
        assert_eq!(out.data, [0.0, 1.0, 2.0, 3.0]);
        // And it undoes pixel_shuffle.
        let src = Tensor {
            c: 2,
            h: 4,
            w: 6,
            data: (0..48).map(|i| i as f32).collect(),
        };
        assert_eq!(pixel_shuffle(&pixel_unshuffle(&src, 2), 2), src);
    }

    #[test]
    fn smooth_activations_and_normalize() {
        let src = Tensor {
            c: 2,
            h: 1,
            w: 2,
            data: vec![0.0, 2.0, -100.0, 1.0],
        };
        let s = sigmoid(&src).data;
        assert_eq!(s[0], 0.5);
        assert!((s[1] - 0.880_797_1).abs() < 1e-6 && (0.0..1e-40).contains(&s[2]));
        let si = silu(&src).data;
        assert!((si[1] - 1.761_594_2).abs() < 1e-6 && si[2].abs() < 1e-30);
        let n = normalize(&src, &[0.5, 1.0], 10.0);
        assert_eq!(n.data, [-5.0, 15.0, -1010.0, 0.0]);
    }

    #[test]
    fn mish_norms_and_slice() {
        let src = Tensor {
            c: 2,
            h: 1,
            w: 2,
            data: vec![1.0, 3.0, 3.0, -1.0],
        };
        let m = mish(&Tensor {
            data: vec![0.0, 1.0, -200.0, 200.0],
            ..src.clone()
        })
        .data;
        assert_eq!((m[0], m[2], m[3]), (0.0, 0.0, 200.0));
        assert!((m[1] - 0.865_098_4).abs() < 1e-6, "{}", m[1]);
        // Pixel 0 holds (1, 3): mean 2, deviation 1. Pixel 1 holds (3, -1): mean 1, deviation 2.
        let l = layer_norm_c(&src, 0.0, &[1.0, 2.0], &[0.5, 0.0]);
        assert_eq!(l.data, [-0.5, 1.5, 2.0, -2.0]);
        // One group over everything: mean 1.5, variance 2.75.
        let g = group_norm(&src, 1, 0.0, &[1.0, 1.0], &[0.0, 1.0]).data;
        let r = 1.0 / 2.75_f32.sqrt();
        for (got, want) in g
            .iter()
            .zip([-0.5 * r, 1.5 * r, 1.5 * r + 1.0, -2.5 * r + 1.0])
        {
            assert!((got - want).abs() < 1e-6, "{g:?}");
        }
        // One group per channel: each plane is normalised alone.
        let g = group_norm(&src, 2, 0.0, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(g.data, [-1.0, 1.0, 1.0, -1.0]);
        let s = slice(&src, 1, 2);
        assert_eq!((s.c, s.h, s.w, s.data), (1, 1, 2, vec![3.0, -1.0]));
    }

    #[test]
    fn prelu_applies_only_to_negatives() {
        let src = Tensor {
            c: 2,
            h: 1,
            w: 1,
            data: vec![-1.0, 3.0],
        };
        assert_eq!(prelu(&src, &[0.5, 0.25]).data, [-0.5, 3.0]);
    }
}
