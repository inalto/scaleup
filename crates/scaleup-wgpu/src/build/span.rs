//! SPAN, from spandrel's `span.py` (`SPAN.forward`, `SPAB`, `Conv3XC`):
//! `[(x - rgb_mean) · 255] → conv_1 → block_1..6 → conv_2`, then
//! `conv_cat(cat[conv_1 out, conv_2 out, block_1 out, block_6's out1]) → upsampler.0 →
//! pixel shuffle`. The output is not de-normalised. `SPAB` returns `out1`, its first
//! convolution's output, after the in-place SiLU (`nn.SiLU(inplace=True)`) has overwritten it.
//!
//! Every `Conv3XC` runs as one 3×3 convolution. In eval mode PyTorch rebuilds it on each forward
//! pass from the training branches (`sk`, `conv.0-2`) and overwrites `eval_conv`, so the stored
//! `eval_conv` tensors are stale in real checkpoints: they are checked for shape, never used.

use scaleup_core::{Result, ScaleUpError};

use super::{Blob, Builder};
use crate::arch::ArchSpec;
use crate::kernels::Smooth;
use crate::ncnn::{Activation, Graph, LayerKind};
use crate::statedict::{StateDict, TensorF32};

const RGB_MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];
const IMG_RANGE: f32 = 255.0;

pub(super) fn build(spec: &ArchSpec, sd: &StateDict) -> Result<Graph> {
    let ArchSpec::Span { scale, norm, .. } = *spec else {
        unreachable!("build dispatches on the spec")
    };
    let (mut b, input) = Builder::new(sd, 3);
    let x = if norm {
        let kind = LayerKind::Normalize {
            mean: RGB_MEAN.to_vec(),
            range: IMG_RANGE,
        };
        b.push("normalize", kind, &[input], 3)
    } else {
        // A buffer whose presence is the flag.
        b.tensor("no_norm")?;
        input
    };
    let feature = conv3xc(&mut b, x, "conv_1")?;
    let mut t = feature;
    let (mut out_b1, mut out_b5_2) = (feature, feature);
    for i in 1..=6 {
        let (out, out1) = spab(&mut b, t, &format!("block_{i}"))?;
        if i == 1 {
            out_b1 = out;
        }
        if i == 6 {
            out_b5_2 = out1;
        }
        t = out;
    }
    let out_b6 = conv3xc(&mut b, t, "conv_2")?;
    let cat = b.concat("cat", &[feature, out_b6, out_b1, out_b5_2]);
    let t = b.conv(cat, "conv_cat", true, Activation::None)?;
    let t = b.conv(t, "upsampler.0", true, Activation::None)?;
    let t = b.pixel_shuffle("upsampler.1", t, scale)?;
    b.finish(t)
}

/// `SPAB.forward`: returns `out` and `out1`, which the in-place SiLU has already activated.
fn spab(b: &mut Builder<'_>, x: Blob, p: &str) -> Result<(Blob, Blob)> {
    let t = conv3xc(b, x, &format!("{p}.c1_r"))?;
    let out1 = b.smooth(&format!("{p}.act1"), t, Smooth::Silu);
    let t = conv3xc(b, out1, &format!("{p}.c2_r"))?;
    let t = b.smooth(&format!("{p}.act1_2"), t, Smooth::Silu);
    let out3 = conv3xc(b, t, &format!("{p}.c3_r"))?;
    let sig = b.smooth(&format!("{p}.sigmoid"), out3, Smooth::Sigmoid);
    let sim = b.add_scalar(&format!("{p}.sim_att"), sig, -0.5);
    let sum = b.sum(&format!("{p}.residual"), out3, x, [1.0, 1.0])?;
    let out = b.mul(p, sum, sim)?;
    Ok((out, out1))
}

/// `Conv3XC.update_params`: `sk` (1×1) plus `conv.2 ∘ conv.1 ∘ conv.0` (1×1, 3×3, 1×1) folded
/// into one 3×3 convolution with zero padding, as `eval_conv` computes it.
fn conv3xc(b: &mut Builder<'_>, src: Blob, p: &str) -> Result<Blob> {
    let c_in = b.channels[src] as usize;
    let key = |k: &str| format!("{p}.{k}");
    let sk = shaped(b, &key("sk.weight"), &[None, Some(c_in), Some(1), Some(1)])?;
    let out = sk.shape[0];
    let sk_b = shaped(b, &key("sk.bias"), &[Some(out)])?;
    let w1 = shaped(
        b,
        &key("conv.0.weight"),
        &[None, Some(c_in), Some(1), Some(1)],
    )?;
    let m1 = w1.shape[0];
    let b1 = shaped(b, &key("conv.0.bias"), &[Some(m1)])?;
    let w2 = shaped(
        b,
        &key("conv.1.weight"),
        &[None, Some(m1), Some(3), Some(3)],
    )?;
    let m2 = w2.shape[0];
    let b2 = shaped(b, &key("conv.1.bias"), &[Some(m2)])?;
    let w3 = shaped(
        b,
        &key("conv.2.weight"),
        &[Some(out), Some(m2), Some(1), Some(1)],
    )?;
    let b3 = shaped(b, &key("conv.2.bias"), &[Some(out)])?;
    shaped(
        b,
        &key("eval_conv.weight"),
        &[Some(out), Some(c_in), Some(3), Some(3)],
    )?;
    shaped(b, &key("eval_conv.bias"), &[Some(out)])?;

    // w12[n][i][t] = Σm w2[n][m][t] · w1[m][i]; b12[n] = b2[n] + Σm,t w2[n][m][t] · b1[m].
    let mut w12 = vec![0.0_f64; m2 * c_in * 9];
    let mut b12: Vec<f64> = b2.data.iter().map(|&v| f64::from(v)).collect();
    for (n, b12) in b12.iter_mut().enumerate() {
        for m in 0..m1 {
            for t in 0..9 {
                let v = f64::from(w2.data[(n * m1 + m) * 9 + t]);
                *b12 += v * f64::from(b1.data[m]);
                for i in 0..c_in {
                    w12[(n * c_in + i) * 9 + t] += v * f64::from(w1.data[m * c_in + i]);
                }
            }
        }
    }
    let mut weights = vec![0.0_f32; out * c_in * 9];
    let mut biases = vec![0.0_f32; out];
    for o in 0..out {
        let bias = f64::from(b3.data[o])
            + f64::from(sk_b.data[o])
            + (0..m2)
                .map(|n| f64::from(w3.data[o * m2 + n]) * b12[n])
                .sum::<f64>();
        biases[o] = bias as f32;
        for i in 0..c_in {
            for t in 0..9 {
                let mut sum = (0..m2)
                    .map(|n| f64::from(w3.data[o * m2 + n]) * w12[(n * c_in + i) * 9 + t])
                    .sum::<f64>();
                if t == 4 {
                    sum += f64::from(sk.data[o * c_in + i]);
                }
                weights[(o * c_in + i) * 9 + t] = sum as f32;
            }
        }
    }
    b.conv_with(src, p, 3, weights, biases, Activation::None)
}

/// Tensor `key`, whose shape must be `want` (`None` for any non-zero size).
fn shaped<'w>(b: &Builder<'w>, key: &str, want: &[Option<usize>]) -> Result<&'w TensorF32> {
    let t = b.tensor(key)?;
    let ok = t.shape.len() == want.len()
        && t.shape
            .iter()
            .zip(want)
            .all(|(&d, w)| d > 0 && w.is_none_or(|w| d == w));
    if !ok {
        let want: Vec<String> = want
            .iter()
            .map(|w| w.map_or("_".to_string(), |w| w.to_string()))
            .collect();
        return Err(ScaleUpError::Model(format!(
            "{key} has shape {:?}, expected [{}]",
            t.shape,
            want.join(", ")
        )));
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::super::build;
    use super::super::tests::{assert_close, fixture, io, random_dict, run_gpu};
    use super::*;
    use crate::arch::detect;
    use crate::cpu_ref::run_graph;
    use crate::kernels::test_gpu::Rng;
    use crate::ncnn::infer_scale;

    /// `tools/make_sd_fixtures.py` builds spandrel's SPAN (4 features, 4×, normalised input) and
    /// records its eval-mode output; the checkpoint's `eval_conv` tensors are stale.
    #[test]
    fn span_fixture_matches_cpu_ref() {
        let sd = StateDict::load(&fixture("tiny_span.pth")).unwrap();
        let spec = detect(&sd).unwrap();
        assert_eq!(
            spec,
            ArchSpec::Span {
                in_ch: 3,
                out_ch: 3,
                feat: 4,
                scale: 4,
                norm: true
            }
        );
        let graph = build(&spec, &sd).unwrap();
        assert_eq!(infer_scale(&graph).unwrap(), 4);
        let (input, want) = io("tiny_span_io.safetensors");
        let got = run_graph(&graph, &input);
        assert_eq!((got.c, got.h, got.w), (want.c, want.h, want.w));
        assert_close(&got.data, &want.data, 1e-4);
        if let Some(gpu) = run_gpu(&graph, &input) {
            assert_close(&gpu, &got.data, 1e-4);
        }
    }

    /// The concatenation order is `[conv_1 out, conv_2 out, block_1 out, block_6's out1]`, the
    /// last one after its in-place SiLU.
    #[test]
    fn span_concat_order() {
        let sd = StateDict::load(&fixture("tiny_span.pth")).unwrap();
        let graph = build(&detect(&sd).unwrap(), &sd).unwrap();
        let producer = |blob: usize| {
            let l = graph.layers.iter().find(|l| l.outputs[0] == blob).unwrap();
            l.name.as_str()
        };
        let cat = graph.layers.iter().find(|l| l.name == "cat").unwrap();
        let names: Vec<&str> = cat.inputs.iter().map(|&b| producer(b)).collect();
        assert_eq!(names, ["conv_1", "conv_2", "block_1", "block_6.act1"]);
    }

    fn conv3xc_entries(p: &str, c_in: usize, out: usize) -> Vec<(String, Vec<usize>)> {
        let e = |k: &str, s: Vec<usize>| (format!("{p}.{k}"), s);
        vec![
            e("sk.weight", vec![out, c_in, 1, 1]),
            e("sk.bias", vec![out]),
            e("conv.0.weight", vec![2 * c_in, c_in, 1, 1]),
            e("conv.0.bias", vec![2 * c_in]),
            e("conv.1.weight", vec![2 * out, 2 * c_in, 3, 3]),
            e("conv.1.bias", vec![2 * out]),
            e("conv.2.weight", vec![out, 2 * out, 1, 1]),
            e("conv.2.bias", vec![out]),
            e("eval_conv.weight", vec![out, c_in, 3, 3]),
            e("eval_conv.bias", vec![out]),
        ]
    }

    fn span_entries(feat: usize, scale: usize) -> Vec<(String, Vec<usize>)> {
        let mut e = conv3xc_entries("conv_1", 3, feat);
        for i in 1..=6 {
            for c in ["c1_r", "c2_r", "c3_r"] {
                e.extend(conv3xc_entries(&format!("block_{i}.{c}"), feat, feat));
            }
        }
        e.extend(conv3xc_entries("conv_2", feat, feat));
        e.push(("conv_cat.weight".into(), vec![feat, 4 * feat, 1, 1]));
        e.push(("conv_cat.bias".into(), vec![feat]));
        e.push((
            "upsampler.0.weight".into(),
            vec![3 * scale * scale, feat, 3, 3],
        ));
        e.push(("upsampler.0.bias".into(), vec![3 * scale * scale]));
        e
    }

    #[test]
    fn span_without_norm_skips_normalisation_and_uses_its_flag() {
        let mut sd = random_dict(&span_entries(4, 2), &mut Rng(3), 0.3, false);
        sd.tensors.insert(
            "no_norm".into(),
            TensorF32 {
                shape: vec![1],
                data: vec![0.0],
            },
        );
        let spec = detect(&sd).unwrap();
        assert!(matches!(
            spec,
            ArchSpec::Span {
                norm: false,
                scale: 2,
                ..
            }
        ));
        let graph = build(&spec, &sd).unwrap();
        assert!(
            !graph
                .layers
                .iter()
                .any(|l| matches!(l.kind, LayerKind::Normalize { .. }))
        );
        // The same weights with normalisation: one layer more, and `no_norm` is then unexpected.
        let mut normed = sd.clone();
        normed.tensors.remove("no_norm");
        let with_norm = build(&detect(&normed).unwrap(), &normed).unwrap();
        assert_eq!(with_norm.layers.len(), graph.layers.len() + 1);
        let err = build(
            &ArchSpec::Span {
                in_ch: 3,
                out_ch: 3,
                feat: 4,
                scale: 2,
                norm: true,
            },
            &sd,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no_norm"), "{err}");
    }

    #[test]
    fn span_bad_branch_shape_names_the_key() {
        let mut sd = random_dict(&span_entries(4, 2), &mut Rng(4), 0.3, false);
        let spec = detect(&sd).unwrap();
        sd.tensors
            .get_mut("block_3.c2_r.conv.1.weight")
            .unwrap()
            .shape = vec![8, 8, 1, 9];
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(err.contains("block_3.c2_r.conv.1.weight"), "{err}");
        let mut sd = random_dict(&span_entries(4, 2), &mut Rng(4), 0.3, false);
        sd.tensors.remove("conv_2.eval_conv.bias");
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(err.contains("conv_2.eval_conv.bias"), "{err}");
    }
}
