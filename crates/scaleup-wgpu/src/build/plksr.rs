//! RealPLKSR, from spandrel's `RealPLKSR.py` (`RealPLKSR.forward`, `PLKBlock`, `DCCM`,
//! `PLKConv2d`, `EA`): `feats.0 → n_blocks × PLKBlock → (dropout, identity in eval) →
//! feats.{n_blocks+2} → pixel shuffle`, plus the input repeated `scale²` times per channel before
//! the shuffle, which is the input upscaled by nearest neighbour after it.
//!
//! `PLKBlock`: `[LayerNorm over channels] → channel_mixer (3×3, Mish, 3×3) → lk (k×k on the first
//! pdim channels, the others pass through) → [attn: x · sigmoid(3×3(x))] → refine (1×1) →
//! [GroupNorm(4) when there is no layer norm] → + block input`. GroupNorm normalises over the whole
//! input, so a tiled run gives each tile its own statistics, as the ONNX export of such a model does.

use scaleup_core::{Result, ScaleUpError};

use super::{Blob, Builder};
use crate::arch::ArchSpec;
use crate::kernels::Smooth;
use crate::ncnn::{Activation, Graph};
use crate::statedict::StateDict;

/// `PLKBlock`'s GroupNorm group count, which the weights do not record (spandrel hard-codes it).
const NORM_GROUPS: u32 = 4;
const LAYER_NORM_EPS: f32 = 1e-6;
/// `nn.GroupNorm`'s default.
const GROUP_NORM_EPS: f32 = 1e-5;

pub(super) fn build(spec: &ArchSpec, sd: &StateDict) -> Result<Graph> {
    let ArchSpec::RealPlksr {
        n_blocks,
        scale,
        use_ea,
        layer_norm,
        dysample,
        ..
    } = *spec
    else {
        unreachable!("build dispatches on the spec")
    };
    if dysample {
        return Err(ScaleUpError::Model(
            "DySample upsampler not supported; use the ONNX path".to_string(),
        ));
    }
    let (mut b, input) = Builder::new(sd, 3);
    let mut t = b.conv(input, "feats.0", true, Activation::None)?;
    for i in 1..=n_blocks {
        t = block(&mut b, t, &format!("feats.{i}"), use_ea, layer_norm)?;
    }
    t = b.conv(
        t,
        &format!("feats.{}", n_blocks + 2),
        true,
        Activation::None,
    )?;
    t = b.pixel_shuffle("to_img", t, scale)?;
    let base = b.interp("repeat_op", input, scale);
    t = b.sum("out", t, base, [1.0, 1.0])?;
    b.finish(t)
}

/// `PLKBlock.forward`.
fn block(b: &mut Builder<'_>, x: Blob, p: &str, use_ea: bool, layer_norm: bool) -> Result<Blob> {
    let mut t = x;
    if layer_norm {
        t = b.layer_norm(t, &format!("{p}.layer_norm"), LAYER_NORM_EPS)?;
    }
    t = b.conv(t, &format!("{p}.channel_mixer.0"), true, Activation::None)?;
    t = b.smooth(&format!("{p}.channel_mixer.1"), t, Smooth::Mish);
    t = b.conv(t, &format!("{p}.channel_mixer.2"), true, Activation::None)?;
    t = partial_conv(b, t, &format!("{p}.lk"))?;
    if use_ea {
        let a = b.conv(t, &format!("{p}.attn.f.0"), true, Activation::None)?;
        let a = b.smooth(&format!("{p}.attn.f.1"), a, Smooth::Sigmoid);
        t = b.mul(&format!("{p}.attn"), t, a)?;
    }
    t = b.conv(t, &format!("{p}.refine"), true, Activation::None)?;
    if !layer_norm {
        t = b.group_norm(t, &format!("{p}.norm"), NORM_GROUPS, GROUP_NORM_EPS)?;
    }
    b.sum(p, t, x, [1.0, 1.0])
}

/// `PLKConv2d.forward`: `{p}.conv` on the first `pdim` channels, where `pdim` is its channel count;
/// the other channels are kept.
fn partial_conv(b: &mut Builder<'_>, x: Blob, p: &str) -> Result<Blob> {
    let key = format!("{p}.conv.weight");
    let dim = b.channels[x];
    let pdim = match b.tensor(&key)?.shape[..] {
        [o, ..] if o > 0 && o <= dim as usize => o as u32,
        ref shape => {
            return Err(ScaleUpError::Model(format!(
                "{key} has shape {shape:?}, expected at most {dim} channels"
            )));
        }
    };
    let head = b.slice(&format!("{p}.head"), x, 0, pdim);
    let head = b.conv(head, &format!("{p}.conv"), true, Activation::None)?;
    if pdim == dim {
        return Ok(head);
    }
    let tail = b.slice(&format!("{p}.tail"), x, pdim, dim);
    Ok(b.concat(p, &[head, tail]))
}

#[cfg(test)]
mod tests {
    use super::super::build;
    use super::super::tests::{assert_close, assert_close_rel, fixture, io, random_dict, run_gpu};
    use super::*;
    use crate::arch::detect;
    use crate::cpu_ref::run_graph;
    use crate::kernels::test_gpu::Rng;
    use crate::ncnn::{LayerKind, infer_scale};

    /// Each fixture of `tools/make_sd_fixtures.py` (dim 8, 2 blocks, 2×, pdim 2) against the output
    /// spandrel's module computed, then the GPU against `cpu_ref`.
    #[test]
    fn realplksr_fixture_matches_cpu_ref() {
        for (name, kernel, use_ea, layer_norm) in [
            ("tiny_realplksr_gn_ea", 7, true, false),
            ("tiny_realplksr_ln", 5, false, true),
            ("tiny_realplksr_ln_ea", 17, true, true),
            ("tiny_realplksr_gn", 9, false, false),
        ] {
            let sd = StateDict::load(&fixture(&format!("{name}.pth"))).unwrap();
            let spec = detect(&sd).unwrap();
            assert_eq!(
                spec,
                ArchSpec::RealPlksr {
                    dim: 8,
                    n_blocks: 2,
                    scale: 2,
                    kernel,
                    split_ratio: 0.25,
                    use_ea,
                    layer_norm,
                    dysample: false
                },
                "{name}"
            );
            let graph = build(&spec, &sd).unwrap();
            assert_eq!(infer_scale(&graph).unwrap(), 2);
            let (input, want) = io(&format!("{name}_io.safetensors"));
            let got = run_graph(&graph, &input);
            assert_eq!((got.c, got.h, got.w), (want.c, want.h, want.w), "{name}");
            assert_close_rel(&got.data, &want.data, 1e-5);
            if let Some(gpu) = run_gpu(&graph, &input) {
                assert_close(&gpu, &got.data, 1e-4);
            }
        }
    }

    fn entries(dim: usize, pdim: usize, k: usize, ea: bool, ln: bool) -> Vec<(String, Vec<usize>)> {
        let mut e = vec![
            ("feats.0.weight".to_string(), vec![dim, 3, 3, 3]),
            ("feats.0.bias".to_string(), vec![dim]),
        ];
        for i in 1..=2 {
            let p = format!("feats.{i}");
            let mut convs = vec![
                ("channel_mixer.0", vec![2 * dim, dim, 3, 3]),
                ("channel_mixer.2", vec![dim, 2 * dim, 3, 3]),
                ("lk.conv", vec![pdim, pdim, k, k]),
                ("refine", vec![dim, dim, 1, 1]),
            ];
            if ea {
                convs.push(("attn.f.0", vec![dim, dim, 3, 3]));
            }
            for (name, shape) in convs {
                e.push((format!("{p}.{name}.bias"), vec![shape[0]]));
                e.push((format!("{p}.{name}.weight"), shape));
            }
            let norm = if ln { "layer_norm" } else { "norm" };
            e.push((format!("{p}.{norm}.weight"), vec![dim]));
            e.push((format!("{p}.{norm}.bias"), vec![dim]));
        }
        e.push(("feats.4.weight".to_string(), vec![12, dim, 3, 3]));
        e.push(("feats.4.bias".to_string(), vec![12]));
        e
    }

    #[test]
    fn realplksr_dysample_is_rejected_by_name() {
        let mut sd = random_dict(&entries(8, 2, 5, true, true), &mut Rng(1), 0.2, false);
        sd.tensors.insert(
            "to_img.init_pos".into(),
            crate::statedict::TensorF32 {
                shape: vec![1, 8, 1, 1],
                data: vec![0.0; 8],
            },
        );
        let spec = detect(&sd).unwrap();
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(
            err.ends_with("DySample upsampler not supported; use the ONNX path"),
            "{err}"
        );
    }

    /// The large kernel may cover every channel: no slicing then.
    #[test]
    fn realplksr_full_width_large_kernel() {
        let sd = random_dict(&entries(8, 8, 3, false, true), &mut Rng(2), 0.2, false);
        let spec = detect(&sd).unwrap();
        assert!(matches!(
            spec,
            ArchSpec::RealPlksr {
                split_ratio: 1.0,
                ..
            }
        ));
        let graph = build(&spec, &sd).unwrap();
        let slices = graph
            .layers
            .iter()
            .filter(|l| matches!(l.kind, LayerKind::Slice { .. }))
            .count();
        assert_eq!(slices, 2, "one head slice per block, no tail");
        let out = run_graph(&graph, &Rng(3).tensor(3, 4, 5));
        assert_eq!((out.c, out.h, out.w), (3, 8, 10));
    }

    /// Blocks after the first must have the same layout: a stray attention or a missing norm is
    /// named.
    #[test]
    fn realplksr_inconsistent_blocks_name_the_key() {
        let mut sd = random_dict(&entries(8, 2, 5, false, false), &mut Rng(4), 0.2, false);
        let spec = detect(&sd).unwrap();
        sd.tensors.remove("feats.2.norm.bias");
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(err.contains("feats.2.norm.bias"), "{err}");

        let mut sd = random_dict(&entries(8, 2, 5, false, false), &mut Rng(4), 0.2, false);
        let extra = sd.tensors["feats.1.refine.weight"].clone();
        sd.tensors.insert("feats.2.attn.f.0.weight".into(), extra);
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(err.contains("feats.2.attn.f.0.weight"), "{err}");
    }
}
