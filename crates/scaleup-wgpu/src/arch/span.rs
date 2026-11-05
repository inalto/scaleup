//! SPAN (spandrel `SPANArch`).
//!
//! Rule: detect all of {`conv_1.sk.weight`, `block_1.c1_r.sk.weight`,
//! `block_1.c1_r.eval_conv.weight`, `block_1.c3_r.eval_conv.weight`, `conv_cat.weight`,
//! `conv_2.sk.weight`, `conv_2.eval_conv.weight`, `upsampler.0.weight`}:
//! `in_ch = conv_1.sk.weight.shape[1]`, `feat = conv_1.sk.weight.shape[0]`, `(scale,out_ch)` from
//! `upsampler.0.weight.shape[0]` as in Compact, `norm = !has("no_norm")`; constants
//! `img_range = 255`, `rgb_mean = (0.4488, 0.4371, 0.4040)`.

use super::{ArchSpec, dim, missing_keys, report, scale_and_out};
use crate::statedict::StateDict;
use scaleup_core::Result;

const KEYS: [&str; 8] = [
    "conv_1.sk.weight",
    "block_1.c1_r.sk.weight",
    "block_1.c1_r.eval_conv.weight",
    "block_1.c3_r.eval_conv.weight",
    "conv_cat.weight",
    "conv_2.sk.weight",
    "conv_2.eval_conv.weight",
    "upsampler.0.weight",
];

pub(super) fn missing(sd: &StateDict) -> Option<String> {
    report(&missing_keys(sd, &KEYS))
}

pub(super) fn load(sd: &StateDict) -> Result<ArchSpec> {
    let in_ch = dim(sd, "conv_1.sk.weight", 4, 1)?;
    let feat = dim(sd, "conv_1.sk.weight", 4, 0)?;
    let up = "upsampler.0.weight";
    let (scale, out_ch) = scale_and_out(up, dim(sd, up, 4, 0)?, in_ch)?;
    Ok(ArchSpec::Span {
        in_ch,
        out_ch,
        feat,
        scale,
        norm: !sd.has("no_norm"),
    })
}

#[cfg(test)]
mod tests {
    use super::super::detect;
    use super::super::tests::{dict, e};
    use super::*;

    fn span(feat: usize, up: usize) -> StateDict {
        let conv = [feat, feat, 3, 3];
        dict(&[
            e("conv_1.sk.weight", &[feat, 3, 1, 1]),
            e("conv_1.eval_conv.weight", &[feat, 3, 3, 3]),
            e("block_1.c1_r.sk.weight", &conv),
            e("block_1.c1_r.eval_conv.weight", &conv),
            e("block_1.c3_r.eval_conv.weight", &conv),
            e("conv_cat.weight", &[feat, 4 * feat, 1, 1]),
            e("conv_2.sk.weight", &[feat, feat, 1, 1]),
            e("conv_2.eval_conv.weight", &conv),
            e("upsampler.0.weight", &[up, feat, 3, 3]),
            e("upsampler.0.bias", &[up]),
        ])
    }

    #[test]
    fn span_48_feat_4x() {
        let spec = detect(&span(48, 48)).unwrap();
        assert_eq!(
            spec,
            ArchSpec::Span {
                in_ch: 3,
                out_ch: 3,
                feat: 48,
                scale: 4,
                norm: true
            }
        );
        assert_eq!((spec.name(), spec.external_scale()), ("SPAN", 4));

        let mut no_norm = span(48, 12);
        no_norm.tensors.insert(
            "no_norm".to_string(),
            crate::statedict::TensorF32 {
                shape: vec![1],
                data: vec![0.0_f32],
            },
        );
        assert!(matches!(
            detect(&no_norm).unwrap(),
            ArchSpec::Span {
                scale: 2,
                norm: false,
                ..
            }
        ));
    }
}
