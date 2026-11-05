//! SRVGGNetCompact (spandrel `CompactArch`).
//!
//! Rule: detect all of {`body.0.weight`, `body.1.weight`}: `highest = seq_len("body")-1`,
//! `in_nc = body.0.weight.shape[1]`, `num_feat = body.0.weight.shape[0]`,
//! `num_conv = (highest-2)/2`, `ps = body.{highest}.bias.len()`, `(scale, out_nc)` = first
//! `out_nc ∈ [in_nc, 3, 4, 1]` with `ps % out_nc == 0` and `ps/out_nc` a perfect square →
//! `scale = sqrt(ps/out_nc)`.

use super::{ArchSpec, count, dim, missing_keys, report, scale_and_out};
use crate::statedict::StateDict;
use scaleup_core::{Result, ScaleUpError};

const KEYS: [&str; 2] = ["body.0.weight", "body.1.weight"];

pub(super) fn missing(sd: &StateDict) -> Option<String> {
    report(&missing_keys(sd, &KEYS))
}

pub(super) fn load(sd: &StateDict) -> Result<ArchSpec> {
    let highest = sd.seq_len("body") - 1;
    // body.0 conv, body.1 PReLU, num_conv × (conv, PReLU), then the last conv.
    if highest < 2 || !highest.is_multiple_of(2) {
        return Err(ScaleUpError::Model(format!(
            "Compact: body.{highest} is the last layer, expected a convolution at an even index ≥ 2"
        )));
    }
    let in_nc = dim(sd, "body.0.weight", 4, 1)?;
    let num_feat = dim(sd, "body.0.weight", 4, 0)?;
    let last = format!("body.{highest}.bias");
    let (scale, out_nc) = scale_and_out(&last, dim(sd, &last, 1, 0)?, in_nc)?;
    Ok(ArchSpec::Compact {
        in_nc,
        out_nc,
        num_feat,
        num_conv: count((highest - 2) / 2, "Compact: body")?,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::super::detect;
    use super::super::tests::{dict, e};
    use super::*;

    /// Compact with `num_conv` hidden convolutions and a last convolution of `ps` channels.
    fn compact(in_nc: usize, num_feat: usize, num_conv: usize, ps: usize) -> StateDict {
        let mut entries = vec![
            e("body.0.weight", &[num_feat, in_nc, 3, 3]),
            e("body.0.bias", &[num_feat]),
            e("body.1.weight", &[num_feat]),
        ];
        for i in 0..num_conv {
            let c = 2 + 2 * i;
            entries.push(e(&format!("body.{c}.weight"), &[num_feat, num_feat, 3, 3]));
            entries.push(e(&format!("body.{c}.bias"), &[num_feat]));
            entries.push(e(&format!("body.{}.weight", c + 1), &[num_feat]));
        }
        let last = 2 + 2 * num_conv;
        entries.push(e(&format!("body.{last}.weight"), &[ps, num_feat, 3, 3]));
        entries.push(e(&format!("body.{last}.bias"), &[ps]));
        dict(&entries)
    }

    #[test]
    fn compact_16_feat_2x() {
        let spec = detect(&compact(3, 16, 8, 12)).unwrap();
        assert_eq!(
            spec,
            ArchSpec::Compact {
                in_nc: 3,
                out_nc: 3,
                num_feat: 16,
                num_conv: 8,
                scale: 2
            }
        );
        assert_eq!(
            (spec.name(), spec.external_scale(), spec.input_multiple_of()),
            ("Compact", 2, 1)
        );
    }

    #[test]
    fn compact_out_nc_resolution_when_ps_48() {
        let spec = detect(&compact(3, 64, 16, 48)).unwrap();
        assert_eq!(
            spec,
            ArchSpec::Compact {
                in_nc: 3,
                out_nc: 3,
                num_feat: 64,
                num_conv: 16,
                scale: 4
            }
        );
        // One input channel: 48 = 4² · 3, since 48 / 1 is not a square.
        let gray = detect(&compact(1, 8, 2, 48)).unwrap();
        assert!(
            matches!(
                gray,
                ArchSpec::Compact {
                    in_nc: 1,
                    out_nc: 3,
                    scale: 4,
                    ..
                }
            ),
            "{gray:?}"
        );
    }

    #[test]
    fn compact_rejects_unresolvable_pixelshuffle() {
        let err = detect(&compact(3, 8, 2, 10)).unwrap_err().to_string();
        assert!(err.contains("body.6.bias") && err.contains("10"), "{err}");
    }

    #[test]
    fn compact_rejects_odd_layer_count() {
        let mut sd = compact(3, 8, 2, 12);
        sd.tensors.remove("body.6.weight");
        sd.tensors.remove("body.6.bias");
        let err = detect(&sd).unwrap_err().to_string();
        assert!(err.contains("body.5"), "{err}");
    }
}
