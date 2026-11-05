//! RealPLKSR (spandrel `PLKSRArch`, the `channel_mixer` branch).
//!
//! Rule: detect all of {`feats.0.weight`, `feats.1.refine.weight`,
//! `feats.1.channel_mixer.0.weight`} and one of {`feats.1.lk.conv.weight`,
//! `feats.1.lk.mn_conv.weight`, `feats.1.lk.convs.0.weight`}: `dim = feats.0.weight.shape[0]`,
//! `n_blocks = seq_len("feats") - 3`, `scale = sqrt(feats.{last}.weight.shape[0] / 3)`,
//! `kernel = feats.1.lk.conv.weight.shape[2]` (17 if sparse variant),
//! `split_ratio = lk.conv.weight.shape[0] / dim`, `use_ea = has("feats.1.attn.f.0.weight")`,
//! `layer_norm = has("feats.1.layer_norm.bias")`, `dysample = has("to_img.init_pos")`. PLKSR
//! (typo key `channe_mixer`) → `Err(Model("PLKSR (non-Real) not supported"))`.
//!
//! spandrel also accepts `channe_mixer` in place of `channel_mixer` when detecting, which is how
//! plain PLKSR is recognised (and rejected here). RealPLKSR has no sparse variant: spandrel reads
//! `feats.1.lk.conv.weight` unconditionally, so its absence is an error, not a kernel of 17.

use super::{ArchSpec, count, dim, missing_keys, report};
use crate::statedict::StateDict;
use scaleup_core::{Result, ScaleUpError};

const KEYS: [&str; 2] = ["feats.0.weight", "feats.1.refine.weight"];
const LK: [&str; 3] = [
    "feats.1.lk.conv.weight",
    "feats.1.lk.convs.0.weight",
    "feats.1.lk.mn_conv.weight",
];
const MIXER: [&str; 2] = [
    "feats.1.channel_mixer.0.weight",
    "feats.1.channe_mixer.0.weight",
];

pub(super) fn missing(sd: &StateDict) -> Option<String> {
    let mut missing: Vec<String> = missing_keys(sd, &KEYS)
        .into_iter()
        .map(str::to_string)
        .collect();
    for group in [&LK[..], &MIXER[..]] {
        if !group.iter().any(|k| sd.has(k)) {
            missing.push(format!("one of {}", group.join(" / ")));
        }
    }
    report(&missing.iter().map(String::as_str).collect::<Vec<_>>())
}

pub(super) fn load(sd: &StateDict) -> Result<ArchSpec> {
    if sd.has(MIXER[1]) {
        return Err(ScaleUpError::Model(format!(
            "PLKSR (non-Real) not supported ({} is the PLKSR layout)",
            MIXER[1]
        )));
    }
    let dim_ = dim(sd, "feats.0.weight", 4, 0)?;
    if dim(sd, "feats.0.weight", 4, 1)? != 3 {
        return Err(ScaleUpError::Model(
            "RealPLKSR: feats.0.weight must take 3 input channels".to_string(),
        ));
    }
    // feats: conv, n_blocks PLK blocks, a parameterless dropout, the last conv.
    let seq = sd.seq_len("feats");
    if seq < 4 {
        return Err(ScaleUpError::Model(format!(
            "RealPLKSR: feats has {seq} layers, expected at least 4"
        )));
    }
    let last = format!("feats.{}.weight", seq - 1);
    let out = dim(sd, &last, 4, 0)?;
    let scale = (out / 3).isqrt();
    if scale == 0 || 3 * scale * scale != out {
        return Err(ScaleUpError::Model(format!(
            "RealPLKSR: {last} has {out} output channels, expected 3 · scale²"
        )));
    }
    let lk = LK[0];
    if !sd.has(lk) {
        return Err(ScaleUpError::Model(format!(
            "RealPLKSR: no {lk} (sparse large-kernel variants are not supported)"
        )));
    }
    let kernel = dim(sd, lk, 4, 2)?;
    let pdim = dim(sd, lk, 4, 0)?;
    Ok(ArchSpec::RealPlksr {
        dim: dim_,
        n_blocks: count(seq - 3, "RealPLKSR: feats")?,
        scale,
        kernel,
        split_ratio: pdim as f32 / dim_ as f32,
        use_ea: sd.has("feats.1.attn.f.0.weight"),
        layer_norm: sd.has("feats.1.layer_norm.bias"),
        dysample: sd.has("to_img.init_pos"),
    })
}

#[cfg(test)]
mod tests {
    use super::super::detect;
    use super::super::tests::{dict, e};
    use super::*;

    /// RealPLKSR with `pdim` large-kernel channels; `typo` writes PLKSR's `channe_mixer`.
    fn plksr(
        dim: usize,
        n_blocks: usize,
        pdim: usize,
        k: usize,
        scale: usize,
        typo: bool,
    ) -> Vec<(String, Vec<usize>)> {
        let mixer = if typo {
            "channe_mixer"
        } else {
            "channel_mixer"
        };
        let mut entries = vec![e("feats.0.weight", &[dim, 3, 3, 3])];
        for b in 1..=n_blocks {
            entries.push(e(
                &format!("feats.{b}.{mixer}.0.weight"),
                &[2 * dim, dim, 3, 3],
            ));
            entries.push(e(
                &format!("feats.{b}.{mixer}.2.weight"),
                &[dim, 2 * dim, 3, 3],
            ));
            entries.push(e(&format!("feats.{b}.lk.conv.weight"), &[pdim, pdim, k, k]));
            entries.push(e(&format!("feats.{b}.attn.f.0.weight"), &[dim, dim, 3, 3]));
            entries.push(e(&format!("feats.{b}.refine.weight"), &[dim, dim, 1, 1]));
            entries.push(e(&format!("feats.{b}.norm.weight"), &[dim]));
        }
        entries.push(e(
            &format!("feats.{}.weight", n_blocks + 2),
            &[3 * scale * scale, dim, 3, 3],
        ));
        entries
    }

    #[test]
    fn realplksr_defaults() {
        let spec = detect(&dict(&plksr(64, 28, 16, 17, 4, false))).unwrap();
        assert_eq!(
            spec,
            ArchSpec::RealPlksr {
                dim: 64,
                n_blocks: 28,
                scale: 4,
                kernel: 17,
                split_ratio: 0.25,
                use_ea: true,
                layer_norm: false,
                dysample: false
            }
        );
        assert_eq!((spec.name(), spec.external_scale()), ("RealPLKSR", 4));
    }

    #[test]
    fn realplksr_flags() {
        let mut entries: Vec<_> = plksr(32, 4, 8, 13, 2, false)
            .into_iter()
            .filter(|(k, _)| !k.contains(".attn."))
            .collect();
        entries.push(e("feats.1.layer_norm.bias", &[32]));
        entries.push(e("to_img.init_pos", &[1, 8, 1, 1]));
        let spec = detect(&dict(&entries)).unwrap();
        assert_eq!(
            spec,
            ArchSpec::RealPlksr {
                dim: 32,
                n_blocks: 4,
                scale: 2,
                kernel: 13,
                split_ratio: 0.25,
                use_ea: false,
                layer_norm: true,
                dysample: true
            }
        );
    }

    #[test]
    fn realplksr_rejects_sparse_kernel() {
        let entries: Vec<_> = plksr(32, 2, 8, 17, 2, false)
            .into_iter()
            .map(|(k, s)| (k.replace("lk.conv.", "lk.convs.0."), s))
            .collect();
        let err = detect(&dict(&entries)).unwrap_err().to_string();
        assert!(err.contains("feats.1.lk.conv.weight"), "{err}");
    }

    #[test]
    fn realplksr_rejects_bad_last_layer() {
        let mut entries = plksr(32, 2, 8, 17, 2, false);
        entries.last_mut().unwrap().1[0] = 13;
        let err = detect(&dict(&entries)).unwrap_err().to_string();
        assert!(
            err.contains("feats.4.weight") && err.contains("13"),
            "{err}"
        );
    }

    #[test]
    fn plksr_typo_rejected() {
        let err = detect(&dict(&plksr(64, 28, 16, 17, 4, true)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("PLKSR (non-Real) not supported"), "{err}");
    }
}
