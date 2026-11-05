//! Which network a state dict holds, and its hyperparameters, read from key names and tensor shapes
//! with the rules of spandrel 0.4.2. Detectors run in spandrel's registry order (Compact, SPAN,
//! ESRGAN, PLKSR); the first whose keys are all present decides, and its errors are final.

mod compact;
mod esrgan;
mod plksr;
mod span;

pub use esrgan::{OldArch, esrgan_to_old_arch};

use crate::statedict::{StateDict, TensorF32};
use scaleup_core::{Result, ScaleUpError};

/// A detected network. Fields are the network's own hyperparameters, as the builder needs them.
#[derive(Debug, Clone, PartialEq)]
pub enum ArchSpec {
    /// RRDBNet (ESRGAN, BSRGAN, Real-ESRGAN, ESRGAN+). With pixel-unshuffle, `in_nc` and `scale`
    /// are those of the network behind the unshuffle (a 2× Real-ESRGAN has `in_nc` 12, `scale` 4,
    /// `shuffle_factor` 2); [`ArchSpec::external_scale`] gives the image scale. `gc` is the growth
    /// channel count of the dense blocks, 32 in every published model.
    Esrgan {
        in_nc: u32,
        out_nc: u32,
        nf: u32,
        nb: u32,
        gc: u32,
        scale: u32,
        plus: bool,
        shuffle_factor: Option<u32>,
    },
    /// SRVGGNetCompact (Real-ESRGAN Compact), PReLU activations.
    Compact {
        in_nc: u32,
        out_nc: u32,
        num_feat: u32,
        num_conv: u32,
        scale: u32,
    },
    /// SPAN; `img_range` 255 and `rgb_mean` (0.4488, 0.4371, 0.4040) are constants the weights do
    /// not record.
    Span {
        in_ch: u32,
        out_ch: u32,
        feat: u32,
        scale: u32,
        norm: bool,
    },
    /// RealPLKSR, 3 channels in and out; `norm_groups` is 4, which the weights do not record.
    RealPlksr {
        dim: u32,
        n_blocks: u32,
        scale: u32,
        kernel: u32,
        split_ratio: f32,
        use_ea: bool,
        layer_norm: bool,
        dysample: bool,
    },
}

impl ArchSpec {
    /// Output size over input size of the whole model.
    pub fn external_scale(&self) -> u32 {
        match *self {
            ArchSpec::Esrgan {
                scale,
                shuffle_factor,
                ..
            } => scale / shuffle_factor.unwrap_or(1),
            ArchSpec::Compact { scale, .. }
            | ArchSpec::Span { scale, .. }
            | ArchSpec::RealPlksr { scale, .. } => scale,
        }
    }

    /// Input width and height must be multiples of this: the pixel-unshuffle factor, else 1.
    pub fn input_multiple_of(&self) -> u32 {
        match *self {
            ArchSpec::Esrgan {
                shuffle_factor: Some(f),
                ..
            } => f,
            _ => 1,
        }
    }

    /// Channels of the image the model takes: before the pixel unshuffle, for ESRGAN.
    pub fn input_channels(&self) -> u32 {
        match *self {
            ArchSpec::Esrgan {
                in_nc,
                shuffle_factor,
                ..
            } => in_nc / shuffle_factor.map_or(1, |f| f * f),
            ArchSpec::Compact { in_nc, .. } => in_nc,
            ArchSpec::Span { in_ch, .. } => in_ch,
            ArchSpec::RealPlksr { .. } => 3,
        }
    }

    /// Channels of the image the model returns.
    pub fn output_channels(&self) -> u32 {
        match *self {
            ArchSpec::Esrgan { out_nc, .. } | ArchSpec::Compact { out_nc, .. } => out_nc,
            ArchSpec::Span { out_ch, .. } => out_ch,
            ArchSpec::RealPlksr { .. } => 3,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ArchSpec::Esrgan { .. } => "ESRGAN",
            ArchSpec::Compact { .. } => "Compact",
            ArchSpec::Span { .. } => "SPAN",
            ArchSpec::RealPlksr { .. } => "RealPLKSR",
        }
    }
}

/// A detector: its name, what its key condition misses (`None` when it matches), and the loader
/// that reads the hyperparameters once it matches.
type Detector = (
    &'static str,
    fn(&StateDict) -> Option<String>,
    fn(&StateDict) -> Result<ArchSpec>,
);

const DETECTORS: [Detector; 4] = [
    ("Compact", compact::missing, compact::load),
    ("SPAN", span::missing, span::load),
    ("ESRGAN", esrgan::missing, esrgan::load),
    ("RealPLKSR", plksr::missing, plksr::load),
];

/// Tries each detector in turn. When none matches, the error lists the keys each one missed.
pub fn detect(sd: &StateDict) -> Result<ArchSpec> {
    let mut misses = Vec::new();
    for (name, missing, load) in DETECTORS {
        match missing(sd) {
            None => return load(sd),
            Some(m) => misses.push(format!("{name}: {m}")),
        }
    }
    Err(ScaleUpError::Model(format!(
        "unsupported architecture ({})",
        misses.join("; ")
    )))
}

/// The keys of `keys` that `sd` lacks.
fn missing_keys<'k>(sd: &StateDict, keys: &[&'k str]) -> Vec<&'k str> {
    keys.iter().copied().filter(|k| !sd.has(k)).collect()
}

/// `None` when nothing is missing, else "missing a, b".
fn report(missing: &[&str]) -> Option<String> {
    (!missing.is_empty()).then(|| format!("missing {}", missing.join(", ")))
}

/// Named weight tensors: a state dict, or a view of one under other names.
pub trait Tensors {
    fn tensor(&self, key: &str) -> Result<&TensorF32>;
    fn names(&self) -> Vec<&str>;
    /// Keys of the underlying file that no name of this view reaches.
    fn unreachable(&self) -> Vec<&str> {
        Vec::new()
    }
}

impl Tensors for StateDict {
    fn tensor(&self, key: &str) -> Result<&TensorF32> {
        self.get(key)
    }

    fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }
}

/// The shape of `key`, which must have `rank` dimensions.
fn shape<'a>(sd: &'a impl Tensors, key: &str, rank: usize) -> Result<&'a [usize]> {
    let shape = &sd.tensor(key)?.shape;
    if shape.len() != rank {
        return Err(ScaleUpError::Model(format!(
            "{key} has shape {shape:?}, expected {rank} dimensions"
        )));
    }
    Ok(shape)
}

/// Dimension `axis` of `key` (of rank `rank`), non-zero.
fn dim(sd: &impl Tensors, key: &str, rank: usize, axis: usize) -> Result<u32> {
    let shape = shape(sd, key, rank)?;
    match u32::try_from(shape[axis]) {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(ScaleUpError::Model(format!(
            "{key} has shape {shape:?}, dimension {axis} is out of range"
        ))),
    }
}

/// A layer count `n` read from key indices of `what`, as u32.
fn count(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| ScaleUpError::Model(format!("{what} has {n} layers, too many")))
}

/// spandrel's `get_scale_and_output_channels`: the first `c` of `[in_ch, 3, 4, 1]` such that
/// `x = scale² · c`. `key` names the tensor `x` comes from.
fn scale_and_out(key: &str, x: u32, in_ch: u32) -> Result<(u32, u32)> {
    [in_ch, 3, 4, 1]
        .into_iter()
        .find_map(|c| {
            let s = x.is_multiple_of(c).then(|| (x / c).isqrt())?;
            (s > 0 && s * s * c == x).then_some((s, c))
        })
        .ok_or_else(|| {
            ScaleUpError::Model(format!(
                "{key}: {x} channels is not scale² · out_nc for out_nc in [{in_ch}, 3, 4, 1]"
            ))
        })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::statedict::TensorF32;
    use std::path::Path;

    /// A state dict of zero-filled tensors with the given shapes.
    pub(crate) fn dict(entries: &[(String, Vec<usize>)]) -> StateDict {
        let tensors = entries
            .iter()
            .map(|(k, shape)| {
                let n = shape.iter().product();
                (
                    k.clone(),
                    TensorF32 {
                        shape: shape.clone(),
                        data: vec![0.0_f32; n],
                    },
                )
            })
            .collect();
        StateDict { tensors }
    }

    pub(crate) fn e(key: &str, shape: &[usize]) -> (String, Vec<usize>) {
        (key.to_string(), shape.to_vec())
    }

    #[test]
    fn detect_error_lists_missing_keys() {
        let err = detect(&dict(&[e("body.0.weight", &[8, 3, 3, 3])]))
            .unwrap_err()
            .to_string();
        for needle in [
            "Compact: missing body.1.weight;",
            "SPAN: missing conv_1.sk.weight,",
            "upsampler.0.weight",
            "ESRGAN:",
            "model.1.sub.0.RDB1.conv1.0.weight",
            "conv_first.weight",
            "RRDB_trunk.0.RDB1.conv1.weight",
            "model.1.sub.0.RDB1.conv1x1.weight",
            "RealPLKSR: missing feats.0.weight",
            "one of feats.1.lk.conv.weight",
            "feats.1.channel_mixer.0.weight",
        ] {
            assert!(err.contains(needle), "{needle} not in {err}");
        }
        // Compact found body.0.weight, so it does not list it.
        assert!(!err.contains("missing body.0.weight"), "{err}");
    }

    #[test]
    fn scale_and_out_order() {
        // spandrel tries in_ch first, then 3, 4, 1.
        assert_eq!(scale_and_out("k", 48, 3).unwrap(), (4, 3));
        assert_eq!(scale_and_out("k", 64, 3).unwrap(), (4, 4));
        assert_eq!(scale_and_out("k", 16, 1).unwrap(), (4, 1));
        assert_eq!(scale_and_out("k", 3, 3).unwrap(), (1, 3));
        let err = scale_and_out("body.6.bias", 10, 3).unwrap_err().to_string();
        assert!(err.contains("body.6.bias") && err.contains("10"), "{err}");
    }

    #[test]
    fn spec_accessors() {
        let compact = ArchSpec::Compact {
            in_nc: 3,
            out_nc: 3,
            num_feat: 16,
            num_conv: 4,
            scale: 2,
        };
        assert_eq!(
            (
                compact.name(),
                compact.external_scale(),
                compact.input_multiple_of()
            ),
            ("Compact", 2, 1)
        );
        assert_eq!(
            (compact.input_channels(), compact.output_channels()),
            (3, 3)
        );
        let x2 = ArchSpec::Esrgan {
            in_nc: 12,
            out_nc: 3,
            nf: 64,
            nb: 23,
            gc: 32,
            scale: 4,
            plus: false,
            shuffle_factor: Some(2),
        };
        assert_eq!((x2.input_channels(), x2.output_channels()), (3, 3));
        let gray = ArchSpec::Esrgan {
            in_nc: 1,
            out_nc: 1,
            nf: 64,
            nb: 23,
            gc: 32,
            scale: 4,
            plus: false,
            shuffle_factor: None,
        };
        assert_eq!((gray.input_channels(), gray.output_channels()), (1, 1));
    }

    /// Detects every checkpoint downloaded by `tools/fetch_sources.py` (not part of CI). The
    /// expected values are what spandrel 0.4.2 reports for each file.
    #[test]
    #[ignore = "needs the real weights in models/cache/src"]
    fn real_checkpoints_detect() {
        let dir = Path::new("F:/Apps/ScaleUP/scaleup/models/cache/src");
        if !dir.is_dir() {
            eprintln!("skipping: {} is absent", dir.display());
            return;
        }
        let rrdb = |in_nc, nb, scale, shuffle_factor| ArchSpec::Esrgan {
            in_nc,
            out_nc: 3,
            nf: 64,
            nb,
            gc: 32,
            scale,
            plus: false,
            shuffle_factor,
        };
        let compact = |num_conv| ArchSpec::Compact {
            in_nc: 3,
            out_nc: 3,
            num_feat: 64,
            num_conv,
            scale: 4,
        };
        let expected = [
            ("4x-UltraSharp.pth", rrdb(3, 23, 4, None)),
            ("4x_foolhardy_Remacri.pth", rrdb(3, 23, 4, None)),
            ("RealESRGAN_x2plus.pth", rrdb(12, 23, 4, Some(2))),
            ("RealESRGAN_x4plus.pth", rrdb(3, 23, 4, None)),
            ("RealESRGAN_x4plus_anime_6B.pth", rrdb(3, 6, 4, None)),
            ("realesr-animevideov3.pth", compact(16)),
            ("realesr-general-x4v3.pth", compact(32)),
            (
                "4xNomosUni_span_multijpg.safetensors",
                ArchSpec::Span {
                    in_ch: 3,
                    out_ch: 3,
                    feat: 48,
                    scale: 4,
                    norm: true,
                },
            ),
        ];
        for (file, want) in expected {
            let sd = StateDict::load(&dir.join(file)).unwrap();
            let got = detect(&sd).unwrap_or_else(|e| panic!("{file}: {e}"));
            eprintln!(
                "{file}: {got:?}, external scale {}, multiple of {}",
                got.external_scale(),
                got.input_multiple_of()
            );
            assert_eq!(got, want, "{file}");
        }
        let x2 = detect(&StateDict::load(&dir.join("RealESRGAN_x2plus.pth")).unwrap()).unwrap();
        assert_eq!((x2.external_scale(), x2.input_multiple_of()), (2, 2));

        let dat = StateDict::load(&dir.join("4xRealWebPhoto_v4_dat2.safetensors")).unwrap();
        let err = detect(&dat).unwrap_err().to_string();
        eprintln!("4xRealWebPhoto_v4_dat2.safetensors: {err}");
        for name in [
            "Compact: missing",
            "SPAN: missing",
            "ESRGAN: ",
            "RealPLKSR: missing",
        ] {
            assert!(err.contains(name), "{name} not in {err}");
        }
    }
}
