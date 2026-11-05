//! RRDBNet: ESRGAN, BSRGAN/RealSR, Real-ESRGAN and ESRGAN+ (spandrel `ESRGANArch`).
//!
//! Rule: detect any of: {`model.0.weight`, `model.1.sub.0.RDB1.conv1.0.weight`} |
//! {`conv_first.weight`, `body.0.rdb1.conv1.weight`, `conv_body.weight`, `conv_last.weight`} |
//! {`conv_first.weight`, `RRDB_trunk.0.RDB1.conv1.weight`, `trunk_conv.weight`,
//! `conv_last.weight`} | {`model.0.weight`, `model.1.sub.0.RDB1.conv1x1.weight`}. After conversion
//! to old arch: `in_nc = model.0.weight.shape[1]`, `nf = model.0.weight.shape[0]`,
//! `nb = seq_len("model.1.sub") - 1`, `seq = seq_len("model")`,
//! `out_nc = model.{seq-1}.weight.shape[0]`, `scale = 2^((seq-5)/3)`,
//! `plus = any key contains ".conv1x1."`; `shuffle_factor = sqrt(in_nc/out_nc)` when
//! `in_nc ∈ {4·out_nc, 16·out_nc}`, then external `in_nc /= f²`, `scale /= f`. New→old rename:
//! `conv_first→model.0`, `trunk_conv|conv_body→model.1.sub.{nb}`,
//! `RRDB_trunk.(\d+).RDB(\d).conv(\d).(weight|bias)` and `body.(\d+).rdb(\d).conv(\d).(w|b)` →
//! `model.1.sub.\1.RDB\2.conv\3.0.\4`, `upconv{n}|conv_up{n}→model.{3n}`,
//! `HRconv|conv_hr→model.{3·max_up+2}`, `conv_last→model.{3·max_up+4}`.
//!
//! As in spandrel, `max_up` above is already `3n` (HRconv lands on `model.{max_up+2}`), keys the
//! rename does not cover are dropped, and `nb` is the highest block index found plus one. Beyond
//! spandrel, `seq - 5` must be a multiple of 3 and the trunk convolution `model.1.sub.{nb}` must
//! exist. `gc = model.1.sub.0.RDB1.conv1.0.weight.shape[0]` (spandrel hard-codes 32).

use super::{ArchSpec, Tensors, dim, missing_keys};
use crate::statedict::{StateDict, TensorF32, seq_len_of};
use regex::Regex;
use scaleup_core::{Result, ScaleUpError};
use std::collections::BTreeMap;

const ALTERNATIVES: [&[&str]; 4] = [
    &["model.0.weight", "model.1.sub.0.RDB1.conv1.0.weight"],
    &[
        "conv_first.weight",
        "body.0.rdb1.conv1.weight",
        "conv_body.weight",
        "conv_last.weight",
    ],
    &[
        "conv_first.weight",
        "RRDB_trunk.0.RDB1.conv1.weight",
        "trunk_conv.weight",
        "conv_last.weight",
    ],
    &["model.0.weight", "model.1.sub.0.RDB1.conv1x1.weight"],
];

pub(super) fn missing(sd: &StateDict) -> Option<String> {
    let misses: Vec<Vec<&str>> = ALTERNATIVES
        .iter()
        .map(|keys| missing_keys(sd, keys))
        .collect();
    if misses.iter().any(Vec::is_empty) {
        return None;
    }
    let alternatives: Vec<String> = misses.iter().map(|m| m.join(", ")).collect();
    Some(format!("missing {}", alternatives.join(" | or ")))
}

pub(super) fn load(sd: &StateDict) -> Result<ArchSpec> {
    let sd = OldArch::new(sd)?;
    let in_nc = dim(&sd, "model.0.weight", 4, 1)?;
    let nf = dim(&sd, "model.0.weight", 4, 0)?;
    // model.0 conv, model.1 trunk, log2(scale) × (upsample, conv, LeakyReLU), HR conv,
    // LeakyReLU, last conv.
    let seq = sd.seq_len("model");
    if seq < 5 || !(seq - 5).is_multiple_of(3) {
        return Err(ScaleUpError::Model(format!(
            "ESRGAN: the last layer is model.{}, expected model.{{4 + 3·log2(scale)}}",
            seq.saturating_sub(1)
        )));
    }
    let scale = u32::try_from((seq - 5) / 3)
        .ok()
        .and_then(|log2| 1_u32.checked_shl(log2))
        .ok_or_else(|| {
            ScaleUpError::Model(format!("ESRGAN: model.{} implies an absurd scale", seq - 1))
        })?;
    let out_nc = dim(&sd, &format!("model.{}.weight", seq - 1), 4, 0)?;
    let nb = sd.seq_len("model.1.sub").saturating_sub(1);
    let trunk = format!("model.1.sub.{nb}.weight");
    let Some(nb) = u32::try_from(nb).ok().filter(|&n| n > 0 && sd.has(&trunk)) else {
        return Err(ScaleUpError::Model(format!(
            "ESRGAN: no trunk convolution {trunk} after the RRDB blocks"
        )));
    };
    let gc = dim(&sd, "model.1.sub.0.RDB1.conv1.0.weight", 4, 0)?;
    let plus = sd.names().iter().any(|k| k.contains(".conv1x1."));
    let shuffle_factor = if in_nc == 4 * out_nc {
        Some(2)
    } else if in_nc == 16 * out_nc {
        Some(4)
    } else {
        None
    };
    if let Some(f) = shuffle_factor
        && !scale.is_multiple_of(f)
    {
        return Err(ScaleUpError::Model(format!(
            "ESRGAN: {in_nc} input channels mean a pixel-unshuffle by {f}, more than the network's scale {scale}"
        )));
    }
    Ok(ArchSpec::Esrgan {
        in_nc,
        out_nc,
        nf,
        nb,
        gc,
        scale,
        plus,
        shuffle_factor,
    })
}

/// Renames the keys of a new-arch (Real-ESRGAN `conv_first`/`body`, BSRGAN `RRDB_trunk`) state
/// dict to the old `model.N` layout, as spandrel's `_new_to_old_arch` does. An old-arch dict (no
/// `conv_first.weight`) is returned as is. Copies every tensor: [`OldArch`] reads the same names
/// without copying.
pub fn esrgan_to_old_arch(sd: &StateDict) -> Result<StateDict> {
    let old = OldArch::new(sd)?;
    let mut tensors = BTreeMap::new();
    for name in old.names() {
        tensors.insert(name.to_string(), old.tensor(name)?.clone());
    }
    Ok(StateDict { tensors })
}

/// The tensors of an ESRGAN state dict under their old-arch (`model.N`) names, read in place.
pub struct OldArch<'a> {
    sd: &'a StateDict,
    /// Old-arch name to the key in `sd`; `None` when `sd` already uses old-arch names.
    renamed: Option<BTreeMap<String, &'a str>>,
}

impl<'a> OldArch<'a> {
    pub fn new(sd: &'a StateDict) -> Result<Self> {
        Ok(Self {
            sd,
            renamed: old_arch_names(sd)?,
        })
    }

    pub fn has(&self, key: &str) -> bool {
        match &self.renamed {
            Some(names) => names.contains_key(key),
            None => self.sd.has(key),
        }
    }

    pub fn seq_len(&self, prefix: &str) -> usize {
        seq_len_of(self.names().into_iter(), prefix)
    }
}

impl Tensors for OldArch<'_> {
    fn tensor(&self, key: &str) -> Result<&TensorF32> {
        match &self.renamed {
            Some(names) => match names.get(key) {
                Some(original) => self.sd.get(original),
                None => Err(ScaleUpError::Model(format!(
                    "state dict has no tensor {key} (old-arch name)"
                ))),
            },
            None => self.sd.get(key),
        }
    }

    fn names(&self) -> Vec<&str> {
        match &self.renamed {
            Some(names) => names.keys().map(String::as_str).collect(),
            None => self.sd.names(),
        }
    }
}

/// Old-arch name to new-arch key, or `None` for an old-arch dict.
fn old_arch_names(sd: &StateDict) -> Result<Option<BTreeMap<String, &str>>> {
    if !sd.has("conv_first.weight") {
        return Ok(None);
    }
    let rdb = [
        Regex::new(r"^RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)$").unwrap(),
        Regex::new(r"^body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)$").unwrap(),
    ];
    // spandrel takes the block count from the first naming that matches any key.
    let nb = rdb
        .iter()
        .find_map(|re| {
            sd.tensors
                .keys()
                .filter_map(|k| re.captures(k)?[1].parse::<usize>().ok())
                .max()
        })
        .map(|max| max + 1)
        .ok_or_else(|| {
            ScaleUpError::Model(
                "ESRGAN: conv_first.weight but no RRDB_trunk.N.RDBn.convn or body.N.rdbn.convn key"
                    .to_string(),
            )
        })?;
    let up = Regex::new(r"^(?:upconv|conv_up)(\d)\.(weight|bias)$").unwrap();
    let max_up = sd
        .tensors
        .keys()
        .filter_map(|k| Some(3 * up.captures(k)?[1].parse::<usize>().ok()?))
        .max()
        .unwrap_or(0);

    let mut old = BTreeMap::new();
    for key in sd.tensors.keys() {
        let renamed = if let Some(c) = rdb.iter().find_map(|re| re.captures(key)) {
            Some(format!(
                "model.1.sub.{}.RDB{}.conv{}.0.{}",
                &c[1], &c[2], &c[3], &c[4]
            ))
        } else if let Some(c) = up.captures(key) {
            Some(format!(
                "model.{}.{}",
                3 * c[1].parse::<usize>().unwrap(),
                &c[2]
            ))
        } else {
            let (layer, kind) = key.rsplit_once('.').unwrap_or((key, ""));
            let index = match layer {
                "conv_first" => Some("0".to_string()),
                "trunk_conv" | "conv_body" => Some(format!("1.sub.{nb}")),
                "HRconv" | "conv_hr" => Some((max_up + 2).to_string()),
                "conv_last" => Some((max_up + 4).to_string()),
                _ => None,
            };
            index
                .filter(|_| kind == "weight" || kind == "bias")
                .map(|i| format!("model.{i}.{kind}"))
        };
        if let Some(name) = renamed {
            old.insert(name, key.as_str());
        }
    }
    Ok(Some(old))
}

#[cfg(test)]
mod tests {
    use super::super::detect;
    use super::super::tests::dict;
    use super::*;
    use std::path::PathBuf;

    #[derive(Clone, Copy)]
    struct Rrdb {
        in_nc: usize,
        out_nc: usize,
        nf: usize,
        nb: usize,
        gc: usize,
        log2_scale: usize,
        plus: bool,
    }

    const BASE: Rrdb = Rrdb {
        in_nc: 3,
        out_nc: 3,
        nf: 64,
        nb: 23,
        gc: 32,
        log2_scale: 2,
        plus: false,
    };

    /// Shapes of RRDB block `b`, dense block `r`, convolution `c` (1-based), and of the ESRGAN+
    /// 1×1 convolution of a dense block.
    fn rdb_conv(n: &Rrdb, c: usize) -> Vec<usize> {
        let out = if c == 5 { n.nf } else { n.gc };
        vec![out, n.nf + (c - 1) * n.gc, 3, 3]
    }

    /// The keys an old-arch (`model.N`) checkpoint holds.
    fn old_arch(n: Rrdb) -> StateDict {
        let conv = |o, i| vec![o, i, 3, 3];
        let mut entries = vec![
            ("model.0.weight".to_string(), conv(n.nf, n.in_nc)),
            ("model.0.bias".to_string(), vec![n.nf]),
        ];
        for b in 0..n.nb {
            for r in 1..=3 {
                for c in 1..=5 {
                    let p = format!("model.1.sub.{b}.RDB{r}.conv{c}.0");
                    entries.push((format!("{p}.weight"), rdb_conv(&n, c)));
                    entries.push((format!("{p}.bias"), vec![rdb_conv(&n, c)[0]]));
                }
                if n.plus {
                    let p = format!("model.1.sub.{b}.RDB{r}.conv1x1");
                    entries.push((format!("{p}.weight"), vec![n.gc, n.nf, 1, 1]));
                }
            }
        }
        entries.push((format!("model.1.sub.{}.weight", n.nb), conv(n.nf, n.nf)));
        entries.push((format!("model.1.sub.{}.bias", n.nb), vec![n.nf]));
        for i in 1..=n.log2_scale {
            entries.push((format!("model.{}.weight", 3 * i), conv(n.nf, n.nf)));
            entries.push((format!("model.{}.bias", 3 * i), vec![n.nf]));
        }
        let hr = 3 * n.log2_scale + 2;
        entries.push((format!("model.{hr}.weight"), conv(n.nf, n.nf)));
        entries.push((format!("model.{hr}.bias"), vec![n.nf]));
        entries.push((format!("model.{}.weight", hr + 2), conv(n.out_nc, n.nf)));
        entries.push((format!("model.{}.bias", hr + 2), vec![n.out_nc]));
        dict(&entries)
    }

    /// The keys a new-arch checkpoint holds: Real-ESRGAN names, or BSRGAN/RealSR ones.
    fn new_arch(n: Rrdb, bsrgan: bool) -> StateDict {
        let (trunk, rdb, body, up, hr) = if bsrgan {
            ("RRDB_trunk", "RDB", "trunk_conv", "upconv", "HRconv")
        } else {
            ("body", "rdb", "conv_body", "conv_up", "conv_hr")
        };
        let conv = |o, i| vec![o, i, 3, 3];
        let mut entries = vec![
            ("conv_first.weight".to_string(), conv(n.nf, n.in_nc)),
            ("conv_first.bias".to_string(), vec![n.nf]),
        ];
        for b in 0..n.nb {
            for r in 1..=3 {
                for c in 1..=5 {
                    let p = format!("{trunk}.{b}.{rdb}{r}.conv{c}");
                    entries.push((format!("{p}.weight"), rdb_conv(&n, c)));
                    entries.push((format!("{p}.bias"), vec![rdb_conv(&n, c)[0]]));
                }
            }
        }
        entries.push((format!("{body}.weight"), conv(n.nf, n.nf)));
        entries.push((format!("{body}.bias"), vec![n.nf]));
        for i in 1..=n.log2_scale {
            entries.push((format!("{up}{i}.weight"), conv(n.nf, n.nf)));
            entries.push((format!("{up}{i}.bias"), vec![n.nf]));
        }
        entries.push((format!("{hr}.weight"), conv(n.nf, n.nf)));
        entries.push((format!("{hr}.bias"), vec![n.nf]));
        entries.push(("conv_last.weight".to_string(), conv(n.out_nc, n.nf)));
        entries.push(("conv_last.bias".to_string(), vec![n.out_nc]));
        dict(&entries)
    }

    fn spec(n: Rrdb, shuffle_factor: Option<u32>) -> ArchSpec {
        ArchSpec::Esrgan {
            in_nc: n.in_nc as u32,
            out_nc: n.out_nc as u32,
            nf: n.nf as u32,
            nb: n.nb as u32,
            gc: n.gc as u32,
            scale: 1 << n.log2_scale,
            plus: n.plus,
            shuffle_factor,
        }
    }

    #[test]
    fn esrgan_old_arch_23_64_4x() {
        let got = detect(&old_arch(BASE)).unwrap();
        assert_eq!(got, spec(BASE, None));
        assert_eq!(
            (got.name(), got.external_scale(), got.input_multiple_of()),
            ("ESRGAN", 4, 1)
        );
    }

    #[test]
    fn esrgan_new_arch_same_result() {
        for n in [
            BASE,
            Rrdb {
                nb: 6,
                log2_scale: 1,
                ..BASE
            },
            Rrdb { in_nc: 12, ..BASE },
        ] {
            let old = old_arch(n);
            for bsrgan in [false, true] {
                let new = new_arch(n, bsrgan);
                assert_eq!(detect(&new).unwrap(), detect(&old).unwrap());
                let renamed = esrgan_to_old_arch(&new).unwrap();
                assert_eq!(
                    renamed.tensors.keys().collect::<Vec<_>>(),
                    old.tensors.keys().collect::<Vec<_>>()
                );
            }
        }
        // An old-arch dict comes back unchanged.
        assert_eq!(esrgan_to_old_arch(&old_arch(BASE)).unwrap(), old_arch(BASE));
    }

    #[test]
    fn old_arch_view_reads_new_arch_tensors_in_place() {
        let new = new_arch(BASE, false);
        let view = OldArch::new(&new).unwrap();
        for (old, original) in [
            ("model.0.weight", "conv_first.weight"),
            ("model.1.sub.23.bias", "conv_body.bias"),
            (
                "model.1.sub.4.RDB2.conv5.0.weight",
                "body.4.rdb2.conv5.weight",
            ),
            ("model.6.weight", "conv_up2.weight"),
            ("model.10.weight", "conv_last.weight"),
        ] {
            assert!(view.has(old), "{old}");
            assert!(std::ptr::eq(
                view.tensor(old).unwrap(),
                new.get(original).unwrap()
            ));
        }
        assert_eq!(view.seq_len("model"), 11);
        assert!(!view.has("conv_first.weight"));
        let err = view.tensor("model.9.weight").unwrap_err().to_string();
        assert!(err.contains("model.9.weight"), "{err}");
        // An old-arch dict is read under its own names.
        let old = old_arch(BASE);
        let view = OldArch::new(&old).unwrap();
        assert_eq!(view.names(), old.names());
    }

    #[test]
    fn esrgan_params_ema_wrapper() {
        // The fixture is an old-arch 4× RRDBNet (nf 8, nb 2, gc 4) saved as {"params_ema": ...}.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sd/tiny_esrgan.pth");
        let wrapped = detect(&StateDict::load(&path).unwrap()).unwrap();
        let tiny = Rrdb {
            nf: 8,
            nb: 2,
            gc: 4,
            ..BASE
        };
        assert_eq!(wrapped, detect(&old_arch(tiny)).unwrap());
        assert_eq!(wrapped, spec(tiny, None));
    }

    #[test]
    fn esrgan_plus_detected() {
        let plus = Rrdb { plus: true, ..BASE };
        assert_eq!(detect(&old_arch(plus)).unwrap(), spec(plus, None));
        // The conv1x1 alternative alone detects ESRGAN; loading then names the missing key.
        let mut sd = old_arch(plus);
        sd.tensors.remove("model.1.sub.0.RDB1.conv1.0.weight");
        let err = detect(&sd).unwrap_err().to_string();
        assert!(!err.contains("unsupported architecture"), "{err}");
        assert!(err.contains("model.1.sub.0.RDB1.conv1.0.weight"), "{err}");
    }

    /// RealESRGAN_x2plus: 2× by unshuffling by 2 into a 4× network.
    #[test]
    fn esrgan_unshuffle_12ch_reports_scale_2_and_multiple_of_2() {
        let n = Rrdb { in_nc: 12, ..BASE };
        let got = detect(&old_arch(n)).unwrap();
        assert_eq!(got, spec(n, Some(2)));
        assert_eq!((got.external_scale(), got.input_multiple_of()), (2, 2));
    }

    /// 1× by unshuffling by 4 into a 4× network.
    #[test]
    fn esrgan_unshuffle_48ch_reports_scale_1_and_multiple_of_4() {
        let n = Rrdb { in_nc: 48, ..BASE };
        let got = detect(&old_arch(n)).unwrap();
        assert_eq!(got, spec(n, Some(4)));
        assert_eq!((got.external_scale(), got.input_multiple_of()), (1, 4));
    }

    #[test]
    fn esrgan_1x_restoration_scale_1() {
        let n = Rrdb {
            log2_scale: 0,
            ..BASE
        };
        let got = detect(&old_arch(n)).unwrap();
        assert_eq!(got, spec(n, None));
        assert_eq!((got.external_scale(), got.input_multiple_of()), (1, 1));
    }

    #[test]
    fn esrgan_rejects_gap_in_layers() {
        let mut sd = old_arch(BASE);
        sd.tensors.remove("model.10.weight");
        sd.tensors.remove("model.10.bias");
        let err = detect(&sd).unwrap_err().to_string();
        assert!(err.contains("model.8"), "{err}");
    }

    #[test]
    fn esrgan_rejects_unshuffle_beyond_scale() {
        // A 1× network cannot unshuffle by 2: the external scale would be 1/2.
        let n = Rrdb {
            in_nc: 12,
            log2_scale: 0,
            ..BASE
        };
        let err = detect(&old_arch(n)).unwrap_err().to_string();
        assert!(err.contains("unshuffle"), "{err}");
    }
}
