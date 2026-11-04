//! PyTorch state dicts: the named weight tensors of a `.safetensors` or `.pth` file, upcast to f32.

mod pth;
mod safetensors;

use regex::Regex;
use scaleup_core::{Result, ScaleUpError};
use std::collections::BTreeMap;
use std::path::Path;

/// A dense row-major tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Tensor name to tensor, as `state_dict()` returned them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StateDict {
    pub tensors: BTreeMap<String, TensorF32>,
}

/// Element types a weight file may store; all are read as f32. The integer ones hold counters
/// such as BatchNorm's `num_batches_tracked`, never weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 | DType::I64 => 8,
        }
    }
}

/// One little-endian element of `bytes` (exactly `dtype.size()` long) as f32.
fn decode(dtype: DType, bytes: &[u8]) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
        DType::F16 => half::f16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        DType::BF16 => half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        DType::F64 => f64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        DType::I64 => i64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        DType::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f32,
    }
}

/// Prefixes that wrappers such as `DataParallel` add to every key (spandrel removes the same).
const COMMON_PREFIXES: [&str; 2] = ["module.", "netG."];

impl StateDict {
    /// Reads a `.safetensors` or `.pth` file. A `.pth` training checkpoint that wraps the weights
    /// (`params_ema`, `params`, `state_dict`, `model`, ...) is unwrapped, and a prefix shared by
    /// every key (`module.`, `netG.`) is removed.
    pub fn load(path: &Path) -> Result<StateDict> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let mut tensors = match ext.as_deref() {
            Some("safetensors") => safetensors::load(path)?,
            Some("pth") => pth::load(path)?,
            _ => {
                return Err(ScaleUpError::Model(format!(
                    "{}: not a .pth or .safetensors file",
                    path.display()
                )));
            }
        };
        if tensors.is_empty() {
            return Err(ScaleUpError::Model(format!(
                "{}: holds no tensors",
                path.display()
            )));
        }
        for prefix in COMMON_PREFIXES {
            if tensors.keys().all(|k| k.starts_with(prefix)) {
                tensors = tensors
                    .into_iter()
                    .map(|(k, v)| (k[prefix.len()..].to_string(), v))
                    .collect();
            }
        }
        Ok(StateDict { tensors })
    }

    pub fn get(&self, key: &str) -> Result<&TensorF32> {
        self.tensors
            .get(key)
            .ok_or_else(|| ScaleUpError::Model(format!("state dict has no tensor {key}")))
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Length of the sequence `<prefix>.0`, `<prefix>.1`, ...: the largest index N found in a
    /// key `<prefix>.N` or `<prefix>.N.<rest>`, plus one; 0 if there is none.
    pub fn seq_len(&self, prefix: &str) -> usize {
        self.tensors
            .keys()
            .filter_map(|k| k.strip_prefix(prefix)?.strip_prefix('.'))
            .filter_map(|rest| rest.split('.').next()?.parse::<usize>().ok())
            .max()
            .map_or(0, |n| n + 1)
    }

    pub fn keys_matching(&self, re: &Regex) -> Vec<&str> {
        self.tensors
            .keys()
            .filter(|k| re.is_match(k))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sd")
            .join(name)
    }

    fn dict(entries: &[(&str, &[usize])]) -> StateDict {
        let tensors = entries
            .iter()
            .map(|(k, shape)| {
                let n = shape.iter().product();
                (
                    k.to_string(),
                    TensorF32 {
                        shape: shape.to_vec(),
                        data: vec![0.0_f32; n],
                    },
                )
            })
            .collect();
        StateDict { tensors }
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.to_string(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        serialize_to_file(views, None, path).unwrap();
    }

    #[test]
    fn safetensors_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        let a: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 3.0).collect();
        let b = [1.5_f64, -2.0];
        write_safetensors(
            &path,
            &[
                (
                    "conv.weight",
                    Dtype::F32,
                    vec![2, 3, 2, 2],
                    a.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                (
                    "conv.bias",
                    Dtype::F64,
                    vec![2],
                    b.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                (
                    "bn.num_batches_tracked",
                    Dtype::I64,
                    vec![],
                    1234_i64.to_le_bytes().to_vec(),
                ),
            ],
        );
        let sd = StateDict::load(&path).unwrap();
        assert_eq!(sd.tensors.len(), 3);
        let steps = sd.get("bn.num_batches_tracked").unwrap();
        assert!(steps.shape.is_empty());
        assert_eq!(steps.data, vec![1234.0_f32]);
        let w = sd.get("conv.weight").unwrap();
        assert_eq!(w.shape, vec![2, 3, 2, 2]);
        assert_eq!(w.data, a);
        assert_eq!(sd.get("conv.bias").unwrap().data, vec![1.5_f32, -2.0]);
    }

    #[test]
    fn safetensors_rejects_unsupported_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        write_safetensors(&path, &[("mask", Dtype::U8, vec![2], vec![0, 1])]);
        let err = StateDict::load(&path).unwrap_err().to_string();
        assert!(err.contains("mask") && err.contains("U8"), "{err}");
    }

    #[test]
    fn f16_upcast_exact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("half.safetensors");
        // Max, min subnormal, a negative fraction and zero are all exact in f32.
        let h = [65504.0_f32, 5.9604645e-8, -0.333_251_95, 0.0];
        let bf = [3.0_f32, -1.0e30, 0.0078125];
        write_safetensors(
            &path,
            &[
                (
                    "h",
                    Dtype::F16,
                    vec![4],
                    h.iter()
                        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                        .collect(),
                ),
                (
                    "bf",
                    Dtype::BF16,
                    vec![3],
                    bf.iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect(),
                ),
            ],
        );
        let sd = StateDict::load(&path).unwrap();
        let expect_h: Vec<f32> = h.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        let expect_bf: Vec<f32> = bf
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        assert_eq!(sd.get("h").unwrap().data, expect_h);
        assert_eq!(expect_h[..2], [65504.0_f32, 5.9604645e-8]);
        assert_eq!(sd.get("bf").unwrap().data, expect_bf);
    }

    #[test]
    fn pth_fixture_loads() {
        let esrgan = StateDict::load(&fixture("tiny_esrgan.pth")).unwrap();
        assert_eq!(esrgan.tensors.len(), 72);
        let first = esrgan.get("model.0.weight").unwrap();
        assert_eq!(first.shape, vec![8, 3, 3, 3]);
        // Value printed by torch for the seeded fixture.
        assert_eq!(first.data[0], 0.146_199_21_f32);
        assert_eq!(
            esrgan
                .get("model.1.sub.1.RDB3.conv5.0.weight")
                .unwrap()
                .shape,
            vec![8, 24, 3, 3]
        );
        assert_eq!(
            esrgan.get("model.10.weight").unwrap().shape,
            vec![3, 8, 3, 3]
        );
        assert_eq!(esrgan.get("model.10.bias").unwrap().data.len(), 3);

        let compact = StateDict::load(&fixture("tiny_compact.pth")).unwrap();
        assert_eq!(compact.tensors.len(), 11);
        assert_eq!(
            compact.get("body.0.weight").unwrap().shape,
            vec![8, 3, 3, 3]
        );
        assert_eq!(compact.get("body.1.weight").unwrap().shape, vec![8]);
        assert_eq!(
            compact.get("body.6.weight").unwrap().shape,
            vec![12, 8, 3, 3]
        );
        for t in compact.tensors.values().chain(esrgan.tensors.values()) {
            assert_eq!(t.data.len(), t.shape.iter().product::<usize>());
            assert!(t.data.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn pth_legacy_fixture_loads() {
        let legacy = StateDict::load(&fixture("tiny_compact_legacy.pth")).unwrap();
        let zip = StateDict::load(&fixture("tiny_compact.pth")).unwrap();
        assert_eq!(legacy, zip);
    }

    #[test]
    fn pth_views_and_dtypes() {
        let sd = StateDict::load(&fixture("tiny_views.pth")).unwrap();
        // base = arange(24).reshape(4, 6); "transposed" = base.t(), "slice" = base[1:, 2:4].
        let t = sd.get("transposed").unwrap();
        assert_eq!(t.shape, vec![6, 4]);
        let expect: Vec<f32> = (0..6)
            .flat_map(|i| (0..4).map(move |j| (j * 6 + i) as f32))
            .collect();
        assert_eq!(t.data, expect);
        let s = sd.get("slice").unwrap();
        assert_eq!(s.shape, vec![3, 2]);
        assert_eq!(s.data, vec![8.0_f32, 9.0, 14.0, 15.0, 20.0, 21.0]);
        assert_eq!(
            sd.get("half").unwrap().data,
            vec![
                0.5_f32,
                -2.0,
                65504.0,
                half::f16::from_f32(6.1035156e-05).to_f32()
            ]
        );
        assert_eq!(
            sd.get("bf16").unwrap().data,
            vec![1.0_f32, -3.5, 2.0_f32.powi(100)]
        );
        assert_eq!(sd.get("f64").unwrap().data, vec![0.25_f32, -1.0]);
        assert_eq!(sd.get("param").unwrap().data, vec![7.0_f32, 8.0]);
        let steps = sd.get("num_batches_tracked").unwrap();
        assert!(steps.shape.is_empty());
        assert_eq!(steps.data, vec![1234.0_f32]);
        // Non-tensor entries ("iter", "name", "meta") are not tensors and are left out.
        assert_eq!(sd.tensors.len(), 7);
    }

    #[test]
    fn wrapper_params_ema_unwrapped() {
        let sd = StateDict::load(&fixture("tiny_esrgan.pth")).unwrap();
        assert!(sd.has("model.0.weight"));
        assert!(sd.tensors.keys().all(|k| k.starts_with("model.")));
    }

    #[test]
    fn common_prefix_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        let one = 1.0_f32.to_le_bytes().to_vec();
        write_safetensors(
            &path,
            &[
                ("module.body.0.weight", Dtype::F32, vec![1], one.clone()),
                ("module.body.1.weight", Dtype::F32, vec![1], one),
            ],
        );
        let sd = StateDict::load(&path).unwrap();
        assert!(sd.has("body.0.weight") && sd.has("body.1.weight"));
    }

    #[test]
    fn load_rejects_unknown_extension() {
        let err = StateDict::load(Path::new("model.onnx"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("model.onnx"), "{err}");
    }

    #[test]
    fn get_names_missing_key() {
        let sd = dict(&[("a.weight", &[1])]);
        assert_eq!(sd.get("a.weight").unwrap().shape, vec![1]);
        let err = sd.get("b.weight").unwrap_err().to_string();
        assert!(err.contains("b.weight"), "{err}");
    }

    #[test]
    fn has_reports_presence() {
        let sd = dict(&[("a.weight", &[1])]);
        assert!(sd.has("a.weight"));
        assert!(!sd.has("a"));
    }

    #[test]
    fn seq_len_counts() {
        let sd = dict(&[
            ("body.0.weight", &[1]),
            ("body.1.weight", &[1]),
            ("body.10.bias", &[1]),
            ("body.4", &[1]),
            ("bodyx.20.weight", &[1]),
            ("body.alpha", &[1]),
        ]);
        assert_eq!(sd.seq_len("body"), 11);
        assert_eq!(sd.seq_len("head"), 0);
        let esrgan = StateDict::load(&fixture("tiny_esrgan.pth")).unwrap();
        assert_eq!(esrgan.seq_len("model.1.sub"), 3);
        assert_eq!(esrgan.seq_len("model"), 11);
        let compact = StateDict::load(&fixture("tiny_compact.pth")).unwrap();
        assert_eq!(compact.seq_len("body"), 7);
    }

    #[test]
    fn keys_matching_filters() {
        let sd = dict(&[
            ("model.1.sub.0.RDB1.conv1.0.weight", &[1]),
            ("model.1.sub.0.RDB2.conv1.0.weight", &[1]),
            ("model.1.sub.0.RDB2.conv1.0.bias", &[1]),
        ]);
        let re = Regex::new(r"RDB\d\.conv1\.0\.weight$").unwrap();
        assert_eq!(
            sd.keys_matching(&re),
            vec![
                "model.1.sub.0.RDB1.conv1.0.weight",
                "model.1.sub.0.RDB2.conv1.0.weight"
            ]
        );
    }

    /// Loads every checkpoint downloaded by `tools/fetch_sources.py` (not part of CI).
    #[test]
    #[ignore = "needs the real weights in models/cache/src"]
    fn real_checkpoints_load() {
        let dir = Path::new("F:/Apps/ScaleUP/scaleup/models/cache/src");
        let Ok(entries) = std::fs::read_dir(dir) else {
            eprintln!("skipping: {} is absent", dir.display());
            return;
        };
        let mut loaded = 0;
        for entry in entries {
            let path = entry.unwrap().path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "pth" && ext != "safetensors" {
                continue;
            }
            let sd = StateDict::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert!(!sd.tensors.is_empty(), "{}", path.display());
            let mut values = 0;
            let mut sum = 0.0_f64;
            for (key, t) in &sd.tensors {
                assert_eq!(t.data.len(), t.shape.iter().product::<usize>(), "{key}");
                assert!(
                    t.data.iter().all(|v| v.is_finite()),
                    "{}: {key}",
                    path.display()
                );
                values += t.data.len();
                sum += t.data.iter().map(|&v| f64::from(v)).sum::<f64>();
            }
            // The sum lets a run be compared with torch's view of the same file.
            eprintln!(
                "{}: {} tensors, {values} values, sum {sum:.6}",
                path.file_name().unwrap().to_string_lossy(),
                sd.tensors.len()
            );
            loaded += 1;
        }
        assert!(loaded > 0, "no checkpoint in {}", dir.display());
    }
}
