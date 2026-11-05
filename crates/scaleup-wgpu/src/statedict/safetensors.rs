//! `.safetensors`: a JSON header, then each tensor as contiguous little-endian data.

use super::{DType, TensorF32, decode};
use safetensors::{Dtype, SafeTensors};
use scaleup_core::{Result, ScaleUpError};
use std::collections::BTreeMap;
use std::path::Path;

pub(super) fn load(path: &Path) -> Result<BTreeMap<String, TensorF32>> {
    let err = |msg: String| ScaleUpError::Model(format!("{}: {msg}", path.display()));
    let bytes = std::fs::read(path)?;
    let file = SafeTensors::deserialize(&bytes).map_err(|e| err(e.to_string()))?;
    let mut out = BTreeMap::new();
    for (name, view) in file.iter() {
        let dtype = match view.dtype() {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::F64 => DType::F64,
            Dtype::I64 => DType::I64,
            Dtype::I32 => DType::I32,
            other => {
                return Err(err(format!(
                    "tensor {name} has dtype {other:?}; only F32, F16, BF16, F64, I64 and I32 are supported"
                )));
            }
        };
        let data = view
            .data()
            .chunks_exact(dtype.size())
            .map(|b| decode(dtype, b))
            .collect();
        out.insert(
            name.to_string(),
            TensorF32 {
                shape: view.shape().to_vec(),
                data,
            },
        );
    }
    Ok(out)
}
