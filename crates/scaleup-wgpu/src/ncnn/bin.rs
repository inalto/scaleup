//! The binary half of an ncnn model.

use scaleup_core::{Result, ScaleUpError};

use super::param::WeightSpec;

/// Values read for one layer: weights plus biases, or slopes (in `weights`).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerWeights {
    pub layer: usize,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

const TAG_F32: u32 = 0;
const TAG_F16: u32 = 0x0130_6B47;

/// Cursor over the file; running out of bytes names the layer being read.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, layer: &str) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        let Some(end) = end else {
            return Err(ScaleUpError::Model(format!(
                "weight file too short at layer {layer}: need {n} bytes at offset {}, file has {}",
                self.pos,
                self.bytes.len()
            )));
        };
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn raw_f32(&mut self, count: usize, layer: &str) -> Result<Vec<f32>> {
        let bytes = self.take(count.saturating_mul(4), layer)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// A weight block: a `u32` tag, then f32 values or f16 values padded to 4 bytes.
    fn tagged(&mut self, count: usize, layer: &str) -> Result<Vec<f32>> {
        let t = self.take(4, layer)?;
        match u32::from_le_bytes([t[0], t[1], t[2], t[3]]) {
            TAG_F32 => self.raw_f32(count, layer),
            TAG_F16 => {
                let size = count.saturating_mul(2).next_multiple_of(4);
                let bytes = self.take(size, layer)?;
                Ok(bytes
                    .chunks_exact(2)
                    .take(count)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect())
            }
            other => Err(ScaleUpError::Model(format!(
                "unsupported weight tag {other:#010x} at layer {layer}"
            ))),
        }
    }
}

/// Reads the weights of `specs` in order. A file that is too short, too long or tagged in an unknown
/// format is refused rather than misread.
pub fn load_bin(bytes: &[u8], specs: &[WeightSpec]) -> Result<Vec<LayerWeights>> {
    let mut r = Reader { bytes, pos: 0 };
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        out.push(match spec {
            WeightSpec::Conv {
                layer,
                name,
                count,
                bias,
            } => LayerWeights {
                layer: *layer,
                weights: r.tagged(*count, name)?,
                biases: r.raw_f32(*bias, name)?,
            },
            WeightSpec::PRelu { layer, name, count } => LayerWeights {
                layer: *layer,
                weights: r.raw_f32(*count, name)?,
                biases: Vec::new(),
            },
        });
    }
    if r.pos != bytes.len() {
        return Err(ScaleUpError::Model(format!(
            "weight file has {} bytes the graph does not use",
            bytes.len() - r.pos
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv_spec(count: usize, bias: usize) -> Vec<WeightSpec> {
        vec![WeightSpec::Conv {
            layer: 1,
            name: "conv_a".into(),
            count,
            bias,
        }]
    }

    #[test]
    fn f32_tag_decodes() {
        let mut b = vec![0u8; 4];
        for v in [1.5f32, -2.0, 0.25] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let out = load_bin(&b, &conv_spec(2, 1)).unwrap();
        assert_eq!(out[0].weights, [1.5, -2.0]);
        assert_eq!(out[0].biases, [0.25]);
    }

    #[test]
    fn fp16_tag_decodes() {
        let mut b = 0x0130_6B47u32.to_le_bytes().to_vec();
        // 1.0, -2.0, 0.333251953125 (0x3555), and a subnormal, in half precision.
        for h in [0x3C00u16, 0xC000, 0x3555, 0x0001] {
            b.extend_from_slice(&h.to_le_bytes());
        }
        b.extend_from_slice(&0.5f32.to_le_bytes());
        let out = load_bin(&b, &conv_spec(4, 1)).unwrap();
        assert_eq!(out[0].weights, [1.0, -2.0, 0.333_251_95, 5.960_464_5e-8]);
        assert_eq!(out[0].biases, [0.5]);
    }

    #[test]
    fn fp16_block_is_padded_to_four_bytes() {
        let mut b = 0x0130_6B47u32.to_le_bytes().to_vec();
        for h in [0x3C00u16, 0x3C00, 0x3C00] {
            b.extend_from_slice(&h.to_le_bytes());
        }
        b.extend_from_slice(&[0, 0]);
        b.extend_from_slice(&0.5f32.to_le_bytes());
        let out = load_bin(&b, &conv_spec(3, 1)).unwrap();
        assert_eq!(out[0].biases, [0.5]);
    }

    #[test]
    fn unknown_tag_fails_with_layer_name() {
        let b = [9u8, 9, 9, 9, 0, 0, 0, 0];
        let err = load_bin(&b, &conv_spec(1, 0)).unwrap_err().to_string();
        assert!(err.contains("conv_a"), "{err}");
    }

    #[test]
    fn weight_count_mismatch_fails_with_layer_name() {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(&1.0f32.to_le_bytes());
        let err = load_bin(&b, &conv_spec(2, 0)).unwrap_err().to_string();
        assert!(err.contains("conv_a"), "{err}");
    }

    #[test]
    fn trailing_bytes_are_an_error() {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b.extend_from_slice(&1.0f32.to_le_bytes());
        assert!(load_bin(&b, &conv_spec(1, 0)).is_err());
    }

    #[test]
    fn prelu_slopes_are_raw_f32_without_a_tag() {
        let mut b = Vec::new();
        for v in [0.1f32, 0.2] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let spec = [WeightSpec::PRelu {
            layer: 2,
            name: "p".into(),
            count: 2,
        }];
        let out = load_bin(&b, &spec).unwrap();
        assert_eq!(out[0].layer, 2);
        assert_eq!(out[0].weights, [0.1, 0.2]);
    }
}
