//! The text half of an ncnn model.

use std::collections::HashMap;

use scaleup_core::{Result, ScaleUpError};

use super::graph::{Activation, BinOp, Graph, Layer, LayerKind};

/// Weights one layer takes from the `.bin`, in file order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightSpec {
    /// A tagged weight block of `count` values, then `bias` raw f32 values.
    Conv {
        layer: usize,
        name: String,
        count: usize,
        bias: usize,
    },
    /// `count` raw f32 slopes.
    PRelu {
        layer: usize,
        name: String,
        count: usize,
    },
}

const MAGIC: &str = "7767517";

/// A layer's `key=value` list. Arrays are written `-(23300 + key)=len,v1,v2,...`.
#[derive(Default)]
struct Params {
    scalars: HashMap<i32, f64>,
    arrays: HashMap<i32, Vec<f64>>,
}

impl Params {
    fn parse(tokens: &[&str], layer: &str) -> Result<Self> {
        let bad = |t: &str| ScaleUpError::Model(format!("bad parameter '{t}' at {layer}"));
        let mut out = Self::default();
        for t in tokens {
            let (k, v) = t.split_once('=').ok_or_else(|| bad(t))?;
            let k: i32 = k.parse().map_err(|_| bad(t))?;
            if k <= -23300 {
                let mut values = v.split(',');
                let len: usize = values
                    .next()
                    .and_then(|n| n.parse().ok())
                    .ok_or_else(|| bad(t))?;
                let values = values
                    .map(|x| x.parse::<f64>().map_err(|_| bad(t)))
                    .collect::<Result<Vec<f64>>>()?;
                if values.len() != len {
                    return Err(bad(t));
                }
                out.arrays.insert(-k - 23300, values);
            } else {
                out.scalars.insert(k, v.parse().map_err(|_| bad(t))?);
            }
        }
        Ok(out)
    }

    fn get(&self, key: i32, default: f64) -> f64 {
        self.scalars.get(&key).copied().unwrap_or(default)
    }

    fn uint(&self, key: i32, default: u32) -> u32 {
        self.get(key, f64::from(default)) as u32
    }

    fn array(&self, key: i32) -> Vec<f32> {
        self.arrays
            .get(&key)
            .map_or_else(Vec::new, |v| v.iter().map(|&x| x as f32).collect())
    }
}

fn unsupported(name: &str, what: &str) -> ScaleUpError {
    ScaleUpError::Model(format!("unsupported {what} at {name}"))
}

fn parse_conv(p: &Params, name: &str) -> Result<(LayerKind, usize, usize)> {
    let out = p.uint(0, 0);
    let k = p.uint(1, 0);
    let pad = p.uint(4, 0);
    let dilation = p.uint(2, 1);
    let stride = p.uint(3, 1);
    let same = p.uint(11, k) == k
        && p.uint(12, dilation) == dilation
        && p.uint(13, stride) == stride
        && p.uint(14, pad) == pad
        && p.uint(15, pad) == pad
        && p.uint(16, p.uint(14, pad)) == pad;
    if !same || !matches!(k, 1 | 3) || dilation != 1 || stride != 1 || pad != k / 2 || out == 0 {
        return Err(unsupported(
            name,
            "convolution geometry (need 1x1 or 3x3, stride 1, same padding)",
        ));
    }
    let act = match p.uint(9, 0) {
        0 => Activation::None,
        1 => Activation::Relu,
        2 => Activation::Leaky(p.array(10).first().copied().unwrap_or(0.0)),
        other => return Err(unsupported(name, &format!("activation type {other}"))),
    };
    let bias = p.uint(5, 0) != 0;
    let count = p.get(6, 0.0) as usize;
    let kind = LayerKind::Conv {
        out,
        k,
        pad,
        bias,
        act,
        weights: Vec::new(),
        biases: Vec::new(),
    };
    Ok((kind, count, if bias { out as usize } else { 0 }))
}

fn parse_interp(p: &Params, name: &str) -> Result<LayerKind> {
    let (h, w) = (p.get(1, 1.0), p.get(2, 1.0));
    let fixed_size = p.uint(3, 0) != 0 || p.uint(4, 0) != 0;
    if p.uint(0, 0) != 1 || h != w || h < 1.0 || h.fract() != 0.0 || fixed_size {
        return Err(unsupported(
            name,
            "Interp (need nearest with an integer scale)",
        ));
    }
    Ok(LayerKind::InterpNearest { scale: h as u32 })
}

fn parse_binary(p: &Params, name: &str) -> Result<LayerKind> {
    let op = match p.uint(0, 0) {
        0 => BinOp::Add,
        2 => BinOp::Mul,
        other => return Err(unsupported(name, &format!("BinaryOp type {other}"))),
    };
    let scalar = (p.uint(1, 0) != 0).then(|| p.get(2, 0.0) as f32);
    Ok(LayerKind::BinaryOp { op, scalar })
}

/// Parses the text graph; weights stay empty until [`super::bin::load_bin`] fills them.
pub fn parse_param(text: &str) -> Result<(Graph, Vec<WeightSpec>)> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    if lines.next() != Some(MAGIC) {
        return Err(ScaleUpError::Model(
            "not an ncnn param file (bad magic number)".into(),
        ));
    }
    let header = lines
        .next()
        .ok_or_else(|| ScaleUpError::Model("ncnn param file has no header".into()))?;
    let layer_count: usize = header
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| ScaleUpError::Model(format!("bad ncnn header '{header}'")))?;

    let mut blobs: HashMap<String, usize> = HashMap::new();
    let mut layers = Vec::with_capacity(layer_count);
    let mut specs = Vec::new();
    let mut input_blob = None;
    for line in lines {
        let t: Vec<&str> = line.split_whitespace().collect();
        let bad = || ScaleUpError::Model(format!("bad ncnn layer line '{line}'"));
        let [ty, name, n_in, n_out, rest @ ..] = t.as_slice() else {
            return Err(bad());
        };
        let n_in: usize = n_in.parse().map_err(|_| bad())?;
        let n_out: usize = n_out.parse().map_err(|_| bad())?;
        if rest.len() < n_in + n_out {
            return Err(bad());
        }
        let mut inputs = Vec::with_capacity(n_in);
        for blob in &rest[..n_in] {
            let id = blobs.get(*blob).copied().ok_or_else(|| {
                ScaleUpError::Model(format!("layer {name} reads unknown blob {blob}"))
            })?;
            inputs.push(id);
        }
        let mut outputs = Vec::with_capacity(n_out);
        for blob in &rest[n_in..n_in + n_out] {
            let next = blobs.len();
            outputs.push(*blobs.entry((*blob).to_string()).or_insert(next));
        }
        let p = Params::parse(&rest[n_in + n_out..], name)?;
        let index = layers.len();
        let kind = match *ty {
            "Input" => {
                if input_blob
                    .replace(*outputs.first().ok_or_else(bad)?)
                    .is_some()
                {
                    return Err(unsupported(name, "second Input"));
                }
                LayerKind::Input
            }
            "Convolution" => {
                let (kind, count, bias) = parse_conv(&p, name)?;
                specs.push(WeightSpec::Conv {
                    layer: index,
                    name: (*name).to_string(),
                    count,
                    bias,
                });
                kind
            }
            "Split" => LayerKind::Split,
            "Concat" => {
                if p.uint(0, 0) != 0 {
                    return Err(unsupported(name, "Concat axis"));
                }
                LayerKind::Concat
            }
            "Eltwise" => {
                if p.uint(0, 1) != 1 {
                    return Err(unsupported(name, "Eltwise op"));
                }
                LayerKind::Eltwise { coeffs: p.array(1) }
            }
            "BinaryOp" => parse_binary(&p, name)?,
            "ReLU" => LayerKind::LeakyRelu {
                slope: p.get(0, 0.0) as f32,
            },
            "PReLU" => {
                specs.push(WeightSpec::PRelu {
                    layer: index,
                    name: (*name).to_string(),
                    count: p.uint(0, 1) as usize,
                });
                LayerKind::PRelu { slopes: Vec::new() }
            }
            "Interp" => parse_interp(&p, name)?,
            "PixelShuffle" => {
                if p.uint(1, 0) != 0 {
                    return Err(unsupported(name, "PixelShuffle mode"));
                }
                LayerKind::PixelShuffle {
                    factor: p.uint(0, 1),
                }
            }
            other => {
                return Err(ScaleUpError::Model(format!(
                    "unsupported ncnn layer {other} at {name}"
                )));
            }
        };
        layers.push(Layer {
            name: (*name).to_string(),
            kind,
            inputs,
            outputs,
        });
    }
    let input_blob =
        input_blob.ok_or_else(|| ScaleUpError::Model("ncnn graph has no Input layer".into()))?;
    let output_blob = layers
        .last()
        .and_then(|l: &Layer| l.outputs.first().copied())
        .ok_or_else(|| ScaleUpError::Model("ncnn graph has no output".into()))?;
    let blob_count = blobs.len();
    Ok((
        Graph {
            layers,
            blob_count,
            input_blob,
            output_blob,
        },
        specs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_layer() {
        let text = "7767517\n2 2\nInput in 0 1 data\nSoftmax sm 1 1 data out\n";
        let err = parse_param(text).unwrap_err().to_string();
        assert!(
            err.contains("unsupported ncnn layer Softmax at sm"),
            "{err}"
        );
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(parse_param("123\n0 0\n").is_err());
    }

    #[test]
    fn array_param_syntax() {
        let text = "7767517\n3 3\nInput in 0 1 a\nSplit s 1 2 a b c\nEltwise e 2 1 b c d 0=1 -23301=2,0.2,0.3\n";
        let (g, _) = parse_param(text).unwrap();
        assert_eq!(
            g.layers[2].kind,
            LayerKind::Eltwise {
                coeffs: vec![0.2, 0.3]
            }
        );
    }

    #[test]
    fn conv_with_fused_leaky_activation() {
        let text = "7767517\n2 2\nInput in 0 1 a\nConvolution c 1 1 a b 0=8 1=3 11=3 2=1 12=1 3=1 13=1 4=1 14=1 15=1 16=1 5=1 6=216 9=2 -23310=1,2.000000e-01\n";
        let (g, specs) = parse_param(text).unwrap();
        let LayerKind::Conv {
            out,
            k,
            pad,
            act,
            bias,
            ..
        } = &g.layers[1].kind
        else {
            panic!()
        };
        assert_eq!((*out, *k, *pad, *bias), (8, 3, 1, true));
        assert_eq!(*act, Activation::Leaky(0.2));
        assert_eq!(
            specs,
            vec![WeightSpec::Conv {
                layer: 1,
                name: "c".into(),
                count: 216,
                bias: 8
            }]
        );
    }

    #[test]
    fn rejects_stride_and_dilation_and_wrong_padding() {
        for bad in ["3=2", "2=2", "4=0"] {
            let text = format!(
                "7767517\n2 2\nInput in 0 1 a\nConvolution c 1 1 a b 0=8 1=3 4=1 5=1 6=216 {bad}\n"
            );
            assert!(parse_param(&text).is_err(), "{bad}");
        }
    }

    #[test]
    fn binary_op_and_interp_and_prelu() {
        let text = "7767517\n5 5\nInput in 0 1 a\nBinaryOp m 1 1 a b 0=2 1=1 2=2.5\nInterp i 1 1 b c 0=1 1=4.0 2=4.0\nPReLU p 1 1 c d 0=3\nPixelShuffle ps 1 1 d e 0=2\n";
        let (g, specs) = parse_param(text).unwrap();
        assert_eq!(
            g.layers[1].kind,
            LayerKind::BinaryOp {
                op: BinOp::Mul,
                scalar: Some(2.5)
            }
        );
        assert_eq!(g.layers[2].kind, LayerKind::InterpNearest { scale: 4 });
        assert_eq!(g.layers[4].kind, LayerKind::PixelShuffle { factor: 2 });
        assert_eq!(
            specs,
            vec![WeightSpec::PRelu {
                layer: 3,
                name: "p".into(),
                count: 3
            }]
        );
        assert_eq!(g.input_blob, 0);
        assert_eq!(g.output_blob, 4);
        assert_eq!(g.blob_count, 5);
    }

    #[test]
    fn rejects_bilinear_interp() {
        let text = "7767517\n2 2\nInput in 0 1 a\nInterp i 1 1 a b 0=2 1=2.0 2=2.0\n";
        assert!(parse_param(text).is_err());
    }
}
