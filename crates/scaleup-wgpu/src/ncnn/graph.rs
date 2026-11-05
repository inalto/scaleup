//! The in-memory network: layers in file order (ncnn files are topologically sorted).

use std::path::Path;

use scaleup_core::{Result, ScaleUpError};

use super::{bin, param};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    None,
    Relu,
    Leaky(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Mul,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    Input,
    /// Weights are `[out][in][k][k]`; `biases` is empty without a bias term.
    Conv {
        out: u32,
        k: u32,
        pad: u32,
        bias: bool,
        act: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },
    Split,
    Concat,
    /// Weighted sum of the inputs (empty `coeffs` means all ones).
    Eltwise {
        coeffs: Vec<f32>,
    },
    /// Two blobs, or one blob and `scalar`.
    BinaryOp {
        op: BinOp,
        scalar: Option<f32>,
    },
    LeakyRelu {
        slope: f32,
    },
    /// One slope per channel, or a single slope shared by all.
    PRelu {
        slopes: Vec<f32>,
    },
    InterpNearest {
        scale: u32,
    },
    PixelShuffle {
        factor: u32,
    },
    /// The inverse of `PixelShuffle`: each `factor`×`factor` block becomes `factor²` channels.
    PixelUnshuffle {
        factor: u32,
    },
    Sigmoid,
    /// `x · sigmoid(x)`.
    Silu,
    /// `(x - mean[c]) · range`, one mean per channel.
    Normalize {
        mean: Vec<f32>,
        range: f32,
    },
    /// `x · tanh(ln(1 + eˣ))`.
    Mish,
    /// Each pixel normalised over its channels, then `gamma[c] · x + beta[c]`.
    LayerNormC {
        eps: f32,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    },
    /// Each of `groups` runs of consecutive channels normalised over the whole input, then
    /// `gamma[c] · x + beta[c]`.
    GroupNorm {
        groups: u32,
        eps: f32,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    },
    /// Channels `from..to` of the input.
    Slice {
        from: u32,
        to: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub name: String,
    pub kind: LayerKind,
    /// Blob indices.
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub layers: Vec<Layer>,
    pub blob_count: usize,
    pub input_blob: usize,
    pub output_blob: usize,
}

/// Reads `<name>.param` and the `<name>.bin` next to it.
pub fn load_graph(param_path: &Path) -> Result<Graph> {
    let text = std::fs::read_to_string(param_path)?;
    let (mut graph, specs) = param::parse_param(&text)?;
    let bytes = std::fs::read(param_path.with_extension("bin"))?;
    for w in bin::load_bin(&bytes, &specs)? {
        match &mut graph.layers[w.layer].kind {
            LayerKind::Conv {
                weights, biases, ..
            } => {
                *weights = w.weights;
                *biases = w.biases;
            }
            LayerKind::PRelu { slopes } => *slopes = w.weights,
            _ => unreachable!("weight specs only name convolutions and PReLU"),
        }
    }
    // Checks every weight block against the channel count it will really be applied to.
    infer_channels(&graph, 3)?;
    Ok(graph)
}

/// Output size / input size of the network, which must be a whole number. Sizes are tracked as
/// fractions of the input: a pixel unshuffle divides them before later layers multiply them.
pub fn infer_scale(graph: &Graph) -> Result<u32> {
    let mut scale = vec![(1u64, 1u64); graph.blob_count];
    for layer in &graph.layers {
        let (num, den) = layer.inputs.first().map_or((1, 1), |&b| scale[b]);
        let (num, den) = match layer.kind {
            LayerKind::InterpNearest { scale: f } | LayerKind::PixelShuffle { factor: f } => {
                (num * u64::from(f), den)
            }
            LayerKind::PixelUnshuffle { factor } => (num, den * u64::from(factor)),
            _ => (num, den),
        };
        let g = gcd(num, den);
        for &b in &layer.outputs {
            scale[b] = (num / g, den / g);
        }
    }
    match scale[graph.output_blob] {
        (num, 1) => u32::try_from(num)
            .map_err(|_| ScaleUpError::Model(format!("the network scales by {num}, too much"))),
        (num, den) => Err(ScaleUpError::Model(format!(
            "the network scales by {num}/{den}, not a whole number"
        ))),
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// Channel count of every blob for an input with `input_channels` channels.
pub fn infer_channels(graph: &Graph, input_channels: u32) -> Result<Vec<u32>> {
    let mut ch = vec![0u32; graph.blob_count];
    for layer in &graph.layers {
        let err = |what: String| ScaleUpError::Model(format!("{what} at layer {}", layer.name));
        let first = layer.inputs.first().map_or(input_channels, |&b| ch[b]);
        let output = match &layer.kind {
            LayerKind::Input => input_channels,
            LayerKind::Conv {
                out,
                k,
                weights,
                biases,
                bias,
                ..
            } => {
                let expected = *out as usize * first as usize * (*k as usize).pow(2);
                let have = weights.len();
                // An empty block means weights were not loaded yet (a bare parse).
                if have != 0 && have != expected {
                    return Err(err(format!(
                        "weight count {have} does not match {out} outputs x {first} inputs x {k}x{k}"
                    )));
                }
                if have != 0 && *bias && biases.len() != *out as usize {
                    return Err(err(format!("bias count {}", biases.len())));
                }
                *out
            }
            LayerKind::Concat => layer.inputs.iter().map(|&b| ch[b]).sum(),
            LayerKind::Eltwise { coeffs } => {
                if layer.inputs.iter().any(|&b| ch[b] != first) {
                    return Err(err("Eltwise over different channel counts".into()));
                }
                if !coeffs.is_empty() && coeffs.len() != layer.inputs.len() {
                    return Err(err("Eltwise coefficient count".into()));
                }
                first
            }
            LayerKind::BinaryOp { scalar, .. } => {
                let expected_inputs = if scalar.is_some() { 1 } else { 2 };
                if layer.inputs.len() != expected_inputs {
                    return Err(err("BinaryOp input count".into()));
                }
                if layer.inputs.iter().any(|&b| ch[b] != first) {
                    return Err(err("BinaryOp over different channel counts".into()));
                }
                first
            }
            LayerKind::PRelu { slopes } => {
                if !slopes.is_empty() && slopes.len() != 1 && slopes.len() != first as usize {
                    return Err(err(format!("{} slopes for {first} channels", slopes.len())));
                }
                first
            }
            LayerKind::PixelShuffle { factor } => {
                let r2 = factor * factor;
                if r2 == 0 || !first.is_multiple_of(r2) {
                    return Err(err(format!("PixelShuffle {factor} over {first} channels")));
                }
                first / r2
            }
            LayerKind::PixelUnshuffle { factor } => first * factor * factor,
            LayerKind::Normalize { mean, .. } => {
                if mean.len() != first as usize {
                    return Err(err(format!("{} means for {first} channels", mean.len())));
                }
                first
            }
            LayerKind::LayerNormC { gamma, beta, .. } => {
                if gamma.len() != first as usize || beta.len() != first as usize {
                    return Err(err(format!(
                        "{} scales and {} shifts for {first} channels",
                        gamma.len(),
                        beta.len()
                    )));
                }
                first
            }
            LayerKind::GroupNorm {
                groups,
                gamma,
                beta,
                ..
            } => {
                if *groups == 0 || !first.is_multiple_of(*groups) {
                    return Err(err(format!("{groups} groups over {first} channels")));
                }
                if gamma.len() != first as usize || beta.len() != first as usize {
                    return Err(err(format!(
                        "{} scales and {} shifts for {first} channels",
                        gamma.len(),
                        beta.len()
                    )));
                }
                first
            }
            LayerKind::Slice { from, to } => {
                if from >= to || *to > first {
                    return Err(err(format!(
                        "channels {from}..{to} of {first} do not exist"
                    )));
                }
                to - from
            }
            LayerKind::Split
            | LayerKind::LeakyRelu { .. }
            | LayerKind::InterpNearest { .. }
            | LayerKind::Sigmoid
            | LayerKind::Silu
            | LayerKind::Mish => first,
        };
        for &b in &layer.outputs {
            ch[b] = output;
        }
    }
    Ok(ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn upscayl_models() -> Option<PathBuf> {
        let dir = PathBuf::from("F:/Apps/ScaleUP/Upscayl/resources/models");
        dir.is_dir().then_some(dir)
    }

    #[test]
    fn parses_real_upscayl_params() {
        let Some(dir) = upscayl_models() else {
            eprintln!("skipped: no Upscayl install");
            return;
        };
        for (name, layers) in [
            ("upscayl-standard-4x", 999),
            ("high-fidelity-4x", 999),
            ("ultrasharp-4x", 999),
            ("remacri-4x", 1370),
            ("ultramix-balanced-4x", 1370),
            ("digital-art-4x", 268),
            ("upscayl-lite-4x", 72),
        ] {
            let graph = load_graph(&dir.join(format!("{name}.param")))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(graph.layers.len(), layers, "{name}");
            assert_eq!(infer_scale(&graph).unwrap(), 4, "{name}");
            let channels = infer_channels(&graph, 3).unwrap();
            assert_eq!(channels[graph.output_blob], 3, "{name}");
        }
    }

    #[test]
    fn channel_mismatch_between_param_and_weights_names_the_layer() {
        let d = tempfile::tempdir().unwrap();
        // 3 input channels x 2 outputs x 1x1 would be 6 weights; the file claims and holds 4.
        let param = d.path().join("m.param");
        std::fs::write(
            &param,
            "7767517\n2 2\nInput in 0 1 a\nConvolution conv_x 1 1 a b 0=2 1=1 6=4\n",
        )
        .unwrap();
        let mut bin = vec![0u8; 4];
        bin.extend_from_slice(&[0u8; 16]);
        std::fs::write(d.path().join("m.bin"), bin).unwrap();
        let err = load_graph(&param).unwrap_err().to_string();
        assert!(err.contains("conv_x"), "{err}");
    }

    #[test]
    fn scale_of_parallel_paths_is_not_multiplied() {
        // The lite model upsamples on two parallel paths, each 4x: the network is 4x, not 16x.
        let graph = Graph {
            layers: vec![
                Layer {
                    name: "in".into(),
                    kind: LayerKind::Input,
                    inputs: vec![],
                    outputs: vec![0],
                },
                Layer {
                    name: "s".into(),
                    kind: LayerKind::Split,
                    inputs: vec![0],
                    outputs: vec![1, 2],
                },
                Layer {
                    name: "ps".into(),
                    kind: LayerKind::PixelShuffle { factor: 4 },
                    inputs: vec![1],
                    outputs: vec![3],
                },
                Layer {
                    name: "i".into(),
                    kind: LayerKind::InterpNearest { scale: 4 },
                    inputs: vec![2],
                    outputs: vec![4],
                },
                Layer {
                    name: "add".into(),
                    kind: LayerKind::BinaryOp {
                        op: BinOp::Add,
                        scalar: None,
                    },
                    inputs: vec![3, 4],
                    outputs: vec![5],
                },
            ],
            blob_count: 6,
            input_blob: 0,
            output_blob: 5,
        };
        assert_eq!(infer_scale(&graph).unwrap(), 4);
    }

    fn chain(kinds: Vec<LayerKind>) -> Graph {
        let mut layers = vec![Layer {
            name: "in".into(),
            kind: LayerKind::Input,
            inputs: vec![],
            outputs: vec![0],
        }];
        for (i, kind) in kinds.into_iter().enumerate() {
            layers.push(Layer {
                name: format!("l{i}"),
                kind,
                inputs: vec![i],
                outputs: vec![i + 1],
            });
        }
        let n = layers.len();
        Graph {
            layers,
            blob_count: n,
            input_blob: 0,
            output_blob: n - 1,
        }
    }

    #[test]
    fn normalize_needs_one_mean_per_channel() {
        let ok = chain(vec![
            LayerKind::Normalize {
                mean: vec![0.5; 3],
                range: 255.0,
            },
            LayerKind::Sigmoid,
            LayerKind::Silu,
        ]);
        assert_eq!(infer_channels(&ok, 3).unwrap(), [3, 3, 3, 3]);
        let bad = chain(vec![LayerKind::Normalize {
            mean: vec![0.5; 2],
            range: 255.0,
        }]);
        let err = infer_channels(&bad, 3).unwrap_err().to_string();
        assert!(
            err.contains("2 means for 3 channels") && err.contains("l0"),
            "{err}"
        );
    }

    #[test]
    fn norms_and_slice_check_their_channel_counts() {
        let norm = |n: usize| LayerKind::LayerNormC {
            eps: 1e-6,
            gamma: vec![1.0; n],
            beta: vec![0.0; n],
        };
        let group = |groups: u32, n: usize| LayerKind::GroupNorm {
            groups,
            eps: 1e-5,
            gamma: vec![1.0; n],
            beta: vec![0.0; n],
        };
        let ok = chain(vec![
            LayerKind::PixelUnshuffle { factor: 2 },
            norm(12),
            group(4, 12),
            LayerKind::Mish,
            LayerKind::Slice { from: 2, to: 7 },
        ]);
        assert_eq!(infer_channels(&ok, 3).unwrap(), [3, 12, 12, 12, 12, 5]);
        for (kind, needle) in [
            (norm(2), "2 scales and 2 shifts for 3 channels"),
            (group(2, 3), "2 groups over 3 channels"),
            (group(0, 3), "0 groups over 3 channels"),
            (group(3, 2), "2 scales and 2 shifts for 3 channels"),
            (LayerKind::Slice { from: 1, to: 4 }, "channels 1..4 of 3"),
            (LayerKind::Slice { from: 2, to: 2 }, "channels 2..2 of 3"),
        ] {
            let err = infer_channels(&chain(vec![kind]), 3)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle) && err.contains("l0"), "{err}");
        }
    }

    #[test]
    fn scale_through_pixel_unshuffle() {
        // RealESRGAN_x2plus: unshuffle by 2, then a 4x network.
        let x2 = chain(vec![
            LayerKind::PixelUnshuffle { factor: 2 },
            LayerKind::InterpNearest { scale: 2 },
            LayerKind::InterpNearest { scale: 2 },
        ]);
        assert_eq!(infer_scale(&x2).unwrap(), 2);
        assert_eq!(infer_channels(&x2, 3).unwrap(), [3, 12, 12, 12]);
        let x1 = chain(vec![
            LayerKind::PixelUnshuffle { factor: 4 },
            LayerKind::InterpNearest { scale: 4 },
        ]);
        assert_eq!(infer_scale(&x1).unwrap(), 1);
        // Half the input size is not an integer scale.
        let half = chain(vec![
            LayerKind::PixelUnshuffle { factor: 4 },
            LayerKind::InterpNearest { scale: 2 },
        ]);
        let err = infer_scale(&half).unwrap_err().to_string();
        assert!(err.contains("1/2"), "{err}");
    }
}
