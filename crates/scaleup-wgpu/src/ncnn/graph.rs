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

/// Output size / input size of the network.
pub fn infer_scale(graph: &Graph) -> u32 {
    let mut scale = vec![1u32; graph.blob_count];
    for layer in &graph.layers {
        let input = layer.inputs.first().map_or(1, |&b| scale[b]);
        let output = match layer.kind {
            LayerKind::InterpNearest { scale } => input * scale,
            LayerKind::PixelShuffle { factor } => input * factor,
            _ => input,
        };
        for &b in &layer.outputs {
            scale[b] = output;
        }
    }
    scale[graph.output_blob]
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
            LayerKind::Split | LayerKind::LeakyRelu { .. } | LayerKind::InterpNearest { .. } => {
                first
            }
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
            assert_eq!(infer_scale(&graph), 4, "{name}");
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
        assert_eq!(infer_scale(&graph), 4);
    }
}
