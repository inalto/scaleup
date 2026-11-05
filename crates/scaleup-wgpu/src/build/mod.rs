//! [`Graph`]s from a detected architecture and its weights, laid out as spandrel's forward pass.

mod compact;
pub(crate) mod esrgan;

use std::cell::RefCell;
use std::collections::BTreeSet;

use scaleup_core::{Result, ScaleUpError};

use crate::arch::{ArchSpec, Tensors};
use crate::ncnn::{Activation, Graph, Layer, LayerKind, infer_channels};
use crate::statedict::{StateDict, TensorF32};

/// The architectures [`build`] turns into a graph, as OpenModelDB names them.
pub const NATIVE_ARCHS: [&str; 3] = ["esrgan", "esrgan+", "compact"];

/// The network `spec` describes, with the weights of `sd` copied into its layers. Every tensor of
/// `sd` must be used: a leftover one means a variant this builder does not know.
pub fn build(spec: &ArchSpec, sd: &StateDict) -> Result<Graph> {
    let (inputs, outputs) = (spec.input_channels(), spec.output_channels());
    if (inputs, outputs) != (3, 3) {
        return Err(ScaleUpError::Model(format!(
            "this {} model takes {inputs} channels and returns {outputs}; only RGB models (3 in, 3 out) run",
            spec.name()
        )));
    }
    match spec {
        ArchSpec::Esrgan { .. } => esrgan::build(spec, sd),
        ArchSpec::Compact { .. } => compact::build(spec, sd),
        ArchSpec::Span { .. } | ArchSpec::RealPlksr { .. } => Err(ScaleUpError::Model(format!(
            "{} models are not supported natively yet; use an ONNX export of the model",
            spec.name()
        ))),
    }
}

type Blob = usize;

/// Appends layers in execution order, tracking the channel count of every blob and which weight
/// tensors were used.
struct Builder<'w> {
    weights: &'w dyn Tensors,
    used: RefCell<BTreeSet<String>>,
    layers: Vec<Layer>,
    channels: Vec<u32>,
}

impl<'w> Builder<'w> {
    /// A builder whose input blob (returned) holds `input_channels` channels.
    fn new(weights: &'w dyn Tensors, input_channels: u32) -> (Self, Blob) {
        let mut b = Self {
            weights,
            used: RefCell::new(BTreeSet::new()),
            layers: Vec::new(),
            channels: Vec::new(),
        };
        let input = b.push("input", LayerKind::Input, &[], input_channels);
        (b, input)
    }

    fn tensor(&self, key: &str) -> Result<&'w TensorF32> {
        self.used.borrow_mut().insert(key.to_string());
        self.weights.tensor(key)
    }

    fn push(&mut self, name: &str, kind: LayerKind, inputs: &[Blob], channels: u32) -> Blob {
        let out = self.channels.len();
        self.channels.push(channels);
        self.layers.push(Layer {
            name: name.to_string(),
            kind,
            inputs: inputs.to_vec(),
            outputs: vec![out],
        });
        out
    }

    /// Convolution `{prefix}.weight` (`[out][in][k][k]`, odd `k`), plus `{prefix}.bias` when `bias`.
    fn conv(&mut self, src: Blob, prefix: &str, bias: bool, act: Activation) -> Result<Blob> {
        let key = format!("{prefix}.weight");
        let w = self.tensor(&key)?;
        let c_in = self.channels[src] as usize;
        let shape_ok = matches!(w.shape[..], [o, i, k, k2]
            if o > 0 && i == c_in && k == k2 && k % 2 == 1 && o <= u32::MAX as usize);
        if !shape_ok {
            return Err(ScaleUpError::Model(format!(
                "{key} has shape {:?}, expected [out, {c_in}, k, k] with an odd k",
                w.shape
            )));
        }
        let (out, k) = (w.shape[0], w.shape[2]);
        let biases = if bias {
            let key = format!("{prefix}.bias");
            let b = self.tensor(&key)?;
            if b.shape != [out] {
                return Err(ScaleUpError::Model(format!(
                    "{key} has shape {:?}, expected [{out}]",
                    b.shape
                )));
            }
            b.data.clone()
        } else {
            Vec::new()
        };
        let kind = LayerKind::Conv {
            out: out as u32,
            k: k as u32,
            pad: k as u32 / 2,
            bias,
            act,
            weights: w.data.clone(),
            biases,
        };
        Ok(self.push(prefix, kind, &[src], out as u32))
    }

    /// PReLU with the slopes of `key`: one per channel, or one for all.
    fn prelu(&mut self, src: Blob, key: &str) -> Result<Blob> {
        let slopes = self.tensor(key)?;
        let c = self.channels[src] as usize;
        if slopes.shape != [c] && slopes.shape != [1] {
            return Err(ScaleUpError::Model(format!(
                "{key} has shape {:?}, expected [{c}] or [1]",
                slopes.shape
            )));
        }
        let kind = LayerKind::PRelu {
            slopes: slopes.data.clone(),
        };
        Ok(self.push(key, kind, &[src], c as u32))
    }

    fn concat(&mut self, name: &str, inputs: &[Blob]) -> Blob {
        let c = inputs.iter().map(|&b| self.channels[b]).sum();
        self.push(name, LayerKind::Concat, inputs, c)
    }

    /// `coeffs[0] · a + coeffs[1] · b`.
    fn sum(&mut self, name: &str, a: Blob, b: Blob, coeffs: [f32; 2]) -> Result<Blob> {
        let (ca, cb) = (self.channels[a], self.channels[b]);
        if ca != cb {
            return Err(ScaleUpError::Model(format!(
                "{name} adds {ca} channels to {cb}"
            )));
        }
        let kind = LayerKind::Eltwise {
            coeffs: coeffs.to_vec(),
        };
        Ok(self.push(name, kind, &[a, b], ca))
    }

    fn interp(&mut self, name: &str, src: Blob, scale: u32) -> Blob {
        let c = self.channels[src];
        self.push(name, LayerKind::InterpNearest { scale }, &[src], c)
    }

    fn pixel_shuffle(&mut self, name: &str, src: Blob, factor: u32) -> Result<Blob> {
        let c = self.channels[src];
        let r2 = factor * factor;
        if !c.is_multiple_of(r2) {
            return Err(ScaleUpError::Model(format!(
                "{name}: {c} channels cannot be pixel-shuffled by {factor}"
            )));
        }
        Ok(self.push(name, LayerKind::PixelShuffle { factor }, &[src], c / r2))
    }

    fn pixel_unshuffle(&mut self, name: &str, src: Blob, factor: u32) -> Blob {
        let c = self.channels[src] * factor * factor;
        self.push(name, LayerKind::PixelUnshuffle { factor }, &[src], c)
    }

    /// The graph ending at `output`, which must hold 3 channels.
    fn finish(self, output: Blob) -> Result<Graph> {
        let used = self.used.into_inner();
        let unused: Vec<&str> = self
            .weights
            .names()
            .into_iter()
            .filter(|k| !used.contains(*k))
            .collect();
        if !unused.is_empty() {
            return Err(ScaleUpError::Model(format!(
                "unexpected tensors for this architecture: {}",
                unused.join(", ")
            )));
        }
        let graph = Graph {
            blob_count: self.channels.len(),
            layers: self.layers,
            input_blob: 0,
            output_blob: output,
        };
        let channels = infer_channels(&graph, 3)?;
        if channels[output] != 3 {
            return Err(ScaleUpError::Model(format!(
                "the network returns {} channels, expected 3",
                channels[output]
            )));
        }
        Ok(graph)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::arch::detect;
    use crate::kernels::test_gpu::Rng;
    use std::path::PathBuf;

    pub(crate) fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sd")
            .join(name)
    }

    /// A state dict of tensors with the given shapes, filled from `rng` scaled by `gain`. With
    /// `pointwise`, 3×3 kernels only have their centre tap, so the network maps each pixel alone.
    pub(crate) fn random_dict(
        entries: &[(String, Vec<usize>)],
        rng: &mut Rng,
        gain: f32,
        pointwise: bool,
    ) -> StateDict {
        let tensors = entries
            .iter()
            .map(|(k, shape)| {
                let n = shape.iter().product();
                let mut data: Vec<f32> = rng.vec(n).into_iter().map(|v| v * gain).collect();
                if pointwise && shape.len() == 4 && shape[2] == 3 {
                    for (i, v) in data.iter_mut().enumerate() {
                        if i % 9 != 4 {
                            *v = 0.0;
                        }
                    }
                }
                (
                    k.clone(),
                    TensorF32 {
                        shape: shape.clone(),
                        data,
                    },
                )
            })
            .collect();
        StateDict { tensors }
    }

    /// `{key}.weight` (centre taps) and `{key}.bias` if present, applied to one pixel.
    pub(crate) fn pointwise_conv(sd: &StateDict, key: &str, x: &[f32]) -> Vec<f32> {
        let w = sd.get(&format!("{key}.weight")).unwrap();
        let (o, i, k) = (w.shape[0], w.shape[1], w.shape[2]);
        assert_eq!(i, x.len(), "{key}");
        let centre = (k / 2) * k + k / 2;
        let bias = sd.tensors.get(&format!("{key}.bias"));
        (0..o)
            .map(|oc| {
                let b = bias.map_or(0.0, |b| b.data[oc]);
                b + (0..i)
                    .map(|ic| w.data[(oc * i + ic) * k * k + centre] * x[ic])
                    .sum::<f32>()
            })
            .collect()
    }

    pub(crate) fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < tol, "max abs diff {worst}");
    }

    /// Within `tol` relative to the largest magnitude of `want` (at least 1).
    pub(crate) fn assert_close_rel(got: &[f32], want: &[f32], tol: f32) {
        let scale = want.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
        assert_close(got, want, tol * scale);
    }

    #[test]
    fn span_and_plksr_are_not_supported_yet() {
        for spec in [
            ArchSpec::Span {
                in_ch: 3,
                out_ch: 3,
                feat: 48,
                scale: 4,
                norm: true,
            },
            ArchSpec::RealPlksr {
                dim: 64,
                n_blocks: 28,
                scale: 4,
                kernel: 17,
                split_ratio: 0.25,
                use_ea: true,
                layer_norm: true,
                dysample: false,
            },
        ] {
            let err = build(&spec, &StateDict::default()).unwrap_err().to_string();
            assert!(
                err.contains(spec.name()) && err.contains("not supported natively yet"),
                "{err}"
            );
        }
    }

    #[test]
    fn non_rgb_models_are_rejected() {
        let gray = ArchSpec::Compact {
            in_nc: 1,
            out_nc: 3,
            num_feat: 8,
            num_conv: 2,
            scale: 4,
        };
        let err = build(&gray, &StateDict::default()).unwrap_err().to_string();
        assert!(
            err.contains("takes 1 channels") && err.contains("3 in, 3 out"),
            "{err}"
        );
        let rgba_out = ArchSpec::Esrgan {
            in_nc: 3,
            out_nc: 4,
            nf: 8,
            nb: 1,
            gc: 4,
            scale: 4,
            plus: false,
            shuffle_factor: None,
        };
        let err = build(&rgba_out, &StateDict::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("returns 4"), "{err}");
    }

    #[test]
    fn conv_shape_mismatch_names_the_key() {
        let mut rng = Rng(3);
        let sd = random_dict(
            &[
                ("a.weight".into(), vec![4, 2, 3, 3]),
                ("a.bias".into(), vec![4]),
                ("b.weight".into(), vec![4, 3, 2, 2]),
                ("c.weight".into(), vec![4, 3, 3, 3]),
                ("c.bias".into(), vec![5]),
            ],
            &mut rng,
            1.0,
            false,
        );
        let (mut b, input) = Builder::new(&sd, 3);
        for (prefix, needle) in [("a", "a.weight"), ("b", "b.weight"), ("c", "c.bias")] {
            let err = b
                .conv(input, prefix, true, Activation::None)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err}");
        }
        let err = b
            .conv(input, "d", true, Activation::None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("d.weight"), "{err}");
    }

    #[test]
    fn unused_tensors_are_an_error_naming_them() {
        let mut sd = StateDict::load(&fixture("tiny_compact.pth")).unwrap();
        let spec = detect(&sd).unwrap();
        sd.tensors.insert(
            "body.3.running_mean".into(),
            TensorF32 {
                shape: vec![8],
                data: vec![0.0; 8],
            },
        );
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(err.contains("body.3.running_mean"), "{err}");
    }
}
