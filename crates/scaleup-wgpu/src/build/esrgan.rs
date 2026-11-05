//! RRDBNet, from spandrel's `RRDB.py` (`RRDBNet.forward`, `RRDB`, `ResidualDenseBlock_5C`) with
//! old-arch weight names: `[pixel unshuffle] → model.0 → nb × RRDB → model.1.sub.{nb} → + trunk →
//! log2(scale) × (nearest 2× → model.{3i} → leaky) → model.{hr} → leaky → model.{hr+2}`.

use scaleup_core::Result;

use super::{Blob, Builder};
use crate::arch::{ArchSpec, OldArch};
use crate::ncnn::{Activation, Graph};
use crate::statedict::StateDict;

const LEAKY: Activation = Activation::Leaky(0.2);
/// Residual scaling of the dense blocks and of the RRDB.
const RESIDUAL: f32 = 0.2;

pub(super) fn build(spec: &ArchSpec, sd: &StateDict) -> Result<Graph> {
    let ArchSpec::Esrgan {
        nb,
        scale,
        plus,
        shuffle_factor,
        ..
    } = *spec
    else {
        unreachable!("build dispatches on the spec")
    };
    let weights = OldArch::new(sd)?;
    let (mut b, input) = Builder::new(&weights, 3);
    let x = match shuffle_factor {
        Some(f) => b.pixel_unshuffle("pixel_unshuffle", input, f),
        None => input,
    };
    let fea = b.conv(x, "model.0", true, Activation::None)?;
    let mut t = fea;
    for i in 0..nb {
        t = rrdb(&mut b, t, &format!("model.1.sub.{i}"), plus)?;
    }
    t = b.conv(t, &format!("model.1.sub.{nb}"), true, Activation::None)?;
    t = b.sum("model.1", t, fea, [1.0, 1.0])?;
    // The detector only reports powers of two.
    let stages = scale.trailing_zeros();
    for i in 1..=stages {
        t = b.interp(&format!("model.{}", 3 * i - 1), t, 2);
        t = b.conv(t, &format!("model.{}", 3 * i), true, LEAKY)?;
    }
    let hr = 3 * stages + 2;
    t = b.conv(t, &format!("model.{hr}"), true, LEAKY)?;
    t = b.conv(t, &format!("model.{}", hr + 2), true, Activation::None)?;
    b.finish(t)
}

/// `RRDB.forward`: three dense blocks, then `0.2 · out + x`.
fn rrdb(b: &mut Builder<'_>, x: Blob, prefix: &str, plus: bool) -> Result<Blob> {
    let mut out = x;
    for r in 1..=3 {
        out = dense_block(b, out, &format!("{prefix}.RDB{r}"), plus)?;
    }
    b.sum(prefix, out, x, [RESIDUAL, 1.0])
}

/// `ResidualDenseBlock_5C.forward`: each convolution sees the block input and every earlier output;
/// ESRGAN+ adds a 1×1 shortcut to the second output and the second output to the fourth.
fn dense_block(b: &mut Builder<'_>, x: Blob, p: &str, plus: bool) -> Result<Blob> {
    let x1 = b.conv(x, &format!("{p}.conv1.0"), true, LEAKY)?;
    let c1 = b.concat(&format!("{p}.cat1"), &[x, x1]);
    let mut x2 = b.conv(c1, &format!("{p}.conv2.0"), true, LEAKY)?;
    if plus {
        let shortcut = b.conv(x, &format!("{p}.conv1x1"), false, Activation::None)?;
        x2 = b.sum(&format!("{p}.plus2"), x2, shortcut, [1.0, 1.0])?;
    }
    let c2 = b.concat(&format!("{p}.cat2"), &[c1, x2]);
    let x3 = b.conv(c2, &format!("{p}.conv3.0"), true, LEAKY)?;
    let c3 = b.concat(&format!("{p}.cat3"), &[c2, x3]);
    let mut x4 = b.conv(c3, &format!("{p}.conv4.0"), true, LEAKY)?;
    if plus {
        x4 = b.sum(&format!("{p}.plus4"), x4, x2, [1.0, 1.0])?;
    }
    let c4 = b.concat(&format!("{p}.cat4"), &[c3, x4]);
    let x5 = b.conv(c4, &format!("{p}.conv5.0"), true, Activation::None)?;
    b.sum(p, x5, x, [RESIDUAL, 1.0])
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::build;
    use super::super::tests::{assert_close_rel, fixture, pointwise_conv, random_dict};
    use crate::arch::detect;
    use crate::cpu_ref::run_graph;
    use crate::exec::plan;
    use crate::kernels::test_gpu::Rng;
    use crate::ncnn::{LayerKind, infer_scale};
    use crate::statedict::StateDict;

    pub(crate) struct Rrdb {
        pub in_nc: usize,
        pub nf: usize,
        pub nb: usize,
        pub gc: usize,
        pub log2_scale: usize,
        pub plus: bool,
    }

    /// Old-arch key names and shapes of an RRDBNet with 3 output channels.
    pub(crate) fn entries(n: &Rrdb) -> Vec<(String, Vec<usize>)> {
        let conv = |o, i| vec![o, i, 3, 3];
        let mut e = vec![
            ("model.0.weight".to_string(), conv(n.nf, n.in_nc)),
            ("model.0.bias".to_string(), vec![n.nf]),
        ];
        for b in 0..n.nb {
            for r in 1..=3 {
                for c in 1..=5 {
                    let out = if c == 5 { n.nf } else { n.gc };
                    let p = format!("model.1.sub.{b}.RDB{r}.conv{c}.0");
                    e.push((format!("{p}.weight"), conv(out, n.nf + (c - 1) * n.gc)));
                    e.push((format!("{p}.bias"), vec![out]));
                }
                if n.plus {
                    let p = format!("model.1.sub.{b}.RDB{r}.conv1x1");
                    e.push((format!("{p}.weight"), vec![n.gc, n.nf, 1, 1]));
                }
            }
        }
        let mut layer = |i: usize, o, c_in| {
            e.push((format!("model.{i}.weight"), conv(o, c_in)));
            e.push((format!("model.{i}.bias"), vec![o]));
        };
        layer(1000, n.nf, n.nf);
        for i in 1..=n.log2_scale {
            layer(3 * i, n.nf, n.nf);
        }
        let hr = 3 * n.log2_scale + 2;
        layer(hr, n.nf, n.nf);
        layer(hr + 2, 3, n.nf);
        // The trunk convolution, named after the blocks.
        for (k, _) in e.iter_mut() {
            if let Some(rest) = k.strip_prefix("model.1000.") {
                *k = format!("model.1.sub.{}.{rest}", n.nb);
            }
        }
        e
    }

    fn lrelu(v: Vec<f32>) -> Vec<f32> {
        v.into_iter()
            .map(|x| if x < 0.0 { 0.2 * x } else { x })
            .collect()
    }

    fn add(a: &[f32], b: &[f32], ca: f32) -> Vec<f32> {
        a.iter().zip(b).map(|(a, b)| ca * a + b).collect()
    }

    /// `RRDB.py`'s forward pass for one pixel of a network whose convolutions are all pointwise.
    fn rrdbnet_pixel(sd: &StateDict, n: &Rrdb, x: &[f32]) -> Vec<f32> {
        let conv = |key: &str, x: &[f32]| pointwise_conv(sd, key, x);
        let rdb = |p: &str, x: &[f32]| {
            let x1 = lrelu(conv(&format!("{p}.conv1.0"), x));
            let mut x2 = lrelu(conv(&format!("{p}.conv2.0"), &[x, &x1].concat()));
            if n.plus {
                x2 = add(&x2, &conv(&format!("{p}.conv1x1"), x), 1.0);
            }
            let x3 = lrelu(conv(&format!("{p}.conv3.0"), &[x, &x1, &x2].concat()));
            let mut x4 = lrelu(conv(&format!("{p}.conv4.0"), &[x, &x1, &x2, &x3].concat()));
            if n.plus {
                x4 = add(&x4, &x2, 1.0);
            }
            let x5 = conv(&format!("{p}.conv5.0"), &[x, &x1, &x2, &x3, &x4].concat());
            add(&x5, x, 0.2)
        };
        let fea = conv("model.0", x);
        let mut t = fea.clone();
        for b in 0..n.nb {
            let p = format!("model.1.sub.{b}");
            let mut out = t.clone();
            for r in 1..=3 {
                out = rdb(&format!("{p}.RDB{r}"), &out);
            }
            t = add(&out, &t, 0.2);
        }
        t = add(&conv(&format!("model.1.sub.{}", n.nb), &t), &fea, 1.0);
        for i in 1..=n.log2_scale {
            t = lrelu(conv(&format!("model.{}", 3 * i), &t));
        }
        let hr = 3 * n.log2_scale + 2;
        t = lrelu(conv(&format!("model.{hr}"), &t));
        conv(&format!("model.{}", hr + 2), &t)
    }

    /// Builds a random pointwise network and compares the graph, run by `cpu_ref`, with the forward
    /// pass applied pixel by pixel.
    fn assert_matches_forward(n: Rrdb, seed: u32) {
        let mut rng = Rng(seed);
        let sd = random_dict(&entries(&n), &mut rng, 1.0, true);
        let spec = detect(&sd).unwrap();
        let graph = build(&spec, &sd).unwrap();
        let input = rng.tensor(3, 3, 4);
        let out = run_graph(&graph, &input);
        let s = 1u32 << n.log2_scale;
        assert_eq!((out.c, out.h, out.w), (3, 3 * s, 4 * s));
        for y in 0..out.h {
            for x in 0..out.w {
                let px: Vec<f32> = (0..3).map(|c| input.get(c, y / s, x / s)).collect();
                let want = rrdbnet_pixel(&sd, &n, &px);
                let got: Vec<f32> = (0..3).map(|c| out.get(c, y, x)).collect();
                assert_close_rel(&got, &want, 1e-5);
            }
        }
    }

    #[test]
    fn esrgan_fixture_graph_layer_counts() {
        let sd = StateDict::load(&fixture("tiny_esrgan.pth")).unwrap();
        let spec = detect(&sd).unwrap();
        let graph = build(&spec, &sd).unwrap();
        let (nb, log2_scale) = (2, 2);
        let count = |f: fn(&LayerKind) -> bool| graph.layers.iter().filter(|l| f(&l.kind)).count();
        // model.0, 5 per dense block, the trunk conv, one per upsampling stage, HR and last.
        let convs = 1 + nb * 3 * 5 + 1 + log2_scale + 2;
        assert_eq!(count(|k| matches!(k, LayerKind::Conv { .. })), convs);
        // Every 4-d tensor of the file is one convolution.
        assert_eq!(
            sd.tensors.values().filter(|t| t.shape.len() == 4).count(),
            convs
        );
        assert_eq!(count(|k| matches!(k, LayerKind::Concat)), nb * 3 * 4);
        assert_eq!(
            count(|k| matches!(k, LayerKind::InterpNearest { scale: 2 })),
            log2_scale
        );
        assert_eq!(count(|k| matches!(k, LayerKind::PixelUnshuffle { .. })), 0);
        assert_eq!(infer_scale(&graph).unwrap(), 4);
    }

    #[test]
    fn esrgan_matches_rrdb_forward_pointwise() {
        for (log2_scale, seed) in [(2, 1), (1, 2), (0, 3)] {
            let n = Rrdb {
                in_nc: 3,
                nf: 6,
                nb: 2,
                gc: 4,
                log2_scale,
                plus: false,
            };
            assert_matches_forward(n, seed);
        }
    }

    #[test]
    fn esrgan_plus_matches_rrdb_forward_pointwise() {
        let n = Rrdb {
            in_nc: 3,
            nf: 6,
            nb: 2,
            gc: 4,
            log2_scale: 2,
            plus: true,
        };
        assert_matches_forward(n, 4);
    }

    /// Review focus 4: a 2× model built on a 4× network (RealESRGAN_x2plus) and a 1× one.
    #[test]
    fn esrgan_unshuffle_reports_external_scale_and_needs_multiples() {
        for (in_nc, factor, external) in [(12, 2, 2), (48, 4, 1)] {
            let n = Rrdb {
                in_nc,
                nf: 6,
                nb: 1,
                gc: 4,
                log2_scale: 2,
                plus: false,
            };
            let mut rng = Rng(in_nc as u32);
            let sd = random_dict(&entries(&n), &mut rng, 0.3, false);
            let spec = detect(&sd).unwrap();
            assert_eq!(
                (spec.external_scale(), spec.input_multiple_of()),
                (external, factor)
            );
            let graph = build(&spec, &sd).unwrap();
            assert_eq!(graph.layers[1].kind, LayerKind::PixelUnshuffle { factor });
            assert_eq!(infer_scale(&graph).unwrap(), external);
            let (h, w) = (2 * factor, 3 * factor);
            let p = plan(&graph, 3, h, w).unwrap();
            assert_eq!(
                p.blob_shapes[graph.output_blob],
                (3, h * external, w * external)
            );
            let out = run_graph(&graph, &rng.tensor(3, h, w));
            assert_eq!((out.h, out.w), (h * external, w * external));
            let err = plan(&graph, 3, h + 1, w).unwrap_err().to_string();
            assert!(err.contains(&format!("multiple of {factor}")), "{err}");
        }
    }

    #[test]
    fn esrgan_new_arch_builds_the_same_graph() {
        let old = StateDict::load(&fixture("tiny_esrgan.pth")).unwrap();
        let mut new = StateDict::default();
        for (key, t) in &old.tensors {
            let renamed = match key.as_str() {
                "model.0.weight" | "model.0.bias" => key.replace("model.0", "conv_first"),
                k if k.starts_with("model.1.sub.2.") => key.replace("model.1.sub.2", "conv_body"),
                k if k.starts_with("model.1.sub.") => {
                    // model.1.sub.B.RDBr.convc.0.w -> body.B.rdbr.convc.w
                    key.replace("model.1.sub.", "body.")
                        .replace(".RDB", ".rdb")
                        .replace(".0.weight", ".weight")
                        .replace(".0.bias", ".bias")
                }
                k if k.starts_with("model.3.") => key.replace("model.3", "conv_up1"),
                k if k.starts_with("model.6.") => key.replace("model.6", "conv_up2"),
                k if k.starts_with("model.8.") => key.replace("model.8", "conv_hr"),
                k if k.starts_with("model.10.") => key.replace("model.10", "conv_last"),
                other => panic!("unexpected key {other}"),
            };
            new.tensors.insert(renamed, t.clone());
        }
        assert!(new.has("body.1.rdb3.conv5.weight"));
        let graph = |sd: &StateDict| build(&detect(sd).unwrap(), sd).unwrap();
        assert_eq!(graph(&new), graph(&old));
    }

    #[test]
    fn esrgan_plus_without_its_shortcut_names_the_key() {
        let n = Rrdb {
            in_nc: 3,
            nf: 6,
            nb: 1,
            gc: 4,
            log2_scale: 2,
            plus: true,
        };
        let mut sd = random_dict(&entries(&n), &mut Rng(5), 0.3, false);
        let spec = detect(&sd).unwrap();
        sd.tensors.remove("model.1.sub.0.RDB2.conv1x1.weight");
        let err = build(&spec, &sd).unwrap_err().to_string();
        assert!(err.contains("model.1.sub.0.RDB2.conv1x1.weight"), "{err}");
    }
}
