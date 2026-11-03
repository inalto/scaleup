//! Runs a [`Graph`]: shape inference and buffer liveness up front, then one pass over the layers.

use std::sync::Mutex;

use scaleup_core::{Image32, Result, ScaleUpError};
use wgpu::util::DeviceExt;

use crate::kernels::{
    Buf, ConvOp, Kernels, Recorder, create_tensor_buffer, read_back, tensor_bytes,
};
use crate::ncnn::{BinOp, Graph, LayerKind};

/// One operation of a run: an index into `Graph::layers`, skipping `Input` and `Split`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub layer: usize,
}

/// Everything about a run that follows from the graph and the input size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub steps: Vec<Step>,
    /// (c, h, w) of every blob.
    pub blob_shapes: Vec<(u32, u32, u32)>,
    /// The blob each blob's data really lives in: `Split` outputs share their input's storage.
    pub root: Vec<usize>,
    /// Per root blob: the last step that reads it (`usize::MAX` for the network output).
    pub blob_last_use: Vec<usize>,
    /// Per root blob: its buffer slot.
    pub blob_slot: Vec<usize>,
    /// Bytes of each buffer slot.
    pub slot_sizes: Vec<u64>,
}

impl Plan {
    /// The largest single buffer this plan needs.
    pub fn max_slot_bytes(&self) -> u64 {
        self.slot_sizes.iter().copied().max().unwrap_or(0)
    }
}

/// Shape of a layer's output, given its input shapes.
fn output_shape(
    graph: &Graph,
    index: usize,
    shapes: &[(u32, u32, u32)],
) -> Result<(u32, u32, u32)> {
    let layer = &graph.layers[index];
    let err = |what: &str| ScaleUpError::Model(format!("{what} at layer {}", layer.name));
    let &first = layer
        .inputs
        .first()
        .ok_or_else(|| err("layer without input"))?;
    let (c, h, w) = shapes[first];
    let same_shapes = || layer.inputs.iter().all(|&b| shapes[b] == (c, h, w));
    Ok(match &layer.kind {
        LayerKind::Conv { out, .. } => (*out, h, w),
        LayerKind::Concat => {
            if !layer
                .inputs
                .iter()
                .all(|&b| (shapes[b].1, shapes[b].2) == (h, w))
            {
                return Err(err("Concat over different sizes"));
            }
            (layer.inputs.iter().map(|&b| shapes[b].0).sum(), h, w)
        }
        LayerKind::Eltwise { .. } => {
            if layer.inputs.len() != 2 {
                return Err(err("Eltwise over other than two inputs is unsupported"));
            }
            if !same_shapes() {
                return Err(err("Eltwise over different shapes"));
            }
            (c, h, w)
        }
        LayerKind::BinaryOp { .. } => {
            if !same_shapes() {
                return Err(err("BinaryOp over different shapes"));
            }
            (c, h, w)
        }
        LayerKind::LeakyRelu { .. } | LayerKind::PRelu { .. } => (c, h, w),
        LayerKind::InterpNearest { scale } => (c, h * scale, w * scale),
        LayerKind::PixelShuffle { factor } => {
            let r2 = factor * factor;
            if r2 == 0 || !c.is_multiple_of(r2) {
                return Err(err("PixelShuffle channel count"));
            }
            (c / r2, h * factor, w * factor)
        }
        LayerKind::Input | LayerKind::Split => (c, h, w),
    })
}

/// Hands out buffer slots: the tightest free slot that fits, else the largest free one grown to fit,
/// else a new one.
#[derive(Default)]
struct Slots {
    sizes: Vec<u64>,
    free: Vec<usize>,
}

impl Slots {
    fn take(&mut self, bytes: u64) -> usize {
        let fitting = self
            .free
            .iter()
            .enumerate()
            .filter(|&(_, &s)| self.sizes[s] >= bytes)
            .min_by_key(|&(_, &s)| self.sizes[s])
            .map(|(i, _)| i);
        let largest = || {
            self.free
                .iter()
                .enumerate()
                .max_by_key(|&(_, &s)| self.sizes[s])
                .map(|(i, _)| i)
        };
        if let Some(i) = fitting.or_else(largest) {
            let slot = self.free.swap_remove(i);
            self.sizes[slot] = self.sizes[slot].max(bytes);
            slot
        } else {
            self.sizes.push(bytes);
            self.sizes.len() - 1
        }
    }
}

/// Infers shapes and assigns buffers. A blob keeps its buffer from the step that writes it through
/// the last step that reads it, so a step's output never shares a buffer with its inputs.
pub fn plan(graph: &Graph, in_c: u32, in_h: u32, in_w: u32) -> Result<Plan> {
    let n = graph.blob_count;
    let mut shapes = vec![(0, 0, 0); n];
    let mut root: Vec<usize> = (0..n).collect();
    let mut steps = Vec::new();
    let mut last_use = vec![0usize; n];
    let mut birth_step = vec![None; n];

    for (index, layer) in graph.layers.iter().enumerate() {
        match layer.kind {
            LayerKind::Input => {
                for &o in &layer.outputs {
                    shapes[o] = (in_c, in_h, in_w);
                    birth_step[o] = Some(0);
                }
            }
            LayerKind::Split => {
                let &src = layer.inputs.first().ok_or_else(|| {
                    ScaleUpError::Model(format!("Split without input at layer {}", layer.name))
                })?;
                for &o in &layer.outputs {
                    shapes[o] = shapes[src];
                    root[o] = root[src];
                }
            }
            _ => {
                let shape = output_shape(graph, index, &shapes)?;
                let step = steps.len();
                for &i in &layer.inputs {
                    last_use[root[i]] = step;
                }
                for &o in &layer.outputs {
                    shapes[o] = shape;
                    last_use[o] = step;
                    birth_step[o] = Some(step);
                }
                steps.push(Step { layer: index });
            }
        }
    }
    last_use[root[graph.output_blob]] = usize::MAX;

    let bytes = |b: usize| {
        let (c, h, w) = shapes[b];
        tensor_bytes(c, h, w)
    };
    let mut slots = Slots::default();
    let mut blob_slot = vec![usize::MAX; n];
    let mut active: Vec<usize> = Vec::new();
    let input = root[graph.input_blob];
    blob_slot[input] = slots.take(bytes(input));
    active.push(input);
    for (i, step) in steps.iter().enumerate() {
        active.retain(|&b| {
            let keep = last_use[b] >= i;
            if !keep {
                slots.free.push(blob_slot[b]);
            }
            keep
        });
        for &o in &graph.layers[step.layer].outputs {
            blob_slot[o] = slots.take(bytes(o));
            active.push(o);
        }
    }
    Ok(Plan {
        steps,
        blob_shapes: shapes,
        root,
        blob_last_use: last_use,
        blob_slot,
        slot_sizes: slots.sizes,
    })
}

/// A layer's weights on the GPU.
struct LayerBuffers {
    a: wgpu::Buffer,
    b: Option<wgpu::Buffer>,
}

/// Runs plans on one device. Runs are serialised: they share the buffer pool, and the GPU would
/// interleave them anyway.
pub struct Executor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Kernels,
    weights: Vec<Option<LayerBuffers>>,
    pool: Mutex<Vec<wgpu::Buffer>>,
}

impl Executor {
    /// Uploads the weights of `graph`.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, graph: &Graph) -> Result<Self> {
        let kernels = Kernels::new(&device, &device.limits());
        let upload = |data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weights"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let weights = graph
            .layers
            .iter()
            .map(|l| match &l.kind {
                LayerKind::Conv {
                    weights, biases, ..
                } => Some(LayerBuffers {
                    a: upload(weights),
                    b: (!biases.is_empty()).then(|| upload(biases)),
                }),
                LayerKind::PRelu { slopes } => Some(LayerBuffers {
                    a: upload(slopes),
                    b: None,
                }),
                _ => None,
            })
            .collect();
        Ok(Self {
            device,
            queue,
            kernels,
            weights,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// The buffers for `plan`, reusing pooled ones. Pooled buffers this plan does not use are dropped.
    fn acquire(&self, pool: &mut Vec<wgpu::Buffer>, plan: &Plan) -> Vec<wgpu::Buffer> {
        plan.slot_sizes
            .iter()
            .map(|&size| {
                let fitting = pool
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.size() >= size)
                    .min_by_key(|(_, b)| b.size())
                    .map(|(i, _)| i);
                match fitting {
                    Some(i) => pool.swap_remove(i),
                    None => create_tensor_buffer(&self.device, size),
                }
            })
            .collect()
    }

    fn record_step(
        &self,
        rec: &mut Recorder<'_>,
        graph: &Graph,
        layer_index: usize,
        buf: &dyn Fn(usize) -> Buf,
    ) {
        let layer = &graph.layers[layer_index];
        let dst = buf(layer.outputs[0]);
        let input = |i: usize| buf(layer.inputs[i]);
        let weights = self.weights[layer_index].as_ref();
        let k = &self.kernels;
        match &layer.kind {
            LayerKind::Conv { k: size, act, .. } => {
                let w = weights.expect("convolutions have weights");
                k.conv(
                    rec,
                    &ConvOp {
                        src: &input(0),
                        dst: &dst,
                        weights: &w.a,
                        biases: w.b.as_ref(),
                        k: *size,
                        act: *act,
                    },
                );
            }
            LayerKind::Concat => {
                let inputs: Vec<Buf> = (0..layer.inputs.len()).map(input).collect();
                let refs: Vec<&Buf> = inputs.iter().collect();
                k.concat(rec, &refs, &dst);
            }
            LayerKind::Eltwise { coeffs } => {
                let c = |i: usize| coeffs.get(i).copied().unwrap_or(1.0);
                k.weighted_sum(rec, &input(0), &input(1), &dst, c(0), c(1));
            }
            LayerKind::BinaryOp { op, scalar } => match (op, scalar) {
                (BinOp::Add, Some(s)) => k.scalar(rec, &input(0), &dst, true, *s),
                (BinOp::Mul, Some(s)) => k.scalar(rec, &input(0), &dst, false, *s),
                (BinOp::Add, None) => k.weighted_sum(rec, &input(0), &input(1), &dst, 1.0, 1.0),
                (BinOp::Mul, None) => k.mul(rec, &input(0), &input(1), &dst),
            },
            LayerKind::LeakyRelu { slope } => k.leaky(rec, &input(0), &dst, *slope),
            LayerKind::PRelu { slopes } => {
                let w = weights.expect("PReLU has slopes");
                k.prelu(rec, &input(0), &dst, &w.a, slopes.len() as u32);
            }
            LayerKind::InterpNearest { scale } => k.interp(rec, &input(0), &dst, *scale),
            LayerKind::PixelShuffle { factor } => k.pixel_shuffle(rec, &input(0), &dst, *factor),
            LayerKind::Input | LayerKind::Split => unreachable!("not planned as steps"),
        }
    }

    pub fn run(&self, graph: &Graph, plan: &Plan, input: &Image32) -> Result<Image32> {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let slots = self.acquire(&mut pool, plan);
        let buf = |blob: usize| {
            let (c, h, w) = plan.blob_shapes[blob];
            Buf {
                buffer: slots[plan.blob_slot[plan.root[blob]]].clone(),
                c,
                h,
                w,
            }
        };
        self.queue.write_buffer(
            &slots[plan.blob_slot[plan.root[graph.input_blob]]],
            0,
            bytemuck::cast_slice(&input.data),
        );

        let mut rec = Recorder::new(&self.kernels, &self.device, &self.queue);
        for step in &plan.steps {
            self.record_step(&mut rec, graph, step.layer, &buf);
        }
        rec.flush();

        let (c, h, w) = plan.blob_shapes[graph.output_blob];
        let output = buf(graph.output_blob);
        let data = read_back(
            &self.device,
            &self.queue,
            &output.buffer,
            (c * h * w) as usize,
        )?;
        drop(output);
        if let Some(e) = pollster::block_on(memory.pop()) {
            pool.clear();
            return Err(ScaleUpError::Backend(format!(
                "out of GPU memory for a {}x{} input, try a smaller tile size: {e}",
                input.width, input.height
            )));
        }
        pool.clear();
        pool.extend(slots);
        Ok(Image32 {
            width: w,
            height: h,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_ref::{self, Tensor};
    use crate::kernels::test_gpu::{Rng, gpu};
    use crate::ncnn::{Activation, BinOp, Layer, LayerKind};

    fn layer(name: &str, kind: LayerKind, inputs: &[usize], outputs: &[usize]) -> Layer {
        Layer {
            name: name.into(),
            kind,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        }
    }

    fn conv_kind(rng: &mut Rng, c_in: u32, out: u32, k: u32, act: Activation) -> LayerKind {
        LayerKind::Conv {
            out,
            k,
            pad: k / 2,
            bias: true,
            act,
            weights: rng.vec((out * c_in * k * k) as usize),
            biases: rng.vec(out as usize),
        }
    }

    /// Input -> Split -> (Conv, Conv) -> Eltwise.
    fn split_graph(rng: &mut Rng) -> Graph {
        Graph {
            layers: vec![
                layer("in", LayerKind::Input, &[], &[0]),
                layer("split", LayerKind::Split, &[0], &[1, 2]),
                layer("c1", conv_kind(rng, 3, 3, 3, Activation::None), &[1], &[3]),
                layer("c2", conv_kind(rng, 3, 3, 3, Activation::None), &[2], &[4]),
                layer(
                    "sum",
                    LayerKind::Eltwise {
                        coeffs: vec![0.2, 1.0],
                    },
                    &[3, 4],
                    &[5],
                ),
            ],
            blob_count: 6,
            input_blob: 0,
            output_blob: 5,
        }
    }

    /// Every supported kind at least once, ending in 3 channels at 4x.
    fn mixed_graph(rng: &mut Rng) -> Graph {
        Graph {
            layers: vec![
                layer("in", LayerKind::Input, &[], &[0]),
                layer(
                    "c0",
                    conv_kind(rng, 3, 8, 3, Activation::Leaky(0.2)),
                    &[0],
                    &[1],
                ),
                layer("split", LayerKind::Split, &[1], &[2, 3, 4]),
                layer("c1", conv_kind(rng, 8, 8, 3, Activation::Relu), &[2], &[5]),
                layer("r", LayerKind::LeakyRelu { slope: 0.1 }, &[3], &[6]),
                layer("e", LayerKind::Eltwise { coeffs: vec![] }, &[5, 6], &[7]),
                layer("p", LayerKind::PRelu { slopes: rng.vec(8) }, &[7], &[8]),
                layer("cat", LayerKind::Concat, &[8, 4], &[9]),
                layer(
                    "c2",
                    conv_kind(rng, 16, 12, 1, Activation::None),
                    &[9],
                    &[10],
                ),
                layer(
                    "m",
                    LayerKind::BinaryOp {
                        op: BinOp::Mul,
                        scalar: Some(0.5),
                    },
                    &[10],
                    &[11],
                ),
                layer("up", LayerKind::InterpNearest { scale: 2 }, &[11], &[12]),
                layer("split2", LayerKind::Split, &[12], &[13, 14]),
                layer(
                    "add",
                    LayerKind::BinaryOp {
                        op: BinOp::Add,
                        scalar: None,
                    },
                    &[13, 14],
                    &[15],
                ),
                layer("ps", LayerKind::PixelShuffle { factor: 2 }, &[15], &[16]),
            ],
            blob_count: 17,
            input_blob: 0,
            output_blob: 16,
        }
    }

    /// No two blobs whose lifetimes overlap may share a buffer, and every slot fits its blobs.
    fn assert_slots_sound(graph: &Graph, p: &Plan) {
        let mut birth = vec![None; graph.blob_count];
        birth[p.root[graph.input_blob]] = Some(0usize);
        for (i, step) in p.steps.iter().enumerate() {
            for &o in &graph.layers[step.layer].outputs {
                birth[p.root[o]] = Some(i);
            }
        }
        let live: Vec<usize> = (0..graph.blob_count).filter(|&b| p.root[b] == b).collect();
        for &a in &live {
            let (c, h, w) = p.blob_shapes[a];
            assert!(
                u64::from(c * h * w) * 4 <= p.slot_sizes[p.blob_slot[a]],
                "blob {a} too big"
            );
            for &b in &live {
                if a < b && p.blob_slot[a] == p.blob_slot[b] {
                    let (ba, bb) = (birth[a].unwrap(), birth[b].unwrap());
                    // Blob `x` holds its buffer from its birth through its last use, inclusive.
                    let apart = p.blob_last_use[a] < bb || p.blob_last_use[b] < ba;
                    assert!(
                        apart,
                        "blobs {a} and {b} overlap in slot {}",
                        p.blob_slot[a]
                    );
                }
            }
        }
    }

    #[test]
    fn plan_shapes_for_rrdb_block() {
        let mut rng = Rng(1);
        let g = Graph {
            layers: vec![
                layer("in", LayerKind::Input, &[], &[0]),
                layer(
                    "c",
                    conv_kind(&mut rng, 3, 64, 3, Activation::None),
                    &[0],
                    &[1],
                ),
                layer("up", LayerKind::InterpNearest { scale: 2 }, &[1], &[2]),
                layer("ps", LayerKind::PixelShuffle { factor: 4 }, &[2], &[3]),
            ],
            blob_count: 4,
            input_blob: 0,
            output_blob: 3,
        };
        let p = plan(&g, 3, 10, 20).unwrap();
        assert_eq!(
            p.blob_shapes,
            [(3, 10, 20), (64, 10, 20), (64, 20, 40), (4, 80, 160)]
        );
        assert_eq!(p.steps.len(), 3);
        assert_eq!(p.blob_last_use[p.root[3]], usize::MAX);
        assert_slots_sound(&g, &p);
    }

    #[test]
    fn split_keeps_buffer_alive_until_last_consumer() {
        let mut rng = Rng(2);
        let g = split_graph(&mut rng);
        let p = plan(&g, 3, 4, 4).unwrap();
        // Both Split outputs live in the input's buffer.
        assert_eq!(p.root[1], 0);
        assert_eq!(p.root[2], 0);
        // Its last reader is the second convolution, the second step.
        assert_eq!(p.blob_last_use[0], 1);
        // So the first convolution's output must not get the input's buffer.
        assert_ne!(p.blob_slot[3], p.blob_slot[0]);
        assert_slots_sound(&g, &p);
    }

    #[test]
    fn freed_buffers_are_reused() {
        let mut rng = Rng(3);
        let mut layers = vec![layer("in", LayerKind::Input, &[], &[0])];
        for i in 0..20 {
            layers.push(layer(
                "c",
                conv_kind(&mut rng, 3, 3, 1, Activation::None),
                &[i],
                &[i + 1],
            ));
        }
        let g = Graph {
            layers,
            blob_count: 21,
            input_blob: 0,
            output_blob: 20,
        };
        let p = plan(&g, 3, 8, 8).unwrap();
        assert!(p.slot_sizes.len() <= 2, "{} slots", p.slot_sizes.len());
        assert_slots_sound(&g, &p);
    }

    #[test]
    fn real_models_plan_soundly() {
        let dir = std::path::Path::new("F:/Apps/ScaleUP/Upscayl/resources/models");
        if !dir.is_dir() {
            return;
        }
        for name in ["upscayl-lite-4x", "digital-art-4x", "remacri-4x"] {
            let g = crate::ncnn::load_graph(&dir.join(format!("{name}.param"))).unwrap();
            let p = plan(&g, 3, 64, 48).unwrap();
            assert_eq!(p.blob_shapes[g.output_blob], (3, 256, 192), "{name}");
            assert_slots_sound(&g, &p);
        }
    }

    #[test]
    fn eltwise_shape_mismatch_is_an_error_naming_the_layer() {
        let g = Graph {
            layers: vec![
                layer("in", LayerKind::Input, &[], &[0]),
                layer("up", LayerKind::InterpNearest { scale: 2 }, &[0], &[1]),
                layer(
                    "bad_sum",
                    LayerKind::Eltwise { coeffs: vec![] },
                    &[0, 1],
                    &[2],
                ),
            ],
            blob_count: 3,
            input_blob: 0,
            output_blob: 2,
        };
        let err = plan(&g, 3, 4, 4).unwrap_err().to_string();
        assert!(err.contains("bad_sum"), "{err}");
    }

    #[test]
    fn cpu_ref_graph_runs_split_graph() {
        let mut rng = Rng(5);
        let g = split_graph(&mut rng);
        let input = rng.tensor(3, 5, 6);
        let out = cpu_ref::run_graph(&g, &input);
        assert_eq!((out.c, out.h, out.w), (3, 5, 6));
    }

    fn to_image(t: &Tensor) -> Image32 {
        Image32 {
            width: t.w,
            height: t.h,
            data: t.data.clone(),
        }
    }

    fn assert_executor_matches(g: &Graph, input: &Tensor) {
        let Some(gpu) = gpu() else { return };
        let exec = Executor::new(gpu.device.clone(), gpu.queue.clone(), g).unwrap();
        let p = plan(g, 3, input.h, input.w).unwrap();
        let want = cpu_ref::run_graph(g, input);
        let got = exec.run(g, &p, &to_image(input)).unwrap();
        assert_eq!((got.width, got.height), (want.w, want.h));
        let worst = got
            .data
            .iter()
            .zip(&want.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "max abs diff {worst}");
    }

    #[test]
    fn executor_matches_cpu_ref_on_split_graph() {
        let mut rng = Rng(6);
        let g = split_graph(&mut rng);
        let input = rng.tensor(3, 9, 13);
        assert_executor_matches(&g, &input);
    }

    #[test]
    fn executor_matches_cpu_ref_on_random_graph() {
        let mut rng = Rng(7);
        let g = mixed_graph(&mut rng);
        let input = rng.tensor(3, 13, 9);
        assert_executor_matches(&g, &input);
        // Same executor shape again with other sizes: pooled buffers must not leak state.
        let input = rng.tensor(3, 5, 7);
        assert_executor_matches(&g, &input);
    }
}
