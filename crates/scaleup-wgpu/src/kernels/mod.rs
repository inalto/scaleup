//! Compute kernels and the recorder that batches their dispatches.
//!
//! Every kernel uses one bind group shape: `src`, `dst`, two auxiliary read-only buffers and a
//! dynamically offset uniform block. Long work is cut into bands and separate submissions, so a large
//! tile cannot trip the operating system's GPU watchdog.

use bytemuck::{Pod, Zeroable};
use scaleup_core::{Result, ScaleUpError};

use crate::cpu_ref::Tensor;
use crate::ncnn::Activation;

/// Multiply-adds in one convolution dispatch.
const DISPATCH_WORK: u64 = 400_000_000;
/// Multiply-adds between queue submissions.
const SUBMIT_WORK: u64 = 1_600_000_000;
const UNIFORM_ARENA: u64 = 1 << 20;
/// Output channels written per convolution pass.
const CHANNELS_PER_PASS: u32 = 4;

/// A tensor on the GPU: f32, planar CHW.
#[derive(Clone)]
pub struct Buf {
    pub buffer: wgpu::Buffer,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

/// The uniform block shared by all kernels.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Params {
    mode: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    k: u32,
    act: u32,
    c0: u32,
    y0: u32,
    y1: u32,
    factor: u32,
    has_bias: u32,
    slope: f32,
    coeff_a: f32,
    coeff_b: f32,
    _pad: u32,
}

const ACT_NONE: u32 = 0;
const ACT_RELU: u32 = 1;
const ACT_LEAKY: u32 = 2;

const MODE_LEAKY: u32 = 0;
const MODE_PRELU: u32 = 1;
const MODE_SIGMOID: u32 = 2;
const MODE_SILU: u32 = 3;
const MODE_MISH: u32 = 4;
const MODE_WEIGHTED_SUM: u32 = 0;
const MODE_MUL: u32 = 1;
const MODE_SCALAR_ADD: u32 = 2;
const MODE_SCALAR_MUL: u32 = 3;
const MODE_NORMALIZE: u32 = 4;
const MODE_LAYER_NORM: u32 = 0;
const MODE_GROUP_NORM: u32 = 1;

/// Element-wise activations without a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Smooth {
    Sigmoid,
    Silu,
    Mish,
}

/// Workgroup memory the fast convolution needs, as the plan's gate states it.
const TILED_WORKGROUP_STORAGE: u32 = 16_384;

/// Whether this device can use the shared-memory convolution.
pub fn use_tiled(limits: &wgpu::Limits) -> bool {
    limits.max_compute_workgroup_storage_size >= TILED_WORKGROUP_STORAGE
}

/// `[out][in][3][3]` scalars rearranged to `[block][in][3][3]` vec4s, four outputs per block; missing
/// outputs of the last block are zero.
pub fn pack_weights(weights: &[f32], out: u32, c_in: u32) -> Vec<[f32; 4]> {
    let cells = (c_in * 9) as usize;
    let mut packed = vec![[0.0; 4]; out.div_ceil(4) as usize * cells];
    for o in 0..out as usize {
        for cell in 0..cells {
            packed[o / 4 * cells + cell][o % 4] = weights[o * cells + cell];
        }
    }
    packed
}

/// Convolution description; `weights` are `[out][in][k][k]`.
pub struct ConvOp<'a> {
    pub src: &'a Buf,
    pub dst: &'a Buf,
    pub weights: &'a wgpu::Buffer,
    /// `None` when the layer has no bias.
    pub biases: Option<&'a wgpu::Buffer>,
    pub k: u32,
    pub act: Activation,
    /// [`pack_weights`] of `weights`; with it, a capable device uses the fast kernel.
    pub packed: Option<&'a wgpu::Buffer>,
}

/// Records dispatches, submitting whenever enough work has piled up.
pub struct Recorder<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: Option<wgpu::CommandEncoder>,
    uniforms: Vec<u8>,
    work: u64,
    stride: u64,
    /// Uniforms of this recorder alone: recorders on other threads must not share it.
    arena: wgpu::Buffer,
}

impl<'a> Recorder<'a> {
    pub fn new(k: &'a Kernels, device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            encoder: None,
            uniforms: Vec::new(),
            work: 0,
            stride: k.uniform_stride,
            arena: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniforms"),
                size: UNIFORM_ARENA,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        let device = self.device;
        self.encoder
            .get_or_insert_with(|| device.create_command_encoder(&Default::default()))
    }

    fn push_uniform(&mut self, p: &Params) -> u32 {
        if self.uniforms.len() as u64 + self.stride > UNIFORM_ARENA {
            self.flush();
        }
        let offset = self.uniforms.len();
        self.uniforms.extend_from_slice(bytemuck::bytes_of(p));
        self.uniforms.resize(offset + self.stride as usize, 0);
        offset as u32
    }

    fn add_work(&mut self, work: u64) {
        self.work += work;
        if self.work >= SUBMIT_WORK {
            self.flush();
        }
    }

    /// Submits what was recorded and waits for it, which also bounds how much is queued.
    pub fn flush(&mut self) {
        if !self.uniforms.is_empty() {
            self.queue.write_buffer(&self.arena, 0, &self.uniforms);
            self.uniforms.clear();
        }
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit([encoder.finish()]);
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.work = 0;
    }
}

/// The compiled kernels of one device.
pub struct Kernels {
    layout: wgpu::BindGroupLayout,
    conv: wgpu::ComputePipeline,
    activation: wgpu::ComputePipeline,
    eltwise: wgpu::ComputePipeline,
    interp: wgpu::ComputePipeline,
    shuffle: wgpu::ComputePipeline,
    unshuffle: wgpu::ComputePipeline,
    norm: wgpu::ComputePipeline,
    group_stats: wgpu::ComputePipeline,
    /// The shared-memory convolution, on devices with the workgroup storage for it.
    tiled: Option<wgpu::ComputePipeline>,
    /// Bound where a kernel does not use an auxiliary buffer.
    dummy: wgpu::Buffer,
    uniform_stride: u64,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Kernels {
    pub fn new(device: &wgpu::Device, limits: &wgpu::Limits) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kernel"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size_of::<Params>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kernel"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label: &str, source: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    // No kernel reads workgroup memory before writing it, and FXC takes a minute
                    // over the generated initialisation of the tiled kernel's cache.
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let stride = (size_of::<Params>() as u64)
            .next_multiple_of(u64::from(limits.min_uniform_buffer_offset_alignment));
        Self {
            conv: pipeline("conv", include_str!("conv.wgsl")),
            activation: pipeline("activation", include_str!("activation.wgsl")),
            eltwise: pipeline("eltwise", include_str!("eltwise.wgsl")),
            interp: pipeline("interp", include_str!("interp_nearest.wgsl")),
            shuffle: pipeline("shuffle", include_str!("pixel_shuffle.wgsl")),
            unshuffle: pipeline("unshuffle", include_str!("pixel_unshuffle.wgsl")),
            norm: pipeline("norm", include_str!("layernorm_c.wgsl")),
            group_stats: pipeline("group stats", include_str!("group_stats.wgsl")),
            tiled: use_tiled(limits)
                .then(|| {
                    // Shader compilers differ between drivers. The direct kernel gives the same
                    // result, so a refusal costs speed, not the run.
                    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
                    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
                    let compiled = pipeline("conv tiled", include_str!("conv3x3_tiled.wgsl"));
                    let invalid = pollster::block_on(validation.pop());
                    let failed = pollster::block_on(internal.pop());
                    match invalid.or(failed) {
                        None => Some(compiled),
                        Some(e) => {
                            tracing::warn!("using the direct convolution kernel: {e}");
                            None
                        }
                    }
                })
                .flatten(),
            dummy: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("unused"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            uniform_stride: stride,
            layout,
        }
    }

    fn bind(
        &self,
        rec: &Recorder<'_>,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        a: Option<&wgpu::Buffer>,
        b: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        rec.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a.unwrap_or(&self.dummy).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b.unwrap_or(&self.dummy).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &rec.arena,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<Params>() as u64),
                    }),
                },
            ],
        })
    }

    fn dispatch(
        &self,
        rec: &mut Recorder<'_>,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        params: &Params,
        groups: [u32; 3],
        work: u64,
    ) {
        let offset = rec.push_uniform(params);
        {
            let mut pass = rec.encoder().begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[offset]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        rec.add_work(work);
    }

    /// A pixel-wise kernel over all channels of `dst`.
    fn per_pixel(
        &self,
        rec: &mut Recorder<'_>,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        params: &Params,
        dst: &Buf,
    ) {
        let groups = [dst.w.div_ceil(8), dst.h.div_ceil(8), dst.c];
        let work = u64::from(dst.w) * u64::from(dst.h) * u64::from(dst.c);
        self.dispatch(rec, pipeline, bind_group, params, groups, work);
    }

    /// Whether [`Kernels::conv`] will use [`ConvOp::packed`] for 3x3 convolutions.
    pub fn has_tiled(&self) -> bool {
        self.tiled.is_some()
    }

    pub fn conv(&self, rec: &mut Recorder<'_>, op: &ConvOp<'_>) {
        let (src, dst) = (op.src, op.dst);
        let fast = match (&self.tiled, op.packed) {
            (Some(pipeline), Some(packed)) if op.k == 3 => Some((pipeline, packed)),
            _ => None,
        };
        let (pipeline, weights, side) = match fast {
            Some((pipeline, packed)) => (pipeline, packed, 16),
            None => (&self.conv, op.weights, 8),
        };
        let bg = self.bind(rec, &src.buffer, &dst.buffer, Some(weights), op.biases);
        let (act, slope) = match op.act {
            Activation::None => (ACT_NONE, 0.0),
            Activation::Relu => (ACT_RELU, 0.0),
            Activation::Leaky(s) => (ACT_LEAKY, s),
        };
        let macs_per_row = u64::from(dst.w)
            * u64::from(src.c)
            * u64::from(CHANNELS_PER_PASS)
            * u64::from(op.k * op.k);
        let band = ((DISPATCH_WORK / macs_per_row.max(1)) as u32)
            .clamp(side, dst.h.max(side))
            .next_multiple_of(side);
        for c0 in (0..dst.c).step_by(CHANNELS_PER_PASS as usize) {
            for y0 in (0..dst.h).step_by(band as usize) {
                let y1 = (y0 + band).min(dst.h);
                let params = Params {
                    c_in: src.c,
                    c_out: dst.c,
                    h: dst.h,
                    w: dst.w,
                    k: op.k,
                    act,
                    slope,
                    c0,
                    y0,
                    y1,
                    has_bias: u32::from(op.biases.is_some()),
                    ..Params::default()
                };
                let groups = [dst.w.div_ceil(side), (y1 - y0).div_ceil(side), 1];
                let work = macs_per_row * u64::from(y1 - y0);
                self.dispatch(rec, pipeline, &bg, &params, groups, work);
            }
        }
    }

    pub fn leaky(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, slope: f32) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, None, None);
        let params = Params {
            mode: MODE_LEAKY,
            h: dst.h,
            w: dst.w,
            slope,
            ..Params::default()
        };
        self.per_pixel(rec, &self.activation, &bg, &params, dst);
    }

    /// `slopes` holds `slope_count` values: one per channel, or one for all.
    pub fn prelu(
        &self,
        rec: &mut Recorder<'_>,
        src: &Buf,
        dst: &Buf,
        slopes: &wgpu::Buffer,
        slope_count: u32,
    ) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, Some(slopes), None);
        let params = Params {
            mode: MODE_PRELU,
            h: dst.h,
            w: dst.w,
            factor: slope_count,
            ..Params::default()
        };
        self.per_pixel(rec, &self.activation, &bg, &params, dst);
    }

    pub fn smooth(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, f: Smooth) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, None, None);
        let params = Params {
            mode: match f {
                Smooth::Sigmoid => MODE_SIGMOID,
                Smooth::Silu => MODE_SILU,
                Smooth::Mish => MODE_MISH,
            },
            h: dst.h,
            w: dst.w,
            ..Params::default()
        };
        self.per_pixel(rec, &self.activation, &bg, &params, dst);
    }

    /// `dst = (src - mean[c]) * range`; `mean` holds one value per channel.
    pub fn normalize(
        &self,
        rec: &mut Recorder<'_>,
        src: &Buf,
        dst: &Buf,
        mean: &wgpu::Buffer,
        range: f32,
    ) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, Some(mean), None);
        let params = Params {
            mode: MODE_NORMALIZE,
            h: dst.h,
            w: dst.w,
            coeff_a: range,
            ..Params::default()
        };
        self.per_pixel(rec, &self.eltwise, &bg, &params, dst);
    }

    /// Each pixel normalised over its channels, then scaled by `gamma` and shifted by `beta` (one
    /// value per channel each).
    pub fn layer_norm(
        &self,
        rec: &mut Recorder<'_>,
        src: &Buf,
        dst: &Buf,
        gamma: &wgpu::Buffer,
        beta: &wgpu::Buffer,
        eps: f32,
    ) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, Some(gamma), Some(beta));
        let params = Params {
            mode: MODE_LAYER_NORM,
            c_in: dst.c,
            h: dst.h,
            w: dst.w,
            slope: eps,
            ..Params::default()
        };
        // One invocation handles every channel of its pixel.
        let groups = [dst.w.div_ceil(8), dst.h.div_ceil(8), 1];
        let work = 3 * u64::from(dst.w) * u64::from(dst.h) * u64::from(dst.c);
        self.dispatch(rec, &self.norm, &bg, &params, groups, work);
    }

    /// Group normalisation of `src` into `dst`.
    pub fn group_norm(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, op: &GroupNormOp<'_>) {
        let per_group = dst.c / op.groups;
        let work = u64::from(dst.w) * u64::from(dst.h) * u64::from(dst.c);
        let bg = self.bind(rec, &src.buffer, op.stats, None, None);
        let params = Params {
            h: dst.h,
            w: dst.w,
            factor: per_group,
            slope: op.eps,
            ..Params::default()
        };
        let groups = [op.groups, 1, 1];
        self.dispatch(rec, &self.group_stats, &bg, &params, groups, 2 * work);
        let aux = (Some(op.gamma_beta), Some(op.stats));
        let bg = self.bind(rec, &src.buffer, &dst.buffer, aux.0, aux.1);
        let params = Params {
            mode: MODE_GROUP_NORM,
            c_in: dst.c,
            h: dst.h,
            w: dst.w,
            factor: per_group,
            ..Params::default()
        };
        self.per_pixel(rec, &self.norm, &bg, &params, dst);
    }

    /// `dst = coeff_a * a + coeff_b * b`.
    pub fn weighted_sum(
        &self,
        rec: &mut Recorder<'_>,
        a: &Buf,
        b: &Buf,
        dst: &Buf,
        coeff_a: f32,
        coeff_b: f32,
    ) {
        let bg = self.bind(rec, &a.buffer, &dst.buffer, Some(&b.buffer), None);
        let params = Params {
            mode: MODE_WEIGHTED_SUM,
            h: dst.h,
            w: dst.w,
            coeff_a,
            coeff_b,
            ..Params::default()
        };
        self.per_pixel(rec, &self.eltwise, &bg, &params, dst);
    }

    pub fn mul(&self, rec: &mut Recorder<'_>, a: &Buf, b: &Buf, dst: &Buf) {
        let bg = self.bind(rec, &a.buffer, &dst.buffer, Some(&b.buffer), None);
        let params = Params {
            mode: MODE_MUL,
            h: dst.h,
            w: dst.w,
            ..Params::default()
        };
        self.per_pixel(rec, &self.eltwise, &bg, &params, dst);
    }

    pub fn scalar(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, add: bool, value: f32) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, None, None);
        let params = Params {
            mode: if add {
                MODE_SCALAR_ADD
            } else {
                MODE_SCALAR_MUL
            },
            h: dst.h,
            w: dst.w,
            coeff_a: value,
            ..Params::default()
        };
        self.per_pixel(rec, &self.eltwise, &bg, &params, dst);
    }

    pub fn interp(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, scale: u32) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, None, None);
        let params = Params {
            h: dst.h,
            w: dst.w,
            factor: scale,
            c_in: src.c,
            ..Params::default()
        };
        // Source rows and columns are addressed through the source size.
        let params = Params {
            y0: src.h,
            y1: src.w,
            ..params
        };
        self.per_pixel(rec, &self.interp, &bg, &params, dst);
    }

    pub fn pixel_shuffle(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, r: u32) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, None, None);
        let params = Params {
            h: dst.h,
            w: dst.w,
            factor: r,
            y0: src.h,
            y1: src.w,
            ..Params::default()
        };
        self.per_pixel(rec, &self.shuffle, &bg, &params, dst);
    }

    /// `dst` sides are `src` sides divided by `r`.
    pub fn pixel_unshuffle(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, r: u32) {
        let bg = self.bind(rec, &src.buffer, &dst.buffer, None, None);
        let params = Params {
            h: dst.h,
            w: dst.w,
            factor: r,
            y0: src.h,
            y1: src.w,
            ..Params::default()
        };
        self.per_pixel(rec, &self.unshuffle, &bg, &params, dst);
    }

    /// Channel concatenation is a sequence of copies in planar layout.
    pub fn concat(&self, rec: &mut Recorder<'_>, inputs: &[&Buf], dst: &Buf) {
        let mut offset = 0;
        let encoder = rec.encoder();
        for src in inputs {
            let size = tensor_bytes(src.c, src.h, src.w);
            encoder.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, offset, size);
            offset += size;
        }
    }

    /// Channels `from..from + dst.c` of `src`: one copy in planar layout.
    pub fn slice(&self, rec: &mut Recorder<'_>, src: &Buf, dst: &Buf, from: u32) {
        let offset = tensor_bytes(from, src.h, src.w);
        let size = tensor_bytes(dst.c, dst.h, dst.w);
        rec.encoder()
            .copy_buffer_to_buffer(&src.buffer, offset, &dst.buffer, 0, size);
    }
}

/// Parameters of [`Kernels::group_norm`]: `groups` runs of consecutive channels.
pub struct GroupNormOp<'a> {
    pub groups: u32,
    pub eps: f32,
    /// One scale per channel, then one shift per channel.
    pub gamma_beta: &'a wgpu::Buffer,
    /// Scratch space, two values per group.
    pub stats: &'a wgpu::Buffer,
}

/// Bytes of an f32 tensor.
pub fn tensor_bytes(c: u32, h: u32, w: u32) -> u64 {
    u64::from(c) * u64::from(h) * u64::from(w) * 4
}

/// A storage buffer usable as a tensor: kernels, copies in and out.
pub fn create_tensor_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tensor"),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Copies `count` f32 values out of `buffer`.
pub fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>> {
    let size = count as u64 * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| ScaleUpError::Backend(format!("GPU wait failed: {e}")))?;
    rx.recv()
        .map_err(|e| ScaleUpError::Backend(e.to_string()))?
        .map_err(|e| ScaleUpError::Backend(format!("reading GPU results: {e}")))?;
    let view = slice
        .get_mapped_range()
        .map_err(|e| ScaleUpError::Backend(format!("reading GPU results: {e}")))?;
    let data = bytemuck::pod_collect_to_vec::<u8, f32>(&view);
    drop(view);
    staging.unmap();
    Ok(data)
}

/// Uploads a [`Tensor`] (tests and input images).
pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, t: &Tensor) -> Buf {
    let buffer = create_tensor_buffer(device, tensor_bytes(t.c, t.h, t.w));
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&t.data));
    Buf {
        buffer,
        c: t.c,
        h: t.h,
        w: t.w,
    }
}

#[cfg(test)]
pub(crate) mod test_gpu {
    use std::sync::OnceLock;

    use super::*;
    use scaleup_core::DevicePreference;

    pub struct Gpu {
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        pub kernels: Kernels,
    }

    /// The shared test device, or `None` (the test then skips) on a machine without an adapter.
    pub fn gpu() -> Option<&'static Gpu> {
        static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
        GPU.get_or_init(|| {
            let o = crate::device::open(DevicePreference::Auto).ok()?;
            let kernels = Kernels::new(&o.device, &o.limits);
            Some(Gpu {
                device: o.device,
                queue: o.queue,
                kernels,
            })
        })
        .as_ref()
    }

    /// Small deterministic generator, values in -1..1.
    pub struct Rng(pub u32);

    impl Rng {
        pub fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        }

        pub fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }

        pub fn tensor(&mut self, c: u32, h: u32, w: u32) -> Tensor {
            Tensor {
                c,
                h,
                w,
                data: self.vec((c * h * w) as usize),
            }
        }
    }

    pub fn raw(g: &Gpu, data: &[f32]) -> wgpu::Buffer {
        let b = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (data.len() as u64 * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    }

    pub fn empty(g: &Gpu, c: u32, h: u32, w: u32) -> Buf {
        Buf {
            buffer: create_tensor_buffer(&g.device, tensor_bytes(c, h, w)),
            c,
            h,
            w,
        }
    }

    pub fn assert_matches(g: &Gpu, dst: &Buf, want: &Tensor) {
        assert_eq!((dst.c, dst.h, dst.w), (want.c, want.h, want.w));
        let got = read_back(&g.device, &g.queue, &dst.buffer, want.data.len()).unwrap();
        let worst = got
            .iter()
            .zip(&want.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "max abs diff {worst}");
    }

    /// As [`assert_matches`], within 1e-5 of the largest magnitude of `want` (at least 10).
    pub fn assert_matches_rel(g: &Gpu, dst: &Buf, want: &Tensor) {
        assert_eq!((dst.c, dst.h, dst.w), (want.c, want.h, want.w));
        let scale = want.data.iter().fold(10.0_f32, |m, v| m.max(v.abs()));
        let got = read_back(&g.device, &g.queue, &dst.buffer, want.data.len()).unwrap();
        let worst = got
            .iter()
            .zip(&want.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5 * scale,
            "max abs diff {worst} at scale {scale}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::test_gpu::*;
    use super::*;
    use crate::cpu_ref;

    fn run_conv(g: &Gpu, src: &Tensor, out: u32, k: u32, act: Activation, seed: u32, bias: bool) {
        let mut rng = Rng(seed);
        let weights = rng.vec((out * src.c * k * k) as usize);
        let biases = if bias { rng.vec(out as usize) } else { vec![] };
        let want = cpu_ref::conv(src, out, k, &weights, &biases, act);
        let s = upload(&g.device, &g.queue, src);
        let d = empty(g, out, src.h, src.w);
        let (wb, bb) = (raw(g, &weights), raw(g, &biases));
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.conv(
            &mut rec,
            &ConvOp {
                src: &s,
                dst: &d,
                weights: &wb,
                biases: bias.then_some(&bb),
                k,
                act,
                packed: None,
            },
        );
        rec.flush();
        assert_matches(g, &d, &want);
    }

    #[test]
    fn conv_matches_ref_for_all_activations() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(1);
        let src = rng.tensor(6, 13, 11);
        run_conv(g, &src, 7, 3, Activation::None, 2, true);
        run_conv(g, &src, 8, 3, Activation::Leaky(0.2), 3, true);
        run_conv(g, &src, 5, 3, Activation::Relu, 4, false);
        run_conv(g, &src, 3, 1, Activation::None, 5, true);
    }

    /// Shader compilers differ: FXC behind DX12 refuses code that Vulkan accepts. Every API on this
    /// machine must compile the kernels and compute the right numbers.
    #[test]
    fn kernels_work_on_every_adapter_of_this_machine() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let mut rng = Rng(81);
        let src = rng.tensor(9, 17, 33);
        let weights = rng.vec(7 * 9 * 9);
        let biases = rng.vec(7);
        let want = cpu_ref::conv(&src, 7, 3, &weights, &biases, Activation::Leaky(0.2));
        for adapter in adapters {
            let info = adapter.get_info();
            let limits = adapter.limits();
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: limits.clone(),
                    ..Default::default()
                }))
                .unwrap();
            let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let kernels = Kernels::new(&device, &limits);
            assert!(pollster::block_on(validation.pop()).is_none(), "{info:?}");
            assert!(pollster::block_on(internal.pop()).is_none(), "{info:?}");
            assert_eq!(
                kernels.has_tiled(),
                use_tiled(&limits),
                "{}: fast kernel refused",
                info.name
            );

            let mk = |data: &[f32]| {
                let b = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (data.len() as u64 * 4).max(16),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
                b
            };
            let s = upload(&device, &queue, &src);
            let d = Buf {
                buffer: create_tensor_buffer(&device, tensor_bytes(7, 17, 33)),
                c: 7,
                h: 17,
                w: 33,
            };
            let packed: Vec<f32> = pack_weights(&weights, 7, 9).into_iter().flatten().collect();
            let (wb, bb, pb) = (mk(&weights), mk(&biases), mk(&packed));
            let mut rec = Recorder::new(&kernels, &device, &queue);
            kernels.conv(
                &mut rec,
                &ConvOp {
                    src: &s,
                    dst: &d,
                    weights: if kernels.has_tiled() { &pb } else { &wb },
                    biases: Some(&bb),
                    k: 3,
                    act: Activation::Leaky(0.2),
                    packed: kernels.has_tiled().then_some(&pb),
                },
            );
            rec.flush();
            let got = read_back(&device, &queue, &d.buffer, want.data.len()).unwrap();
            let worst = got
                .iter()
                .zip(&want.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-3, "{}: max abs diff {worst}", info.name);
            println!(
                "{} {:?}: ok, tiled {}",
                info.name,
                info.backend,
                kernels.has_tiled()
            );
        }
    }

    #[test]
    fn pack_weights_layout() {
        // 5 outputs, 1 input, 3x3: weight value = 10 * output + cell.
        let weights: Vec<f32> = (0..5)
            .flat_map(|o| (0..9).map(move |cell| (10 * o + cell) as f32))
            .collect();
        let packed = pack_weights(&weights, 5, 1);
        assert_eq!(packed.len(), 18, "two blocks of 9 cells");
        assert_eq!(packed[0], [0.0, 10.0, 20.0, 30.0]);
        assert_eq!(packed[8], [8.0, 18.0, 28.0, 38.0]);
        assert_eq!(packed[9], [40.0, 0.0, 0.0, 0.0]);
        assert_eq!(packed[17], [48.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn tiled_gate_follows_workgroup_storage() {
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_compute_workgroup_storage_size = 16_352;
        assert!(!use_tiled(&limits));
        limits.max_compute_workgroup_storage_size = 32_768;
        assert!(use_tiled(&limits));
    }

    fn run_tiled_conv(g: &Gpu, src: &Tensor, out: u32, act: Activation, seed: u32, bias: bool) {
        let mut rng = Rng(seed);
        let weights = rng.vec((out * src.c * 9) as usize);
        let biases = if bias { rng.vec(out as usize) } else { vec![] };
        let want = cpu_ref::conv(src, out, 3, &weights, &biases, act);
        let s = upload(&g.device, &g.queue, src);
        let d = empty(g, out, src.h, src.w);
        let (wb, bb) = (raw(g, &weights), raw(g, &biases));
        let packed: Vec<f32> = pack_weights(&weights, out, src.c)
            .into_iter()
            .flatten()
            .collect();
        let pb = raw(g, &packed);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.conv(
            &mut rec,
            &ConvOp {
                src: &s,
                dst: &d,
                weights: &wb,
                biases: bias.then_some(&bb),
                k: 3,
                act,
                packed: Some(&pb),
            },
        );
        rec.flush();
        assert_matches(g, &d, &want);
    }

    #[test]
    fn tiled_conv_matches_ref() {
        let Some(g) = gpu() else { return };
        assert!(g.kernels.tiled.is_some(), "this test wants the fast path");
        let mut rng = Rng(71);
        // Sizes across tile edges; channel counts below, at and above a slab; a partial last block.
        for (c_in, out, h, w) in [
            (3, 4, 13, 11),
            (8, 8, 16, 16),
            (9, 7, 17, 33),
            (20, 6, 40, 5),
        ] {
            let src = rng.tensor(c_in, h, w);
            run_tiled_conv(g, &src, out, Activation::None, 72, true);
            run_tiled_conv(g, &src, out, Activation::Leaky(0.2), 73, false);
            run_tiled_conv(g, &src, out, Activation::Relu, 74, true);
        }
    }

    #[test]
    fn tiled_conv_cut_into_bands_matches_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(75);
        let src = rng.tensor(64, 300, 40);
        run_tiled_conv(g, &src, 8, Activation::Leaky(0.2), 76, true);
    }

    #[test]
    fn conv_zero_padding_matches_ref_at_border() {
        let Some(g) = gpu() else { return };
        // All ones with all-one weights: a corner sums 4 cells per channel, not 9.
        let src = Tensor {
            c: 2,
            h: 4,
            w: 4,
            data: vec![1.0; 32],
        };
        let weights = vec![1.0; 18];
        let s = upload(&g.device, &g.queue, &src);
        let d = empty(g, 1, 4, 4);
        let wb = raw(g, &weights);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.conv(
            &mut rec,
            &ConvOp {
                src: &s,
                dst: &d,
                weights: &wb,
                biases: None,
                k: 3,
                act: Activation::None,
                packed: None,
            },
        );
        rec.flush();
        let got = read_back(&g.device, &g.queue, &d.buffer, 16).unwrap();
        assert_eq!(got[0], 8.0, "corner");
        assert_eq!(got[1], 12.0, "edge");
        assert_eq!(got[5], 18.0, "inside");
    }

    #[test]
    fn conv_large_enough_to_be_cut_into_bands_matches_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(9);
        // 64x64 inputs x 4 x 9 x 300 x 300 exceeds one dispatch's budget many times over.
        let c_in = 64;
        let src = rng.tensor(c_in, 300, 40);
        run_conv(g, &src, 8, 3, Activation::Leaky(0.2), 10, true);
    }

    #[test]
    fn leaky_and_prelu_match_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(11);
        let src = rng.tensor(5, 9, 13);
        let s = upload(&g.device, &g.queue, &src);
        let d1 = empty(g, 5, 9, 13);
        let d2 = empty(g, 5, 9, 13);
        let d3 = empty(g, 5, 9, 13);
        let slopes = rng.vec(5);
        let (sb, one) = (raw(g, &slopes), raw(g, &[0.5]));
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.leaky(&mut rec, &s, &d1, 0.2);
        g.kernels.prelu(&mut rec, &s, &d2, &sb, 5);
        g.kernels.prelu(&mut rec, &s, &d3, &one, 1);
        rec.flush();
        assert_matches(g, &d1, &cpu_ref::leaky(&src, 0.2));
        assert_matches(g, &d2, &cpu_ref::prelu(&src, &slopes));
        assert_matches(g, &d3, &cpu_ref::prelu(&src, &[0.5]));
    }

    #[test]
    fn smooth_activations_and_normalize_match_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(12);
        let mut src = rng.tensor(3, 9, 13);
        // Far into both tails, where a naive formula overflows.
        src.data[..4].copy_from_slice(&[-120.0, 120.0, -30.0, 30.0]);
        let src = cpu_ref::scalar_mul(&src, 8.0);
        let s = upload(&g.device, &g.queue, &src);
        let ds: Vec<Buf> = (0..4).map(|_| empty(g, 3, 9, 13)).collect();
        let mean = [0.4488, 0.4371, 0.4040];
        let mb = raw(g, &mean);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.smooth(&mut rec, &s, &ds[0], Smooth::Sigmoid);
        g.kernels.smooth(&mut rec, &s, &ds[1], Smooth::Silu);
        g.kernels.normalize(&mut rec, &s, &ds[2], &mb, 255.0);
        g.kernels.smooth(&mut rec, &s, &ds[3], Smooth::Mish);
        rec.flush();
        assert_matches(g, &ds[0], &cpu_ref::sigmoid(&src));
        assert_matches_rel(g, &ds[1], &cpu_ref::silu(&src));
        assert_matches_rel(g, &ds[2], &cpu_ref::normalize(&src, &mean, 255.0));
        assert_matches_rel(g, &ds[3], &cpu_ref::mish(&src));
    }

    /// Sizes that are not multiples of the workgroup sides, groups larger than the reduction's
    /// thread count, and a common offset that a one-pass variance would lose.
    #[test]
    fn norms_and_slice_match_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(13);
        let src = cpu_ref::scalar_add(&rng.tensor(12, 21, 19), 40.0);
        let s = upload(&g.device, &g.queue, &src);
        let (gamma, beta) = (rng.vec(12), rng.vec(12));
        let (gb, bb) = (raw(g, &gamma), raw(g, &beta));
        let both = raw(g, &[gamma.as_slice(), beta.as_slice()].concat());
        let ds: Vec<Buf> = (0..3).map(|_| empty(g, 12, 21, 19)).collect();
        let part = empty(g, 5, 21, 19);
        let stats: Vec<wgpu::Buffer> = (0..2)
            .map(|_| create_tensor_buffer(&g.device, 64))
            .collect();
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.layer_norm(&mut rec, &s, &ds[0], &gb, &bb, 1e-6);
        for (i, groups) in [4, 1].into_iter().enumerate() {
            let op = GroupNormOp {
                groups,
                eps: 1e-5,
                gamma_beta: &both,
                stats: &stats[i],
            };
            g.kernels.group_norm(&mut rec, &s, &ds[1 + i], &op);
        }
        g.kernels.slice(&mut rec, &s, &part, 3);
        rec.flush();
        let want = cpu_ref::layer_norm_c(&src, 1e-6, &gamma, &beta);
        assert_matches(g, &ds[0], &want);
        for (i, groups) in [4, 1].into_iter().enumerate() {
            let want = cpu_ref::group_norm(&src, groups, 1e-5, &gamma, &beta);
            assert_matches(g, &ds[1 + i], &want);
        }
        assert_matches(g, &part, &cpu_ref::slice(&src, 3, 8));
    }

    #[test]
    fn eltwise_and_binary_ops_match_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(21);
        let a = rng.tensor(4, 7, 10);
        let b = rng.tensor(4, 7, 10);
        let (ba, bb) = (
            upload(&g.device, &g.queue, &a),
            upload(&g.device, &g.queue, &b),
        );
        let ds: Vec<Buf> = (0..4).map(|_| empty(g, 4, 7, 10)).collect();
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.weighted_sum(&mut rec, &ba, &bb, &ds[0], 0.2, 1.0);
        g.kernels.mul(&mut rec, &ba, &bb, &ds[1]);
        g.kernels.scalar(&mut rec, &ba, &ds[2], true, 0.75);
        g.kernels.scalar(&mut rec, &ba, &ds[3], false, 0.2);
        rec.flush();
        assert_matches(g, &ds[0], &cpu_ref::eltwise(&[&a, &b], &[0.2, 1.0]));
        assert_matches(g, &ds[1], &cpu_ref::binary_mul(&a, &b));
        assert_matches(g, &ds[2], &cpu_ref::scalar_add(&a, 0.75));
        assert_matches(g, &ds[3], &cpu_ref::scalar_mul(&a, 0.2));
    }

    #[test]
    fn concat_matches_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(31);
        let (a, b, c) = (
            rng.tensor(2, 5, 6),
            rng.tensor(3, 5, 6),
            rng.tensor(1, 5, 6),
        );
        let (ba, bb, bc) = (
            upload(&g.device, &g.queue, &a),
            upload(&g.device, &g.queue, &b),
            upload(&g.device, &g.queue, &c),
        );
        let d = empty(g, 6, 5, 6);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.concat(&mut rec, &[&ba, &bb, &bc], &d);
        rec.flush();
        assert_matches(g, &d, &cpu_ref::concat(&[&a, &b, &c]));
    }

    #[test]
    fn interp_matches_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(41);
        let a = rng.tensor(3, 5, 7);
        let s = upload(&g.device, &g.queue, &a);
        let d2 = empty(g, 3, 10, 14);
        let d4 = empty(g, 3, 20, 28);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.interp(&mut rec, &s, &d2, 2);
        g.kernels.interp(&mut rec, &s, &d4, 4);
        rec.flush();
        assert_matches(g, &d2, &cpu_ref::interp_nearest(&a, 2));
        assert_matches(g, &d4, &cpu_ref::interp_nearest(&a, 4));
    }

    #[test]
    fn pixel_shuffle_layout_matches_ncnn() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(51);
        let a = rng.tensor(48, 5, 7);
        let s = upload(&g.device, &g.queue, &a);
        let d = empty(g, 3, 20, 28);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        g.kernels.pixel_shuffle(&mut rec, &s, &d, 4);
        rec.flush();
        assert_matches(g, &d, &cpu_ref::pixel_shuffle(&a, 4));
    }

    #[test]
    fn pixel_unshuffle_matches_ref() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(52);
        for (r, c, h, w) in [(2, 3, 10, 14), (4, 3, 8, 12), (2, 5, 2, 18)] {
            let a = rng.tensor(c, h, w);
            let s = upload(&g.device, &g.queue, &a);
            let d = empty(g, c * r * r, h / r, w / r);
            let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
            g.kernels.pixel_unshuffle(&mut rec, &s, &d, r);
            rec.flush();
            assert_matches(g, &d, &cpu_ref::pixel_unshuffle(&a, r));
        }
    }

    #[test]
    fn workgroup_storage_within_16k() {
        // All kernels must compile on a device limited to the WebGPU-style downlevel defaults.
        let Ok(o) = crate::device::open_with_limits(
            scaleup_core::DevicePreference::Auto,
            wgpu::Limits::downlevel_defaults(),
        ) else {
            return;
        };
        assert!(o.limits.max_compute_workgroup_storage_size <= 16_384);
        let scope = o.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _ = Kernels::new(&o.device, &o.limits);
        assert!(pollster::block_on(scope.pop()).is_none());
    }

    #[test]
    fn many_dispatches_overflow_the_uniform_arena_safely() {
        let Some(g) = gpu() else { return };
        let mut rng = Rng(61);
        let a = rng.tensor(2, 3, 3);
        let mut want = a.clone();
        let mut src = upload(&g.device, &g.queue, &a);
        let mut rec = Recorder::new(&g.kernels, &g.device, &g.queue);
        // Far more steps than arena slots at the largest alignment wgpu allows.
        let steps = (UNIFORM_ARENA / g.kernels.uniform_stride) as usize + 10;
        for _ in 0..steps {
            let dst = empty(g, 2, 3, 3);
            g.kernels.scalar(&mut rec, &src, &dst, true, 0.001);
            want = cpu_ref::scalar_add(&want, 0.001);
            src = dst;
        }
        rec.flush();
        let got = read_back(&g.device, &g.queue, &src.buffer, want.data.len()).unwrap();
        for (a, b) in got.iter().zip(&want.data) {
            assert!((a - b).abs() < 1e-2, "{a} {b}");
        }
    }
}
