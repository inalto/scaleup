// Group norm statistics: workgroup g reduces the p.factor channels of group g (contiguous in planar
// layout) and writes dst[2g] = mean, dst[2g + 1] = 1 / sqrt(variance + p.slope). The variance is
// computed around the mean in a second pass, which keeps f32 accurate on large tiles.
struct Params {
    mode: u32, c_in: u32, c_out: u32, h: u32,
    w: u32, k: u32, act: u32, c0: u32,
    y0: u32, y1: u32, factor: u32, has_bias: u32,
    slope: f32, coeff_a: f32, coeff_b: f32, pad: u32,
};
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> aux_a: array<f32>;
@group(0) @binding(3) var<storage, read> aux_b: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

const THREADS: u32 = 256u;
var<workgroup> part: array<f32, 256>;

fn reduce(li: u32) {
    for (var stride = THREADS / 2u; stride > 0u; stride /= 2u) {
        if (li < stride) {
            part[li] += part[li + stride];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let n = p.factor * p.h * p.w;
    let base = wg.x * n;
    var s = 0.0;
    for (var i = li; i < n; i += THREADS) {
        s += src[base + i];
    }
    part[li] = s;
    workgroupBarrier();
    reduce(li);
    let mean = part[0] / f32(n);
    workgroupBarrier();
    var q = 0.0;
    for (var i = li; i < n; i += THREADS) {
        let d = src[base + i] - mean;
        q += d * d;
    }
    part[li] = q;
    workgroupBarrier();
    reduce(li);
    if (li == 0u) {
        dst[2u * wg.x] = mean;
        dst[2u * wg.x + 1u] = 1.0 / sqrt(part[0] / f32(n) + p.slope);
    }
}
