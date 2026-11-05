// mode 0: layer norm over the p.c_in channels of each pixel, eps p.slope; aux_a: gamma, aux_b: beta.
// mode 1: group norm, groups of p.factor channels; aux_a: gamma then beta (2 * p.c_in values),
// aux_b: [mean, 1 / sqrt(var + eps)] per group, from group_stats.wgsl.
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) { return; }
    let plane = p.h * p.w;
    let px = id.y * p.w + id.x;
    if (p.mode == 1u) {
        let i = id.z * plane + px;
        let g = id.z / p.factor;
        dst[i] = (src[i] - aux_b[2u * g]) * aux_b[2u * g + 1u] * aux_a[id.z] + aux_a[p.c_in + id.z];
        return;
    }
    let n = f32(p.c_in);
    var mean = 0.0;
    for (var c = 0u; c < p.c_in; c++) {
        mean += src[c * plane + px];
    }
    mean /= n;
    var var_sum = 0.0;
    for (var c = 0u; c < p.c_in; c++) {
        let d = src[c * plane + px] - mean;
        var_sum += d * d;
    }
    let r = 1.0 / sqrt(var_sum / n + p.slope);
    for (var c = 0u; c < p.c_in; c++) {
        let i = c * plane + px;
        dst[i] = aux_a[c] * ((src[i] - mean) * r) + aux_b[c];
    }
}
