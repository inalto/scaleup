// mode 0: leaky ReLU with p.slope. mode 1: PReLU, aux_a holds p.factor slopes (one, or per channel).
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
    let i = (id.z * p.h + id.y) * p.w + id.x;
    var slope = p.slope;
    if (p.mode == 1u) {
        var s = id.z;
        if (p.factor == 1u) { s = 0u; }
        slope = aux_a[s];
    }
    let v = src[i];
    if (v < 0.0) {
        dst[i] = v * slope;
    } else {
        dst[i] = v;
    }
}
