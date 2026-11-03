// mode 0: coeff_a * src + coeff_b * aux_a. mode 1: src * aux_a.
// mode 2: src + coeff_a. mode 3: src * coeff_a.
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
    switch p.mode {
        case 0u: { dst[i] = p.coeff_a * src[i] + p.coeff_b * aux_a[i]; }
        case 1u: { dst[i] = src[i] * aux_a[i]; }
        case 2u: { dst[i] = src[i] + p.coeff_a; }
        default: { dst[i] = src[i] * p.coeff_a; }
    }
}
