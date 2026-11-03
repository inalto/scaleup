// out[c][y*r+dy][x*r+dx] = in[c*r*r + dy*r + dx][y][x], with r = p.factor.
// Source size: p.y0 rows, p.y1 columns.
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
    let r = p.factor;
    let sc = id.z * r * r + (id.y % r) * r + (id.x % r);
    dst[(id.z * p.h + id.y) * p.w + id.x] = src[(sc * p.y0 + id.y / r) * p.y1 + id.x / r];
}
