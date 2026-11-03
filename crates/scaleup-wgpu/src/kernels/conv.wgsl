// Convolution with zero padding. aux_a: weights [out][in][k][k]. aux_b: biases.
// One dispatch writes channels c0..c0+4 of rows y0..y1.
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
    let x = id.x;
    let y = p.y0 + id.y;
    if (x >= p.w || y >= p.y1) { return; }
    let half_k = i32(p.k / 2u);
    let hi = min(p.c0 + 4u, p.c_out);
    for (var co = p.c0; co < hi; co++) {
        var sum = 0.0;
        if (p.has_bias != 0u) { sum = aux_b[co]; }
        var wi = co * p.c_in * p.k * p.k;
        for (var ci = 0u; ci < p.c_in; ci++) {
            let plane = ci * p.h * p.w;
            for (var ky = 0u; ky < p.k; ky++) {
                let sy = i32(y) + i32(ky) - half_k;
                for (var kx = 0u; kx < p.k; kx++) {
                    let sx = i32(x) + i32(kx) - half_k;
                    if (sy >= 0 && sx >= 0 && sy < i32(p.h) && sx < i32(p.w)) {
                        sum += aux_a[wi] * src[plane + u32(sy) * p.w + u32(sx)];
                    }
                    wi += 1u;
                }
            }
        }
        if (p.act == 1u) {
            sum = max(sum, 0.0);
        } else if (p.act == 2u && sum < 0.0) {
            sum *= p.slope;
        }
        dst[(co * p.h + y) * p.w + x] = sum;
    }
}
