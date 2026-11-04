// 3x3 convolution over 16x16 output tiles. One dispatch writes channels c0..c0+4 of rows y0..y1.
//
// The 18x18 input neighbourhood of a tile is loaded into workgroup memory one slab of 8 input
// channels at a time, so each of the 256 invocations reads it from fast shared memory instead of
// the storage buffer. aux_a: weights packed as vec4 per output block, [block][in][3][3].
// aux_b: biases. Pixels outside the input read as zero.
struct Params {
    mode: u32, c_in: u32, c_out: u32, h: u32,
    w: u32, k: u32, act: u32, c0: u32,
    y0: u32, y1: u32, factor: u32, has_bias: u32,
    slope: f32, coeff_a: f32, coeff_b: f32, pad: u32,
};
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> aux_b: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

const INPUT_SIDE: u32 = 18u;
const SLAB: u32 = 8u;
const PLANE: u32 = 324u; // 18 * 18
const CACHE: u32 = 2592u; // 8 * 324, 10368 bytes
const LOADS: u32 = 11u; // CACHE / 256, rounded up

var<workgroup> cache: array<f32, 2592>;

fn bias(i: u32) -> f32 {
    if (p.c0 + i < p.c_out) { return aux_b[p.c0 + i]; }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let x0 = i32(wg.x * 16u) - 1;
    let y0 = i32(p.y0 + wg.y * 16u) - 1;
    let block = p.c0 / 4u;

    var sum = vec4<f32>(0.0);
    if (p.has_bias != 0u) {
        sum = vec4<f32>(bias(0u), bias(1u), bias(2u), bias(3u));
    }

    for (var base = 0u; base < p.c_in; base += SLAB) {
        // Every invocation loads its share of the slab. The trip count must not depend on the
        // invocation: FXC (DX12) refuses a barrier that follows such a loop.
        for (var n = 0u; n < LOADS; n++) {
            let i = li + n * 256u;
            if (i < CACHE) {
                let ch = base + i / PLANE;
                let rest = i % PLANE;
                let sy = y0 + i32(rest / INPUT_SIDE);
                let sx = x0 + i32(rest % INPUT_SIDE);
                var v = 0.0;
                if (ch < p.c_in && sy >= 0 && sx >= 0 && sy < i32(p.h) && sx < i32(p.w)) {
                    v = src[(ch * p.h + u32(sy)) * p.w + u32(sx)];
                }
                cache[i] = v;
            }
        }
        workgroupBarrier();

        let channels = min(SLAB, p.c_in - base);
        for (var c = 0u; c < channels; c++) {
            var wi = (block * p.c_in + base + c) * 9u;
            let plane = c * PLANE;
            for (var ky = 0u; ky < 3u; ky++) {
                let row = plane + (lid.y + ky) * INPUT_SIDE + lid.x;
                sum += weights[wi] * cache[row];
                sum += weights[wi + 1u] * cache[row + 1u];
                sum += weights[wi + 2u] * cache[row + 2u];
                wi += 3u;
            }
        }
        workgroupBarrier();
    }

    let x = wg.x * 16u + lid.x;
    let y = p.y0 + wg.y * 16u + lid.y;
    if (x >= p.w || y >= p.y1) { return; }
    if (p.act == 1u) {
        sum = max(sum, vec4<f32>(0.0));
    } else if (p.act == 2u) {
        sum = select(sum, sum * p.slope, sum < vec4<f32>(0.0));
    }
    // Written out rather than looped: FXC unrolls loops over vector components, very slowly.
    let at = y * p.w + x;
    let plane_size = p.h * p.w;
    dst[p.c0 * plane_size + at] = sum.x;
    if (p.c0 + 1u < p.c_out) { dst[(p.c0 + 1u) * plane_size + at] = sum.y; }
    if (p.c0 + 2u < p.c_out) { dst[(p.c0 + 2u) * plane_size + at] = sum.z; }
    if (p.c0 + 3u < p.c_out) { dst[(p.c0 + 3u) * plane_size + at] = sum.w; }
}
