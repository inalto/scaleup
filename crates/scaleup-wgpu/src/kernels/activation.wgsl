// mode 0: leaky ReLU with p.slope. mode 1: PReLU, aux_a holds p.factor slopes (one, or per channel).
// mode 2: sigmoid. mode 3: SiLU, x * sigmoid(x). mode 4: Mish, x * tanh(ln(1 + exp(x))).
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

// exp(-x) overflows to infinity for very negative x, which gives the right limit, 0.
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) { return; }
    let i = (id.z * p.h + id.y) * p.w + id.x;
    let v = src[i];
    if (p.mode == 2u) {
        dst[i] = sigmoid(v);
        return;
    }
    if (p.mode == 3u) {
        dst[i] = v * sigmoid(v);
        return;
    }
    if (p.mode == 4u) {
        // ln(1 + exp(v)) is v to f32 precision from 20 up; exp only ever sees up to 20, so it
        // cannot overflow even where a compiler evaluates both sides of the select. tanh is 1 in
        // f32 from 20 up, and FXC (DX12) computes it through exp, which turns NaN for large inputs.
        let softplus = select(log(1.0 + exp(min(v, 20.0))), v, v >= 20.0);
        dst[i] = v * tanh(min(softplus, 20.0));
        return;
    }
    var slope = p.slope;
    if (p.mode == 1u) {
        var s = id.z;
        if (p.factor == 1u) { s = 0u; }
        slope = aux_a[s];
    }
    if (v < 0.0) {
        dst[i] = v * slope;
    } else {
        dst[i] = v;
    }
}
