//! SRVGGNetCompact, from spandrel's `SRVGG.py`: `body.0 → PReLU body.1 → num_conv × (conv, PReLU)
//! → body.{last} → pixel shuffle`, plus the input upscaled by nearest neighbour.

use scaleup_core::Result;

use super::Builder;
use crate::arch::ArchSpec;
use crate::ncnn::{Activation, Graph};
use crate::statedict::StateDict;

pub(super) fn build(spec: &ArchSpec, sd: &StateDict) -> Result<Graph> {
    let ArchSpec::Compact {
        num_conv, scale, ..
    } = *spec
    else {
        unreachable!("build dispatches on the spec")
    };
    let (mut b, input) = Builder::new(sd, 3);
    let mut t = input;
    for i in 0..=num_conv {
        t = b.conv(t, &format!("body.{}", 2 * i), true, Activation::None)?;
        t = b.prelu(t, &format!("body.{}.weight", 2 * i + 1))?;
    }
    t = b.conv(
        t,
        &format!("body.{}", 2 * num_conv + 2),
        true,
        Activation::None,
    )?;
    t = b.pixel_shuffle("upsampler", t, scale)?;
    let base = b.interp("base", input, scale);
    t = b.sum("out", t, base, [1.0, 1.0])?;
    b.finish(t)
}

#[cfg(test)]
mod tests {
    use super::super::build;
    use super::super::tests::{
        assert_close, assert_close_rel, fixture, pointwise_conv, random_dict,
    };
    use crate::arch::detect;
    use crate::cpu_ref::run_graph;
    use crate::exec::{Executor, plan};
    use crate::kernels::test_gpu::{Rng, gpu};
    use crate::ncnn::infer_scale;
    use crate::statedict::StateDict;
    use scaleup_core::Image32;

    fn entries(num_feat: usize, num_conv: usize, scale: usize) -> Vec<(String, Vec<usize>)> {
        let mut e = Vec::new();
        let mut c_in = 3;
        for i in 0..=num_conv {
            e.push((format!("body.{}.weight", 2 * i), vec![num_feat, c_in, 3, 3]));
            e.push((format!("body.{}.bias", 2 * i), vec![num_feat]));
            e.push((format!("body.{}.weight", 2 * i + 1), vec![num_feat]));
            c_in = num_feat;
        }
        let last = 2 * num_conv + 2;
        let ps = 3 * scale * scale;
        e.push((format!("body.{last}.weight"), vec![ps, num_feat, 3, 3]));
        e.push((format!("body.{last}.bias"), vec![ps]));
        e
    }

    /// `SRVGG.py`'s forward pass for one input pixel of a pointwise network: the `s`×`s` block of
    /// output pixels it becomes, as `[c][dy][dx]`.
    fn srvgg_pixel(sd: &StateDict, num_conv: usize, s: usize, x: &[f32]) -> Vec<f32> {
        let mut t = x.to_vec();
        for i in 0..=num_conv {
            t = pointwise_conv(sd, &format!("body.{}", 2 * i), &t);
            let slopes = &sd.get(&format!("body.{}.weight", 2 * i + 1)).unwrap().data;
            for (c, v) in t.iter_mut().enumerate() {
                if *v < 0.0 {
                    *v *= slopes[c];
                }
            }
        }
        let t = pointwise_conv(sd, &format!("body.{}", 2 * num_conv + 2), &t);
        // PixelShuffle keeps [c][dy][dx] order; the nearest-upscaled input is added to each.
        t.iter()
            .enumerate()
            .map(|(i, v)| v + x[i / (s * s)])
            .collect()
    }

    #[test]
    fn compact_matches_srvgg_forward_pointwise() {
        for (scale, seed) in [(4, 1), (2, 2), (1, 3)] {
            let num_conv = 2;
            let mut rng = Rng(seed);
            let sd = random_dict(&entries(5, num_conv, scale), &mut rng, 1.0, true);
            let spec = detect(&sd).unwrap();
            let graph = build(&spec, &sd).unwrap();
            assert_eq!(infer_scale(&graph).unwrap() as usize, scale);
            let input = rng.tensor(3, 3, 2);
            let out = run_graph(&graph, &input);
            let s = scale as u32;
            for y in 0..input.h {
                for x in 0..input.w {
                    let px: Vec<f32> = (0..3).map(|c| input.get(c, y, x)).collect();
                    let want = srvgg_pixel(&sd, num_conv, scale, &px);
                    let mut got = Vec::new();
                    for c in 0..3 {
                        for dy in 0..s {
                            for dx in 0..s {
                                got.push(out.get(c, y * s + dy, x * s + dx));
                            }
                        }
                    }
                    assert_close_rel(&got, &want, 1e-5);
                }
            }
        }
    }

    #[test]
    fn compact_fixture_runs_and_matches_cpu_ref() {
        let sd = StateDict::load(&fixture("tiny_compact.pth")).unwrap();
        let graph = build(&detect(&sd).unwrap(), &sd).unwrap();
        let input = Rng(7).tensor(3, 9, 12);
        let want = run_graph(&graph, &input);
        assert_eq!((want.c, want.h, want.w), (3, 18, 24));
        let Some(gpu) = gpu() else { return };
        let exec = Executor::new(gpu.device.clone(), gpu.queue.clone(), &graph).unwrap();
        let p = plan(&graph, 3, input.h, input.w).unwrap();
        let image = Image32 {
            width: input.w,
            height: input.h,
            data: input.data.clone(),
        };
        let got = exec.run(&graph, &p, &image).unwrap();
        assert_eq!((got.width, got.height), (24, 18));
        assert_close(&got.data, &want.data, 1e-4);
    }
}
