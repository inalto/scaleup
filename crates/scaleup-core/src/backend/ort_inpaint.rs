//! Inpainting networks on ONNX Runtime: LaMa, MI-GAN and the anime/illustration LaMa, packed as
//! measured in `docs/OUTPAINT_SPIKE.md` (section 2 and "Plan 12 candidates").
//!
//! The graph tells the tensor types and the window size; what it cannot tell (the range of a
//! float32 output, whether the graph works on a GPU, the context factor) comes from the catalogue
//! entry's `layout` ([`InpaintLayout`]), with defaults that keep a model given by file path running
//! as it always has.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ::ort::session::Session;
use ::ort::value::{TensorElementType, TensorRef};

use crate::backend::ort::{backend_err, builder_for, cpu_device};
use crate::backend::{Backend, DeviceInfo, Inpainter, OrtBackend};
use crate::image32::Image32;
use crate::registry::{InpaintLayout, OutputRange};
use crate::{Result, ScaleUpError};

/// Window side when the graph leaves the spatial size dynamic (MI-GAN and the anime LaMa run at
/// 512; the anime LaMa needs a multiple of 8).
const DEFAULT_WINDOW: u32 = 512;

/// How a network wants its window packed: tensor types, value ranges and mask polarity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Packing {
    /// `image` uint8 0..255 and `mask` uint8 with 255 = known, 0 = hole; output uint8 0..255.
    MiGan,
    /// `image` float32 0..1 and `mask` float32 with 1 = hole, 0 = known; output float32 in
    /// `output`.
    LaMa { output: OutputRange },
}

/// How to run one inpainting network: the packing and window read from its graph, plus the rules
/// its catalogue entry gives (or the defaults of its packing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub packing: Packing,
    /// Side of the square window, in pixels.
    pub window: u32,
    /// Runs on the chosen GPU; `false` keeps it on the CPU whatever device was chosen.
    pub gpu: bool,
    /// See [`Inpainter::max_context_factor`].
    pub max_context_factor: u32,
}

/// One graph input as ONNX Runtime reports it: name, element type, shape (-1 where dynamic).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InputSig {
    pub name: String,
    pub ty: Option<TensorElementType>,
    pub shape: Vec<i64>,
}

impl InputSig {
    fn describe(&self) -> String {
        let ty = self
            .ty
            .map_or("non-tensor".to_string(), |t| format!("{t:?}"));
        format!("{} {ty} {:?}", self.name, self.shape)
    }
}

impl Layout {
    /// The layout of a graph with these inputs: exactly `image` `[N,3,H,W]` and `mask`
    /// `[N,1,H,W]`, both uint8 (MI-GAN) or both float32 (LaMa), with a square window. A dynamic
    /// side means [`DEFAULT_WINDOW`]. Anything else is a [`ScaleUpError::Model`] naming the inputs
    /// found.
    ///
    /// `hint` (the entry's `layout`) gives what the graph cannot; each absent field takes the
    /// default of the packing. Float32: output 0..255, CPU only (the LaMa export throws inside
    /// DirectML, docs/OUTPAINT_SPIKE.md section 3, and CUDA is untested), context factor up to 4.
    /// Uint8: the chosen device, context factor 1 (a wider context made MI-GAN invent more,
    /// section 5). `hint.output` does not apply to uint8 graphs.
    pub(crate) fn detect(inputs: &[InputSig], hint: &InpaintLayout) -> Result<Layout> {
        let unknown = || {
            let found: Vec<String> = inputs.iter().map(InputSig::describe).collect();
            ScaleUpError::Model(format!(
                "not an inpainting model (expected inputs `image` [N,3,H,W] and `mask` [N,1,H,W], \
                 both uint8 as in MI-GAN or both float32 as in LaMa); inputs found: {}",
                if found.is_empty() {
                    "none".to_string()
                } else {
                    found.join(", ")
                }
            ))
        };
        let get = |name: &str| inputs.iter().find(|i| i.name == name);
        let (Some(image), Some(mask)) = (get("image"), get("mask")) else {
            return Err(unknown());
        };
        if inputs.len() != 2 {
            return Err(unknown());
        }
        let (packing, gpu, max_context_factor) = match (image.ty, mask.ty) {
            (Some(TensorElementType::Uint8), Some(TensorElementType::Uint8)) => {
                (Packing::MiGan, true, 1)
            }
            (Some(TensorElementType::Float32), Some(TensorElementType::Float32)) => {
                let output = hint.output.unwrap_or(OutputRange::Byte);
                (Packing::LaMa { output }, false, 4)
            }
            _ => return Err(unknown()),
        };
        let (&[_, 3, ih, iw], &[_, 1, mh, mw]) = (image.shape.as_slice(), mask.shape.as_slice())
        else {
            return Err(unknown());
        };
        let side = |d: i64| u32::try_from(d).ok().filter(|&d| d > 0);
        let window = match (side(ih), side(iw)) {
            (None, None) => DEFAULT_WINDOW,
            (Some(h), Some(w)) if h == w => w,
            _ => return Err(unknown()),
        };
        let mask_fits = |d: i64| d < 0 || d == i64::from(window);
        if !(mask_fits(mh) && mask_fits(mw)) {
            return Err(unknown());
        }
        Ok(Layout {
            packing,
            window,
            gpu: hint.gpu.unwrap_or(gpu),
            max_context_factor: hint.max_context_factor.unwrap_or(max_context_factor),
        })
    }

    /// The device this network runs on when `selected` was chosen: `selected` when the layout
    /// allows a GPU, else the CPU.
    pub(crate) fn device_for(&self, selected: &DeviceInfo) -> DeviceInfo {
        if self.gpu {
            selected.clone()
        } else {
            cpu_device()
        }
    }
}

fn commit(device: &DeviceInfo, path: &Path) -> Result<Session> {
    builder_for(device)?
        .commit_from_file(path)
        .map_err(|e| ScaleUpError::Model(format!("{}: {e}", path.display())))
}

fn signature(session: &Session) -> Vec<InputSig> {
    session
        .inputs()
        .iter()
        .map(|i| InputSig {
            name: i.name().to_string(),
            ty: i.dtype().tensor_type(),
            shape: i.dtype().tensor_shape().map_or(Vec::new(), |s| s.to_vec()),
        })
        .collect()
}

impl OrtBackend {
    /// Loads the inpainting network at `path`. Its packing and window are read from the graph;
    /// `hint` is its catalogue entry's `layout` ([`InpaintLayout::default`] for a bare file), see
    /// [`Layout::detect`].
    ///
    /// The graph is opened once to read its layout: on this backend's device when `hint` says the
    /// graph works on a GPU, else on the CPU, which every graph loads on. It is opened again only
    /// when the layout then names another device (MI-GAN given by path on a GPU), because the
    /// LaMa graphs take 16 to 20 s to open. A file that is missing or of neither packing is a
    /// [`ScaleUpError::Model`].
    pub fn load_inpainter(&self, path: &Path, hint: &InpaintLayout) -> Result<Box<dyn Inpainter>> {
        if !path.is_file() {
            return Err(ScaleUpError::Model(format!(
                "inpainting model file not found: {}",
                path.display()
            )));
        }
        let started = Instant::now();
        let selected = self.device();
        let first = if hint.gpu == Some(true) {
            selected.clone()
        } else {
            cpu_device()
        };
        let session = commit(&first, path)?;
        let layout = Layout::detect(&signature(&session), hint)
            .map_err(|e| ScaleUpError::Model(format!("{}: {e}", path.display())))?;
        let Some(output_name) = session.outputs().first().map(|o| o.name().to_string()) else {
            return Err(ScaleUpError::Model(format!(
                "{}: model has no outputs",
                path.display()
            )));
        };
        let device = layout.device_for(selected);
        let session = if device == first {
            session
        } else {
            drop(session);
            commit(&device, path)?
        };
        let why = if !layout.gpu && selected.kind.is_gpu() {
            " (kept on the CPU: this model is not known to work on a GPU)"
        } else {
            ""
        };
        tracing::info!(
            "inpainting model {} ({layout:?}) on {}{why}; loaded in {:.1} s",
            path.display(),
            device.name,
            started.elapsed().as_secs_f32()
        );
        Ok(Box::new(OrtInpainter {
            session: Mutex::new(session),
            layout,
            output_name,
        }))
    }
}

struct OrtInpainter {
    session: Mutex<Session>,
    layout: Layout,
    output_name: String,
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl OrtInpainter {
    /// Runs the graph on packed `image` and `mask` tensors; returns the output data.
    fn run<T>(&self, image: &[T], mask: &[T]) -> Result<Vec<T>>
    where
        T: ::ort::value::PrimitiveTensorElementType + Copy + std::fmt::Debug + 'static,
    {
        let s = self.layout.window as usize;
        let model_err = |e: ::ort::Error| ScaleUpError::Model(e.to_string());
        let image = TensorRef::from_array_view(([1usize, 3, s, s], image)).map_err(backend_err)?;
        let mask = TensorRef::from_array_view(([1usize, 1, s, s], mask)).map_err(backend_err)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| ScaleUpError::Backend("session mutex poisoned".into()))?;
        let outputs = session
            .run(::ort::inputs!["image" => image, "mask" => mask])
            .map_err(model_err)?;
        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<T>()
            .map_err(model_err)?;
        let s = s as i64;
        if **shape != [1, 3, s, s] {
            return Err(ScaleUpError::Model(format!(
                "expected inpainting output [1,3,{s},{s}], got {:?}",
                &**shape
            )));
        }
        Ok(data.to_vec())
    }
}

impl Inpainter for OrtInpainter {
    fn window(&self) -> u32 {
        self.layout.window
    }

    fn max_context_factor(&self) -> u32 {
        self.layout.max_context_factor
    }

    fn fill(&self, rgb: &Image32, known: &[bool]) -> Result<Image32> {
        let window = self.layout.window;
        let n = window as usize * window as usize;
        if (rgb.width, rgb.height) != (window, window) || known.len() != n {
            return Err(ScaleUpError::Model(format!(
                "the inpainting model takes a {0}x{0} window and mask, got {1}x{2} and {3}",
                window,
                rgb.width,
                rgb.height,
                known.len()
            )));
        }
        let data = match self.layout.packing {
            Packing::MiGan => {
                let image: Vec<u8> = rgb.data.iter().map(|&v| to_u8(v)).collect();
                let mask: Vec<u8> = known.iter().map(|&k| if k { 255 } else { 0 }).collect();
                self.run(&image, &mask)?
                    .into_iter()
                    .map(|v| f32::from(v) / 255.0)
                    .collect()
            }
            Packing::LaMa { output } => {
                let mask: Vec<f32> = known.iter().map(|&k| if k { 0.0 } else { 1.0 }).collect();
                let out = self.run(&rgb.data, &mask)?.into_iter();
                // Rounded to 8 bits either way, so a known pixel comes back as it went in.
                match output {
                    OutputRange::Byte => out.map(|v| v.clamp(0.0, 255.0).round() / 255.0).collect(),
                    OutputRange::Unit => out.map(|v| f32::from(to_u8(v)) / 255.0).collect(),
                }
            }
        };
        Ok(Image32 {
            width: window,
            height: window,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::fake::MeanInpainter;
    use crate::backend::{DeviceKind, DevicePreference};
    use crate::registry::Registry;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    fn cpu() -> OrtBackend {
        OrtBackend::new(DevicePreference::Cpu).unwrap()
    }

    /// The `layout` of catalogue entry `id`, as the embedded manifest gives it.
    fn catalogue_layout(id: &str) -> InpaintLayout {
        let dir = tempfile::tempdir().unwrap();
        Registry::with_dir(dir.path().to_path_buf())
            .get(id)
            .unwrap()
            .layout
    }

    /// The three fixtures, each with the layout of the catalogue model whose signature it copies.
    fn fixtures() -> [(&'static str, InpaintLayout); 3] {
        [
            ("inpaint_mean_lama.onnx", catalogue_layout("lama-512")),
            (
                "inpaint_mean_migan.onnx",
                catalogue_layout("migan-512-places2"),
            ),
            (
                "inpaint_mean_lama_manga.onnx",
                catalogue_layout("lama-manga"),
            ),
        ]
    }

    fn load(name: &str, hint: &InpaintLayout) -> Box<dyn Inpainter> {
        cpu().load_inpainter(&fixture(name), hint).unwrap()
    }

    /// A window of `k/255` values (exact in 8 bits) with the left `known_cols` columns known.
    fn window(side: u32, known_cols: u32) -> (Image32, Vec<bool>) {
        let mut img = Image32::new(side, side);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = ((i * 37 + 11) % 256) as f32 / 255.0;
        }
        let known = (0..side * side).map(|i| i % side < known_cols).collect();
        (img, known)
    }

    fn assert_matches_mean(model: &dyn Inpainter, known_cols: u32) {
        let side = model.window();
        let (img, known) = window(side, known_cols);
        let got = model.fill(&img, &known).unwrap();
        let want = MeanInpainter { window: side }.fill(&img, &known).unwrap();
        assert_eq!((got.width, got.height), (side, side));
        let worst = got
            .data
            .iter()
            .zip(&want.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1.0 / 255.0 + 1e-6, "max difference {worst}");
    }

    /// A window whose known left half is a checkerboard of 64 and 192 (mean 128) and whose hole
    /// (the right half) is white garbage; the model's output for it.
    fn fill_grey(model: &dyn Inpainter) -> (Image32, Image32) {
        let side = model.window();
        let half = side / 2;
        let mut img = Image32::new(side, side);
        let known: Vec<bool> = (0..side * side).map(|i| i % side < half).collect();
        for c in 0..3 {
            for y in 0..side {
                for x in 0..side {
                    let v = if x >= half {
                        1.0
                    } else if (x + y) % 2 == 0 {
                        64.0 / 255.0
                    } else {
                        192.0 / 255.0
                    };
                    img.set(c, x, y, v);
                }
            }
        }
        let got = model.fill(&img, &known).unwrap();
        (img, got)
    }

    #[test]
    fn each_fixture_gets_the_layout_of_its_catalogue_model() {
        let [lama, migan, manga] = fixtures();
        let lama = load(lama.0, &lama.1);
        // The window comes from the graph's fixed input shape.
        assert_eq!((lama.window(), lama.max_context_factor()), (16, 4));
        let migan = load(migan.0, &migan.1);
        // Dynamic spatial size: the 512 window MI-GAN is run at.
        assert_eq!((migan.window(), migan.max_context_factor()), (512, 1));
        let manga = load(manga.0, &manga.1);
        // Dynamic too; the entry keeps the context factor at 1.
        assert_eq!((manga.window(), manga.max_context_factor()), (512, 1));
    }

    #[test]
    fn every_fixture_fills_like_the_mean_inpainter() {
        for (name, hint) in fixtures() {
            let model = load(name, &hint);
            let known_cols = model.window() * 3 / 8;
            assert_matches_mean(model.as_ref(), known_cols);
        }
    }

    #[test]
    fn every_fixture_returns_mid_grey_over_mid_grey() {
        // A wrong output range gives black (0..1 read as 0..255) or white (0..255 read as 0..1); an
        // inverted mask keeps the white hole and repaints the known pixels.
        for (name, hint) in fixtures() {
            let model = load(name, &hint);
            let (img, got) = fill_grey(model.as_ref());
            let half = model.window() / 2;
            for c in 0..3 {
                for y in 0..img.height {
                    for x in 0..img.width {
                        let (want, what) = if x < half {
                            // Known pixels come back exactly.
                            (img.get(c, x, y), "known")
                        } else {
                            (128.0 / 255.0, "hole")
                        };
                        let v = got.get(c, x, y);
                        assert!(
                            (v - want).abs() <= 1e-6,
                            "{name}: {what} pixel ({c},{x},{y}) is {}, not {}",
                            v * 255.0,
                            want * 255.0
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_output_range_comes_from_the_layout_not_the_pixels() {
        let hole = |model: &dyn Inpainter| fill_grey(model).1.get(0, model.window() - 1, 0);
        // The anime LaMa fixture without its entry (as a bare file): read as 0..255, so black.
        let manga = load("inpaint_mean_lama_manga.onnx", &InpaintLayout::default());
        assert_eq!(manga.max_context_factor(), 4);
        assert!(hole(manga.as_ref()) <= 1.0 / 255.0);
        // The LaMa fixture told its output is 0..1: clipped to white.
        let unit = InpaintLayout {
            output: Some(OutputRange::Unit),
            ..InpaintLayout::default()
        };
        let lama = load("inpaint_mean_lama.onnx", &unit);
        assert_eq!(hole(lama.as_ref()), 1.0);
    }

    #[test]
    fn fill_rejects_a_window_of_the_wrong_size() {
        let model = load("inpaint_mean_lama.onnx", &InpaintLayout::default());
        let (img, known) = window(8, 4);
        assert!(matches!(
            model.fill(&img, &known),
            Err(ScaleUpError::Model(_))
        ));
        let (img, _) = window(16, 4);
        assert!(matches!(
            model.fill(&img, &[true; 3]),
            Err(ScaleUpError::Model(_))
        ));
    }

    #[test]
    fn a_graph_of_neither_layout_names_its_inputs() {
        let err = cpu()
            .load_inpainter(&fixture("nearest2x.onnx"), &InpaintLayout::default())
            .err()
            .expect("an upscaler is not an inpainter");
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
        assert!(
            err.to_string()
                .contains("inputs found: input Float32 [1, 3, -1, -1]"),
            "{err}"
        );
    }

    fn sig(name: &str, ty: TensorElementType, shape: &[i64]) -> InputSig {
        InputSig {
            name: name.into(),
            ty: Some(ty),
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn layout_and_window_come_from_the_inputs() {
        use TensorElementType::{Float32, Uint8};
        let pair = |ty, side: i64| {
            vec![
                sig("image", ty, &[1, 3, side, side]),
                sig("mask", ty, &[1, 1, side, side]),
            ]
        };
        let none = InpaintLayout::default();
        let detect = |inputs: &[InputSig]| {
            let l = Layout::detect(inputs, &none).unwrap();
            (l.packing, l.window)
        };
        let lama = Packing::LaMa {
            output: OutputRange::Byte,
        };
        assert_eq!(detect(&pair(Float32, 512)), (lama, 512));
        assert_eq!(detect(&pair(Float32, 16)), (lama, 16));
        assert_eq!(detect(&pair(Uint8, -1)), (Packing::MiGan, 512));
        // Input order does not matter.
        let mut swapped = pair(Uint8, 256);
        swapped.reverse();
        assert_eq!(detect(&swapped), (Packing::MiGan, 256));

        let rejected = [
            vec![sig("image", Float32, &[1, 3, 512, 512])],
            vec![
                sig("image", Float32, &[1, 3, 512, 512]),
                sig("mask", Uint8, &[1, 1, 512, 512]),
            ],
            vec![
                sig("image", Float32, &[1, 3, 512, 256]),
                sig("mask", Float32, &[1, 1, 512, 256]),
            ],
            vec![
                sig("image", Float32, &[1, 3, 512, 512]),
                sig("mask", Float32, &[1, 1, 256, 256]),
            ],
            vec![
                sig("image", Float32, &[1, 4, 512, 512]),
                sig("mask", Float32, &[1, 1, 512, 512]),
            ],
            [pair(Float32, 512), vec![sig("extra", Float32, &[1])]].concat(),
            Vec::new(),
        ];
        for inputs in rejected {
            let err = Layout::detect(&inputs, &none).unwrap_err();
            assert!(matches!(err, ScaleUpError::Model(_)), "{inputs:?}");
            assert!(err.to_string().contains("inputs found: "), "{err}");
        }
    }

    #[test]
    fn the_entry_fills_in_what_the_signature_cannot_tell() {
        use TensorElementType::{Float32, Uint8};
        let float_dynamic = vec![
            sig("image", Float32, &[-1, 3, -1, -1]),
            sig("mask", Float32, &[-1, 1, -1, -1]),
        ];
        let migan = vec![
            sig("image", Uint8, &[-1, 3, -1, -1]),
            sig("mask", Uint8, &[-1, 1, -1, -1]),
        ];
        // No entry (a model given by path): the packing's defaults, as before plan 12.
        let none = InpaintLayout::default();
        assert_eq!(
            Layout::detect(&float_dynamic, &none).unwrap(),
            Layout {
                packing: Packing::LaMa {
                    output: OutputRange::Byte
                },
                window: 512,
                gpu: false,
                max_context_factor: 4,
            }
        );
        assert_eq!(
            Layout::detect(&migan, &none).unwrap(),
            Layout {
                packing: Packing::MiGan,
                window: 512,
                gpu: true,
                max_context_factor: 1,
            }
        );
        // The catalogue's anime LaMa: same signature as a float LaMa, other conventions.
        assert_eq!(
            Layout::detect(&float_dynamic, &catalogue_layout("lama-manga")).unwrap(),
            Layout {
                packing: Packing::LaMa {
                    output: OutputRange::Unit
                },
                window: 512,
                gpu: true,
                max_context_factor: 1,
            }
        );
        // Each field alone; `output` does not apply to a uint8 graph.
        let cpu_only = InpaintLayout {
            gpu: Some(false),
            output: Some(OutputRange::Unit),
            max_context_factor: Some(3),
        };
        let l = Layout::detect(&migan, &cpu_only).unwrap();
        assert_eq!(
            (l.packing, l.gpu, l.max_context_factor),
            (Packing::MiGan, false, 3)
        );
    }

    #[test]
    fn the_device_follows_the_layout_not_the_model_name() {
        let gpu = DeviceInfo {
            kind: DeviceKind::DirectMl,
            index: 0,
            name: "DirectML GPU 0".into(),
        };
        let float_sig = [
            sig("image", TensorElementType::Float32, &[1, 3, 512, 512]),
            sig("mask", TensorElementType::Float32, &[1, 1, 512, 512]),
        ];
        let device = |hint: &InpaintLayout| {
            Layout::detect(&float_sig, hint)
                .unwrap()
                .device_for(&gpu)
                .kind
        };
        // lama-512 has no layout: a float graph stays on the CPU.
        assert_eq!(device(&catalogue_layout("lama-512")), DeviceKind::Cpu);
        assert_eq!(
            device(&catalogue_layout("lama-manga")),
            DeviceKind::DirectMl
        );
        let migan = [
            sig("image", TensorElementType::Uint8, &[1, 3, -1, -1]),
            sig("mask", TensorElementType::Uint8, &[1, 1, -1, -1]),
        ];
        let l = Layout::detect(&migan, &catalogue_layout("migan-512-places2")).unwrap();
        assert_eq!(l.device_for(&gpu), gpu);
        assert_eq!(l.device_for(&cpu_device()), cpu_device());
    }

    #[test]
    fn missing_inpainter_file_is_a_model_error() {
        let err = cpu()
            .load_inpainter(
                Path::new("no-such-inpainter.onnx"),
                &InpaintLayout::default(),
            )
            .err()
            .unwrap();
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
    }
}
