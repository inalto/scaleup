//! Inpainting networks on ONNX Runtime: LaMa and MI-GAN, packed as measured in
//! `docs/OUTPAINT_SPIKE.md` section 2.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ::ort::session::Session;
use ::ort::value::{TensorElementType, TensorRef};

use crate::backend::ort::{backend_err, builder_for, cpu_device};
use crate::backend::{Backend, DeviceInfo, DeviceKind, Inpainter, OrtBackend};
use crate::image32::Image32;
use crate::{Result, ScaleUpError};

/// Window side when the graph leaves the spatial size dynamic (MI-GAN runs at 512).
const DEFAULT_WINDOW: u32 = 512;

/// How a network wants its window packed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// `image` uint8 0..255 and `mask` uint8 with 255 = known, 0 = hole; output uint8 0..255.
    MiGan,
    /// `image` float32 0..1 and `mask` float32 with 1 = hole, 0 = known; output float32 0..255.
    LaMa,
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
    /// The layout and window side of a graph with these inputs: exactly `image` `[N,3,H,W]` and
    /// `mask` `[N,1,H,W]`, both uint8 (MI-GAN) or both float32 (LaMa), with a square window. A
    /// dynamic side means [`DEFAULT_WINDOW`]. Anything else is a [`ScaleUpError::Model`] naming the
    /// inputs found.
    pub(crate) fn detect(inputs: &[InputSig]) -> Result<(Layout, u32)> {
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
        let layout = match (image.ty, mask.ty) {
            (Some(TensorElementType::Uint8), Some(TensorElementType::Uint8)) => Layout::MiGan,
            (Some(TensorElementType::Float32), Some(TensorElementType::Float32)) => Layout::LaMa,
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
        Ok((layout, window))
    }

    /// The device a network of this layout runs on when `selected` was chosen. LaMa always runs on
    /// the CPU: it throws inside DirectML (docs/OUTPAINT_SPIKE.md section 3) and CUDA is untested.
    pub(crate) fn device_for(self, selected: &DeviceInfo) -> DeviceInfo {
        match self {
            Layout::LaMa => cpu_device(),
            Layout::MiGan => selected.clone(),
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
    /// Loads the inpainting network at `path` (LaMa or MI-GAN layout, read from the graph).
    ///
    /// The graph is opened on the CPU first to read its layout; MI-GAN then gets a session on this
    /// backend's device, LaMa keeps the CPU session whatever the device (see
    /// [`Layout::device_for`]). A file that is missing or of neither layout is a
    /// [`ScaleUpError::Model`].
    pub fn load_inpainter(&self, path: &Path) -> Result<Box<dyn Inpainter>> {
        if !path.is_file() {
            return Err(ScaleUpError::Model(format!(
                "inpainting model file not found: {}",
                path.display()
            )));
        }
        let started = Instant::now();
        let cpu = cpu_device();
        let session = commit(&cpu, path)?;
        let (layout, window) = Layout::detect(&signature(&session))
            .map_err(|e| ScaleUpError::Model(format!("{}: {e}", path.display())))?;
        let Some(output_name) = session.outputs().first().map(|o| o.name().to_string()) else {
            return Err(ScaleUpError::Model(format!(
                "{}: model has no outputs",
                path.display()
            )));
        };
        let device = layout.device_for(self.device());
        let session = if device.kind == DeviceKind::Cpu {
            session
        } else {
            drop(session);
            commit(&device, path)?
        };
        let why = match (layout, self.device().kind) {
            (Layout::LaMa, kind) if kind != DeviceKind::Cpu => {
                " (LaMa runs on the CPU only: it fails on GPU execution providers)"
            }
            _ => "",
        };
        tracing::info!(
            "inpainting model {} ({layout:?}, {window}x{window}) on {}{why}; loaded in {:.1} s",
            path.display(),
            device.name,
            started.elapsed().as_secs_f32()
        );
        Ok(Box::new(OrtInpainter {
            session: Mutex::new(session),
            layout,
            window,
            output_name,
        }))
    }
}

struct OrtInpainter {
    session: Mutex<Session>,
    layout: Layout,
    window: u32,
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
        let s = self.window as usize;
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
        self.window
    }

    fn max_context_factor(&self) -> u32 {
        match self.layout {
            // A wider, downsampled context made MI-GAN invent more (docs/OUTPAINT_SPIKE.md §5).
            Layout::MiGan => 1,
            Layout::LaMa => 4,
        }
    }

    fn fill(&self, rgb: &Image32, known: &[bool]) -> Result<Image32> {
        let n = self.window as usize * self.window as usize;
        if (rgb.width, rgb.height) != (self.window, self.window) || known.len() != n {
            return Err(ScaleUpError::Model(format!(
                "the inpainting model takes a {0}x{0} window and mask, got {1}x{2} and {3}",
                self.window,
                rgb.width,
                rgb.height,
                known.len()
            )));
        }
        let data = match self.layout {
            Layout::MiGan => {
                let image: Vec<u8> = rgb.data.iter().map(|&v| to_u8(v)).collect();
                let mask: Vec<u8> = known.iter().map(|&k| if k { 255 } else { 0 }).collect();
                self.run(&image, &mask)?
                    .into_iter()
                    .map(|v| f32::from(v) / 255.0)
                    .collect()
            }
            Layout::LaMa => {
                let mask: Vec<f32> = known.iter().map(|&k| if k { 0.0 } else { 1.0 }).collect();
                self.run(&rgb.data, &mask)?
                    .into_iter()
                    .map(|v| v.clamp(0.0, 255.0).round() / 255.0)
                    .collect()
            }
        };
        Ok(Image32 {
            width: self.window,
            height: self.window,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DevicePreference;
    use crate::backend::fake::MeanInpainter;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    fn cpu() -> OrtBackend {
        OrtBackend::new(DevicePreference::Cpu).unwrap()
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

    #[test]
    fn lama_layout_fixture_fills_like_the_mean_inpainter() {
        let model = cpu()
            .load_inpainter(&fixture("inpaint_mean_lama.onnx"))
            .unwrap();
        // The window comes from the graph's fixed input shape.
        assert_eq!(model.window(), 16);
        assert_eq!(model.max_context_factor(), 4);
        assert_matches_mean(model.as_ref(), 5);
    }

    #[test]
    fn migan_layout_fixture_fills_like_the_mean_inpainter() {
        let model = cpu()
            .load_inpainter(&fixture("inpaint_mean_migan.onnx"))
            .unwrap();
        // Dynamic spatial size: the 512 window MI-GAN is run at.
        assert_eq!(model.window(), 512);
        assert_eq!(model.max_context_factor(), 1);
        assert_matches_mean(model.as_ref(), 200);
    }

    #[test]
    fn fill_rejects_a_window_of_the_wrong_size() {
        let model = cpu()
            .load_inpainter(&fixture("inpaint_mean_lama.onnx"))
            .unwrap();
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
            .load_inpainter(&fixture("nearest2x.onnx"))
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
        assert_eq!(
            Layout::detect(&pair(Float32, 512)).unwrap(),
            (Layout::LaMa, 512)
        );
        assert_eq!(
            Layout::detect(&pair(Float32, 16)).unwrap(),
            (Layout::LaMa, 16)
        );
        assert_eq!(
            Layout::detect(&pair(Uint8, -1)).unwrap(),
            (Layout::MiGan, 512)
        );
        // Input order does not matter.
        let mut swapped = pair(Uint8, 256);
        swapped.reverse();
        assert_eq!(Layout::detect(&swapped).unwrap(), (Layout::MiGan, 256));

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
            let err = Layout::detect(&inputs).unwrap_err();
            assert!(matches!(err, ScaleUpError::Model(_)), "{inputs:?}");
            assert!(err.to_string().contains("inputs found: "), "{err}");
        }
    }

    #[test]
    fn lama_runs_on_the_cpu_whatever_the_device() {
        let gpu = DeviceInfo {
            kind: DeviceKind::DirectMl,
            index: 0,
            name: "DirectML GPU 0".into(),
        };
        assert_eq!(Layout::LaMa.device_for(&gpu).kind, DeviceKind::Cpu);
        assert_eq!(Layout::MiGan.device_for(&gpu), gpu);
        assert_eq!(Layout::MiGan.device_for(&cpu_device()), cpu_device());
    }

    #[test]
    fn missing_inpainter_file_is_a_model_error() {
        let err = cpu()
            .load_inpainter(Path::new("no-such-inpainter.onnx"))
            .err()
            .unwrap();
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
    }
}
