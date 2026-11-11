//! High-level entry point: backend + registry, file-to-file upscaling with backend-aware defaults.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::backend::{
    Backend, DeviceInfo, DevicePreference, Inpainter, Model, OrtBackend, create_backend,
    cross_preference, extra_backend, is_extra_backend_model,
};
use crate::extend::{ExtendOptions, Fill, Margins, outpaint, pad_canvas};
use crate::image32::Image32;
use crate::io::{OutputFormat, load_image, save_image};
use crate::pipeline::{CancelToken, Phase, Progress, UpscaleOptions, upscale};
use crate::registry::{ModelKind, Registry, scale_from_name};
use crate::resize::{resize_rgba, target_size};
use crate::tiling::{auto_tile_size, room_for_overlap, tile_grid};
use crate::{Result, ScaleUpError};

/// One input file → one output file.
/// Not `Eq`: percentage margins hold floats.
#[derive(Clone, Debug, PartialEq)]
pub struct FileJob {
    pub input: PathBuf,
    pub output: PathBuf,
    pub format: OutputFormat,
    /// 0..=100, see [`save_image`].
    pub quality: u8,
    /// Replace an existing output file instead of skipping the job.
    pub overwrite: bool,
    /// Run the model twice (second pass on the first result) before the final resize.
    pub double: bool,
    /// Add canvas around the image before upscaling.
    pub extend: Option<ExtendOptions>,
}

/// A model whose catalogue entry names the overlap: the ONNX file itself does not say what it is.
struct WithOverlap {
    inner: Box<dyn Model>,
    overlap: u32,
}

impl Model for WithOverlap {
    fn scale(&self) -> u32 {
        self.inner.scale()
    }

    fn input_multiple(&self) -> u32 {
        self.inner.input_multiple()
    }

    fn max_input_side(&self) -> Option<u32> {
        self.inner.max_input_side()
    }

    fn preferred_overlap(&self) -> u32 {
        self.overlap
    }

    fn run(&self, tile: &Image32) -> Result<Image32> {
        self.inner.run(tile)
    }
}

/// A backend plus the model registry.
pub struct Upscaler {
    backend: Box<dyn Backend>,
    registry: Registry,
    pref: DevicePreference,
    /// Created on demand for model files the main backend cannot load.
    other_backend: OnceLock<Box<dyn Backend>>,
    /// Runs inpainting models, which are always ONNX: the main backend's device when that is ONNX
    /// Runtime, else an ONNX Runtime backend created on demand.
    inpaint_backend: OnceLock<OrtBackend>,
    /// Device of the most recently loaded model.
    active: Mutex<DeviceInfo>,
}

/// Tiles one `upscale` pass over a `w`×`h` image will report (×2 when alpha gets its own pass).
fn pass_tiles(w: u32, h: u32, tile: u32, overlap: u32, alpha: bool) -> u32 {
    let n = tile_grid(w, h, tile, overlap).map_or(1, |jobs| jobs.len() as u32);
    if alpha { 2 * n } else { n }
}

impl Upscaler {
    /// Creates the default backend for `pref` (GPU failure falls back to CPU).
    pub fn new(pref: DevicePreference, registry: Registry) -> Result<Self> {
        let backend = create_backend(pref)?;
        Ok(Self {
            active: Mutex::new(backend.device().clone()),
            backend,
            registry,
            pref,
            other_backend: OnceLock::new(),
            inpaint_backend: OnceLock::new(),
        })
    }

    pub fn device(&self) -> &DeviceInfo {
        self.backend.device()
    }

    /// Where the last loaded model runs. Differs from [`Upscaler::device`] when the model's format
    /// needed the other backend.
    pub fn active_device(&self) -> DeviceInfo {
        self.active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The backend for the model file at `path`: ncnn and PyTorch files need the registered
    /// backend, and ONNX files need ONNX Runtime whatever device was chosen.
    fn backend_for(&self, path: &Path) -> Result<&dyn Backend> {
        if self.backend.handles(path) {
            return Ok(self.backend.as_ref());
        }
        if let Some(b) = self.other_backend.get() {
            return Ok(b.as_ref());
        }
        let created: Box<dyn Backend> = if is_extra_backend_model(path) {
            let extra = extra_backend().ok_or_else(|| {
                ScaleUpError::Model(
                    "ncnn and PyTorch models need the wgpu backend, absent from this build".into(),
                )
            })?;
            (extra.create)(cross_preference(self.pref, true)).map_err(|e| {
                ScaleUpError::Backend(format!("ncnn and PyTorch models need a GPU adapter: {e}"))
            })?
        } else {
            Box::new(OrtBackend::new(cross_preference(self.pref, false))?)
        };
        Ok(self.other_backend.get_or_init(|| created).as_ref())
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Loads a catalogue model id (downloading it if needed) or a model file path.
    pub fn load(
        &self,
        model: &str,
        download_progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Box<dyn Model>> {
        let (path, scale, overlap) = match self.registry.get(model) {
            Some(entry) if entry.kind == ModelKind::Inpaint => {
                return Err(ScaleUpError::InvalidOption(format!(
                    "'{model}' is an inpainting model for canvas extension, not an upscaler; \
                     pass it with --extend-model"
                )));
            }
            Some(entry) => (
                self.registry.ensure(model, download_progress)?,
                Some(entry.scale),
                entry.preferred_overlap(),
            ),
            None => {
                let resolved = self.registry.resolve(model)?;
                (resolved.path, resolved.scale, None)
            }
        };
        let backend = self.backend_for(&path)?;
        let model = backend.load_model(&path, scale)?;
        *self.active.lock().unwrap_or_else(|e| e.into_inner()) = backend.device().clone();
        Ok(match overlap {
            Some(overlap) => Box::new(WithOverlap {
                inner: model,
                overlap,
            }),
            None => model,
        })
    }

    /// Loads an inpainting model: a catalogue id of kind `inpaint` (downloading it if needed) or an
    /// ONNX file path. It always runs on ONNX Runtime, on the main backend's device when that is
    /// ONNX Runtime and otherwise on the device `pref` names there. An upscaling catalogue id, a
    /// missing model or a graph of neither inpainting layout is an error.
    pub fn load_inpainter(
        &self,
        id_or_path: &str,
        download_progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Box<dyn Inpainter>> {
        let path = match self.registry.get(id_or_path) {
            Some(entry) if entry.kind != ModelKind::Inpaint => {
                let ids: Vec<&str> = self.registry.inpainters().map(|e| e.id.as_str()).collect();
                return Err(ScaleUpError::InvalidOption(format!(
                    "'{id_or_path}' is an upscaling model, not an inpainting model (inpainting \
                     models: {})",
                    ids.join(", ")
                )));
            }
            Some(_) => self.registry.ensure(id_or_path, download_progress)?,
            None => self.registry.resolve(id_or_path)?.path,
        };
        self.inpaint_backend()?.load_inpainter(&path)
    }

    fn inpaint_backend(&self) -> Result<&OrtBackend> {
        if let Some(b) = self.inpaint_backend.get() {
            return Ok(b);
        }
        let created = if self.backend.name() == "onnxruntime" {
            OrtBackend::on_device(self.backend.device().clone())
        } else {
            OrtBackend::new(cross_preference(self.pref, false))?
        };
        Ok(self.inpaint_backend.get_or_init(|| created))
    }

    /// `requested`, or when it is 0 the device's default tile size, grown to leave room for `overlap`.
    pub fn effective_tile(&self, requested: u32, overlap: u32) -> u32 {
        if requested == 0 {
            room_for_overlap(auto_tile_size(self.active_device().kind, None), overlap)
        } else {
            requested
        }
    }

    /// Extends and/or upscales `job.input` into `job.output` and returns the output path: load, then
    /// extend (`job.extend`), then upscale, resize and save. An existing output is left alone (and its
    /// path returned) unless `job.overwrite`. The output is written atomically, so a cancelled or
    /// failed run leaves no file behind.
    ///
    /// `inpainter` is required when `job.extend` uses [`Fill::Ai`], else `InvalidOption`. The margins
    /// are resolved against the loaded image before any other work, so a margin or canvas over the
    /// limits is `InvalidOption` with nothing extended or upscaled. Margins that resolve to zero (an
    /// aspect the image already has) skip extension entirely.
    ///
    /// `model == None` extends without upscaling: the output is the extended image at its own size,
    /// and `opts.output_scale`, `opts.output_width`, `opts.tta` and `job.double` are ignored. With no
    /// model, `job.extend` must be set (`InvalidOption` otherwise).
    ///
    /// Phases come in the order `Loading, Extending, Upscaling, Resizing, Saving`, each only when it
    /// applies. The AI fill reports one tile per window; the upscale then counts its own tiles from
    /// zero.
    pub fn run_file(
        &self,
        model: Option<&dyn Model>,
        inpainter: Option<&dyn Inpainter>,
        job: &FileJob,
        opts: &UpscaleOptions,
        progress: &mut dyn FnMut(Progress),
        cancel: &CancelToken,
    ) -> Result<PathBuf> {
        if model.is_none() && job.extend.is_none() {
            return Err(ScaleUpError::InvalidOption(
                "nothing to do: no upscale model and no canvas extension".into(),
            ));
        }
        if let Some(ext) = &job.extend
            && ext.fill == Fill::Ai
            && inpainter.is_none()
        {
            return Err(ScaleUpError::InvalidOption(
                "AI fill needs an inpainting model".into(),
            ));
        }
        if job.output.exists() && !job.overwrite {
            return Ok(job.output.clone());
        }
        progress(Progress::Phase(Phase::Loading));
        let mut image = load_image(&job.input)?;

        if let Some(ext) = &job.extend {
            let margins = ext.margins.resolve(image.width(), image.height())?;
            if margins != Margins::default() {
                image = match (&ext.fill, inpainter) {
                    (Fill::Ai, Some(inpainter)) => {
                        outpaint(inpainter, &image, margins, progress, cancel)?
                    }
                    (Fill::Ai, None) => unreachable!("checked above"),
                    (fill, _) => {
                        progress(Progress::Phase(Phase::Extending));
                        pad_canvas(&image, margins, fill)?
                    }
                };
                if cancel.is_cancelled() {
                    return Err(ScaleUpError::Cancelled);
                }
            }
        }

        let output = match model {
            Some(model) => {
                let opts = UpscaleOptions {
                    tile_size: self.effective_tile(opts.tile_size, opts.overlap_for(model)),
                    ..opts.clone()
                };
                if job.double {
                    self.upscale_twice(model, &image, &opts, progress, cancel)?
                } else {
                    upscale(model, &image, &opts, progress, cancel)?
                }
            }
            None => image,
        };
        if cancel.is_cancelled() {
            return Err(ScaleUpError::Cancelled);
        }

        progress(Progress::Phase(Phase::Saving));
        if let Some(parent) = job.output.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        save_image(&job.output, &output, job.format, job.quality)?;
        Ok(job.output.clone())
    }

    /// Two model passes at the model's scale, then one resize. With `output_scale = s` the result is
    /// `s²` times the input; with `output_width` it is that width. Tile progress spans both passes.
    fn upscale_twice(
        &self,
        model: &dyn Model,
        input: &image::RgbaImage,
        opts: &UpscaleOptions,
        progress: &mut dyn FnMut(Progress),
        cancel: &CancelToken,
    ) -> Result<image::RgbaImage> {
        let (w, h) = input.dimensions();
        let s = model.scale();
        let target = match opts.output_width {
            Some(_) => target_size(w, h, s, opts.output_scale, opts.output_width)?,
            None => {
                let f = target_size(1, 1, s, opts.output_scale, None)?.0;
                (w * f * f, h * f * f)
            }
        };
        let pass_opts = UpscaleOptions {
            output_scale: None,
            output_width: None,
            ..opts.clone()
        };
        let alpha = input.pixels().any(|p| p[3] != 255);
        let second_estimate =
            pass_tiles(w * s, h * s, opts.tile_size, opts.overlap_for(model), alpha);

        let mut first_total = 0;
        let first = upscale(
            model,
            input,
            &pass_opts,
            &mut |p| match p {
                Progress::Tiles { done, total } => {
                    first_total = total;
                    progress(Progress::Tiles {
                        done,
                        total: total + second_estimate,
                    });
                }
                other => progress(other),
            },
            cancel,
        )?;
        let second = upscale(
            model,
            &first,
            &pass_opts,
            &mut |p| match p {
                Progress::Tiles { done, total } => progress(Progress::Tiles {
                    done: first_total + done,
                    total: first_total + total,
                }),
                Progress::Phase(Phase::Upscaling) => {}
                other => progress(other),
            },
            cancel,
        )?;
        if second.dimensions() == target {
            return Ok(second);
        }
        progress(Progress::Phase(Phase::Resizing));
        Ok(resize_rgba(&second, target.0, target.1))
    }

    /// Output path for `input` in `out_dir`: `<stem>_scaleup_<s>x_<model_id>.<ext>`, or
    /// `<stem>_scaleup_<w>px_<model_id>.<ext>` with a custom width. Without an explicit output scale,
    /// `s` comes from the model id (Upscayl's name rule). See [`Upscaler::output_name_for`] for jobs
    /// that extend the canvas.
    pub fn output_name(
        input: &Path,
        out_dir: &Path,
        model_id: &str,
        opts: &UpscaleOptions,
        fmt: OutputFormat,
    ) -> PathBuf {
        Self::output_name_for(input, out_dir, Some(model_id), None, opts, fmt)
    }

    /// [`Upscaler::output_name`] for any job. `extend_fill` is the fill when the job extends the
    /// canvas: `_ext` then goes before the size (`<stem>_scaleup_ext_4x_<model_id>.<ext>`). With no
    /// upscale model the name is `<stem>_scaleup_ext_<fill>.<ext>`, `<fill>` being the fill's CLI word
    /// (`ai`, `mirror`, `edge`, `transparent`, or the colour as `rrggbb`). With neither, it is
    /// `<stem>_scaleup.<ext>`.
    pub fn output_name_for(
        input: &Path,
        out_dir: &Path,
        model_id: Option<&str>,
        extend_fill: Option<&Fill>,
        opts: &UpscaleOptions,
        fmt: OutputFormat,
    ) -> PathBuf {
        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();
        let ext = if extend_fill.is_some() { "_ext" } else { "" };
        let tail = match (model_id, extend_fill) {
            (Some(model_id), _) => {
                let size = match (opts.output_width, opts.output_scale) {
                    (Some(w), _) => format!("{w}px"),
                    (None, Some(s)) => format!("{s}x"),
                    (None, None) => format!("{}x", scale_from_name(model_id)),
                };
                format!("{ext}_{size}_{model_id}")
            }
            (None, Some(fill)) => format!("_ext_{}", fill_word(fill)),
            (None, None) => String::new(),
        };
        out_dir.join(format!("{stem}_scaleup{tail}.{}", fmt.ext()))
    }
}

/// The CLI word for `fill`, as [`Fill::parse`] reads it (a colour without its `#`).
fn fill_word(fill: &Fill) -> String {
    match fill {
        Fill::Ai => "ai".into(),
        Fill::Mirror => "mirror".into(),
        Fill::Edge => "edge".into(),
        Fill::Transparent => "transparent".into(),
        Fill::Color([r, g, b]) => format!("{r:02x}{g:02x}{b:02x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncnn_model_without_its_backend_is_a_clear_error() {
        let d = tempfile::tempdir().unwrap();
        let param = d.path().join("m.param");
        std::fs::write(&param, b"7767517").unwrap();
        let up = cpu_upscaler(d.path());
        let err = up
            .load(param.to_str().unwrap(), &mut |_, _| {})
            .err()
            .expect("no ncnn backend is registered in this test binary");
        assert!(err.to_string().contains("ncnn"), "{err}");
    }

    #[test]
    fn pytorch_model_without_its_backend_is_a_clear_error() {
        let d = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(d.path());
        for name in ["m.pth", "m.safetensors"] {
            let path = d.path().join(name);
            std::fs::write(&path, b"not read").unwrap();
            let err = up
                .load(path.to_str().unwrap(), &mut |_, _| {})
                .err()
                .expect("no wgpu backend is registered in this test binary");
            assert!(err.to_string().contains("wgpu backend"), "{name}: {err}");
        }
    }
    use crate::ScaleUpError;
    use crate::backend::DeviceKind;
    use crate::backend::fake::MeanInpainter;
    use crate::extend::MarginSpec;
    use image::{Rgba, RgbaImage};

    fn fixture() -> String {
        format!(
            "{}/../../tests/fixtures/nearest2x.onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn cpu_upscaler(dir: &Path) -> Upscaler {
        Upscaler::new(
            DevicePreference::Cpu,
            Registry::with_dir(dir.join("models")),
        )
        .unwrap()
    }

    fn write_input(dir: &Path, w: u32, h: u32) -> PathBuf {
        let path = dir.join("in.png");
        let img = RgbaImage::from_fn(w, h, |x, y| Rgba([(x * 50) as u8, (y * 60) as u8, 9, 255]));
        img.save(&path).unwrap();
        path
    }

    fn job(input: PathBuf, output: PathBuf) -> FileJob {
        FileJob {
            input,
            output,
            format: OutputFormat::Png,
            quality: 90,
            overwrite: false,
            double: false,
            extend: None,
        }
    }

    #[test]
    fn output_name_formats() {
        let out = Path::new("out");
        let scale4 = UpscaleOptions {
            output_scale: Some(4),
            ..Default::default()
        };
        assert_eq!(
            Upscaler::output_name(
                Path::new("dir/a.png"),
                out,
                "realesrgan-x4plus",
                &scale4,
                OutputFormat::Png
            ),
            out.join("a_scaleup_4x_realesrgan-x4plus.png")
        );
        let width = UpscaleOptions {
            output_width: Some(1920),
            ..scale4.clone()
        };
        assert_eq!(
            Upscaler::output_name(
                Path::new("a.png"),
                out,
                "realesrgan-x4plus",
                &width,
                OutputFormat::Jpg
            ),
            out.join("a_scaleup_1920px_realesrgan-x4plus.jpg")
        );
        // No explicit scale: the model id's scale (Upscayl name rule).
        assert_eq!(
            Upscaler::output_name(
                Path::new("b.jpeg"),
                out,
                "realesrgan-x2plus",
                &UpscaleOptions::default(),
                OutputFormat::Webp
            ),
            out.join("b_scaleup_2x_realesrgan-x2plus.webp")
        );
    }

    #[test]
    fn catalogue_model_uses_the_overlap_of_its_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let entry = crate::registry::ModelEntry {
            id: "span".into(),
            name: "span".into(),
            arch: "SPAN".into(),
            scale: 2,
            license: "MIT".into(),
            commercial: true,
            size_bytes: 0,
            url: String::new(),
            sha256: String::new(),
            tags: vec![],
            notes: String::new(),
            format: "onnx".into(),
            source: "catalogue".into(),
            kind: ModelKind::Upscale,
        };
        let registry = Registry::from_entries(vec![entry], dir.path().to_path_buf());
        std::fs::copy(fixture(), registry.local_path("span")).unwrap();
        let up = Upscaler::new(DevicePreference::Cpu, registry).unwrap();
        let model = up.load("span", &mut |_, _| {}).unwrap();
        assert_eq!(model.preferred_overlap(), 16);
        assert_eq!(model.scale(), 2);
    }

    #[test]
    fn model_file_of_unknown_architecture_uses_the_safe_overlap() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        assert_eq!(model.preferred_overlap(), 64);
    }

    #[test]
    fn effective_tile_uses_device_default() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        assert_eq!(up.device().kind, DeviceKind::Cpu);
        assert_eq!(up.effective_tile(0, 16), 128);
        assert_eq!(up.effective_tile(0, 64), 256);
        assert_eq!(up.effective_tile(300, 64), 300);
    }

    #[test]
    fn run_file_end_to_end_cpu() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        assert_eq!(model.scale(), 2);
        let input = write_input(dir.path(), 4, 4);
        let output = dir.path().join("sub").join("out.png");
        let mut phases = Vec::new();
        let got = up
            .run_file(
                Some(model.as_ref()),
                None,
                &job(input.clone(), output.clone()),
                &UpscaleOptions::default(),
                &mut |p| {
                    if let Progress::Phase(ph) = p {
                        phases.push(ph)
                    }
                },
                &CancelToken::new(),
            )
            .unwrap();
        assert_eq!(got, output);
        let result = load_image(&output).unwrap();
        assert_eq!(result.dimensions(), (8, 8));
        let src = load_image(&input).unwrap();
        assert_eq!(result.get_pixel(7, 5), src.get_pixel(3, 2));
        assert_eq!(
            phases,
            vec![Phase::Loading, Phase::Upscaling, Phase::Saving]
        );
    }

    #[test]
    fn run_file_skips_existing() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 4, 4);
        let output = dir.path().join("exists.png");
        std::fs::write(&output, b"keep me").unwrap();
        let mut j = job(input, output.clone());
        let opts = UpscaleOptions::default();
        let got = up
            .run_file(
                Some(model.as_ref()),
                None,
                &j,
                &opts,
                &mut |_| {},
                &CancelToken::new(),
            )
            .unwrap();
        assert_eq!(got, output);
        assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
        j.overwrite = true;
        up.run_file(
            Some(model.as_ref()),
            None,
            &j,
            &opts,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!(load_image(&output).unwrap().dimensions(), (8, 8));
    }

    #[test]
    fn double_quadruples() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 5, 3);
        let output = dir.path().join("double.png");
        let mut j = job(input, output.clone());
        j.double = true;
        let mut last_tiles = None;
        up.run_file(
            Some(model.as_ref()),
            None,
            &j,
            &UpscaleOptions::default(),
            &mut |p| {
                if let Progress::Tiles { done, total } = p {
                    assert!(done <= total);
                    last_tiles = Some((done, total));
                }
            },
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!(load_image(&output).unwrap().dimensions(), (20, 12));
        // One progress bar across both passes.
        assert_eq!(last_tiles, Some((2, 2)));
    }

    #[test]
    fn cancelled_run_leaves_no_output() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 300, 300);
        let output = dir.path().join("cancelled.png");
        let token = CancelToken::new();
        let observer = token.clone();
        let opts = UpscaleOptions {
            tile_size: 64,
            ..Default::default()
        };
        let res = up.run_file(
            Some(model.as_ref()),
            None,
            &job(input, output.clone()),
            &opts,
            &mut |p| {
                if let Progress::Tiles { .. } = p {
                    observer.cancel()
                }
            },
            &token,
        );
        assert!(matches!(res, Err(ScaleUpError::Cancelled)));
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("cancelled"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    fn right(px: u32, fill: Fill) -> ExtendOptions {
        ExtendOptions {
            margins: MarginSpec::Pixels(Margins {
                right: px,
                ..Margins::default()
            }),
            fill,
        }
    }

    fn extend_job(input: PathBuf, output: PathBuf, extend: ExtendOptions) -> FileJob {
        FileJob {
            extend: Some(extend),
            ..job(input, output)
        }
    }

    /// Runs `job` and returns the phases and tile events it reported.
    fn run_logged(
        up: &Upscaler,
        model: Option<&dyn Model>,
        inpainter: Option<&dyn Inpainter>,
        job: &FileJob,
        opts: &UpscaleOptions,
    ) -> (Result<PathBuf>, Vec<Progress>) {
        let mut log = Vec::new();
        let res = up.run_file(
            model,
            inpainter,
            job,
            opts,
            &mut |p| log.push(p),
            &CancelToken::new(),
        );
        (res, log)
    }

    fn phases_of(log: &[Progress]) -> Vec<Phase> {
        log.iter()
            .filter_map(|p| match p {
                Progress::Phase(ph) => Some(*ph),
                Progress::Tiles { .. } => None,
            })
            .collect()
    }

    fn assert_left_is_plain_upscale(result: &RgbaImage, plain: &RgbaImage) {
        for y in 0..plain.height() {
            for x in 0..plain.width() {
                assert_eq!(result.get_pixel(x, y), plain.get_pixel(x, y), "at {x},{y}");
            }
        }
    }

    #[test]
    fn run_file_extends_then_upscales() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 4, 4);
        let plain_out = dir.path().join("plain.png");
        up.run_file(
            Some(model.as_ref()),
            None,
            &job(input.clone(), plain_out.clone()),
            &UpscaleOptions::default(),
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        let plain = load_image(&plain_out).unwrap();
        let src = load_image(&input).unwrap();
        let mean = MeanInpainter { window: 16 };
        for (name, fill, inpainter) in [
            ("edge", Fill::Edge, None),
            ("ai", Fill::Ai, Some(&mean as &dyn Inpainter)),
        ] {
            let output = dir.path().join(format!("{name}.png"));
            let j = extend_job(input.clone(), output.clone(), right(2, fill));
            let (res, log) = run_logged(
                &up,
                Some(model.as_ref()),
                inpainter,
                &j,
                &UpscaleOptions::default(),
            );
            assert_eq!(res.unwrap(), output);
            let result = load_image(&output).unwrap();
            assert_eq!(result.dimensions(), (12, 8), "{name}");
            assert_left_is_plain_upscale(&result, &plain);
            assert_eq!(
                phases_of(&log),
                vec![
                    Phase::Loading,
                    Phase::Extending,
                    Phase::Upscaling,
                    Phase::Saving
                ],
                "{name}"
            );
            // The upscale pass counts its own tiles from zero.
            let upscaling = log
                .iter()
                .position(|p| *p == Progress::Phase(Phase::Upscaling))
                .unwrap();
            assert!(matches!(
                log[upscaling + 1],
                Progress::Tiles { done: 1, .. }
            ));
            if name == "edge" {
                // The new column repeats the last source column, upscaled.
                assert_eq!(result.get_pixel(11, 5), src.get_pixel(3, 2));
            } else {
                // The AI path reported its windows before the upscale.
                assert!(matches!(log[2], Progress::Tiles { .. }), "{log:?}");
            }
        }
    }

    #[test]
    fn run_file_extend_only_keeps_the_extended_size() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let input = write_input(dir.path(), 4, 4);
        let output = dir.path().join("ext.png");
        let fill = Fill::Color([1, 2, 3]);
        let j = extend_job(input.clone(), output.clone(), right(2, fill.clone()));
        // Upscale options do not apply without a model.
        let opts = UpscaleOptions {
            output_scale: Some(4),
            output_width: Some(100),
            ..Default::default()
        };
        let (res, log) = run_logged(&up, None, None, &j, &opts);
        assert_eq!(res.unwrap(), output);
        let result = load_image(&output).unwrap();
        assert_eq!(result.dimensions(), (6, 4));
        let src = load_image(&input).unwrap();
        let expected = pad_canvas(
            &src,
            Margins {
                right: 2,
                ..Margins::default()
            },
            &fill,
        )
        .unwrap();
        assert_eq!(result, expected);
        assert_eq!(
            phases_of(&log),
            vec![Phase::Loading, Phase::Extending, Phase::Saving]
        );
    }

    #[test]
    fn run_file_zero_margins_skip_extending() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 4, 4);
        let output = dir.path().join("square.png");
        let extend = ExtendOptions {
            margins: MarginSpec::Aspect { w: 1, h: 1 },
            fill: Fill::Ai,
        };
        // Already square: no inpainter is needed because nothing is extended.
        let mean = MeanInpainter { window: 16 };
        let j = extend_job(input, output.clone(), extend);
        let (res, log) = run_logged(
            &up,
            Some(model.as_ref()),
            Some(&mean),
            &j,
            &UpscaleOptions::default(),
        );
        res.unwrap();
        assert_eq!(load_image(&output).unwrap().dimensions(), (8, 8));
        assert_eq!(
            phases_of(&log),
            vec![Phase::Loading, Phase::Upscaling, Phase::Saving]
        );
    }

    #[test]
    fn run_file_refuses_invalid_extension_before_any_work() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 4, 4);
        let output = dir.path().join("refused.png");
        let opts = UpscaleOptions::default();
        // No model and nothing to extend.
        let (res, log) = run_logged(&up, None, None, &job(input.clone(), output.clone()), &opts);
        assert!(
            matches!(res, Err(ScaleUpError::InvalidOption(_))),
            "{res:?}"
        );
        assert!(log.is_empty());
        // AI fill without an inpainter: never a quiet plain fill.
        let j = extend_job(input.clone(), output.clone(), right(2, Fill::Ai));
        let (res, log) = run_logged(&up, Some(model.as_ref()), None, &j, &opts);
        assert!(
            matches!(res, Err(ScaleUpError::InvalidOption(_))),
            "{res:?}"
        );
        assert!(log.is_empty());
        // A margin over 4x the side: refused right after loading.
        let j = extend_job(input, output.clone(), right(17, Fill::Edge));
        let (res, log) = run_logged(&up, Some(model.as_ref()), None, &j, &opts);
        assert!(
            matches!(res, Err(ScaleUpError::InvalidOption(_))),
            "{res:?}"
        );
        assert_eq!(phases_of(&log), vec![Phase::Loading]);
        assert!(!output.exists());
    }

    #[test]
    fn run_file_skips_existing_with_extension() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 4, 4);
        let output = dir.path().join("exists.png");
        std::fs::write(&output, b"keep me").unwrap();
        let mut j = extend_job(input, output.clone(), right(2, Fill::Mirror));
        let opts = UpscaleOptions::default();
        let got = up
            .run_file(
                Some(model.as_ref()),
                None,
                &j,
                &opts,
                &mut |_| {},
                &CancelToken::new(),
            )
            .unwrap();
        assert_eq!(got, output);
        assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
        j.overwrite = true;
        up.run_file(
            Some(model.as_ref()),
            None,
            &j,
            &opts,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!(load_image(&output).unwrap().dimensions(), (12, 8));
    }

    #[test]
    fn cancelled_extension_leaves_no_output() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        let model = up.load(&fixture(), &mut |_, _| {}).unwrap();
        let input = write_input(dir.path(), 64, 64);
        let mean = MeanInpainter { window: 16 };
        let opts = UpscaleOptions {
            tile_size: 64,
            ..Default::default()
        };
        // Cancelled during the extension (with and without an upscale model), then during the
        // upscale that follows it.
        for (name, with_model, cancel_in) in [
            ("cancel_ext", true, Phase::Extending),
            ("cancel_extonly", false, Phase::Extending),
            ("cancel_up", true, Phase::Upscaling),
        ] {
            let output = dir.path().join(format!("{name}.png"));
            let j = extend_job(input.clone(), output.clone(), right(32, Fill::Ai));
            let token = CancelToken::new();
            let observer = token.clone();
            let mut phase = Phase::Loading;
            let mut extended_tiles = 0;
            let res = up.run_file(
                with_model.then_some(model.as_ref()),
                Some(&mean),
                &j,
                &opts,
                &mut |p| match p {
                    Progress::Phase(ph) => phase = ph,
                    Progress::Tiles { .. } => {
                        if phase == Phase::Extending {
                            extended_tiles += 1;
                        }
                        if phase == cancel_in {
                            observer.cancel()
                        }
                    }
                },
                &token,
            );
            assert!(
                matches!(res, Err(ScaleUpError::Cancelled)),
                "{name}: {res:?}"
            );
            if cancel_in == Phase::Extending {
                assert_eq!(extended_tiles, 1, "{name}: stops after the first window");
            }
            let leftovers: Vec<_> = std::fs::read_dir(dir.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with(name))
                .collect();
            assert!(leftovers.is_empty(), "{name}: {leftovers:?}");
        }
    }

    #[test]
    fn output_name_for_extensions() {
        let out = Path::new("out");
        let scale4 = UpscaleOptions {
            output_scale: Some(4),
            ..Default::default()
        };
        let name = |model: Option<&str>, fill: Option<&Fill>, opts: &UpscaleOptions| {
            Upscaler::output_name_for(
                Path::new("dir/a.png"),
                out,
                model,
                fill,
                opts,
                OutputFormat::Png,
            )
        };
        // Not extending: the same name as `output_name`.
        assert_eq!(
            name(Some("realesrgan-x4plus"), None, &scale4),
            Upscaler::output_name(
                Path::new("dir/a.png"),
                out,
                "realesrgan-x4plus",
                &scale4,
                OutputFormat::Png
            )
        );
        assert_eq!(
            name(Some("realesrgan-x4plus"), Some(&Fill::Ai), &scale4),
            out.join("a_scaleup_ext_4x_realesrgan-x4plus.png")
        );
        let width = UpscaleOptions {
            output_width: Some(1920),
            ..Default::default()
        };
        assert_eq!(
            name(Some("realesrgan-x4plus"), Some(&Fill::Edge), &width),
            out.join("a_scaleup_ext_1920px_realesrgan-x4plus.png")
        );
        // No upscale model: the fill's CLI word, and no size.
        for (fill, word) in [
            (Fill::Ai, "ai"),
            (Fill::Mirror, "mirror"),
            (Fill::Edge, "edge"),
            (Fill::Transparent, "transparent"),
            (Fill::Color([0xab, 0x01, 0xff]), "ab01ff"),
        ] {
            assert_eq!(
                name(None, Some(&fill), &scale4),
                out.join(format!("a_scaleup_ext_{word}.png"))
            );
        }
    }

    fn inpaint_entry(id: &str) -> crate::registry::ModelEntry {
        crate::registry::ModelEntry {
            id: id.into(),
            name: id.into(),
            arch: "LaMa".into(),
            scale: 1,
            license: "MIT".into(),
            commercial: true,
            size_bytes: 0,
            url: String::new(),
            sha256: String::new(),
            tags: vec![],
            notes: String::new(),
            format: "onnx".into(),
            source: "catalogue".into(),
            kind: ModelKind::Inpaint,
        }
    }

    fn inpaint_fixture() -> String {
        format!(
            "{}/../../tests/fixtures/inpaint_mean_lama.onnx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn an_inpainting_model_is_not_loaded_as_an_upscaler() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::from_entries(vec![inpaint_entry("fill")], dir.path().into());
        std::fs::copy(inpaint_fixture(), registry.local_path("fill")).unwrap();
        let up = Upscaler::new(DevicePreference::Cpu, registry).unwrap();
        let err = up.load("fill", &mut |_, _| {}).err().expect("refused");
        assert!(matches!(err, ScaleUpError::InvalidOption(_)), "{err:?}");
        assert!(err.to_string().contains("--extend-model"), "{err}");
    }

    #[test]
    fn load_inpainter_takes_a_catalogue_id_or_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::from_entries(vec![inpaint_entry("fill")], dir.path().into());
        std::fs::copy(inpaint_fixture(), registry.local_path("fill")).unwrap();
        let up = Upscaler::new(DevicePreference::Cpu, registry).unwrap();
        let by_id = up.load_inpainter("fill", &mut |_, _| {}).unwrap();
        assert_eq!(by_id.window(), 16);
        let by_path = up
            .load_inpainter(&inpaint_fixture(), &mut |_, _| {})
            .unwrap();
        assert_eq!(by_path.window(), 16);
    }

    #[test]
    fn load_inpainter_refuses_upscalers_and_missing_models() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        // An upscaling catalogue model, without downloading it.
        let err = up
            .load_inpainter("realesr-general-x4v3", &mut |_, _| {})
            .err()
            .expect("refused");
        assert!(matches!(err, ScaleUpError::InvalidOption(_)), "{err:?}");
        // An upscaling model file: neither inpainting layout.
        let err = up.load_inpainter(&fixture(), &mut |_, _| {}).err().unwrap();
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
        // Nothing there: an error, never a quiet fallback.
        let err = up
            .load_inpainter("no-such-inpainter", &mut |_, _| {})
            .err()
            .unwrap();
        assert!(matches!(err, ScaleUpError::Model(_)), "{err:?}");
    }

    #[test]
    #[ignore = "downloads both inpainting models (~240 MB); run with: \
                cargo test -p scaleup-core --lib real_inpainters -- --ignored --nocapture"]
    fn real_inpainters_extend_a_photo() {
        use crate::extend::{Margins, outpaint};
        use std::time::Instant;
        // SCALEUP_MODELS_DIR may point at a folder that already holds the files.
        let up = Upscaler::new(DevicePreference::Auto, Registry::builtin()).unwrap();
        println!("main device: {}", up.device().name);
        let photo = format!(
            "{}/../../tests/fixtures/images/photo_like_256x192.jpg",
            env!("CARGO_MANIFEST_DIR")
        );
        let img = load_image(Path::new(&photo)).unwrap();
        let (w, h) = img.dimensions();
        let m = Margins {
            left: 0,
            top: 0,
            right: 64,
            bottom: 0,
        };
        for id in ["lama-512", "migan-512-places2"] {
            let reg = up.registry();
            let path = reg.ensure(id, &mut |_, _| {}).unwrap();
            let sha = crate::registry::sha256_of(&mut fs::File::open(&path).unwrap()).unwrap();
            assert_eq!(sha, reg.get(id).unwrap().sha256, "{id}: {}", path.display());
            let t = Instant::now();
            let model = up.load_inpainter(id, &mut |_, _| {}).unwrap();
            let load = t.elapsed().as_secs_f32();
            let t = Instant::now();
            let out = outpaint(model.as_ref(), &img, m, &mut |_| {}, &CancelToken::new()).unwrap();
            let run = t.elapsed().as_secs_f32();
            assert_eq!(out.dimensions(), (w + 64, h));
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(out.get_pixel(x, y), img.get_pixel(x, y), "{id} at {x},{y}");
                }
            }
            let luma: Vec<f64> = (0..h)
                .flat_map(|y| (w..w + 64).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let p = out.get_pixel(x, y);
                    0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
                })
                .collect();
            let mean = luma.iter().sum::<f64>() / luma.len() as f64;
            let sd =
                (luma.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / luma.len() as f64).sqrt();
            println!(
                "{id}: window {}, k max {}, load {load:.1} s, extend {run:.1} s, \
                 new strip luma mean {mean:.1} sd {sd:.1}",
                model.window(),
                model.max_context_factor()
            );
            assert!(sd > 2.0, "{id}: the new strip is flat (sd {sd:.2})");
        }
    }

    #[test]
    fn load_unknown_model_fails() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        assert!(matches!(
            up.load("no-such-model", &mut |_, _| {}),
            Err(ScaleUpError::Model(_))
        ));
    }
}
