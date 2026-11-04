//! High-level entry point: backend + registry, file-to-file upscaling with backend-aware defaults.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::backend::{
    Backend, DeviceInfo, DevicePreference, Model, OrtBackend, create_backend, cross_preference,
    extra_backend, is_ncnn_model,
};
use crate::io::{OutputFormat, load_image, save_image};
use crate::pipeline::{CancelToken, Phase, Progress, UpscaleOptions, upscale};
use crate::registry::{Registry, scale_from_name};
use crate::resize::{resize_rgba, target_size};
use crate::tiling::{auto_tile_size, tile_grid};
use crate::{Result, ScaleUpError};

/// One input file → one output file.
#[derive(Clone, Debug, PartialEq, Eq)]
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
}

/// A backend plus the model registry.
pub struct Upscaler {
    backend: Box<dyn Backend>,
    registry: Registry,
    pref: DevicePreference,
    /// Created on demand for model files the main backend cannot load.
    other_backend: OnceLock<Box<dyn Backend>>,
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

    /// The backend for the model file at `path`: ncnn files need the registered backend, and ONNX
    /// files need ONNX Runtime whatever device was chosen.
    fn backend_for(&self, path: &Path) -> Result<&dyn Backend> {
        if self.backend.handles(path) {
            return Ok(self.backend.as_ref());
        }
        if let Some(b) = self.other_backend.get() {
            return Ok(b.as_ref());
        }
        let created: Box<dyn Backend> = if is_ncnn_model(path) {
            let extra = extra_backend().ok_or_else(|| {
                ScaleUpError::Model(
                    "ncnn models need the wgpu backend, absent from this build".into(),
                )
            })?;
            (extra.create)(cross_preference(self.pref, true)).map_err(|e| {
                ScaleUpError::Backend(format!("ncnn models need a GPU adapter: {e}"))
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
        let (path, scale) = match self.registry.get(model) {
            Some(entry) => (
                self.registry.ensure(model, download_progress)?,
                Some(entry.scale),
            ),
            None => {
                let resolved = self.registry.resolve(model)?;
                (resolved.path, resolved.scale)
            }
        };
        let backend = self.backend_for(&path)?;
        let model = backend.load_model(&path, scale)?;
        *self.active.lock().unwrap_or_else(|e| e.into_inner()) = backend.device().clone();
        Ok(model)
    }

    /// `requested`, or the device's default tile size when it is 0.
    pub fn effective_tile(&self, requested: u32) -> u32 {
        if requested == 0 {
            auto_tile_size(self.active_device().kind, None)
        } else {
            requested
        }
    }

    /// Upscales `job.input` into `job.output` and returns the output path. An existing output is left
    /// alone (and its path returned) unless `job.overwrite`. The output is written atomically, so a
    /// cancelled or failed run leaves no file behind.
    pub fn run_file(
        &self,
        model: &dyn Model,
        job: &FileJob,
        opts: &UpscaleOptions,
        progress: &mut dyn FnMut(Progress),
        cancel: &CancelToken,
    ) -> Result<PathBuf> {
        if job.output.exists() && !job.overwrite {
            return Ok(job.output.clone());
        }
        progress(Progress::Phase(Phase::Loading));
        let input = load_image(&job.input)?;
        let opts = UpscaleOptions {
            tile_size: self.effective_tile(opts.tile_size),
            ..opts.clone()
        };

        let output = if job.double {
            self.upscale_twice(model, &input, &opts, progress, cancel)?
        } else {
            upscale(model, &input, &opts, progress, cancel)?
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
        let second_estimate = pass_tiles(w * s, h * s, opts.tile_size, opts.tile_overlap, alpha);

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
    /// `s` comes from the model id (Upscayl's name rule).
    pub fn output_name(
        input: &Path,
        out_dir: &Path,
        model_id: &str,
        opts: &UpscaleOptions,
        fmt: OutputFormat,
    ) -> PathBuf {
        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();
        let size = match (opts.output_width, opts.output_scale) {
            (Some(w), _) => format!("{w}px"),
            (None, Some(s)) => format!("{s}x"),
            (None, None) => format!("{}x", scale_from_name(model_id)),
        };
        out_dir.join(format!("{stem}_scaleup_{size}_{model_id}.{}", fmt.ext()))
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
    use crate::ScaleUpError;
    use crate::backend::DeviceKind;
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
    fn effective_tile_uses_device_default() {
        let dir = tempfile::tempdir().unwrap();
        let up = cpu_upscaler(dir.path());
        assert_eq!(up.device().kind, DeviceKind::Cpu);
        assert_eq!(up.effective_tile(0), 128);
        assert_eq!(up.effective_tile(300), 300);
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
                model.as_ref(),
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
            .run_file(model.as_ref(), &j, &opts, &mut |_| {}, &CancelToken::new())
            .unwrap();
        assert_eq!(got, output);
        assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
        j.overwrite = true;
        up.run_file(model.as_ref(), &j, &opts, &mut |_| {}, &CancelToken::new())
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
            model.as_ref(),
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
            model.as_ref(),
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
