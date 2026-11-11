//! `scaleup upscale` and `scaleup extend`: single image and folder mode.

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use scaleup_core::io::is_supported_input;
use scaleup_core::{
    CancelToken, ExtendOptions, FileJob, Fill, Inpainter, Model, OutputFormat, ScaleUpError,
    UpscaleOptions, Upscaler,
};

use crate::args::{CanvasArgs, CommonArgs, ExtendArgs, UpscaleArgs, UpscaleOpts, parse_gpu};
use crate::models::registry_for;
use crate::report::Reporter;

pub const EXIT_USAGE: i32 = 1;
pub const EXIT_IO: i32 = 2;
pub const EXIT_MODEL: i32 = 3;
pub const EXIT_CANCELLED: i32 = 4;

pub fn exit_code_for(err: &ScaleUpError) -> i32 {
    match err {
        ScaleUpError::InvalidOption(_) => EXIT_USAGE,
        ScaleUpError::Io(_) | ScaleUpError::Decode(_) | ScaleUpError::Encode(_) => EXIT_IO,
        ScaleUpError::Model(_) | ScaleUpError::Backend(_) | ScaleUpError::Download(_) => EXIT_MODEL,
        ScaleUpError::Cancelled => EXIT_CANCELLED,
    }
}

/// `-f`, else the output path's extension, else png.
pub fn resolve_format(args: &CommonArgs) -> OutputFormat {
    args.format
        .as_deref()
        .and_then(OutputFormat::from_ext)
        .or_else(|| {
            args.output
                .extension()
                .and_then(|e| e.to_str())
                .and_then(OutputFormat::from_ext)
        })
        .unwrap_or(OutputFormat::Png)
}

fn has_trailing_separator(p: &Path) -> bool {
    let s = p.to_string_lossy();
    s.ends_with('/') || s.ends_with('\\')
}

enum Outcome {
    Done,
    Skipped,
    Failed(ScaleUpError),
}

/// The upscaling model of a run; absent for `scaleup extend`.
struct Loaded<'a> {
    model: &'a dyn Model,
    id: &'a str,
}

struct Ctx<'a> {
    upscaler: &'a Upscaler,
    upscale: Option<Loaded<'a>>,
    inpainter: Option<&'a dyn Inpainter>,
    extend: Option<&'a ExtendOptions>,
    opts: &'a UpscaleOptions,
    double: bool,
    args: &'a CommonArgs,
    rep: &'a Reporter,
    cancel: &'a CancelToken,
}

impl Ctx<'_> {
    fn output_name(&self, input: &Path, out_dir: &Path, format: OutputFormat) -> PathBuf {
        Upscaler::output_name_for(
            input,
            out_dir,
            self.upscale.as_ref().map(|u| u.id),
            self.extend.map(|e| &e.fill),
            self.opts,
            format,
        )
    }

    fn process(&self, input: &Path, output: PathBuf, format: OutputFormat) -> Outcome {
        if output.exists() && !self.args.overwrite {
            self.rep.warn(&format!(
                "{} exists, skipping (use --overwrite)",
                output.display()
            ));
            return Outcome::Skipped;
        }
        let job = FileJob {
            input: input.to_path_buf(),
            output,
            format,
            quality: self.args.quality,
            overwrite: self.args.overwrite,
            double: self.double,
            extend: self.extend.cloned(),
        };
        let result = self.upscaler.run_file(
            self.upscale.as_ref().map(|u| u.model),
            self.inpainter,
            &job,
            self.opts,
            &mut |p| self.rep.core_progress(p),
            self.cancel,
        );
        match result {
            Ok(path) => {
                self.rep.done(&path);
                Outcome::Done
            }
            Err(e) => Outcome::Failed(e),
        }
    }
}

/// `scaleup upscale`, extending the canvas first when `--extend`/`--aspect` is given.
pub fn run_upscale(
    args: &UpscaleArgs,
    rep: &Reporter,
    cancel: &CancelToken,
) -> Result<i32, ScaleUpError> {
    if args.opts.model_scale.is_some() {
        rep.warn("-z is ignored (the model scale is read from the model)");
    }
    if args.opts.threads.is_some() {
        rep.warn("-j load:proc:save is ignored; use --jobs for parallel files");
    }
    run(
        &args.common,
        Some(&args.opts),
        &args.opts.canvas,
        rep,
        cancel,
    )
}

/// `scaleup extend`: the canvas grows and nothing is upscaled.
pub fn run_extend(
    args: &ExtendArgs,
    rep: &Reporter,
    cancel: &CancelToken,
) -> Result<i32, ScaleUpError> {
    run(&args.common, None, &args.canvas, rep, cancel)
}

/// Reports a model download in tenths, in verbose mode.
fn download_progress(rep: &Reporter) -> impl FnMut(u64, u64) + '_ {
    let mut last_decile = u64::MAX;
    move |done, total| {
        if total > 0 && done * 10 / total != last_decile {
            last_decile = done * 10 / total;
            rep.info(&format!("download {}%", done * 100 / total));
        }
    }
}

fn run(
    args: &CommonArgs,
    up: Option<&UpscaleOpts>,
    canvas: &CanvasArgs,
    rep: &Reporter,
    cancel: &CancelToken,
) -> Result<i32, ScaleUpError> {
    if let Some(f) = &args.format {
        OutputFormat::parse(f)?;
    }
    let pref = parse_gpu(&args.gpu).map_err(ScaleUpError::InvalidOption)?;

    let batch = args.input.is_dir();
    if !batch && !args.input.is_file() {
        return Err(ScaleUpError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("input '{}' does not exist", args.input.display()),
        )));
    }

    let out_is_dir = batch || args.output.is_dir() || has_trailing_separator(&args.output);
    if out_is_dir {
        std::fs::create_dir_all(&args.output)?;
    } else if let Some(parent) = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty() && !p.is_dir())
    {
        return Err(ScaleUpError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("output directory '{}' does not exist", parent.display()),
        )));
    }

    let registry = registry_for(args.models_dir.as_deref());
    let upscaler = Upscaler::new(pref, registry)?;
    rep.info(&format!("device: {}", upscaler.device().name));

    let mut loaded = None;
    if let Some(up) = up {
        let in_catalogue = upscaler.registry().get(&up.model).is_some();
        if !in_catalogue && !Path::new(&up.model).is_file() {
            return Err(ScaleUpError::Model(format!(
                "'{}' is neither a known model id nor an existing file; search OpenModelDB with `scaleup hub search`",
                up.model
            )));
        }
        let model_id = if in_catalogue {
            up.model.clone()
        } else {
            Path::new(&up.model)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| up.model.clone())
        };
        if in_catalogue && !upscaler.registry().is_downloaded(&up.model) {
            rep.info(&format!("downloading model {}", up.model));
            upscaler.registry().ensure_cancellable(
                &up.model,
                &mut download_progress(rep),
                cancel,
            )?;
        }
        let model = upscaler.load(&up.model, &mut |_, _| {})?;
        rep.info(&format!("model: {model_id} ({}x)", model.scale()));
        let active = upscaler.active_device();
        if active != *upscaler.device() {
            let line = format!("this model runs on {}", active.name);
            // Auto picked nothing in particular; an explicit choice deserves to know it was not used.
            if pref == scaleup_core::DevicePreference::Auto {
                rep.info(&line);
            } else {
                rep.warn(&line);
            }
        }
        loaded = Some((model, model_id));
    }

    // The inpainting model is loaded once per run, and only when the AI fill will use it.
    let extend = canvas.options();
    let mut inpainter = None;
    if extend.as_ref().is_some_and(|e| e.fill == Fill::Ai) {
        let id = canvas.model();
        // A missing catalogue model is downloaded first; Ctrl-C stops the download.
        let mut announced = false;
        let mut tenths = download_progress(rep);
        let mut progress = |done, total| {
            if !announced {
                announced = true;
                rep.info(&format!("downloading model {id}"));
            }
            tenths(done, total);
        };
        inpainter = Some(upscaler.load_inpainter(id, &mut progress, cancel)?);
        rep.info(&format!("inpainting model: {id}"));
    }

    let opts = match up {
        Some(up) => UpscaleOptions {
            tile_size: up.tile,
            tile_overlap: up.tile_overlap,
            tta: up.tta,
            output_scale: up.scale,
            output_width: up.width,
        },
        None => UpscaleOptions::default(),
    };
    let ctx = Ctx {
        upscaler: &upscaler,
        upscale: loaded.as_ref().map(|(model, id)| Loaded {
            model: model.as_ref(),
            id,
        }),
        inpainter: inpainter.as_deref(),
        extend: extend.as_ref(),
        opts: &opts,
        double: up.is_some_and(|u| u.double),
        args,
        rep,
        cancel,
    };
    let format = resolve_format(args);

    if !batch {
        let output = if out_is_dir {
            ctx.output_name(&args.input, &args.output, format)
        } else {
            args.output.clone()
        };
        return match ctx.process(&args.input, output, format) {
            Outcome::Done | Outcome::Skipped => Ok(0),
            Outcome::Failed(e) => Err(e),
        };
    }
    run_batch(&ctx, format)
}

fn run_batch(ctx: &Ctx, format: OutputFormat) -> Result<i32, ScaleUpError> {
    let mut files = Vec::new();
    let mut unsupported = 0usize;
    for entry in std::fs::read_dir(&ctx.args.input)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if is_supported_input(&path) {
            files.push(path);
        } else {
            unsupported += 1;
        }
    }
    files.sort();
    let total = files.len();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.args.jobs)
        .build()
        .map_err(|e| ScaleUpError::Backend(e.to_string()))?;
    let outcomes: Vec<Option<Outcome>> = pool.install(|| {
        files
            .par_iter()
            .enumerate()
            .map(|(i, input)| {
                if ctx.cancel.is_cancelled() {
                    return None;
                }
                ctx.rep.file_start(i + 1, total, input);
                let output = ctx.output_name(input, &ctx.args.output, format);
                Some(ctx.process(input, output, format))
            })
            .collect()
    });

    let (mut done, mut failed, mut skipped) = (0, 0, unsupported);
    let mut first_error: Option<i32> = None;
    for outcome in outcomes.into_iter().flatten() {
        match outcome {
            Outcome::Done => done += 1,
            Outcome::Skipped => skipped += 1,
            Outcome::Failed(ScaleUpError::Cancelled) => {}
            Outcome::Failed(e) => {
                failed += 1;
                ctx.rep.error(&e.to_string());
                first_error.get_or_insert(exit_code_for(&e));
            }
        }
    }
    ctx.rep.summary(done, failed, skipped);
    if ctx.cancel.is_cancelled() {
        return Ok(EXIT_CANCELLED);
    }
    if total > 0 && failed == total {
        return Ok(first_error.unwrap_or(EXIT_IO));
    }
    Ok(0)
}
