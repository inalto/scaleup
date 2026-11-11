# scaleup-core

The ScaleUP upscaling engine: upscale RGBA images with any ONNX super-resolution model on the GPU
(CUDA, DirectML, CoreML via ONNX Runtime) or the CPU, with tiling, alpha, TTA, post-resize,
progress reporting, cancellation, and a model registry with sha256-verified downloads.

## Usage

```rust,no_run
use scaleup_core::{
    CancelToken, DevicePreference, FileJob, OutputFormat, Progress, Registry, UpscaleOptions, Upscaler,
};
use std::path::Path;

fn main() -> scaleup_core::Result<()> {
    // Embedded model catalogue; files live in SCALEUP_MODELS_DIR or <data dir>/scaleup/models.
    let upscaler = Upscaler::new(DevicePreference::Auto, Registry::builtin())?;
    println!("device: {}", upscaler.device().name);

    // A catalogue id (downloaded and verified on first use) or a path to any .onnx file.
    let model = upscaler.load("realesr-general-x4v3", &mut |done, total| eprintln!("{done}/{total} bytes"))?;

    let opts = UpscaleOptions { output_scale: Some(4), ..Default::default() };
    let output = Upscaler::output_name(Path::new("photo.jpg"), Path::new("out"), "realesr-general-x4v3", &opts, OutputFormat::Png);
    let job = FileJob { input: "photo.jpg".into(), output, format: OutputFormat::Png, quality: 90, overwrite: false, double: false, extend: None };

    let cancel = CancelToken::new(); // clone it into another thread and call cancel() to stop
    let saved = upscaler.run_file(Some(model.as_ref()), None, &job, &opts, &mut |p| {
        if let Progress::Tiles { done, total } = p {
            eprintln!("{:.2}%", 100.0 * done as f64 / total as f64);
        }
    }, &cancel)?;
    println!("saved {}", saved.display());
    Ok(())
}
```

For in-memory images use `scaleup_core::upscale(model, &rgba, &opts, &mut progress, &cancel)`.

To add canvas before upscaling, set `FileJob::extend` (margins and a fill); an AI fill needs an
inpainter from `Upscaler::load_inpainter`, passed as `run_file`'s second argument. With no upscale
model (`None`), `run_file` only extends. Name such outputs with `Upscaler::output_name_for`.

## Notes

- Every model must take `[1,3,H,W]` float32 RGB in `0..1` and return `[1,3,H*s,W*s]`.
- WebP output is always lossless (pure-Rust encoder); `quality` applies to JPEG and PNG compression.
- Tests run on CPU only; GPU checks are `#[ignore]`: `cargo test -p scaleup-core -- --ignored`.
- `tests/fixtures/nearest2x.onnx` is regenerated with `python -I tools/make_test_model.py` (needs `onnx`).
