# Contributing to ScaleUP

Thanks for helping. This page covers the checks every change must pass and the two most common
extensions: a new model and a new backend.

## Before you open a pull request

CI runs the same three commands on Linux, Windows and macOS; run them locally first:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Tests that need a GPU or a downloaded model are `#[ignore]`d and run with
`cargo test --workspace -- --ignored`. On Linux the desktop app needs `libgtk-3-dev` and
`libxdo-dev`.

Commit messages follow `type(scope): summary` (`feat(cli): ...`, `fix(core): ...`,
`docs: ...`, `ci: ...`). Keep the changelog's `Unreleased` section up to date for user-visible
changes.

## Adding a model

1. Add the checkpoint to `tools/sources.json` and its license text to `models/licenses/<id>.txt`
   (`tools/fetch_sources.py` downloads both).
2. Export and validate it with `tools/export_onnx.py <id>` (PyTorch/spandrel to ONNX, opset 17,
   dynamic height and width, input `input` float32 NCHW RGB 0..1). The script prints the manifest
   entry with `size_bytes` and `sha256`. See `tools/README.md` for the virtualenv setup.
3. Upload the `.onnx` file to the `models-v1` GitHub release (the `models` workflow does this) and
   add the printed entry to `models/manifest.json`. Set `"commercial": false` and the
   `noncommercial` tag when the license forbids commercial use; the app lists those models behind a
   badge.
4. `cargo test -p scaleup-core` checks that every manifest entry has a license file.

ncnn models do not need a conversion: a `.param`/`.bin` pair in the models folder is picked up by
the `wgpu` backend (`format: "ncnn"` in the manifest, or `scaleup models import <dir>`).

## Adding a backend

A backend is a `scaleup_core::Backend` that loads models and a `scaleup_core::Model` that runs one
tile:

- `Backend` reports the devices it can see (`DeviceInfo { kind, index, name }`), says which model
  files it accepts, and loads a `Model` for a `DevicePreference`.
- `Model` exposes its scale, the input multiple it needs (4 by default) and `run(&Image32) ->
  Image32` for a single tile (planar CHW float32, 0..1). Tiling, overlap, alpha, TTA and resizing
  are handled by `scaleup_core` around it.

Register the backend at startup the way `scaleup_wgpu::register()` does so the CLI and the app see
it; the registry picks a backend by the model's file format. Put the device in the `DeviceKind`
enum if none fits, and add the GPU-less path to the tests: everything in `cargo test --workspace`
must pass on a machine without a GPU, so gate real-device tests behind `#[ignore]`.

## Reporting bugs

Use the bug template: it asks for the OS, GPU, the output of `scaleup devices`, the exact command
or app settings, the model and the log.
