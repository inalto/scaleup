# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-11

### Added

- Canvas extension: `scaleup extend` adds canvas around an image or a folder without upscaling, and
  `scaleup upscale` takes the same options (`--extend`, `--aspect`, `--extend-fill`,
  `--extend-model`). The new area is filled by an inpainting model or by a plain fill (`mirror`,
  `edge`, `transparent` or `#rrggbb`). The app's sidebar has an **Extend canvas** section with the
  same options. Two inpainting models are in the catalogue: `lama-512` (default, LaMa, Apache-2.0,
  runs on the CPU) and `migan-512-places2` (MI-GAN, MIT, fast, may invent objects in the new area).

### Changed

- `scaleup-core` API: `Upscaler::run_file` now takes `Option<&dyn Model>` (`None` extends without
  upscaling) and an `Option<&dyn Inpainter>`; `FileJob` gained an `extend` field and no longer
  implements `Eq`; the new `Upscaler::load_inpainter` takes a `CancelToken` that stops its download.

## [0.2.1] - 2026-11-05

First published release: everything listed under 0.2.0 and 0.1.0 below, with updated dependencies.

### Changed

- Dependencies: `wgpu` 30, `ndarray` 0.17, `sha2` 0.11, `dirs` 7, `pollster` 1.

### Fixed

- The release workflow's check of the tag against the workspace version, which stopped the 0.2.0
  release before it built anything.
- The Mish activation returned NaN for large values on DX12.
- A failed or cancelled model conversion no longer removes an ONNX file converted earlier.
- The tile overlap of a network follows the size of its convolutions, not only their number.

### Known limitations

- RealPLKSR models without layer norm normalise each tile on its own, so their output depends on the
  tile size and faint lines can show where tiles meet (see the user guide, OpenModelDB).

## 0.2.0 - 2026-11-05

Tagged but not published as a release; its content shipped in 0.2.1.

### Added

- OpenModelDB: `scaleup hub search | info | install | import | remove | recommend | refresh` and
  the app's **Browse OpenModelDB…** window list the models of openmodeldb.info with their license
  class and whether they run on this machine, download them (direct hosts and Google Drive; other
  hosts through the browser, then import), check the sha256 and add them to the model list.
  `hub recommend` and the window's **Recommend…** suggest models for a kind of image, a goal and a
  speed.
- PyTorch weights run natively on the `wgpu` backend, with no Python: `.pth` (zip and legacy
  formats) and `.safetensors` files of the ESRGAN, ESRGAN+, Compact (SRVGGNet), SPAN and RealPLKSR
  architectures, detected from the file. The `.pth` reader only builds tensors and never executes
  code from the file.
- Optional conversion of other architectures to ONNX with the user's own Python (spandrel, torch,
  onnx): `scaleup models convert`, `scaleup hub install <id> --convert` and the app's **Convert**
  button.
- Each network asks for the tile overlap it needs.
- CI on Linux, Windows and macOS (`cargo fmt`, `cargo clippy`, `cargo test`).
- Release workflow: archives for Windows x64, Linux x64 and macOS arm64, plus `.deb`, `.msi` and `.dmg` installers.
- Repository documentation: README, user guide, Upscayl compatibility notes, contribution files.

## 0.1.0 - 2026-11-04

Not published as a release; its content shipped in 0.2.0.

### Added

- `scaleup` command line, compatible with the `upscayl-bin` flags (`-i`, `-o`, `-n`, `-s`, `-w`,
  `-f`, `-c`, `-t`, `-g`, `-x`, `-m`), plus `--double`, `--overwrite`, `--jobs`, `--json` and the
  `models` and `devices` subcommands.
- `scaleup-app` desktop app (eframe/egui): file and folder picker, drag and drop, clipboard paste,
  model dropdown with license badges, before/after comparison slider, batch progress, log panel,
  persisted settings.
- ONNX Runtime backend (`ort` 2.0.0-rc.15): CUDA, DirectML and CoreML execution providers with a
  CPU fallback; tiling with reflect padding, alpha channel, TTA, double upscale, Lanczos3 resize.
- `wgpu` backend: ncnn `.param`/`.bin` models executed on Vulkan, Metal and DX12 compute shaders.
- Model registry: embedded catalogue (`models/manifest.json`), sha256-verified cancellable
  downloads, bundled license texts, import of ncnn model folders.

[Unreleased]: https://github.com/inalto/scaleup/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/inalto/scaleup/releases/tag/v0.3.0
[0.2.1]: https://github.com/inalto/scaleup/releases/tag/v0.2.1
