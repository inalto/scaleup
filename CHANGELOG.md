# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI on Linux, Windows and macOS (`cargo fmt`, `cargo clippy`, `cargo test`).
- Release workflow: archives for the four targets, `.deb`, `.msi` and `.dmg` installers.
- Repository documentation: README, user guide, Upscayl compatibility notes, contribution files.

## [0.1.0] - 2026-11-04

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

[Unreleased]: https://github.com/inalto/scaleup/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/inalto/scaleup/releases/tag/v0.1.0
