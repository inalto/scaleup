# ScaleUP user guide

Everything below applies to both the `scaleup` command line and the desktop app; the flag in
brackets is the command-line spelling, the app exposes the same setting in its sidebar.

## Choosing a model

| You have | Use | Why |
|---|---|---|
| photos, scans, screenshots | `realesrgan-x4plus` | best general quality; slow on CPU |
| photos, limited time or no GPU | `realesr-general-x4v3` (default) | 15× smaller, much faster, slightly softer |
| anime, illustrations, line art | `realesrgan-x4plus-anime` | trained on drawn content; keeps lines crisp |
| anime frames, lots of them | `realesr-animevideov3` | fastest model in the catalogue |
| JPEG-heavy web images | `4xNomosUni_span_multijpg` | trained on compression artefacts |
| only 2× needed | `realesrgan-x2plus` | native 2× (no downscale after a 4× run) |
| maximum sharpness, personal use | `4x-UltraSharp`, `4x_foolhardy_Remacri` | non-commercial licenses |

`scaleup models list` prints the catalogue; `scaleup models info <id>` prints the license text.

**"Non-commercial"** models are published under CC-BY-NC-SA-4.0: you may use them and their output
for personal, educational or research purposes, but not for anything commercial (client work,
products, paid content). The app lists them in a collapsed "Non-commercial" group with a license
badge next to the selected model; the command line accepts them like any other id and does not
warn, so check `scaleup models info <id>` (license name and the first line of its text) and
`models/licenses/<id>.txt` for the full terms. Which uses count as commercial is between you and
the model author; when in doubt use a BSD or CC-BY model.

## Scale versus custom width

Every model has a native scale (2× or 4×). With `-s 2|3|4` ScaleUP runs the model at its native
scale and then resizes the result (Lanczos3) to the requested factor, so `-s 2` with a 4× model is
"upscale 4×, then halve". `-w <pixels>` resizes to an exact output width instead and overrides
`-s`. A resize only happens when the requested size differs from the model's output; the output
name says which was used (`_4x_` or `_3840px_`).

## Tile size and VRAM

Images are cut into tiles that overlap by 16 to 64 px, depending on the model (`--tile-overlap`
overrides it), so the whole image never has to fit in GPU memory.
`-t 0` (auto, the default) picks a size by device:

| Device | Auto tile |
|---|---|
| GPU with 6 GiB or more free | 512 |
| other GPU | 256 |
| CPU | 128 |

Lower it (`-t 256`, `-t 128`) when the run fails with an out-of-memory error, when the GPU is shared
with other work, or when a large tile makes the UI stutter. Smaller tiles cost a little speed
(more overlap is recomputed) and nothing in quality, except for RealPLKSR models without layer norm
(see [OpenModelDB](#openmodeldb)). Values below 32 are rejected. Images smaller
than one tile are processed whole with reflect padding.

## TTA

`-x` runs every tile eight times (the four rotations and their mirrors) and averages the results.
It removes some noise and directional artefacts at roughly eight times the run time. Worth it for
a final print, not for a batch.

## Double upscale

`--double` runs the model twice (4× then 4× again gives 16×). The second pass works on the first
pass's output, so artefacts get amplified; use the custom width to come back down to the size you
actually need.

## Formats and quality

`-f png|jpg|webp` picks the output format (default: the extension of `-o`, else PNG). `-c 0..100`
is the quality (default 90): for JPEG it is the usual quality scale (100 = best), for PNG it
selects the compression effort (0 = fastest, 100 = smallest file). WebP output is always lossless
and ignores `-c`. PNG and WebP keep the alpha channel; JPEG flattens it on white.

## Batch folders and output naming

`-i <folder> -o <folder>` processes every image in the input folder (non-images are skipped and
reported). Output files are named

```text
<stem>_scaleup_<scale>x_<model>.<ext>       e.g. photo_scaleup_4x_realesrgan-x4plus.png
<stem>_scaleup_<width>px_<model>.<ext>      when -w is used
```

Existing outputs are skipped unless `--overwrite` is given. `--jobs N` processes N files at once
(useful on CPU; on one GPU it rarely helps). A corrupt file is reported and the batch continues; the
exit code is non-zero only if every file failed. With `-o <file>` a single input is written exactly
there.

## Using your own models

- Any ONNX super-resolution model with input `input` float32 NCHW RGB 0..1 and a single output:
  `-n path/to/model.onnx`. The scale is read from the model when possible; pass `-s` otherwise.
- ncnn `.param`/`.bin` pairs (Upscayl, OpenModelDB "ncnn" downloads) run on the `wgpu` backend:
  `-n path/to/model.param`, or copy a whole folder into the models directory with

  ```bash
  scaleup models import /path/to/models
  ```

  which prints the ids it added. They then show up in `scaleup models list` and in the app's
  dropdown.
- The app's "Custom model file" button does the same for one file.
- PyTorch weights (`.pth`, `.safetensors`) of the ESRGAN, ESRGAN+, Compact (SRVGGNet), SPAN and
  RealPLKSR architectures run on the `wgpu` backend as they are: `-n path/to/model.pth`. The
  architecture is read from the file. RealPLKSR files with the DySample upsampler and models that
  are not RGB to RGB are refused with a message that says so.
- A PyTorch checkpoint of any other architecture can be converted to ONNX with your own Python, see
  [OpenModelDB](#openmodeldb).

## OpenModelDB

[OpenModelDB](https://openmodeldb.info) lists several hundred community models. ScaleUP keeps a copy of its
catalogue in `<models dir>/hub/` and fetches it again when it is older than 24 hours or when you ask.

**In the app**, click **Browse OpenModelDB…** next to the model picker. Search by name, author or tag, and
filter by scale, content, architecture, **Commercial use OK** (on by default: non-commercial and unlicensed
models stay hidden until you turn it off) and **Runnable on this machine**. Each model shows its license
(green: commercial use allowed, amber: share-alike or copyleft, red: non-commercial, grey: unknown) and its
state:

| State | Button |
|---|---|
| Installed | Installed ✓ (Remove in the details) |
| Downloadable | **Install**: download, check the sha256, add to the list. ESRGAN, ESRGAN+, Compact, SPAN and RealPLKSR models run from their PyTorch file, others from an ONNX file |
| Manual download | **Open download page**, save the file, then **Import file…**; it is checked against the catalogue's sha256 |
| Needs conversion | **Convert**: download the PyTorch file and convert it to ONNX with Python |
| Unsupported | none |

Click a model for its description, files and before/after pictures. **Recommend…** asks what is in the image,
what should happen and how fast it must be, and suggests five models with the reasons. Installed models appear
in the model picker under **From OpenModelDB**. A model whose file the catalogue lists without a sha256 is
installed with a warning that it could not be verified.

**RealPLKSR and tiles.** RealPLKSR models without layer norm (such as `4x-NomosWebPhoto-RealPLKSR`
and `4x-RealPLKSR-Pretrain-V4`; ScaleUP does not show which kind a model is) normalise over the whole picture they are given, so each tile is normalised on
its own and the result depends on the tile layout: at the default tile, values differ from a one-tile run by
under 1 level on average but by up to 79 in places, and faint lines can show where tiles meet. A larger overlap
does not help. Set the tile size at least as large as the image's longer side (`-t 1024` for an image up to
1024 px) so the image runs as one tile; that needs more GPU memory, and when the GPU cannot take a tile that
size ScaleUP shrinks it and the effect comes back, so an image larger than the GPU's largest tile has no
remedy yet. The same is expected of the model's ONNX conversion, which is tiled the same way.

**On the command line** the same is available as `scaleup hub search | info | install | import | remove |
recommend | refresh`; see `scaleup hub --help` and the [CLI README](../crates/scaleup-cli/README.md).

**Converting.** `scaleup hub install <id> --convert`, `scaleup models convert <file.pth> [--out dir]` and the
app's Convert button run your own Python 3 (`SCALEUP_PYTHON` if set, else `python3` or `python` on `PATH`) with:

```bash
pip install spandrel torch onnx
```

Without Python or those modules you get that line as the error (exit code 3 on the command line). Python only
runs when you ask for a conversion. Converting loads the downloaded checkpoint with your Python and PyTorch
(through spandrel), so convert only files whose source you trust. A model whose PyTorch file is on a site
ScaleUP cannot download from offers **Open download page** and **Import file…** instead of Convert; on the
command line, `scaleup hub import <id> <file.pth>` converts such a file the same way. Stop (or Ctrl-C) ends a
running conversion and leaves no files behind. A converted local file is marked non-commercial, since its license is not
known.

## Troubleshooting

| Symptom | What to do |
|---|---|
| "GPU not detected", everything runs on the CPU | run `scaleup devices`; it lists what each backend sees. CUDA needs the CUDA 12 and cuDNN 9 runtimes and a recent driver; DirectML needs Windows 10 1903+; on Linux without NVIDIA only ncnn models use the GPU (Vulkan). Pass `-g <index>` to pick a specific device. |
| Out of memory, driver reset, or the process dies mid-image | lower the tile size: `-t 256`, then `-t 128`. |
| "sha256 mismatch" or a model that fails to load after a download | delete the file in the models directory (see below) and run the command again; the download restarts and is verified. |
| Download hangs or fails | check the proxy settings; `SCALEUP_MODELS_DIR` can point at a folder you filled manually from the Releases page. |
| Output looks blurry | the model ran at its native scale and was downscaled: use a 2× model for 2×, or raise the output size. |
| Seams or a grid in the output | report it with the image, model and `-t` value; tiles overlap by the model's own amount (16 to 64 px) and should be invisible, except for RealPLKSR models without layer norm (see OpenModelDB). |
| App window opens blank or crashes at start on Linux | install `libgtk-3-0` and a Vulkan driver (`mesa-vulkan-drivers`), or run the CLI. |

Run with `-v` (or open the app's log panel) to see the device, model and phases.

## Where files live

Models (`SCALEUP_MODELS_DIR` overrides the default, `-m <dir>` overrides it per run):

| OS | Default models directory |
|---|---|
| Windows | `%APPDATA%\scaleup\models` |
| Linux | `~/.local/share/scaleup/models` |
| macOS | `~/Library/Application Support/scaleup/models` |

Desktop app settings (output folder, last model, format, quality, device, advanced options) are
stored by eframe:

| OS | Settings file |
|---|---|
| Windows | `%APPDATA%\ScaleUP\data\app.ron` |
| Linux | `~/.local/share/scaleup/app.ron` |
| macOS | `~/Library/Application Support/ScaleUP/app.ron` |

Delete the settings file to reset the app. The ONNX Runtime provider libraries live next to the
executables (`bin/` in the Windows installer, `/usr/lib/scaleup` for the `.deb`,
`ScaleUP.app/Contents/MacOS` on macOS); keep them together if you move the binaries.
