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

In the app, a single image never reuses an existing file while **Overwrite existing files** is off: its
result gets the next free name, `-2`, `-3` and so on before the extension, for example
`photo_scaleup_4x_realesrgan-x4plus-2.png`. A folder run in the app still skips existing outputs. The
command line keeps the rules above.

## Extending the canvas

`scaleup extend` adds pixels around an image, so the output is wider or taller than the original.
`scaleup upscale` takes the same options and extends the image before it upscales. The new area
is filled by an inpainting model, or by a plain fill that needs no model. The extension is made at
the source resolution, before the upscale, and original pixels are never changed.

| Flag | Meaning | Example |
|---|---|---|
| `--extend <SPEC>` | `64` (all sides), `l,t,r,b`, `20%` or `l%,t%,r%,b%`; pixels and percentages are of the source image | `--extend 0,0,128,0` |
| `--aspect <W:H>` | grow, centred, to this ratio; never crops. Conflicts with `--extend` | `--aspect 16:9` |
| `--extend-fill <FILL>` | `ai` (default), `mirror`, `edge`, `transparent` or `#rrggbb` | `--extend-fill "#202020"` |
| `--extend-model <ID\|PATH>` | inpainting model for the `ai` fill: `lama-512` (default), `migan-512-places2` or `lama-manga`, or a path to an `.onnx` file | `--extend-model migan-512-places2` |

`scaleup extend` needs `--extend` or `--aspect`. The inpainting model is downloaded and loaded once
per run, and only for the `ai` fill. `scaleup models list --kind inpaint` shows the inpainting
models and what each is good for.

```bash
scaleup extend -i photo.jpg -o wide.png --aspect 16:9
scaleup upscale -i photo.jpg -o out/ --extend 10% --extend-fill mirror
```

Output names carry the fill after `_ext_`, so two fills of the same extension never overwrite each
other: `photo_scaleup_ext_<fill>.png` with `scaleup extend`, and
`photo_scaleup_ext_<fill>_4x_<model>.png` when upscaling. `<fill>` is `mirror`, `edge`,
`transparent`, the colour as `rrggbb`, or `ai-<inpainting model>` for the AI fill, for example
`photo_scaleup_ext_ai-lama-512.png` or `photo_scaleup_ext_mirror_4x_<model>.png`.

**In the app**, the section is called **Extend canvas** in the sidebar. Tick **Add canvas before
upscaling** to turn it on. Choose **Pixels**, **Percent** or **Aspect**. Pixels and percent take
one value for all sides (**Same on every side**) or one value per side. Aspect offers 1:1, 4:3,
3:2, 16:9 and 9:16, or a custom ratio. **Fill** offers AI (inpainting), Mirror, Edge pixels,
Colour and Transparent. The AI fill shows the model picker. When the chosen inpainting model is
not downloaded yet, the Start button says **Download and start**, or **Download and extend** when nothing is
upscaled. Under **Output**, the **Scale** row has a **None** choice: the image is only extended, nothing is
upscaled, and the Start button reads **Extend**.

### Editing the canvas in the app

With **Add canvas before upscaling** on and one image loaded, the image area shows the planned canvas: the
picture at its place, the new area showing its fill, and eight handles on the sides and corners of the new
canvas. A chip under the canvas gives the size before and after, for example `256×192 to 512×448`, or asks
for a handle to be dragged while no canvas is added. If the margins cannot be used for the image, the chip
says why and Start is disabled.

Drag a side handle outward to add canvas on that side, or a corner handle to add on two sides. Dragging
inward takes canvas away and stops at 0. Outward drags stop at the limits in [Limits](#limits). Shift moves
all four sides by the same amount, and Alt moves the opposite side by the same amount as well. Each drag
writes the same settings as the sidebar fields, and the fields follow at once. A drag in **Aspect** mode
switches the mode to **Pixels**; in **Percent** mode it writes percentages.

Click a handle to select it, then the arrow keys move it by 1 px, or by 10 px with Shift.

The new area previews the fill: the colour for **Colour**, a checkerboard for **Transparent**, the picture
reflected across its border for **Mirror**, its border pixels stretched for **Edge pixels**, and for AI a
hatched tint labelled with the model, for example `AI fill · LaMa (512)`.

After a run, the image area shows the before and after pictures. Changing the extension shows the planned
canvas again. **Adjust canvas** returns to the planned canvas, **Show result** returns to the last result,
and **Extend canvas…**, shown while Extend is off and an image is loaded, turns Extend on and shows the
planned canvas. A folder has no picture to drag, so it has no canvas to edit.

### Trying several fills

**Try several fills** is a list under the Fill and Model rows of the Extend canvas section, closed until it
is opened. It offers each inpainting model, with its licence and what it is good for, and then **Mirror** and
**Edge pixels**. Tick the fills to try and press the button, which says how many fills it will run: **Try 3
fills**, or **Download and try 3 fills** when a ticked model is not downloaded yet. Until you change a tick,
the list follows the fill chosen above it.

Trying needs a single image, Extend canvas turned on, and at least two fills ticked. The fills run one after
another on the same extension, and the status line says `Fill 2 of 3`. Each finished fill is a step in the
history. A fill that repeats an earlier one writes a new file, with `-2` added before the extension, unless
**Overwrite existing files** is on, when the file is replaced. **Stop** ends the queue: the fills that
finished keep their steps. A fill whose inpainting model cannot be loaded is reported by name, and the
others still run.

### History

Each finished single-image run in the app is a step in a history, and so is each fill of a **Try several
fills** list. The steps appear as a strip of thumbnails under the image area, with the step's label on hover,
for example `Extend +128 px · LaMa (512) · 4×`. The step on screen has an accent outline.

- **Back** and **Forward** show the previous or next step, with its result against its own source. The count
  beside them reads `2 of 3`, or `3 steps` while the image on screen is not a step.
- The shortcuts are `Alt+Left` for back and `Alt+Right` for forward, and also `Ctrl+Z` for back and
  `Ctrl+Shift+Z` or `Ctrl+Y` for forward. On macOS, Option replaces Alt and Command replaces Ctrl. They do
  nothing when there is no step, while a run is in progress, while a text field has the keyboard (Ctrl+Z is
  then the field's own undo), or while a canvas handle has focus, because its arrows move the handle.
- The step actions are **Use as input**, **Open folder**, **Restore settings** and **Remove**. **Use as
  input** makes the step's result the selected image, so an extension can be upscaled next. When the step
  extended the canvas, Use as input also turns Extend canvas off, so the next run does not add the same
  canvas a second time. **Restore
  settings** puts the step's settings back into the sidebar. **Remove** takes the step out of the history and
  keeps its file: removing a step never deletes a file. The actions need a step on screen.
- The history lasts for the session, so closing the app clears it. It holds up to 30 steps, and the oldest is
  dropped when a new one is added to a full history. It covers single images only; a folder run adds no
  steps. A step whose file has been deleted stays in the history, and the image area says that it cannot be
  shown.

### Fills

| Fill | What it does | Needs a model |
|---|---|---|
| `ai` (default) | the inpainting model invents the new pixels | yes |
| `mirror` | reflects the nearest pixels across the border | no |
| `edge` | repeats the nearest border pixel | no |
| `#rrggbb` | a solid colour, opaque | no |
| `transparent` | fully transparent new pixels | no |

Formats without transparency (JPEG) show the transparent area as white.

Transparent pixels of the source are kept as they are and count as known pixels. The AI fill uses
their stored colour, often black, as context, and the pixels it invents are opaque. For an image with
a transparent border, use the `transparent` or `edge` fill.

### Inpainting models

| Model | Licence | Download | Runs on | Good for |
|---|---|---|---|---|
| `lama-512` (default) | Apache-2.0 | about 198 MB (208,044,816 bytes) | CPU only; the device setting does not apply | Photos: soft or plain backgrounds, modest margins. Runs on the CPU. |
| `migan-512-places2` | MIT | about 28 MB (29,546,882 bytes) | the device of the run (GPU when one is used) | Photos, fast on the GPU. May invent objects in the new area. |
| `lama-manga` | Apache-2.0, declared by the uploader | about 197 MB (206,291,843 bytes) | the device of the run (GPU when one is used); CPU if the GPU cannot load it | Illustrations and flat colour, fast on the GPU. Blurs photo detail on large margins. |

The weights are under the licences above; the licence texts are in `models/licenses/lama-512.txt`,
`models/licenses/migan-512-places2.txt` and `models/licenses/lama-manga.txt`. The licences of the
training data were not checked. For `lama-manga`: License as declared by the uploader; the terms of the
weights and of the training data were not verified.

Timings were measured on one machine (with an NVIDIA GTX 1660 SUPER; the CPU model was not
recorded) and will differ on other hardware. In a test run of a debug build, loading LaMa on the CPU
took about 13 s, and extending a 256 × 192 image by 64 px on one side, a single 512 px window, took
about 2 s. Loading MI-GAN took about 0.7 s. The feasibility spike, which ran in Python
and not in the shipped build, measured about 14 s for LaMa on the CPU and about 1 s for MI-GAN on the
GPU for a 3000 × 2000 photo with a 192 px right margin. The same spike measured one 512 px window of
`lama-manga` at about 0.3 s on the GPU and about 2.2 s on the CPU; the model takes roughly 10 to 20 s to
load.

### What the models do well and badly

These statements come from six test images with margins of 64, 128 and 256 px, judged by eye at
small size. Seams at full size were not checked. The results are in `docs/OUTPAINT_SPIKE.md`.

- **LaMa** gives acceptable results for extensions up to about 128 px per side, on photos with
  soft or plain backgrounds such as a wall or a sky. Beyond that the new area becomes blurry or
  flat; at 256 px it is mostly a plain or blurred fill. It does not continue busy textures: on a
  gravel image the new area became a smooth slab with a visible seam. It does not continue a
  person or an object cut by the frame: with 128 px on all four sides of a portrait, the sleeves
  were lost in dark smudges. It did not invent objects in these tests.
- **MI-GAN** is reliable only for small one-sided extensions of simple backgrounds. On the other
  test images it invented objects, faces, text-like marks and repeated patterns, some already at
  64 px. Use it only for small extensions, and check the result.
- **LaMa Manga** (`lama-manga`) was judged on the same six test images and on two illustration-like stand-ins
  from scikit-image's sample data (a flat logo and a black silhouette), with margins of 64 and 192 px. It gave
  clean results on flat colour and silhouettes and invented no objects in these tests. It blurs photo detail
  on large margins, puts a warm red-brown cast on photos at 192 px on four sides, and does not continue
  textures such as gravel.
- **Not tested:** real anime and manga images (`lama-manga` was tested on the two stand-ins only), and text at
  the edge of the frame.

All three models were trained for hole filling, not for extending a picture. For larger extensions,
use a plain fill and check the result.

### Limits

Each margin is at most 4 times the side it extends, and the extended canvas is at most 16384 px per
side. Larger values are refused before any work starts.

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
- The app's "Custom file…" link under the model list does the same for one file.
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
| Installed | Installed (Remove in the details) |
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
| Invented objects, faces or patterns in the new area | use `lama-512` (the default) or a plain fill (`mirror` or `edge`). `migan-512-places2` is only reliable for small one-sided extensions. |
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
