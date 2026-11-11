# scaleup (CLI)

`upscayl-bin` compatible command line for the ScaleUP engine.

```
scaleup upscale -i <file|dir> -o <file|dir> [options]     (or bare: scaleup -i .. -o ..)
scaleup extend -i <file|dir> -o <file|dir> (--extend <spec> | --aspect <W:H>) [options]
scaleup models list [--kind upscale|inpaint|all] | info <id> | download <id> | path <id> | import <dir> | convert <file>
scaleup hub search | info | install | import | remove | recommend | refresh
scaleup devices
scaleup --version
```

## Flags

| Flag | Meaning | Example |
|---|---|---|
| `-i` | input image or folder | `-i photo.jpg` |
| `-o` | output file or folder. A folder (existing, or ending in `/`) is created and files are named `<stem>_scaleup_<s>x_<model>.<ext>`; a single-file target needs an existing parent directory | `-o out/` |
| `-n` | model id from the catalogue, or a path to an `.onnx` file (relative to the CWD) | `-n realesrgan-x4plus` |
| `-m` | models directory (default `SCALEUP_MODELS_DIR`, else the user data dir) | `-m D:\models` |
| `-s` | output scale 2..4 (default: the model scale) | `-s 2` |
| `-w` | output width in pixels, overrides `-s` | `-w 3840` |
| `-f` | `png`, `jpg` or `webp`; default is the output extension, else `png` | `-f webp` |
| `-c` | quality 0..100 (default 90) | `-c 80` |
| `-t` | tile size, `0` = automatic, otherwise at least 32 | `-t 256` |
| `--tile-overlap` | pixels of context around each tile, capped at a quarter tile; default is what the model asks for | `--tile-overlap 32` |
| `-g` | `auto`, `cpu` or a GPU index `0`..`9` | `-g cpu` |
| `-x` | test-time augmentation (8x slower) | `-x` |
| `--double` | run the model twice | `--double` |
| `--overwrite` | replace existing outputs (otherwise they are skipped) | |
| `--jobs N` | files processed in parallel in folder mode | `--jobs 2` |
| `-v` | print phases, device and model | |
| `--json` | one JSON object per line on stderr | |
| `--extend`, `--aspect`, `--extend-fill`, `--extend-model` | add canvas before upscaling, see [Extending the canvas](#extending-the-canvas) | `--extend 64` |
| `-z`, `-j load:proc:save` | accepted for upscayl-bin compatibility and ignored with a warning | |

Folder mode skips files that are not images and reports `N done, M failed, K skipped`. A corrupt file is
reported and does not stop the run; the exit code is non-zero only if every file failed.

## Extending the canvas

`upscale` can grow the canvas around each image before it upscales; `scaleup extend` does only that and
writes the extended image at its own size. Both take the same flags.

| Flag | Meaning | Example |
|---|---|---|
| `--extend <SPEC>` | `64` (all sides), `l,t,r,b`, `20%` or `l%,t%,r%,b%`; pixels and percentages are of the source image | `--extend 0,0,128,0` |
| `--aspect <W:H>` | grow, centred, to this ratio; never crops. Conflicts with `--extend` | `--aspect 16:9` |
| `--extend-fill <FILL>` | `ai` (default), `mirror`, `edge`, `transparent` or `#rrggbb` | `--extend-fill "#202020"` |
| `--extend-model <ID\|PATH>` | inpainting model for the `ai` fill: `lama-512` (default) or `migan-512-places2`, or a path to an `.onnx` file | `--extend-model migan-512-places2` |

`scaleup extend` needs `--extend` or `--aspect`; on `upscale`, `--extend-fill` and `--extend-model` are usage
errors without one of them. `--extend` only takes margins and `--aspect` only takes `W:H`. The inpainting model
is downloaded and loaded once per run, and only for the `ai` fill: the other fills never touch it. `scaleup
models list --kind inpaint` shows the inpainting models (`--kind all` shows both kinds; plain `models list`
shows upscalers only).

Output names get `_ext` before the size: `photo_scaleup_ext_4x_<model>.png` when upscaling, and
`photo_scaleup_ext_<fill>.png` with `extend` (`<fill>` is `ai`, `mirror`, `edge`, `transparent` or `rrggbb`).

```
scaleup extend -i photo.jpg -o wide.png --aspect 16:9
scaleup upscale -i photo.jpg -o out/ --extend 10% --extend-fill mirror
```

## OpenModelDB

Browse [OpenModelDB](https://openmodeldb.info) and use its models like any other. The catalogue is cached for
24 hours in `<models dir>/hub/`. Installed models appear in `scaleup models list` and work with `-n <id>`.

| Command | Example |
|---|---|
| `hub search [text]` with `--scale`, `--tag`, `--arch`, `--commercial`, `--runnable`, `--refresh`, `--limit` | `scaleup hub search anime --scale 2 --commercial` |
| `hub info <id>`: license in words, files, hosts, readiness | `scaleup hub info 2x-MLP-StarSample-V1-0` |
| `hub install <id> [--resource N] [--convert]`; `--convert` downloads the PyTorch file and converts it to ONNX (needs Python, see below) | `scaleup hub install 4x-DRCT-L --convert` |
| `hub import <id> <file>`: a file downloaded by hand (checked against the catalogue's sha256) | `scaleup hub import 1x-DeH264-realplksr model.onnx` |
| `hub remove <id>` | `scaleup hub remove 2x-MLP-StarSample-V1-0` |
| `hub recommend` with `--content`, `--goal`, `--scale`, `--speed`, `--commercial`, `-n` | `scaleup hub recommend --content anime --speed fastest` |
| `hub refresh`: fetch the catalogue again | `scaleup hub refresh` |

The READY column is `installed`, `downloadable`, `manual` (the files are on a host you open in a browser; the
error of `hub install` says which URL, file name and sha256, then use `hub import`), `needs-conversion` or
`unsupported`. A file the catalogue lists without a sha256 installs with a warning.

## Converting PyTorch models

`scaleup models convert <file.pth|file.safetensors> [--out dir]` converts a checkpoint to ONNX (input `input`
float32 [1,3,H,W], opset 17) and lists it as a local model named after the file; its license is unknown, so it
is marked non-commercial. `hub install <id> --convert` does the same for an OpenModelDB model and records it as
installed. Both run your own Python 3: `SCALEUP_PYTHON` (a path to the executable) if set, else `python3` or
`python` on `PATH`, with these modules:

```
pip install spandrel torch onnx
```

Without them the command prints that line and exits 3. Python only runs when you ask for a conversion.

## Exit codes

`0` ok, `1` usage, `2` I/O (missing input, unreadable/undecodable image, unwritable output), `3`
model/backend/download, `4` cancelled. Ctrl-C cancels the current job, leaves no `.tmp` or `.part` file and
exits 4; a second Ctrl-C exits immediately.

## Progress and JSON events

Human mode writes `12.34%` lines on stderr, `Resizing` when a final resize runs, and ` Upscayled Successfully!`
per file. The AI fill reports one progress step per inpainting window, then the upscale counts its own
tiles from zero. With `--json` every line is an object:

```
{"event":"progress","percent":12.34}
{"event":"phase","phase":"saving"}            (verbose, or resizing; "extending" when the canvas grows)
{"event":"file","index":1,"total":3,"path":"..."}
{"event":"done","path":"..."}
{"event":"error","message":"..."}
{"event":"warning","message":"..."}
{"event":"info","message":"..."}               (verbose only)
{"event":"summary","done":3,"failed":0,"skipped":1}
```

## Timings

1024x768 JPEG input, 4x output (4096x3072), Windows 11 release build, `-n <path>.onnx`. Measured once, so
treat as indicative.

| Model | Device | Seconds |
|---|---|---|
| realesr-general-x4v3 | DirectML GPU 0 (`-g auto`) | 19 |
| realesr-general-x4v3 | CPU (`-g cpu`) | 28 |
| realesrgan-x4plus | DirectML GPU 0 | 39 |
| realesrgan-x4plus | DirectML GPU 0, `-x` | 136 |
| realesrgan-x4plus | CPU | not measured (very slow) |
