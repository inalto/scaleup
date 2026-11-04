# scaleup (CLI)

`upscayl-bin` compatible command line for the ScaleUP engine.

```
scaleup upscale -i <file|dir> -o <file|dir> [options]     (or bare: scaleup -i .. -o ..)
scaleup models list | info <id> | download <id> | path <id>
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
| `-z`, `-j load:proc:save` | accepted for upscayl-bin compatibility and ignored with a warning | |

Folder mode skips files that are not images and reports `N done, M failed, K skipped`. A corrupt file is
reported and does not stop the run; the exit code is non-zero only if every file failed.

## Exit codes

`0` ok, `1` usage, `2` I/O (missing input, unreadable/undecodable image, unwritable output), `3`
model/backend/download, `4` cancelled. Ctrl-C cancels the current job, leaves no `.tmp` or `.part` file and
exits 4; a second Ctrl-C exits immediately.

## Progress and JSON events

Human mode writes `12.34%` lines on stderr, `Resizing` when a final resize runs, and ` Upscayled Successfully!`
per file. With `--json` every line is an object:

```
{"event":"progress","percent":12.34}
{"event":"phase","phase":"saving"}            (verbose, or resizing)
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
