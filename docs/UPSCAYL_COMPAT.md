# Upscayl compatibility notes

`scaleup` accepts the `upscayl-bin` command line, so a script that calls
`upscayl-bin -i in.jpg -o out.png -n realesrgan-x4plus -s 4 -f png` works with `scaleup` in its
place (`scaleup -i ... -o ...` is `scaleup upscale -i ... -o ...`). Progress is printed to stderr as
`12.34%` lines, `Resizing` precedes the final resize and ` Upscayled Successfully!` closes a file,
exactly as the original binary does.

## Flag mapping

| `upscayl-bin` | `scaleup` | Difference |
|---|---|---|
| `-h` | `-h`, `--help` | |
| `-i input-path` | `-i` | same: file or folder |
| `-o output-path` | `-o` | same; in folder mode files are named `_scaleup_` (see below) |
| `-z model-scale` | accepted, ignored with a warning | the model scale is read from the model, not guessed from its name |
| `-s output-scale` | `-s 2\|3\|4` | same semantics: model at native scale, then resize |
| `-r resize` | not supported in v1 | use `-w <width>`; `WxH` and the resize-filter choice are not available |
| `-w width` | `-w <px>` | same; overrides `-s` |
| `-c compress` | `-c <0..100>` | **inverted meaning**: `upscayl-bin -c` is a compression amount (0 = none, the default); `scaleup -c` is a *quality* (100 = best, default 90). A script passing `-c 0` now gets the lowest quality; drop the flag or pass `-c 90` |
| `-t tile-size` | `-t <n>` | same; `0` = auto; the multi-GPU `0,0,0` form is not supported |
| `-m model-path` | `-m <dir>` | same; also `SCALEUP_MODELS_DIR` |
| `-n model-name` | `-n <id\|file>` | same ids; also a path to an `.onnx` or `.param` file |
| `-g gpu-id` | `-g auto\|cpu\|<n>` | same indices; `cpu` added; the multi-GPU `0,1` form is not supported |
| `-j load:proc:save` | accepted, ignored with a warning | use `--jobs N` for parallel files |
| `-x` | `-x` | same (TTA) |
| `-f format` | `-f png\|jpg\|webp` | same |
| `-v` | `-v` | prints device, model and phases |
| — | `--double`, `--overwrite`, `--json`, `--jobs` | ScaleUP additions |

Exit codes differ: `scaleup` returns 0 ok, 1 usage, 2 I/O, 3 model/backend, 4 cancelled;
`upscayl-bin` returns 0 or -1. Error lines still contain the word `Error`, so a wrapper that greps
for it keeps working.

## Output names

| | Single file into a folder | Batch folder |
|---|---|---|
| Upscayl app | `<name>_upscayl_<scale>x_<model>.<ext>` | writes into `<out>/upscayl_<format>_<model>_<scale>x/` |
| ScaleUP | `<name>_scaleup_<scale>x_<model>.<ext>` (or `<width>px`) | writes directly into `-o` |

Rename with `rename 's/_scaleup_/_upscayl_/'` (or the equivalent) if downstream tools expect the
old infix. Upscayl's "double upscale" ran the binary twice with the second pass in place; ScaleUP's
`--double` does the same in one run and applies `-c`, `-w` and `-x` once, to the final image.

## Reusing an existing Upscayl models folder

Upscayl ships ncnn `.param`/`.bin` pairs. ScaleUP runs them on its `wgpu` backend without
conversion, so point it at the folder once:

```bash
scaleup models import "<Upscayl>/resources/models"
```

where `<Upscayl>` is the install directory: `%LOCALAPPDATA%\Programs\upscayl` on Windows,
`/opt/Upscayl` (deb/rpm) or the mounted AppImage on Linux, `/Applications/Upscayl.app/Contents`
on macOS. The command copies every pair into the ScaleUP models directory and prints the ids
(`realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-animevideov3`, `remacri`, `ultrasharp`,
`ultramix_balanced`, ...). A custom models folder chosen in Upscayl's settings works the same way,
and a single pair runs directly with `-n path/to/model.param`.

Model ids that already exist in the ScaleUP catalogue as ONNX models (`realesrgan-x4plus`,
`realesrgan-x4plus-anime`, `realesr-animevideov3`) are skipped by the import: the catalogue entry
wins and is downloaded on first use. Imported models have no license information, so they are
listed as non-commercial with license `unknown` until you check the model's own terms. Use
`scaleup models list` to see which format and license each id resolves to.
