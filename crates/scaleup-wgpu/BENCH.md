# scaleup-wgpu benchmarks

Measured on Windows 11 with an NVIDIA GTX 1660 SUPER (6 GB). Input is one synthetic RGB image processed as a single tile, 4× output. Times are whole-process seconds, including model load and writing the PNG, unless noted. They are single runs, except tile 256, which shows two.

| Engine | Model | Tile 256 | Tile 512 |
|---|---|---|---|
| wgpu, Vulkan | `upscayl-standard-4x` | 7.3, 6.6 | 80.8 |
| `upscayl-bin.exe` (ncnn, Vulkan) | `upscayl-standard-4x` | 4.5, 3.9 | **fails**, see below |
| ONNX Runtime, DirectML | `realesrgan-x4plus` | 6.7, 5.3 | 93.4 |
| wgpu, Vulkan | `upscayl-lite-4x` | | 8.2 |
| `upscayl-bin.exe` | `upscayl-lite-4x` | | 2.0 |
| ONNX Runtime, DirectML | `realesr-general-x4v3` | | 7.8 |

The ONNX Runtime rows use the same architecture with different weights, so they compare speed, not output. ONNX Runtime with CUDA is not in the table: this machine initialises DirectML, not CUDA, so it could not be measured.

## Reading the numbers

- **Inference only**, without model load and PNG writing, wgpu takes 3.8 s for standard at tile 256 and 1.0 s for lite at tile 512. Most of the lite run's 8.2 s is PNG encoding, which every ScaleUP backend shares.
- **`RealESRGAN_x4plus.pth` run natively** on `odd_1001x733.png`, whole CLI run at the default tile, release build: 180.0 s, against 180.8 s for the ncnn file of the same network (`upscayl-standard-4x.param`) and 37.3 s for its ONNX export on DirectML (outputs 86.6 dB apart). Reading PyTorch weights costs nothing over the ncnn path; the gap to DirectML is the executor's.
- **`upscayl-bin.exe` fails at tile 512 with the standard model.** After 369 s it reports success and writes an entirely black image. wgpu and DirectML both complete that tile correctly on this card.

## Shared-memory convolution

Same machine, inference only.

| Kernel | Standard, tile 256 | Standard, tile 512 | Lite, tile 512 |
|---|---|---|---|
| Direct (`conv.wgsl`) | 19.0 | 125.0 | 4.65 |
| Tiled (`conv3x3_tiled.wgsl`) | 3.8 | not timed alone | 1.02 |

The tiled kernel's output is byte-identical to the direct kernel's in all three cases. Standard at tile 512 was compared through the CLI, which took 80.8 s in total.

## Tile overlap and seams

Each tile is pasted as it comes out of the network, so a line can show where two tiles meet unless the overlap covers what the network actually looks at. Each network now asks for the overlap it needs through `Model::preferred_overlap`; `--tile-overlap` overrides it.

Input is three 640×480 crops of two JPEG photographs of a Rio hillside: rock and cloud, dense buildings, clear sky and forest. Each result is compared with a whole-image run of the same model, 8-bit PNG against 8-bit PNG, at tile 256 and leaving out 64 input pixels around the image, where a whole run zero-pads and tiles reflect. Everything comes from `ncnn_tool seams`:

- **Worst**: largest single channel difference from the whole-image run, anywhere, in 1/255 steps.
- **Line**: how much that difference jumps across a tile boundary, averaged along 32 output pixels of boundary, and the largest such value. This is the part an eye can see as a line. **Flat** is the same where the picture is flat.
- **≥ 1**: share of boundary segments whose line value is at least 1/255. This is the pass mark used here. It is a chosen threshold, not a perceptual measurement.

Each row reports the worst of the three crops, with the median time. wgpu rows are inference only, on Vulkan. ONNX rows are the whole CLI process on DirectML, so they include about 5 s of model load and PNG writing. Times are single runs on a machine that was also doing other work. **Bold** marks the overlap each network now asks for.

| Network | Engine | Convolutions or architecture | Overlap | Seconds | Worst | Line | Flat | ≥ 1 |
|---|---|---|---|---|---|---|---|---|
| `upscayl-lite-4x` | wgpu | 34 | 16 | 1.8 | 7 | 2.03 | 0.81 | 4.6% |
| `upscayl-lite-4x` | wgpu | 34 | **32** | 2.5 | 1 | 0.03 | 0.03 | 0.0% |
| `upscayl-lite-4x` | wgpu | 34 | 40 | 2.9 | 0 | 0.00 | 0.00 | 0.0% |
| `realesr-general-x4v3` | DirectML | SRVGGNetCompact | 16 | 6.8 | 7 | 1.97 | 0.75 | 4.2% |
| `realesr-general-x4v3` | DirectML | SRVGGNetCompact | **32** | 7.0 | 1 | 0.03 | 0.03 | 0.0% |
| `realesr-general-x4v3` | DirectML | SRVGGNetCompact | 40 | 7.0 | 0 | 0.00 | 0.00 | 0.0% |
| `4xNomosUni_span_multijpg` | DirectML | SPAN | **16** | 5.0 | 1 | 0.06 | 0.03 | 0.0% |
| `4xNomosUni_span_multijpg` | DirectML | SPAN | 32 | 5.1 | 1 | 0.03 | 0.00 | 0.0% |
| `digital-art-4x` | wgpu | 96 | 16 | 7.8 | 112 | 13.75 | 1.97 | 36.4% |
| `digital-art-4x` | wgpu | 96 | 32 | 11.1 | 38 | 3.88 | 0.47 | 7.4% |
| `digital-art-4x` | wgpu | 96 | 40 | 13.5 | 28 | 2.94 | 0.34 | 1.8% |
| `digital-art-4x` | wgpu | 96 | **48** | 14.3 | 6 | 0.50 | 0.19 | 0.0% |
| `digital-art-4x` | wgpu | 96 | 64 | 23.1 | 1 | 0.09 | 0.03 | 0.0% |
| `realesrgan-x4plus-anime` | DirectML | RRDBNet | 16 | 6.7 | 112 | 13.69 | 1.91 | 36.0% |
| `realesrgan-x4plus-anime` | DirectML | RRDBNet | 32 | 7.3 | 40 | 4.09 | 0.47 | 7.4% |
| `realesrgan-x4plus-anime` | DirectML | RRDBNet | 48 | 7.8 | 7 | 0.56 | 0.25 | 0.0% |
| `realesrgan-x4plus-anime` | DirectML | RRDBNet | **64** | 9.6 | 1 | 0.06 | 0.03 | 0.0% |
| `upscayl-standard-4x` | wgpu | 351 | 16 | 26.0 | 80 | 7.62 | 3.62 | 42.5% |
| `upscayl-standard-4x` | wgpu | 351 | 32 | 36.9 | 45 | 5.38 | 2.69 | 17.9% |
| `upscayl-standard-4x` | wgpu | 351 | 48 | 47.7 | 48 | 4.75 | 1.53 | 3.7% |
| `upscayl-standard-4x` | wgpu | 351 | **64** | 78.3 | 7 | 0.75 | 0.47 | 0.0% |
| `realesrgan-x4plus` | DirectML | RRDBNet | 16 | 12.9 | 82 | 7.75 | 3.69 | 43.2% |
| `realesrgan-x4plus` | DirectML | RRDBNet | 32 | 13.4 | 44 | 5.44 | 2.59 | 18.1% |
| `realesrgan-x4plus` | DirectML | RRDBNet | 48 | 14.9 | 48 | 4.75 | 1.59 | 3.7% |
| `realesrgan-x4plus` | DirectML | RRDBNet | **64** | 20.3 | 7 | 0.72 | 0.59 | 0.0% |
| `4x-UltraSharp` | DirectML | RRDBNet | 32 | 15.7 | 5 | 0.41 | 0.34 | 0.0% |
| `4x-UltraSharp` | DirectML | RRDBNet | 48 | 14.2 | 1 | 0.06 | 0.06 | 0.0% |
| `4x-UltraSharp` | DirectML | RRDBNet | **64** | 19.6 | 1 | 0.03 | 0.03 | 0.0% |

- **What is needed follows the depth.** SRVGGNetCompact is exact from overlap 40, which fits its receptive radius of 34 pixels, and one step off at 32. RRDB networks are not so sharply bounded and take 48 at 96 convolutions, 64 at 351.
- **Weights matter too.** `4x-UltraSharp` has the same architecture as `realesrgan-x4plus` and passes at 32. Catalogue models are chosen by architecture, so it runs at 64 and pays for the deeper network's need.
- **Networks of the same shape behave alike on both engines.** `upscayl-lite-4x` tracks `realesr-general-x4v3`, `digital-art-4x` tracks `realesrgan-x4plus-anime`, and `upscayl-standard-4x` tracks `realesrgan-x4plus`.
- **Synthetic images overstate RRDB.** A synthetic texture still differed by 0.076, about 19/255, on `digital-art-4x` at overlap 64. These photographs differ by at most 1/255.
- **Not covered.** No DAT model and no model at tile 512 was measured. `remacri-4x`, `ultramix-balanced-4x`, `high-fidelity-4x` and `ultrasharp-4x` (ncnn) were not run; they have 351 convolutions, so they ask for 64.

### Cost

Tiles per image grow with overlap, as (tile / (tile − 2 × overlap))² for a large image:

| Tile | Overlap 16 | 32 | 48 | 64 |
|---|---|---|---|---|
| 128 | 1.78 | 4.00 | capped at 32 | capped at 32 |
| 256 | 1.31 | 1.78 | 2.56 | 4.00 |
| 512 | 1.14 | 1.31 | 1.52 | 1.78 |

Against the old flat overlap of 16 at tile 256, inference on wgpu takes 1.4× as long on `upscayl-lite-4x` (2.5 s against 1.8 s), 1.8× on `digital-art-4x` (14.3 s against 7.8 s) and 3× on `upscayl-standard-4x` (78.3 s against 26.0 s). A flat overlap of 64 would have cost `upscayl-lite-4x` 2.8× for nothing, and a flat 32 would have left lines on every RRDB network except `4x-UltraSharp`.

The automatic tile size grows to four times the overlap, so the quarter-tile cap does not undo the choice. On the CPU this makes the automatic tile 256 instead of 128 for RRDB networks. CPU runs on the buildings crop, whole CLI process, single runs:

| Network | Before: tile 128, overlap 16 | Now: automatic |
|---|---|---|
| `realesr-general-x4v3` | 11 s | 15 s (tile 128, overlap 32) |
| `realesrgan-x4plus` | 75 s | 198 s (tile 256, overlap 64) |

An explicit tile size is left alone, so `-t 128` still caps the overlap at 32.

## Parity with the original engine

PSNR of wgpu against `upscayl-bin.exe` on the same weights:

| Model | Check | PSNR |
|---|---|---|
| `upscayl-standard-4x` | whole image, tile 256 | 47.75 dB |
| `upscayl-lite-4x` | whole image, tile 512 | 53.20 dB |
| `upscayl-lite-4x` | interior, `parity_with_upscayl_bin` | 45.91 dB |
| `digital-art-4x` | interior, `parity_with_upscayl_bin` | 47.67 dB |

The two engines pad the image border and place tiles differently, which explains part of the difference: `digital-art-4x` fell to 33.10 dB when both ran tile 128 on a high-contrast picture. The remaining difference has not been traced.

## Not done

The optional half-precision storage for activations was not attempted.

## Other APIs on this machine

`kernels_work_on_every_adapter_of_this_machine` compiles the kernels and checks a convolution against the CPU reference on NVIDIA through Vulkan, NVIDIA through DX12, and the Microsoft Basic Render Driver. All three use the tiled kernel. Metal and Linux drivers have not been run.
