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
- **`upscayl-bin.exe` fails at tile 512 with the standard model.** After 369 s it reports success and writes an entirely black image. wgpu and DirectML both complete that tile correctly on this card.

## Shared-memory convolution

Same machine, inference only.

| Kernel | Standard, tile 256 | Standard, tile 512 | Lite, tile 512 |
|---|---|---|---|
| Direct (`conv.wgsl`) | 19.0 | 125.0 | 4.65 |
| Tiled (`conv3x3_tiled.wgsl`) | 3.8 | not timed alone | 1.02 |

The tiled kernel's output is byte-identical to the direct kernel's in all three cases. Standard at tile 512 was compared through the CLI, which took 80.8 s in total.

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
