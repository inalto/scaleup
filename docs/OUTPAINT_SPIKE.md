# Outpainting feasibility spike (Task 1, gate)

Throwaway measurements for the canvas-extension feature. Code: `tools/outpaint_spike.py` (Python 3.11,
onnxruntime-directml 1.24.4, Windows 11, GTX 1660 SUPER). Scratch folder (models, venv, outputs):
`F:/Apps/ScaleUP/outpaint-spike/`. Contact sheets: `F:/Apps/ScaleUP/outpaint-spike/out/sheet_*.png`.
Tasks 2 to 7 should treat tables 2, 3 and 6 as the contract.

## 7. Gate result (stated first)

**YES.** At least one model gives an acceptable 128 px extension on the easy photos: LaMa does on
`coffee_wall`, `chelsea_cat` and `astronaut_cut` (right side +128, and +128 on all four sides for the first
two); it also holds on `rocket_sky` right +128 (towers fade instead of being invented). MI-GAN passes on
`coffee_wall` and `astronaut_cut` but not on `chelsea_cat` (a yellow blob appears inside the blurred background
at +64, +128 and +256) or `rocket_sky` (broken tower fragments). Judged by eye on the sheets; no metric.

## 1. Models

| Model | File | Source (Hugging Face) | Source commit | Bytes | sha256 | Licence |
|---|---|---|---|---|---|---|
| MI-GAN | `migan.onnx` | `andraniksargsyan/migan` | `406830d0fa60666da0071c342ad2fbc8f30c5c64` | 29,546,882 | `593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae` | MIT (HF repo LICENSE and card). Weights licence: `LICENSE-WEIGHTS` in github.com/Picsart-AI-Research/MI-GAN, fetched from `https://raw.githubusercontent.com/Picsart-AI-Research/MI-GAN/main/LICENSE-WEIGHTS`, text is the standard MIT licence, "Copyright (c) 2024 Picsart AI Research (PAIR)". |
| LaMa | `lama_fp32.onnx` | `Carve/LaMa-ONNX` | `c3c0c9e468934d62e79c329e35d82dd09ff8c444` | 208,044,816 | `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` | Apache-2.0 (per the HF model card; nothing else read) |

Not checked (owner should decide before shipping): the upstream training-data terms of either set of weights
(both were trained on Places-style data, which I believe carries research-use wording). I did not verify that.
The MI-GAN repo also contains `migan_pipeline_v2.onnx`; it was not downloaded (owner approved `migan.onnx` only).

## 2. Tensor contract (what Task 4 implements)

Signatures read with onnxruntime (`tools/outpaint_spike.py probe`):

| | MI-GAN | LaMa |
|---|---|---|
| Input `image` | `image`, uint8, `[batch_size, 3, height, width]` | `image`, float32, `[batch, 3, 512, 512]` |
| Input `mask` | `mask`, uint8, `[batch_size, 1, height, width]` | `mask`, float32, `[batch, 1, 512, 512]` |
| Output | `result`, uint8, `[N, 3, H, W]` | `output`, float32, `[batch, 3, 512, 512]` |
| Spatial size | dynamic; 256x384, 500x500, 512x512 and 1024x1024 all ran. Use 512x512. | fixed 512x512 only; a 256x256 input is rejected (INVALID_ARGUMENT) |
| Input value range | 0..255 | 0..1 |
| Output value range | 0..255 (uint8, no scaling) | 0..255 float (saw 0.0..255.5); clip to 0..255 and round |
| Channel order | RGB in, RGB out | RGB in, RGB out |
| Mask polarity | **255 = known pixel, 0 = hole** | **1.0 = hole, 0.0 = known pixel** |
| Hole pixel content in `image` | ignored | ignored |
| Known pixels in output | returned unchanged | returned unchanged (after rounding) |

How each was established (synthetic 512x512 image: left half pure red `(255,0,0)` known, right half pure green
garbage, right half masked):

| Fact | MI-GAN experiment | LaMa experiment |
|---|---|---|
| Mask polarity | `known=255/hole=0`: left stayed `[255,0,0]` and the hole became `[214,8,1]` (red continued). Inverted mask: the left half was repainted `[6,82,14]` and the right half kept the green garbage. A mask of value 1 instead of 255 is treated as hole everywhere (hole mean `[135,125,109]`, a hallucination). | `hole=1`: left `[255.5,0,0]`, hole `[207,23,17]` (red continued). `hole=0`: left repainted `[6,228,54]`, right green kept. |
| Value range | uint8 in and out; inputs are 0..255. | 0..1 input: red continues. `-1..1` input: hole `[254,254,252]` (white, wrong). 0..255 input: hole `[255,255,255]` (wrong). Output is on a 0..255 scale (max 255.5). |
| Channel order | Feeding RGB gave a red continuation in channel 0; feeding BGR gave a continuation in channel 2. So the output order equals the input order. Whether the training order is RGB cannot be proven by quality: on 4 photos with the right half hidden the error against the original was 54.3 (fed RGB) vs 54.2 (fed BGR), indistinguishable. Use RGB (documented upstream order); the model is colour-aware (RGB-fed and BGR-fed outputs differ by 46 grey levels on a natural image). | Same: red channel 0 in gives channel 0 out. Ground-truth test 47.0 (RGB) vs 46.7 (BGR), indecisive. Use RGB. |
| Hole content ignored | Two different garbage fills gave outputs with max absolute difference 0. | Same, difference 0.0. So the prefill may be fed in the hole; it has no effect. |

## 3. Timing, one 512x512 window (half known), 5 runs after 2 warm-ups

| Model | CPU (ORT CPU provider) | GPU (DirectML provider) | Session create |
|---|---|---|---|
| MI-GAN | 709 ms mean (685 to 737) | 54 ms mean (52 to 61) | 1.4 s CPU, 1.0 s DML |
| LaMa fp32 | 2014 ms mean (1987 to 2062) | **fails**: `RUNTIME_EXCEPTION` in MatMul node `/generator/model/model.5/conv1/ffc/convg2g/fu/rttn/MatMul_5` (DirectML, "Parametro non corretto"), with default options, sequential execution and no mem pattern, and with graph optimisation off | 18.9 s CPU |

LaMa therefore runs on CPU only with this ORT build (1.24.4). A CUDA provider was not tried (not installed).
Whole-walk cost scales with window count: a 3000x2000 photo with a 192 px right margin at k=1 is 7 windows
(14 s LaMa CPU, 1 s MI-GAN DML); at k=4 it is 1 window (2.3 s LaMa).

## Method notes

Window walk as specified in Global Constraints, implemented in `outpaint()`. Two things the spec did not
cover, found while running it:

- **The half-known rule cannot be met when the source is shorter than the window.** A side strip window has at
  most `side / (2*S)` known pixels (e.g. 427 px tall source: 0.417). In the first run those windows were skipped
  and the canvas stayed partly empty (the 300 px tall chelsea photo failed the "all pixels filled" check).
  The walk now runs the best-placed window anyway and counts it. This happened on the all-four-side runs of
  `rocket_sky` and `coffee_wall` at 256 (2 windows), and `chelsea_cat` at 128 (1) and 256 (2). Task 5/6 needs
  the same fallback, not a skip.
- Pixels outside the canvas inside a window are reflect-padded and counted as known; the window for a strip
  shorter than the window is centred on the strip.

Window logs per image, margin and model: `out/sheets_log.txt` (windows, minimum known fraction, relaxed
windows, seconds).

## 4. Quality notes (judged by eye on `F:/Apps/ScaleUP/outpaint-spike/out/sheet_<image>_<right|all4>.png`)

Each sheet is margins 64, 128, 256 (rows) by mirror prefill, MI-GAN, LaMa (columns), tiles downscaled to 460 px
wide, so fine seams at 1:1 were **not** inspected. MI-GAN ran on DirectML, LaMa on CPU (the DML results of
MI-GAN were not compared to its CPU output; sizes and logic are identical).

| Image (type) | Margin | MI-GAN | LaMa |
|---|---|---|---|
| `coffee_wall` (400x600 photo, wooden table, soft wall) | right 64 / 128 | Good, grain continues, no seam seen. | Good, slightly smoother grain. |
| | right 256 | Believable table, but a few dark blobs and a hard diagonal wood edge at the corner. | Smooth, plausible, loses grain towards the edge; no invented objects. |
| | all4 64 | Wood continues; a pale band along the top edge. | Good; shadow and table edge continue naturally. |
| | all4 128 | Invents a pale wall/shelf strip at the top and streaks on the left. | Plausible, soft, slightly smeared; best result of this set. |
| | all4 256 | Invents a whole room (corridor, door, white figure). Failure. | Table fades to flat grey-brown; plausible but loses detail. |
| `chelsea_cat` (300x451 photo, blurred background) | right 64 / 128 / 256 | A solid yellow blob is invented in the background in all three; at 256 also a fur-like texture patch. Fails. | Right 64 and 128 clean; a pale pink smear at the top-right corner appears at 128; at 256 a flat pink-beige slab replaces the background. Acceptable to 128. |
| | all4 64 | Cat fur extends but a ghost arch and the yellow blob appear. | Clean, continues blur and fur. |
| | all4 128 / 256 | Invents fur masses and an arch; 256 is incoherent. | Soft vignette of grey-brown smear; no invented objects; at 256 the cat sits on flat grey with horizontal bands. |
| `rocket_sky` (427x640 night photo, steel towers, lights) | right 64 / 128 / 256 | The tower cut by the frame becomes broken, glassy fragments; at 256 an orange strand appears. Fails. | Towers fade into dark sky, no invented objects; at 256 the right 100 px lose detail. Acceptable. |
| | all4 64 | Reasonable sky; invents ground clutter with garbled yellow lettering at the bottom. | Good; sky, tower continue, ground becomes dim clutter. |
| | all4 128 | Sky OK; bottom has tangled metal. | Good, dim. |
| | all4 256 | Invents aircraft-shaped objects, a cloud, a coloured sign. Fails. | Almost flat dark vignette; very little structure but clean. |
| `gravel_texture` (512x512 busy texture) | right 64 | Texture continues, a few dark blobs. | Texture continues but turns visibly smoother at the last ~30 px. |
| | right 128 / 256 | Over-regular dark blob pattern, pale patch at 256. Repeated texture, fails. | Flat grey slab with a clearly visible vertical seam at the old edge. Fails as texture extension. |
| | all4 64 | Regular dotted pattern, repeated. | Texture continues but a smooth frame is visible. |
| | all4 128 / 256 | Dotted checker pattern, pale sky strip at the top, an orange blob at 256. Fails. | Original square floats on flat grey with a visible rectangular boundary. Fails. |
| `astronaut_cut` (512x512 portrait, subject cut at bottom and left by the frame) | right 64 / 128 | Backdrop and shuttle model continue; some dark patches. Acceptable. | Clean, shading continues. Acceptable. |
| | right 256 | Backdrop continues, with a vertical wall edge and dark patches. | Shading continues to black; clean. |
| | all4 64 | Flag continues with a smear at the top-left corner; hands and lap get messy at the bottom. | Flag, suit and helmet continue; lap area dark and smudgy. Best result. |
| | all4 128 | Invents a figure, a bird-like shape and a messy collage on the left and bottom. Fails. | Flag becomes a blurred, shorter flag; suit sleeves lost into dark smudge at the bottom; no invented people. |
| | all4 256 | Invents people, faces and clutter on the left and bottom. Fails. | Subject sits on a flat grey backdrop with a faded flag; no invented objects; the subject body is not continued downwards (it ends in a smudge). |
| `screenshot_text` (916x639 app UI screenshot) | right 64 / 128 / 256 | The black window edge becomes textured noise, then a green streaked slab at 256. Fails. | Continues the black strip. Clean. |
| | all4 64 | Stray textured noise at the right edge, white above, black below. | Clean: white above, black below and right. |
| | all4 128 / 256 | Green streak slab on the right, noisy white and black shapes. Fails. | White and dark panels, some blurry streaks at the top. No invented text or widgets. |
| Anime frame | | **Not covered.** No anime image exists in the allowed offline sources (skimage data, repo fixtures, repo screenshots). Nothing was faked. | |

Text at the frame edge was not tested (the screenshot's borders are plain).

Summary of observed behaviour:

- LaMa fails by smoothing: contexts it cannot continue become blur or flat tone. Never invented objects in these tests.
- MI-GAN fails by inventing: new objects, faces, text-like marks, regular dot patterns. Frequent from +128 on, and almost always at +256 or on four sides.
- Four-sided extension is much worse than one-sided at the same margin, for both models. Corners are where most defects appear.
- Margin bands for LaMa on photos: +64 good, +128 acceptable, +256 soft/plain background but not invented (one-sided OK on soft backgrounds, four-sided loses detail). Textures (`gravel`) are poor beyond +64 for both models.

## 5. Context factor k experiment

Source: `skimage.data.rocket()` upsampled with Lanczos to 3000x2001 (**not** a native 3000 px photo; none is
available offline, so the fine detail is soft). Right-side margin only, margins 192 / 384 / 768 (which the default
rule maps to k = 1 / 2 / 4), k = 1, 2, 4. Sheets: `out/sheet_k_migan.png`, `out/sheet_k_lama.png`; crops are
1400x1000 around the seam shown at 480 px wide (0.34 scale).

| Margin | k=1 windows | k=2 windows | k=4 windows | LaMa seconds (k=1 / 2 / 4) | MI-GAN seconds |
|---|---|---|---|---|---|
| 192 | 7 | 3 | 1 | 14.2 / 6.2 / 2.3 | 1.0 / 0.4 / 0.3 |
| 384 | 14 | 3 | 1 | 28.2 / 6.3 / 2.3 | 1.0 / 0.4 / 0.4 |
| 768 | 21 | 6 | 1 | 42.6 / 12.5 / 2.4 | 1.5 / 0.8 / 0.4 |

| Margin | LaMa k=1 | LaMa k=2 | LaMa k=4 | MI-GAN k=1 | MI-GAN k=2 | MI-GAN k=4 |
|---|---|---|---|---|---|---|
| 192 | Fine; slight dark-green drift beyond the tower. | Fine, blue sky continues. | Fine; tower edge slightly softer. | Green rectangle and horizontal smear at the bottom. | Orange streaks, cracked lines. | Orange streak, tangled bright lines. |
| 384 | Drifts to grey-green, vertical banding. | Blue sky continues, mild darkening. | Blue sky continues; best colour. | Faint banding plus a small object at the bottom. | Long bright orange lines. | Webbing of invented lines. |
| 768 | Strong grey-green drift and vertical banding (each ring shifts colour). | Palette mostly kept, faint lighter patch. | Smooth, correct sky colour, cleanest. | Colour blocks and a rectangular artefact. | Bright invented strands. | Bright invented strands (worst). |

Reading: for LaMa a larger context fixes the colour drift and banding that stack up over several rings, and it
needs far fewer windows (1 at k=4). For MI-GAN a larger k makes inventions worse, so k=1 is the best of a bad set
but the model is not a good fit at large margins in any setting. Caveats: one image, a soft upsampled photo, a hard
tower-at-the-edge case; the k=4 downsample (4x) costs resolution in the fill that would show on a real sharp photo
and was **not verified** at 1:1.

## 6. Verdict

- **Default model: LaMa (`lama_fp32.onnx`).** Its failure mode (blur or flat tone) is tolerable in a canvas
  extender; MI-GAN's (invented objects, patterns, faces) is not, from +128 on. This inverts the plan's assumption
  that MI-GAN is the default; owner decision. Costs of LaMa: 208 MB download, ~2 s per window on CPU, and **no GPU
  path** with ORT 1.24.4 DirectML (it throws), so Task 4 must pin LaMa to the CPU provider (or retest other builds
  or providers).
- **Second catalogue entry: yes, but it is MI-GAN, as the light/fast option.** 29.5 MB, 54 ms per window on
  DirectML, 709 ms on CPU, good enough for margins up to +64, and on easy soft backgrounds (coffee, astronaut)
  up to +128. Label it as lower quality for large or four-sided extensions. If the owner wants a single entry, ship
  LaMa only.
- **Ring depth: window/2 = 256 px at k=1 (256*k at other k), as specified.** Two rings (k=1, >256) visibly
  degrade LaMa (grey-green drift and banding); use larger k instead of more rings (see k rule). Quality bands to
  state in the UI/doc: up to 128 px per side good, 256 px soft, beyond that expect blur.
- **k rule: keep `k = clamp(ceil(max_margin / 192), 1, 4)` for LaMa.** It gave k=1 at 192 (fine), k=2 at 384 (clearly
  better than k=1) and k=4 at 768 (best). For MI-GAN use k=1 whatever the margin (larger k made it worse).
- **Walk implementation requirements from the evidence:** (a) run the best-placed window when the half-known rule
  cannot be met (small sources) instead of skipping; (b) corners fail most, so do corner windows last with three
  known quadrants as specified; (c) known pixels stayed bit-exact in every run (asserted in the spike).
- **Open items, unverified:** anime frames not tested; seams at 1:1 not inspected; no feathering tried (only unknown
  pixels written as specified; no obvious seam was visible at sheet scale except the gravel/LaMa band); LaMa fp16
  and CUDA provider not tried; training-data licence questions in section 1.
