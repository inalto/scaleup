# Outpainting feasibility spike (Task 1, gate)

Throwaway measurements for the canvas-extension feature. Code: `tools/outpaint_spike.py` (Python 3.11,
onnxruntime-directml 1.24.4, Windows 11, GTX 1660 SUPER). Models, venv and outputs were kept in
the spike's scratch folder, outside the repository; the contact sheets are its `out/sheet_*.png`.
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
| Spatial size | dynamic; 256x384, 500x500, 512x512 and 1024x1024 all ran (re-run, logged in `probe_fix1.txt`). Use 512x512. | fixed 512x512 only; a 256x256 input is rejected (INVALID_ARGUMENT) |
| Input value range | 0..255 | 0..1 |
| Output value range | 0..255 (uint8, no scaling) | 0..255 float (observed range 0.0 to 255.0); clip to 0..255 and round |
| Channel order | RGB in, RGB out | RGB in, RGB out |
| Mask polarity | **255 = known pixel, 0 = hole** | **1.0 = hole, 0.0 = known pixel** |
| Hole pixel content in `image` | ignored | ignored |
| Known pixels in output | returned unchanged | returned unchanged (after rounding) |

How each was established (synthetic 512x512 image: left half pure red `(255,0,0)` known, right half pure green
garbage, right half masked):

| Fact | MI-GAN experiment | LaMa experiment |
|---|---|---|
| Mask polarity | `known=255/hole=0`: left stayed `[255,0,0]` and the hole became `[214,8,1]` (red continued). Inverted mask: the left half was repainted `[6,82,14]` and the right half kept the green garbage. A mask of value 1 instead of 255 is treated as hole everywhere (hole mean `[135,125,109]`, a hallucination). | `hole=1`: left `[255,0,0]`, hole `[207.5,22.7,16.6]` (red continued). `hole=0`: left repainted `[5.6,228,53.6]`, right green kept. |
| Value range | uint8 in and out; inputs are 0..255. | 0..1 input: red continues. `-1..1` input: hole `[253,254,252]` (white, wrong). 0..255 input: hole `[255,254,254]` (wrong). Output is on a 0..255 scale (observed max 255.0). Means are computed in float64; an earlier float32 `mean()` printed 255.5 spuriously. |
| Channel order | Feeding RGB gave a red continuation in channel 0; feeding BGR gave a continuation in channel 2. So the output order equals the input order. Whether the training order is RGB cannot be proven by quality (probe re-run and logged in `probe_fix1.txt` in the spike's scratch folder, outside the repository): on 4 photos with the right half hidden the error against the original was 54.3 (fed RGB) vs 54.2 (fed BGR), indistinguishable. Use RGB (documented upstream order); the model is colour-aware (RGB-fed and BGR-fed outputs differ by 46 grey levels on a natural image). | Same: red channel 0 in gives channel 0 out. Ground-truth test 47.0 (RGB) vs 46.7 (BGR), indecisive. Use RGB. |
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
  shorter than the window is centred on the strip. For a source shorter than 512 that centred side window also
  reaches into (and fills part of) the corner regions before the corner windows run, so corners are not always
  filled by a corner window with three known quadrants.
- In the k experiment at k=4 the window (2048 px) is taller than the 2001 px source, so reflect-padded rows
  counted as known there.

Window logs per image, margin and model: `out/sheets_log.txt` (windows, minimum known fraction, relaxed
windows, seconds).

## 4. Quality notes (judged by eye on `out/sheet_<image>_<right|all4>.png` in the spike's scratch folder, outside the repository)

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
- MI-GAN fails by inventing: new objects, faces, text-like marks, regular dot patterns. It already fails at +64 on `chelsea_cat`, `rocket_sky` and `screenshot_text` (right side) and on four sides at +64 on `chelsea_cat`, `rocket_sky`, `astronaut_cut` and `gravel_texture`.
- MI-GAN band supported by the table: one side only, +64 and +128, on `coffee_wall` and `astronaut_cut`. Four sides: only `coffee_wall` +64 (with a pale band along the top); nothing else.
- LaMa band supported by the table (photos with soft or plain backgrounds: coffee, chelsea, astronaut, rocket sky; screenshot with plain borders): one side +64 and +128 acceptable (`chelsea_cat` +128 has a pink smear at the top-right corner); +256 gives a plain or blurred fill, not invented objects. Four sides: +64 clean or acceptable on all of these; +128 good on `coffee_wall` and `rocket_sky`, but a blurred halo on `chelsea_cat` and lost sleeves/body on `astronaut_cut`; +256 is mostly flat tone. These are "acceptable", not "good", judged by eye at sheet scale.
- Textures are excluded: `gravel_texture` fails for LaMa from +128 one-sided and is visibly smoothed at +64, and fails for MI-GAN from +128 (blob pattern).
- Four-sided versus one-sided: much worse for MI-GAN at the same margin; for LaMa mixed (similar on coffee and rocket up to +128, worse on chelsea and astronaut from +128).

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
  extender; MI-GAN's (invented objects, patterns, faces) is not, and it shows from +64 on some images (one side:
  `chelsea_cat`, `rocket_sky`, `screenshot_text`; four sides: `chelsea_cat`, `rocket_sky`, `astronaut_cut`,
  `gravel_texture`); only one-sided +64 and +128 on `coffee_wall` and `astronaut_cut`, and four-sided +64 on `coffee_wall`, held up. This inverts the plan's assumption
  that MI-GAN is the default; owner decision. Costs of LaMa: 208 MB download, ~2 s per window on CPU, and **no GPU
  path** with ORT 1.24.4 DirectML (it throws), so Task 4 must pin LaMa to the CPU provider (or retest other builds
  or providers).
- **Second catalogue entry: yes, but it is MI-GAN, as the light/fast option.** 29.5 MB, 54 ms per window on
  DirectML, 709 ms on CPU, usable only for one-sided extensions of up to +128 on soft photo backgrounds
  (`coffee_wall`, `astronaut_cut` in these tests). It failed at +64 on `chelsea_cat`, `rocket_sky`, `screenshot_text` and on
  most four-sided runs, so label it as an unreliable fast option, not a general one. If the owner wants a single entry, ship
  LaMa only.
- **Ring depth: window/2 = 256 px at k=1 (256*k at other k), as specified.** Two rings (k=1, >256) visibly
  degrade LaMa (grey-green drift and banding); use larger k instead of more rings (see k rule). Quality wording the evidence supports (LaMa,
  photos with soft or plain backgrounds only, textures excluded): up to 128 px per side acceptable, though four-sided 128 can
  blur or lose detail on busy subjects; 256 px gives a plain or blurred fill. Do not claim "good".
- **k rule: keep `k = clamp(ceil(max_margin / 192), 1, 4)` for LaMa.** It gave k=1 at 192 (fine), k=2 at 384 (clearly
  better than k=1) and k=4 at 768 (best). For MI-GAN use k=1 whatever the margin (larger k made it worse).
- **Walk implementation requirements from the evidence:** (a) run the best-placed window when the half-known rule
  cannot be met (small sources) instead of skipping; (b) corners fail most, so do corner windows last with three
  known quadrants as specified; (c) known pixels stayed bit-exact in every run (asserted in the spike).
- **Open items, unverified:** anime frames not tested; seams at 1:1 not inspected; no feathering tried (only unknown
  pixels written as specified; no obvious seam was visible at sheet scale except the gravel/LaMa band); LaMa fp16
  and CUDA provider not tried; training-data licence questions in section 1.


---

## Plan 12 candidates (2026-11-24)

Five more inpainting candidates for the canvas-studio plan (D5, D6, D7), measured with the same method as
above. Code: `tools/outpaint_spike.py` (new commands `probe12`, `conv12`, `sheets12`). Scratch folder (outside the
repository): models in `outpaint-spike/models/<name>/`, outputs in `outpaint-spike/out/plan12/`
(`sheet_<image>_<right|all4>_<a|b>.png`, `probe12.txt`, `conv12.txt`, `sheets12_log.txt`). onnxruntime-directml
1.24.4, Windows 11, GTX 1660 SUPER. Licence statements below are "declared by the uploader; weights and
training-data terms not verified".

### Verdicts first

| Model | Verdict | One line |
|---|---|---|
| `lama_manga` (ogkalu LaMa manga, dynamic) | **ADD** | Clean on flat colour and silhouettes, never invented an object, runs on DirectML (298 ms vs 2.2 s CPU) with the same result as CPU. |
| `cv_lama` (OpenCV LaMa) | SKIP | Same network family as the LaMa already shipped, smaller, but CPU only and drifts to a dark teal/green cast from +192. |
| `aot` (ogkalu AOT-GAN) | SKIP | Small and fast (112 ms DirectML), fine on flat shapes and at +64, but smears, white strips and hard edges at +192 on photos. |
| `qc_lama` (Qualcomm LaMa-Dilated zip) | SKIP | Invents human faces at +192 (three different images). GPU works but is no reason to add zip support. |
| `qc_aot` (Qualcomm AOT-GAN zip) | SKIP | Colour-smear garbage strips from +64 on some images, worse at +192. |

D7 (Qualcomm zips): **not justified.** Both zips do run on DirectML (about 150 ms per window against 1.3 to 2.3 s on CPU),
but the better result is `lama_manga`, a plain single `.onnx` that also runs on DirectML. No zip or
external-data support is needed in the registry.

### Sources, sizes, hashes (all sizes equal the expected ones)

| Key | URL (pinned commit) | Bytes | sha256 | Declared licence |
|---|---|---|---|---|
| `lama_manga` | `huggingface.co/ogkalu/lama-manga-onnx-dynamic` @ `ee4ed4a8447b6730fc41d34f90876b6c48af925a`, `lama-manga-dynamic.onnx` | 206,291,843 | `de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9` | Apache-2.0, declared by the uploader; weights and training-data terms not verified |
| `aot` | `huggingface.co/ogkalu/aot-inpainting` @ `42ffc84ff1bd46dd95f1c5a41e83ee7e98f39189`, `aot.onnx` | 23,068,213 | `ffd39ed8e2a275869d3b49180d030f0d8b8b9c2c20ed0e099ecd207201f0eada` | MIT tag, declared by the uploader; weights and training-data terms not verified |
| `cv_lama` | `huggingface.co/opencv/inpainting_lama` @ `aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f`, `inpainting_lama_2025jan.onnx` | 92,591,623 | `7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2` | Apache, declared by the uploader; weights and training-data terms not verified |
| `qc_lama` zip | `qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/lama_dilated/releases/v0.67.0/lama_dilated-onnx-float.zip` | 169,631,704 | `516c324a61cf70bc210ee38ada4c2a2361c6f1afc422e144bdec85c4d9b2da07` | not read |
| inside it | `lama_dilated.onnx` / `lama_dilated.data` (plus `metadata.json`) | 1,044,115 / 182,568,704 | `199f4e84695b7bf9afba3c803fb8ba341b459d9e699f90b66b8f6df4bfab5d2f` / `689b3d1a87b906cb0ff9a42394c187a86ceae369061384175df821976b5b7d9b` | |
| `qc_aot` zip | `.../models/aotgan/releases/v0.67.0/aotgan-onnx-float.zip` | 56,621,001 | `aed3349473be5dc525eefdc63547b5708fe29a5fe6bec0090a869f60f257e9a2` | not read |
| inside it | `aotgan.onnx` / `aotgan.data` (plus `metadata.json`) | 659,913 / 60,793,600 | `97891cada506ddd4fc46cd3ff9fad2b5e6622319baa0a08314cecf40c277f378` / `9a5fb31bff02a111e16979ef8a463e3d919688343fac1a8268c2c5b2241f05f3` | |

Zips were unzipped with Python `zipfile` into their own folders; only the `.onnx` files were loaded (the `.data` file by
ONNX Runtime as external data). Nothing else was run.

### Facts

| | `lama_manga` | `aot` | `cv_lama` | `qc_lama` | `qc_aot` |
|---|---|---|---|---|---|
| Opset (ai.onnx) | 18 | 18 | 21 | 21 | 21 |
| Input `image` | float32 `[batch,3,h,w]` | float32 `[batch,3,h,w]` | float32 `[batch,3,512,512]` | float32 `[1,3,512,512]` | float32 `[1,3,512,512]` |
| Input `mask` | float32 `[batch,1,h,w]` | float32 `[batch,1,h,w]` | float32 `[batch,1,512,512]` | float32 `[1,1,512,512]` | float32 `[1,1,512,512]` |
| Output | `inpainted` float32 `[batch,3,h,w]` | `inpainted` float32 `[batch,3,h,w]` | `output` float32 `[batch,3,512,512]` | `painted_image` float32 `[1,3,512,512]` | `painted_image` float32 `[1,3,512,512]` |
| Size | dynamic | dynamic | fixed 512 | fixed 512 | fixed 512 |
| Input range | 0..1 | **-1..1** | 0..1 | 0..1 | 0..1 |
| Mask polarity | 1 = hole | 1 = hole | 1 = hole | 1 = hole | 1 = hole |
| Output range | 0..1 | -1..1 | **0..255** | 0..1 | 0..1 |
| Channel order | RGB | RGB | RGB | RGB | RGB |
| Hole pixels in `image` | ignored (identical output for garbage, black, white) | **used**: must be mid grey (0.0 in -1..1); garbage green gave known MAE 22, grey 4.5 | ignored | ignored | ignored |
| Known pixels in output | returned exactly | **changed** (known-area MAE 4.5): composite the original back | returned exactly | returned exactly | returned exactly |

How range and polarity were found (`conv12`): a real photo (`astronaut`, 512x512) with the right 192 columns and an interior
130x130 square hidden, so the hidden pixels have a ground truth. Every combination of input range (0..1, -1..1,
0..255) x polarity (hole=1, known=1) x hole content (green garbage, black, mid grey, white) was run and the output
was scored on three output scales against the original (mean absolute error on the visible part and on the hidden part).
Winners: `lama_manga` 0..1 / hole=1 gives known 0.00 and hidden 49.7 (the same model fed -1..1 scores 87.3 on the hidden
part, the inverted mask 70.3 on the visible part, i.e. it repaints it); `cv_lama` 0..1 / hole=1 with output 0..255 gives
0.00 / 55.8 (raw output range 0..255 observed); `qc_lama` 0..1 / hole=1 gives 0.00 / 50.9; `qc_aot` 0..1 / hole=1 gives
0.00 / 60.6; `aot` -1..1 / hole=1 / grey hole gives 4.47 / 52.8 (best of 24; 0..1 input with a black hole is second at
6.78 / 57.8 and is rejected because its raw output only spans -0.40..1.00 and looks washed out). The vendor
`metadata.json` of both Qualcomm zips says 0..1 RGB in and out; the experiment agrees, the mask polarity is not stated
there and came from the experiment. A first version of the probe left the true pixels in the hole of the test image
and so flattered `aot` and `qc_aot`; it was fixed (hole = green garbage) and re-run before the numbers above were taken
(first run kept as `probe12_run1_leaky.txt`; its timings are still valid and are used below).

### Timing, one 512x512 window (half known), 5 runs after warm-up

| Model | CPU load | CPU window | DirectML load | DirectML first run | DirectML window | DirectML result |
|---|---|---|---|---|---|---|
| `lama_manga` | 19.5 s | 2195 ms (2013 to 2446) | 16.4 s | 5.8 s | **298 ms** | same as CPU (mean abs diff 0.00, max 1 of 255) |
| `aot` | 0.8 s | 1057 ms | 0.7 s | 1.7 s | **112 ms** | same as CPU (0.00, max 1) |
| `cv_lama` | 21.8 s | 2916 ms | n/a | n/a | **fails** (MatMul `/generator/model/model.5/conv1/ffc/convg2g/fu/rttn/MatMul_5`, the same node that stops the Carve LaMa) | n/a |
| `qc_lama` | 1.1 s | 1335 ms | 1.3 s | 0.4 s | **151 ms** | same as CPU (0.00, max 1) |
| `qc_aot` | 0.3 s | 2286 ms | 4.6 s | 0.4 s | **149 ms** | same as CPU (0.00, max 1) |

Load times of the two big graphs (`lama_manga`, `cv_lama`) are 16 to 22 s every start (the first spike saw 19 s for LaMa), so the
app should create the session once and keep it. Other window sizes (app feeds 512; `conv12` and `probe12`):
`lama_manga` ran at 256x256, 384x512 and 640x640 but **rejected 500x500** (a Mul node, 500 against 504, so sizes need
to be a multiple of 8); `aot` ran at all four sizes including 500x500; `cv_lama`, `qc_lama`, `qc_aot` accept 512x512 only.

### Contact sheets

Eight images, +64 and +192 px, one side (right) and four sides, `outpaint()` unchanged, k=1, `lama` on CPU, `migan` on
DirectML, new models on DirectML except `cv_lama` (CPU): 32 sheets in `out/plan12/`. Images: the six of the first spike
plus two **stand-ins** for illustration from scikit-image's bundled data (no downloads): `logo_flat` (the scikit-image
logo on white, flat colour with fine line work) and `horse_silhouette` (black shape on white, inverted to a white horse on
black; 328x400). They are **not** anime or manga. Judged by eye on tiles downscaled to 400 px wide, so seams at 1:1 were
**not** inspected.

| Image / case | `lama_manga` | `cv_lama` | `aot` | `qc_lama` | `qc_aot` | (LaMa, for reference) |
|---|---|---|---|---|---|---|
| `rocket_sky` right +64 / +192 | clean; at +192 darkens to a flat dark blue | clean at +64, like LaMa | clean at +64; at +192 a translucent patch | small dark smear at +64 and a blotch at +192 | dirty smear at +64, white smear at +192 | clean, fades dark |
| `chelsea_cat` right +64 / +192 | +64 clean; +192 a grey-beige textured slab | +64 ok; +192 greenish grey slab | +64 ok; +192 pale smear | +64 ok; +192 slight blotch | +64 ok; +192 blocky | smooth pink slab |
| `coffee_wall` all four +192 | smooth orange blur frame, no objects | strong teal/grey colour frame, white halo | lavender-white frame, hard bands | **invents a dark human face** at the left, orange blob | garbled bands | grey-brown vignette, best of the set |
| `astronaut_cut` right +192 | wall fades to dark red-brown, no edge | wall goes green-grey | **hard black wall edge** | dark band | coloured streak strip | clean fade |
| `astronaut_cut` all four +192 | dark red-brown vignette, flag blurred, no people | green-grey cast | not viewed | not viewed | not viewed | blue-grey, flag blurred |
| `gravel_texture` right +64 / +192 | faint regular hatch pattern, not gravel | thin pale line at +64, dark green slab at +192 | dark band, then a white strip | blurry grey band | orange smear | flat grey slab, visible seam |
| `logo_flat` right +192 | stays white, faint warm tint at the far edge | dark grey-green cloud blob | stays white, **cleanest** | pale grey cloud | brown smear | stays white |
| `logo_flat` all four +192 | white, faint yellow patches | teal frame with a white halo | not viewed | not viewed | not viewed | white, faint haze |
| `horse_silhouette` all four +64 / +192 | black, no artefacts | dark green haze at +192 | black, clean (best here with `lama_manga`) | **dark human face** at +192, grey haze | white/colour garbage at +192 | black, one pink pixel |
| `screenshot_text` all four +192 | white top with pink cast, dark sides | dark teal frame | white top, black sides, grey smudge | **two faces** at the top-left | colour garbage | white top, black sides, blur |

Failure modes: `lama_manga` fails by smoothing and a warm cast (like LaMa, never inventing); `cv_lama` fails by colour
drift to teal; `aot` fails by sharp unrelated bands; `qc_lama` fails by inventing faces; `qc_aot` fails by garbage. Not every
cell above was viewed for all eight images: `astronaut_cut` all four was viewed for `lama_manga` and `cv_lama` only, `logo_flat`
all four for `lama_manga` and `cv_lama` only, and the remaining sheets (`rocket_sky` and `chelsea_cat` four sides,
`coffee_wall` right, `screenshot_text` right, `horse_silhouette` right) were generated but not looked at.

### Per model

**ADD `lama_manga`.** Rust side needs: file `lama-manga-dynamic.onnx` (206,291,843 bytes); inputs `image` float32
`[N,3,H,W]` RGB 0..1 and `mask` float32 `[N,1,H,W]` with **1.0 = hole**, 0.0 = known; output `inpainted` float32 `[N,3,H,W]`
RGB **0..1** (multiply by 255, clip, round; today's `Layout::LaMa` assumes 0..255, so it needs an output-scale field or a
third layout); dynamic spatial size, feed 512x512 (sizes must be multiples of 8); hole content ignored; known pixels
returned unchanged; **GPU-capable** (DirectML, 298 ms against 2195 ms on CPU, same output); keep the session alive (16 to
20 s load). Good for (user text): "Illustrations, logos and flat-colour images; also fast on the GPU." Honest limits: the
"illustration" claim rests on two stand-ins, no real anime or manga was tested; on photos it is about as good as LaMa but
carries a warm red-brown cast at +192 on four sides; textures (gravel) fail.

**SKIP `cv_lama`.** Same family and the same convention as the shipped LaMa (0..1 in, hole=1, output 0..255, fixed 512,
fits the existing `Layout::LaMa`), 93 MB against 208 MB, but CPU only (the same MatMul fails on DirectML), 2.9 s per
window, and a visible teal/green colour drift at +192 on `astronaut_cut`, `coffee_wall`, `chelsea_cat`, `gravel_texture`,
`logo_flat` and `horse_silhouette`. It does not differ usefully from LaMa except for being worse and smaller.

**SKIP `aot`** (owner may reverse for a "small and quick" slot). Convention if added: `image` -1..1 with the hole set
to 0.0 (mid grey), `mask` 1 = hole, output -1..1, known pixels not preserved (composite them back), dynamic size, GPU
capable (112 ms). It is clean on flat shapes and at +64, but at +192 on photos it produced pale smears, a hard black edge
and a white strip; provenance of the weights is unclear (MIT tag declared by the uploader). Not different enough in a
good way to earn a catalogue slot.

**SKIP `qc_lama`.** Float 0..1, hole=1, output 0..1, fixed 512, external `.data` file, GPU-capable (151 ms against 1335 ms
CPU, same result). It invented a recognisable human face in three separate images (`coffee_wall`, `horse_silhouette`,
`screenshot_text`, all at +192), which a canvas extender must not do. D7: the zip support is not justified.

**SKIP `qc_aot`.** Float 0..1, hole=1, output 0..1, fixed 512, external `.data` file, GPU-capable (149 ms). Output quality is
the worst of the set: coloured smears at the borders from +64 on `rocket_sky` and `gravel_texture`, garbage frames at
+192. D7: not justified.

### Not tested

Real anime or manga frames (only the two stand-ins above); seams at 1:1; the k > 1 context factor for any new model;
text on the frame edge; CUDA or other execution providers; fp16 variants; DirectML on any GPU other than the
GTX 1660 SUPER; non-Windows platforms; the Qualcomm licence and every model's weights and training-data terms (only the
uploaders' declarations are recorded); several sheets listed above as generated but not viewed; memory use.
