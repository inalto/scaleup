# Upscayl bundled models: provenance and redistribution

Research notes, 2026-11-03. Sources: `upscayl/upscayl` (AGPL-3.0, `resources/models`, README credits), `upscayl/custom-models` (no LICENSE file; GitHub reports no license; README credits only), OpenModelDB pages. None of these models is added to `models/manifest.json` (plan 07 runs the original ncnn files straight from the user's Upscayl install).

**Repository licenses.** The AGPL-3.0 of `upscayl/upscayl` covers Upscayl's code. It does not relicense third-party model weights, and the repo ships `Real-ESRGAN_LICENSE.txt` (BSD-3-Clause, Xintao Wang) next to them. `upscayl/custom-models` has no license at all, so nothing in it can be assumed redistributable beyond what the original authors' licenses allow.

| Upscayl model | Upstream checkpoint (as far as identifiable) | License | Redistribute ONNX export? |
|---|---|---|---|
| `upscayl-standard-4x` | Real-ESRGAN `RealESRGAN_x4plus` (README says Upscayl uses Real-ESRGAN; no per-file statement) | BSD-3-Clause | Yes, if confirmed as an unmodified conversion; we already ship `realesrgan-x4plus` from the original, which makes this redundant. |
| `upscayl-lite-4x` | Real-ESRGAN general v3 family (`realesr-general-x4v3`), not stated explicitly | BSD-3-Clause | Same as above; redundant. |
| `high-fidelity-4x` | 4x-HFA2k (README credit "Helaman ... HFA2k") | OpenModelDB lists CC-BY-4.0 for the HFA2k entries; the page also lists unrelated CC-BY-NC-SA entries, so confirm on the model's own page before use | Likely yes with attribution, **unverified**; do not ship until confirmed. |
| `digital-art-4x` | Not stated; presumably a Real-ESRGAN anime model (`RealESRGAN_x4plus_anime_6B`) | BSD-3-Clause if so | Unverified provenance; skip, we ship the original anime model. |
| `ultramix-balanced-4x` | Kim2091 `4x-UltraMix_Balanced` (README credit) | Kim2091 UltraSharp family is CC-BY-NC-SA-4.0 | No for a commercial catalogue. Non-commercial only, ShareAlike applies to derivatives (an ONNX conversion is one). |
| `ultrasharp-4x` | Kim2091 `4x-UltraSharp` | CC-BY-NC-SA-4.0 | Non-commercial only (already in sources.json as `commercial:false`). |
| `remacri-4x` | Foolhardy `4x_foolhardy_Remacri` | CC-BY-NC-SA-4.0 | Non-commercial only (already in sources.json as `commercial:false`). |

**Custom-models repo** (not bundled with Upscayl; user-installable): NMKD Siax / Superscale (NMKD's own terms, not verified), Kim2091 `uniscale_restore` (not verified), Phhofm models `4xLSDIR`, `4xLSDIRplusC`, `4xLSDIRCompactC3`, `4xNomos8kSC`, `4xHFA2k` (Phhofm publishes these under CC-BY-4.0 on OpenModelDB/HF; verify per model), `RealESRGAN_General_*` and `realesr-animevideov3-x2/x3/x4` (BSD-3-Clause).

**Conclusion.**
- The ncnn `.bin/.param` files are conversions of the upstream weights and inherit the upstream license; Upscayl's AGPL does not apply to them.
- Commercial-safe and worth exporting from the *original* checkpoints (done): Real-ESRGAN models and the Phips CC-BY-4.0 models.
- Never redistribute the CC-BY-NC-SA conversions in a catalogue marked commercial. They stay behind the `noncommercial` tag with a visible warning.
- Where the Upscayl files themselves are used (plan 07), read them from the user's installation; do not copy them into ScaleUP's release.
- Per-file upstream checkpoints for `upscayl-standard-4x`, `high-fidelity-4x`, `digital-art-4x` and `ultramix-balanced-4x` could not be confirmed from the Upscayl repo; their exact lineage is an open question for the maintainers.
