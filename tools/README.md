# tools

Dev-time scripts (not shipped).

- `fetch_sources.py` — downloads checkpoints listed in `sources.json` into `models/cache/src/` and license texts into `models/licenses/`.
- `export_onnx.py` — converts a checkpoint to ONNX (opset 17, dynamic H/W), validates against PyTorch at 64x64 and 96x80, prints the manifest entry.
- `make_test_model.py` — generates the tiny test fixture.

Setup:

    python -m venv tools/.venv
    tools/.venv/Scripts/python -I -m pip install -r tools/requirements.txt

## Export status (v1)

All models in `sources.json` export and validate (max abs diff < 1e-3 at 64x64 and 96x80) except:

- `4xRealWebPhoto_v4_dat2` (DAT-2): the exported graph bakes in the traced window/reshape sizes and fails at 96x80
  (`Reshape` error inside `layers.0/blocks.2/attn`). Excluded from v1; the manifest entry is removed.

Validation input is a smooth gradient plus mild noise, not uniform noise: SPAN amplifies uniform noise to outputs of +-130,
which makes an absolute-error threshold meaningless (mean error stayed ~1e-5).
