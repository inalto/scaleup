# tools

Dev-time scripts (not shipped).

- `fetch_sources.py` — downloads checkpoints listed in `sources.json` into `models/cache/src/` and license texts into `models/licenses/`.
- `export_onnx.py` — converts a checkpoint to ONNX (opset 17, dynamic H/W), validates against PyTorch at 64x64 and 96x80, prints the manifest entry.
- `make_test_model.py` — generates the tiny test fixture.

Setup:

    python -m venv tools/.venv
    tools/.venv/Scripts/python -I -m pip install -r tools/requirements.txt
