"""Export a PyTorch upscaling checkpoint to ONNX and validate it.

Usage: python -I tools/export_onnx.py <id> [--all] [--out models/cache/onnx]

ONNX contract: input `input` float32 [1,3,H,W] (dynamic H/W, RGB 0..1),
output `output` float32 [1,3,H*s,W*s]. Opset 17, fp32.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from spandrel import ModelLoader

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "models" / "cache" / "src"
RELEASE_URL = "https://github.com/inalto/scaleup/releases/download/models-v1/{id}.onnx"
TOL = 1e-3
SIZES = [(64, 64), (96, 80)]  # (H, W)


def test_input(h: int, w: int) -> torch.Tensor:
    """Smooth gradients plus mild noise, in 0..1.

    Pure uniform noise is out of distribution: some models (SPAN) amplify it to
    outputs of +-130, which makes an absolute-error threshold meaningless.
    """
    g = torch.Generator().manual_seed(h * 1000 + w)
    ys = torch.linspace(0, 1, h).view(1, 1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, 1, w)
    base = torch.cat([ys * xs, ys.expand(1, 1, h, w), xs.expand(1, 1, h, w)], dim=1)
    return (0.9 * base + 0.1 * torch.rand(1, 3, h, w, generator=g)).clamp(0, 1)


def max_diff(model: torch.nn.Module, onnx_path: Path, scale: int) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for h, w in SIZES:
        x = test_input(h, w)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(["output"], {"input": x.numpy()})[0]
        if out.shape != (1, 3, h * scale, w * scale):
            raise RuntimeError(f"bad output shape {out.shape} for input {h}x{w}")
        worst = max(worst, float(np.abs(ref - out).max()))
    return worst


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(entry: dict, out_dir: Path) -> dict:
    src = SRC_DIR / entry["file"]
    desc = ModelLoader().load_from_file(src)
    model = desc.model.eval().float().cpu()
    if desc.scale != entry["scale"]:
        raise RuntimeError(f"scale mismatch: checkpoint {desc.scale} vs sources.json {entry['scale']}")
    scale = desc.scale

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{entry['id']}.onnx"
    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H2", 3: "W2"}},
        opset_version=17,
        dynamo=False,
    )
    try:
        onnx.checker.check_model(onnx.load(str(path)))
        diff = max_diff(model, path, scale)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    try:
        import onnxsim

        simplified, ok = onnxsim.simplify(onnx.load(str(path)))
        if ok:
            tmp = path.with_suffix(".simp.onnx")
            onnx.save(simplified, str(tmp))
            try:
                sdiff = max_diff(model, tmp, scale)
                if sdiff < TOL:
                    tmp.replace(path)
                    diff = sdiff
                else:
                    tmp.unlink()
            except Exception:
                tmp.unlink(missing_ok=True)
    except Exception as e:
        print(f"note: onnxsim skipped for {entry['id']}: {e}", file=sys.stderr)

    if diff >= TOL:
        path.unlink()
        raise RuntimeError(f"maxdiff {diff:.2e} >= {TOL}")

    size = path.stat().st_size
    print(f"OK {entry['id']} scale={scale} maxdiff={diff:.2e} size={size}")
    return {
        "id": entry["id"],
        "arch": entry["arch"],
        "scale": scale,
        "license": entry["license"],
        "commercial": entry["commercial"],
        "size_bytes": size,
        "url": RELEASE_URL.format(id=entry["id"]),
        "sha256": sha256(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "models" / "cache" / "onnx"))
    args = ap.parse_args()
    sources = json.loads((ROOT / "tools" / "sources.json").read_text(encoding="utf-8"))
    if args.all:
        todo = sources
    elif args.id:
        todo = [s for s in sources if s["id"] == args.id]
        if not todo:
            print(f"unknown id {args.id}", file=sys.stderr)
            return 1
    else:
        ap.error("give an id or --all")

    entries, failed = [], 0
    for entry in todo:
        try:
            entries.append(export(entry, Path(args.out)))
        except Exception as e:
            failed += 1
            print(f"FAIL {entry['id']}: {e}", file=sys.stderr)
    print(json.dumps(entries, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
