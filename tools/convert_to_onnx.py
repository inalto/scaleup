"""Convert one PyTorch upscaling checkpoint (.pth or .safetensors) to ONNX.

Usage: python convert_to_onnx.py <input file> <output file>

Self-contained: ScaleUP embeds this file and runs it from a temporary folder, so it cannot import
tools/export_onnx.py. Same ONNX contract as that exporter: input `input` float32 [1,3,H,W] (dynamic
H/W, RGB 0..1), output `output`, opset 17, fp32. Exits non-zero with a one-line reason on failure.
"""
import sys
from pathlib import Path


def fail(reason: str) -> int:
    print(" ".join(reason.split()), file=sys.stderr)
    return 1


def main() -> int:
    if len(sys.argv) != 3:
        return fail("usage: convert_to_onnx.py <input file> <output file>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        import onnx
        import torch
        from spandrel import ModelLoader
    except ImportError as e:
        return fail(f"missing Python module ({e}); pip install spandrel torch onnx")

    try:
        desc = ModelLoader().load_from_file(str(src))
    except Exception as e:
        return fail(f"{src.name}: not a checkpoint spandrel can load: {e}")
    if desc.input_channels != 3 or desc.output_channels != 3:
        return fail(f"{src.name}: only RGB models are supported "
                    f"({desc.input_channels} in, {desc.output_channels} out)")
    model = desc.model.eval().float().cpu()

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.rand(1, 3, 64, 64),
                str(tmp),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H2", 3: "W2"}},
                opset_version=17,
                dynamo=False,
            )
        onnx.checker.check_model(onnx.load(str(tmp)))
        tmp.replace(dst)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        arch = getattr(desc.architecture, "name", "unknown architecture")
        return fail(f"{src.name}: ONNX export failed ({arch}): {e}")
    print(f"OK {dst} scale={desc.scale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
