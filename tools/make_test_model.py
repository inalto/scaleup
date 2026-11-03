"""Generate tests/fixtures/nearest2x.onnx: a 2x nearest-neighbour "super-resolution" model.

The graph is a single opset-18 Resize (mode=nearest, nearest_mode=floor, scales [1,1,2,2])
with dynamic H/W, matching the ScaleUP ONNX contract (input `input` [1,3,H,W] float32,
output `output` [1,3,2H,2W] float32). It is deterministic, so tests can compare ORT
output with the pure-Rust NearestModel exactly.

Usage (from the workspace root): python -I tools/make_test_model.py
Requires: pip install onnx
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "nearest2x.onnx"


def main() -> None:
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    node = helper.make_node(
        "Resize",
        inputs=["input", "", "scales"],
        outputs=["output"],
        mode="nearest",
        nearest_mode="floor",
        coordinate_transformation_mode="asymmetric",
    )
    graph = helper.make_graph(
        [node],
        "nearest2x",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "H", "W"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H2", "W2"])],
        initializer=[scales],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="scaleup"
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
