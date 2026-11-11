"""Generate the tiny ONNX fixtures the Rust tests run.

- tests/fixtures/nearest2x.onnx: a 2x nearest-neighbour "super-resolution" model. A single
  opset-18 Resize (mode=nearest, nearest_mode=floor, scales [1,1,2,2]) with dynamic H/W, matching
  the ScaleUP ONNX contract (input `input` [1,3,H,W] float32, output `output` [1,3,2H,2W] float32).
  Deterministic, so tests can compare ORT output with the pure-Rust NearestModel exactly.
- tests/fixtures/inpaint_mean_lama.onnx and inpaint_mean_migan.onnx: "inpainters" with the input
  and output signature of LaMa and MI-GAN (docs/OUTPAINT_SPIKE.md section 2) that paint every hole
  pixel with the per-channel mean of the known pixels, like the Rust MeanInpainter.
  LaMa: `image` float32 [1,3,16,16] in 0..1, `mask` float32 [1,1,16,16] with 1 = hole, output
  `output` float32 [1,3,16,16] in 0..255 (the real graph is fixed 512x512; the fixture is fixed
  16x16 so the adapter must read the window from the graph). MI-GAN: `image` uint8
  [batch_size,3,height,width] in 0..255, `mask` uint8 [batch_size,1,height,width] with
  255 = known, output `result` uint8 (dynamic size, like the real graph).

Usage (from the workspace root): python -I tools/make_test_model.py
Requires: pip install onnx
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def save(model: onnx.ModelProto, name: str) -> None:
    model.ir_version = 8
    onnx.checker.check_model(model)
    out = FIXTURES / name
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def opset18(graph: onnx.GraphProto) -> onnx.ModelProto:
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="scaleup"
    )


def nearest2x() -> None:
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
    save(opset18(graph), "nearest2x.onnx")


def mean_fill_nodes(img: str, known: str, out: str) -> list:
    """`out = img*known + mean*(1-known)`, `mean` the per-channel mean of `img` where `known` is 1.

    Both inputs float32; `known` is [N,1,H,W] of 0/1. With no known pixel the mean is 0."""
    n = helper.make_node
    return [
        n("Mul", [img, known], ["kept"]),
        n("ReduceSum", ["kept", "hw"], ["sum"], keepdims=1),
        n("ReduceSum", [known, "hw"], ["count"], keepdims=1),
        n("Max", ["count", "one"], ["count1"]),
        n("Div", ["sum", "count1"], ["mean"]),
        n("Sub", ["one", known], ["hole"]),
        n("Mul", ["mean", "hole"], ["painted"]),
        n("Add", ["kept", "painted"], [out]),
    ]


def constants(*extra: str) -> list:
    """`hw` (the spatial axes) and `one`, plus the scalars named in `extra`. Only what the graph
    uses, so ONNX Runtime has no unused initializer to warn about."""
    scalars = {"one": 1.0, "c255": 255.0, "zero": 0.0}
    return [helper.make_tensor("hw", TensorProto.INT64, [2], [2, 3])] + [
        helper.make_tensor(k, TensorProto.FLOAT, [], [scalars[k]]) for k in ("one", *extra)
    ]


def inpaint_mean_lama() -> None:
    n = helper.make_node
    side = 16
    nodes = [
        n("Sub", ["one", "mask"], ["known"]),
        *mean_fill_nodes("image", "known", "filled"),
        n("Mul", ["filled", "c255"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "inpaint_mean_lama",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, side, side]),
            helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, side, side]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, side, side])],
        initializer=constants("c255"),
    )
    save(opset18(graph), "inpaint_mean_lama.onnx")


def inpaint_mean_migan() -> None:
    n = helper.make_node
    nodes = [
        n("Cast", ["image"], ["imagef"], to=TensorProto.FLOAT),
        n("Cast", ["mask"], ["maskf"], to=TensorProto.FLOAT),
        n("Div", ["maskf", "c255"], ["known"]),
        *mean_fill_nodes("imagef", "known", "filled"),
        n("Round", ["filled"], ["rounded"]),
        n("Clip", ["rounded", "zero", "c255"], ["clipped"]),
        n("Cast", ["clipped"], ["result"], to=TensorProto.UINT8),
    ]
    dims = ["batch_size", None, "height", "width"]
    shape = lambda c: [d if d is not None else c for d in dims]  # noqa: E731
    graph = helper.make_graph(
        nodes,
        "inpaint_mean_migan",
        [
            helper.make_tensor_value_info("image", TensorProto.UINT8, shape(3)),
            helper.make_tensor_value_info("mask", TensorProto.UINT8, shape(1)),
        ],
        [helper.make_tensor_value_info("result", TensorProto.UINT8, ["N", 3, "H", "W"])],
        initializer=constants("c255", "zero"),
    )
    save(opset18(graph), "inpaint_mean_migan.onnx")


def main() -> None:
    nearest2x()
    inpaint_mean_lama()
    inpaint_mean_migan()


if __name__ == "__main__":
    main()
