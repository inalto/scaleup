"""Generate the PyTorch state-dict fixtures in tests/fixtures/sd/ (seeded, each < 200 KB).

- tiny_esrgan.pth: ESRGAN old-arch keys (nf=8, nb=2, grow channels 4, 3->3 channels, x4),
  wrapped in {"params_ema": ...}.
- tiny_compact.pth: SRVGGNetCompact (num_feat=8, num_conv=2, x2, PReLU), plain state dict.
- tiny_compact_legacy.pth: the same Compact state dict in the pre-1.6 (non-zip) format.
- tiny_views.pth: tensors that are views (transposed, offset slice) of one storage, f16/bf16/f64
  and int64 tensors and non-tensor values, to check strides, offsets and dtype conversions.
- tiny_span.pth: SPAN (feature_channels=4, x4, norm), saved before any forward pass, so its
  eval_conv tensors are stale, as in real checkpoints (eval mode recomputes them).
- tiny_realplksr_{gn_ea,ln,ln_ea}.pth: RealPLKSR (dim=8, n_blocks=2, x2, split_ratio 0.25) with
  group norm and attention (kernel 7), layer norm without attention (kernel 5), layer norm and
  attention (kernel 17). Every parameter is re-drawn, so no block is close to the identity.
- {name}_io.safetensors next to each of these: a seeded input (`input`, [3, 6, 7], 0..1) and the
  output (`output`) spandrel's module computes for it in eval mode.

Key names come from spandrel's ESRGAN, Compact, SPAN and RealPLKSR architectures.
Usage (from the workspace root): tools/.venv/Scripts/python.exe -I tools/make_sd_fixtures.py
"""

import re
from collections import OrderedDict
from pathlib import Path

import torch
from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact
from safetensors.torch import save_file
from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet
from spandrel.architectures.PLKSR.__arch.RealPLKSR import RealPLKSR
from spandrel.architectures.SPAN.__arch.span import SPAN

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sd"
GROW = 4  # spandrel builds gc=32; 4 keeps the file small, shapes follow the same rule


def esrgan_state_dict() -> OrderedDict:
    nf = 8
    keys = RRDBNet(in_nc=3, out_nc=3, num_filters=nf, num_blocks=2, scale=4).state_dict()
    sd = OrderedDict()
    for key, t in keys.items():
        m = re.search(r"RDB\d\.conv(\d)\.0\.(weight|bias)$", key)
        if m:
            k = int(m.group(1))
            out_ch = GROW if k < 5 else nf
            shape = (out_ch, nf + (k - 1) * GROW, 3, 3) if m.group(2) == "weight" else (out_ch,)
        else:
            shape = tuple(t.shape)
        sd[key] = torch.randn(shape) * 0.1
    return sd


def write_with_io(model: torch.nn.Module, name: str, seed: int) -> None:
    """Saves `model`'s state dict as `{name}.pth`, then runs the model in eval mode on a seeded
    input and saves both as `{name}_io.safetensors`."""
    torch.save({k: v.clone() for k, v in model.state_dict().items()}, OUT / f"{name}.pth")
    torch.manual_seed(seed)
    x = torch.rand(1, 3, 6, 7)
    model.eval()
    with torch.no_grad():
        y = model(x)
    io = {"input": x[0].contiguous(), "output": y[0].contiguous()}
    save_file(io, OUT / f"{name}_io.safetensors")


def span_model() -> SPAN:
    torch.manual_seed(1)
    model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=4, upscale=4)
    # The input is scaled by 255: keep the output near 1 so an absolute tolerance means something.
    with torch.no_grad():
        model.upsampler[0].weight.mul_(0.05)
        model.upsampler[0].bias.mul_(0.05)
    return model


def realplksr_model(seed: int, kernel: int, use_ea: bool, layer_norm: bool) -> RealPLKSR:
    torch.manual_seed(seed)
    model = RealPLKSR(
        dim=8,
        n_blocks=2,
        upscaling_factor=2,
        kernel_size=kernel,
        split_ratio=0.25,
        use_ea=use_ea,
        layer_norm=layer_norm,
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            base = 1.0 if name.endswith("norm.weight") else 0.0
            p.copy_(base + 0.2 * torch.randn_like(p))
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save({"params_ema": esrgan_state_dict()}, OUT / "tiny_esrgan.pth")

    compact = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    torch.save(compact.state_dict(), OUT / "tiny_compact.pth")
    torch.save(
        compact.state_dict(),
        OUT / "tiny_compact_legacy.pth",
        _use_new_zipfile_serialization=False,
    )

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    views = {
        "transposed": base.t(),  # stride (1, 6)
        "slice": base[1:, 2:4],  # offset 8, stride (6, 1)
        "half": torch.tensor([0.5, -2.0, 65504.0, 6.1035156e-05], dtype=torch.float16),
        "bf16": torch.tensor([1.0, -3.5, 2.0**100], dtype=torch.bfloat16),
        "f64": torch.tensor([0.25, -1.0], dtype=torch.float64),
        "param": torch.nn.Parameter(torch.tensor([7.0, 8.0])),
        "num_batches_tracked": torch.tensor(1234),  # 0-dim int64, as BatchNorm stores it
        "iter": 1000,
        "name": "views",
        "meta": {"lr": 0.5, "flags": [True, None]},
    }
    torch.save(views, OUT / "tiny_views.pth")

    write_with_io(span_model(), "tiny_span", seed=2)
    write_with_io(realplksr_model(3, 7, True, False), "tiny_realplksr_gn_ea", seed=4)
    write_with_io(realplksr_model(5, 5, False, True), "tiny_realplksr_ln", seed=6)
    write_with_io(realplksr_model(7, 17, True, True), "tiny_realplksr_ln_ea", seed=8)


if __name__ == "__main__":
    main()
