#!/usr/bin/env python
"""Throwaway feasibility spike for the canvas-extension (outpainting) feature.

Not shipped. It exists to produce the measurements in docs/OUTPAINT_SPIKE.md:
  probe   tensor layouts, value ranges, mask polarity, channel order (by experiment)
  timing  one 512 window on CPU and on the DirectML provider
  sheets  contact sheets: image x margin (right / all four sides) x model
  kexp    context-factor experiment on a ~3000 px photo
  all     probe + timing + sheets + kexp
  probe12   plan 12 candidates: sha256, opset, signatures, convention search, CPU/DML timing, sizes
  sheets12  plan 12 contact sheets (8 images x +64/+192 x LaMa, MI-GAN and the new models)

Setup (scratch folder outside the repo, default F:/Apps/ScaleUP/outpaint-spike):
  python -m venv venv; venv/Scripts/pip install onnxruntime-directml numpy pillow scikit-image
  put migan.onnx and lama_fp32.onnx next to the venv (see docs/OUTPAINT_SPIKE.md for sources).
Run:  venv/Scripts/python tools/outpaint_spike.py all

The window walk (Global Constraints of the plan):
  * the model sees a fixed 512x512 window; a window covers S = 512*k source pixels and is
    resampled to 512 (k = 1 means no resampling)
  * a window runs only if at least half of it is known pixels
  * one ring adds at most S/2 of depth; windows along a side advance by S/2 (half overlap)
  * sides are filled before corners, so a corner window sees three known quadrants
  * only unknown pixels are written; known pixels stay bit-exact; the earlier window wins
  * the canvas starts from a mirror-reflected prefill; pixels outside the canvas (window larger
    than the canvas) are reflect-padded context and count as known
"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPIKE = os.environ.get("SPIKE_DIR", "F:/Apps/ScaleUP/outpaint-spike")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SPIKE, "out")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "plan12"), exist_ok=True)

W = 512  # model window


# --------------------------------------------------------------------------- models
def make_session(model, provider):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider"}[provider]
    return ort.InferenceSession(os.path.join(SPIKE, model + ".onnx"), so, providers=[prov])


class Model:
    """known: HxW bool (True = known pixel). img: HxWx3 uint8 RGB. Returns HxWx3 uint8."""

    def __init__(self, name, provider="cpu"):
        self.name = name
        self.provider = provider
        self.sess = make_session({"migan": "migan", "lama": "lama_fp32"}[name], provider)

    def __call__(self, img, known):
        if self.name == "migan":
            # image uint8 NCHW RGB 0..255, mask uint8 N1HW with 255 = known, 0 = hole
            x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
            m = (known.astype(np.uint8) * 255)[None, None]
            o = self.sess.run(None, {"image": x, "mask": m})[0][0]
            return np.ascontiguousarray(o.transpose(1, 2, 0))
        # lama: image float32 NCHW RGB 0..1, mask float32 N1HW with 1 = hole; output float 0..255
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        m = (~known).astype(np.float32)[None, None]
        o = self.sess.run(None, {"image": np.ascontiguousarray(x), "mask": m})[0][0]
        return np.clip(np.rint(o.transpose(1, 2, 0)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- plan 12 candidates
# name -> (path under SPIKE, conventions found by probe12). Conventions are filled from the probe
# result (docs/OUTPAINT_SPIKE.md, "Plan 12 candidates"); inp/out are (lo, hi) value ranges.
P12 = {
    "lama_manga": dict(path="models/lama_manga/lama-manga-dynamic.onnx", inp=(0, 1), hole1=True, out=(0, 1), prov="dml"),
    # aot: the hole content matters (it is not composited or ignored), so the hole is set to mid grey
    "aot": dict(path="models/aot_ogkalu/aot.onnx", inp=(-1, 1), hole1=True, out=(-1, 1), fill=128, prov="dml"),
    "cv_lama": dict(path="models/opencv_lama/inpainting_lama_2025jan.onnx", inp=(0, 1), hole1=True, out=(0, 255), prov="cpu"),
    "qc_lama": dict(path="models/qc_lama/lama_dilated-onnx-float/lama_dilated_onnx/lama_dilated.onnx", inp=(0, 1), hole1=True, out=(0, 1), prov="dml"),
    "qc_aot": dict(path="models/qc_aotgan/aotgan-onnx-float/aotgan_onnx/aotgan.onnx", inp=(0, 1), hole1=True, out=(0, 1), prov="dml"),
}


def to255(o, rng):
    lo, hi = rng
    return np.clip(np.rint((o - lo) / (hi - lo) * 255.0), 0, 255)


class Gen:
    """A float32 NCHW image + mask model with the conventions in P12 (or given). known: HxW bool."""

    def __init__(self, name, provider="cpu", conv=None, share=None):
        import onnxruntime as ort

        self.name, self.provider = name, provider
        c = dict(P12[name])
        c.update(conv or {})
        self.c = c
        if share is not None:  # reuse an already loaded session (a convention only changes how tensors are packed)
            self.sess = share.sess
        else:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            prov = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider"}[provider]
            self.sess = ort.InferenceSession(os.path.join(SPIKE, c["path"]), so, providers=[prov])
        ins = self.sess.get_inputs()
        self.n_img, self.n_mask = ins[0].name, ins[1].name

    def raw(self, img, known, fill=None):
        lo, hi = self.c["inp"]
        x = img.astype(np.float32) / 255.0 * (hi - lo) + lo
        if fill is not None:  # fill is a 0..255 grey for hole pixels, None = leave the prefill
            x = np.where(known[..., None], x, np.float32(fill / 255.0 * (hi - lo) + lo))
        m = (known if not self.c["hole1"] else ~known).astype(np.float32)[None, None]
        xx = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
        return self.sess.run(None, {self.n_img: xx, self.n_mask: m})[0][0].transpose(1, 2, 0)

    def __call__(self, img, known):
        return to255(self.raw(img, known, self.c.get("fill")), self.c["out"]).astype(np.uint8)


# --------------------------------------------------------------------------- window walk
def reflect_idx(i, n):
    if n == 1:
        return np.zeros_like(i)
    p = 2 * (n - 1)
    i = np.mod(i, p)
    return np.where(i > n - 1, p - i, i)


def outpaint(src, margins, model, k=1, stats=None):
    """margins = (left, top, right, bottom) in source pixels. Returns the extended RGB canvas."""
    l, t, r, b = margins
    h, w = src.shape[:2]
    canvas = np.pad(src, ((t, b), (l, r), (0, 0)), mode="reflect")  # mirror prefill
    hc, wc = canvas.shape[:2]
    known = np.zeros((hc, wc), bool)
    known[t : t + h, l : l + w] = True
    S = W * k
    D = S // 2
    x0, y0, x1, y1 = l, t, l + w, t + h
    rem = {"l": l, "t": t, "r": r, "b": b}

    def window(wx, wy):
        iy = reflect_idx(np.arange(wy, wy + S), hc)
        ix = reflect_idx(np.arange(wx, wx + S), wc)
        inside_y = (np.arange(wy, wy + S) >= 0) & (np.arange(wy, wy + S) < hc)
        inside_x = (np.arange(wx, wx + S) >= 0) & (np.arange(wx, wx + S) < wc)
        img = canvas[np.ix_(iy, ix)]
        kn = known[np.ix_(iy, ix)] | ~(inside_y[:, None] & inside_x[None, :])
        return img, kn, iy, ix, inside_y, inside_x

    def run(wx, wy):
        img, kn, iy, ix, inside_y, inside_x = window(wx, wy)
        frac = kn.mean()
        if stats is not None:
            stats.setdefault("min_known_frac", 1.0)
            stats["min_known_frac"] = min(stats["min_known_frac"], float(frac))
            stats["windows"] = stats.get("windows", 0) + 1
        if frac < 0.5 - 1e-9:
            # Finding: when the source side along a strip is shorter than the window, no
            # placement can reach half known (max = side / (2*S)). Skipping leaves holes, so the
            # window runs anyway; the count is reported.
            if stats is not None:
                stats["relaxed"] = stats.get("relaxed", 0) + 1
        unk = ~kn
        if not unk.any():
            return
        if k > 1:
            im512 = np.asarray(Image.fromarray(img).resize((W, W), Image.LANCZOS))
            kn512 = np.asarray(Image.fromarray(kn.astype(np.uint8) * 255).resize((W, W), Image.BOX)) >= 254
        else:
            im512, kn512 = img, kn
        out = model(im512, kn512)
        if k > 1:
            out = np.asarray(Image.fromarray(out).resize((S, S), Image.BICUBIC))
        # write only unknown pixels that lie inside the canvas; earlier window wins because
        # written pixels become known
        sel = unk & inside_y[:, None] & inside_x[None, :]
        ys, xs = np.nonzero(sel)
        canvas[iy[ys], ix[xs]] = out[ys, xs]
        known[iy[ys], ix[xs]] = True

    def starts(lo, hi):
        """window start positions tiling the strip [lo, hi) with step S/2."""
        n = hi - lo
        if n <= S:
            return [lo + n // 2 - S // 2]
        s = list(range(lo, hi - S, S // 2)) + [hi - S]
        return s

    while any(v > 0 for v in rem.values()):
        d = {s: min(D, v) for s, v in rem.items()}
        # sides first
        if d["r"]:
            for wy in starts(y0, y1):
                run(x1 + d["r"] - S, wy)
        if d["l"]:
            for wy in starts(y0, y1):
                run(x0 - d["l"], wy)
        if d["t"]:
            for wx in starts(x0, x1):
                run(wx, y0 - d["t"])
        if d["b"]:
            for wx in starts(x0, x1):
                run(wx, y1 + d["b"] - S)
        # then corners
        for sx, sy in (("r", "t"), ("l", "t"), ("r", "b"), ("l", "b")):
            if d[sx] and d[sy]:
                wx = x1 + d["r"] - S if sx == "r" else x0 - d["l"]
                wy = y1 + d["b"] - S if sy == "b" else y0 - d["t"]
                run(wx, wy)
        x1 += d["r"]
        x0 -= d["l"]
        y1 += d["b"]
        y0 -= d["t"]
        for s in rem:
            rem[s] -= d[s]
    assert known.all(), "unfilled pixels remain"
    assert np.array_equal(canvas[t : t + h, l : l + w], src), "known pixels changed"
    return canvas


# --------------------------------------------------------------------------- test images
def load_images():
    import skimage.data as sd

    def rgb(a):
        a = np.asarray(a)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        return np.ascontiguousarray(a[..., :3].astype(np.uint8))

    imgs = {}
    imgs["rocket_sky"] = rgb(sd.rocket())  # photo, sky and ground borders
    imgs["coffee_wall"] = rgb(sd.coffee())  # photo, soft wall/table borders
    imgs["chelsea_cat"] = rgb(sd.chelsea())  # photo, extra
    imgs["gravel_texture"] = rgb(sd.gravel())  # busy texture
    imgs["astronaut_cut"] = rgb(sd.astronaut())  # portrait, subject cut at the bottom and left
    shot = os.path.join(REPO, "docs", "screenshots", "idle.png")
    imgs["screenshot_text"] = rgb(Image.open(shot).convert("RGB"))
    # NOT covered: anime frame (no offline source in the allowed set)
    return imgs


# --------------------------------------------------------------------------- probe
def probe():
    print("== graph signatures")
    for name, f in (("migan", "migan"), ("lama", "lama_fp32")):
        s = make_session(f, "cpu")
        for i in s.get_inputs():
            print(name, "input ", i.name, i.type, i.shape)
        for o in s.get_outputs():
            print(name, "output", o.name, o.type, o.shape)

    img = np.zeros((W, W, 3), np.uint8)
    img[:, :256] = (255, 0, 0)  # known half: pure red in RGB order
    img[:, 256:] = (0, 255, 0)  # hole half: green garbage
    hole = np.zeros((W, W), bool)
    hole[:, 256:] = True

    def mean(o, sl):
        return np.round(o[:, sl].reshape(-1, 3).astype(np.float64).mean(0), 1).tolist()

    print("== MI-GAN polarity / channel order (known half red, hole half green garbage)")
    s = make_session("migan", "cpu")
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
    for label, kv in (("known=255 hole=0", 255), ("known=0 hole=255", 0)):
        m = np.where(hole, 255 - kv, kv).astype(np.uint8)[None, None]
        for order, xi in (("RGB", x), ("BGR", np.ascontiguousarray(x[:, ::-1]))):
            o = s.run(None, {"image": xi, "mask": m})[0][0].transpose(1, 2, 0)
            print("migan", label, "fed", order, "| left", mean(o, slice(0, 256)), "right", mean(o, slice(256, 512)))
    # is the hole content ignored?  Same known half, hole filled with two different garbages
    m = np.where(hole, 0, 255).astype(np.uint8)[None, None]
    a = img.copy()
    b = img.copy()
    b[:, 256:] = (0, 0, 255)
    oa = s.run(None, {"image": np.ascontiguousarray(a.transpose(2, 0, 1)[None]), "mask": m})[0]
    ob = s.run(None, {"image": np.ascontiguousarray(b.transpose(2, 0, 1)[None]), "mask": m})[0]
    print("migan hole-content sensitivity (max abs diff of outputs):", int(np.abs(oa.astype(int) - ob.astype(int)).max()))
    m1 = np.where(hole, 0, 1).astype(np.uint8)[None, None]
    o1 = s.run(None, {"image": np.ascontiguousarray(a.transpose(2, 0, 1)[None]), "mask": m1})[0]
    print("migan mask 1 instead of 255, mean over hole:", np.round(o1[0].transpose(1, 2, 0)[:, 256:].reshape(-1, 3).astype(np.float64).mean(0), 1).tolist())
    # natural image: is BGR vs RGB feeding distinguishable?
    nat = load_images()["chelsea_cat"][:W, :W]
    nat = np.asarray(Image.fromarray(nat).resize((W, W)))
    m = np.where(hole, 0, 255).astype(np.uint8)[None, None]
    xr = np.ascontiguousarray(nat.transpose(2, 0, 1)[None])
    orgb = s.run(None, {"image": xr, "mask": m})[0].astype(int)
    obgr = s.run(None, {"image": np.ascontiguousarray(xr[:, ::-1]), "mask": m})[0][:, ::-1].astype(int)
    print("migan natural image: mean abs diff between RGB-fed and BGR-fed (channels realigned), hole only:",
          round(float(np.abs(orgb - obgr)[..., 256:].mean()), 2))
    print("migan output dtype/range on natural image:", ob.dtype, int(orgb.min()), int(orgb.max()))

    print("== LaMa range / polarity (known half red, hole half green garbage)")
    s = make_session("lama_fp32", "cpu")
    fimg = (img.astype(np.float32) / 255.0)
    for rng, (lo, hi) in (("0..1", (0, 1)), ("-1..1", (-1, 1)), ("0..255", (0, 255))):
        xx = (fimg * (hi - lo) + lo).transpose(2, 0, 1)[None].astype(np.float32)
        for pol, m in (("hole=1", hole.astype(np.float32)), ("hole=0", (~hole).astype(np.float32))):
            o = s.run(None, {"image": xx, "mask": m[None, None]})[0][0].transpose(1, 2, 0)
            print("lama input", rng, pol, "| left", mean(o, slice(0, 256)), "right", mean(o, slice(256, 512)),
                  "out range", round(float(o.min()), 1), round(float(o.max()), 1))
    xx = fimg.transpose(2, 0, 1)[None]
    mm = hole.astype(np.float32)[None, None]
    oa = s.run(None, {"image": xx, "mask": mm})[0]
    fb = fimg.copy()
    fb[:, 256:] = (0, 0, 1)
    ob = s.run(None, {"image": fb.transpose(2, 0, 1)[None], "mask": mm})[0]
    print("lama hole-content sensitivity (max abs diff of outputs, 0..255 scale):", round(float(np.abs(oa - ob).max()), 3))
    kn = oa[0][:, :, :256]
    print("lama known half preserved exactly after rounding:", bool(np.all(np.rint(kn[0]) == 255) and np.all(np.rint(kn[1:]) == 0)))
    orgb = s.run(None, {"image": nat.astype(np.float32).transpose(2, 0, 1)[None] / 255.0, "mask": mm})[0]
    obgr = s.run(None, {"image": (nat[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0).copy(), "mask": mm})[0][:, ::-1]
    print("lama natural image: mean abs diff RGB-fed vs BGR-fed (realigned), hole only:",
          round(float(np.abs(orgb - obgr)[..., 256:].mean()), 2))

    # ground-truth channel-order check: hide the right half of real photos, compare to truth
    for mname in ("migan", "lama"):
        sess = make_session("migan" if mname == "migan" else "lama_fp32", "cpu")
        errs = {"RGB": [], "BGR": []}
        for key in ("rocket_sky", "coffee_wall", "chelsea_cat", "astronaut_cut"):
            g = np.asarray(Image.fromarray(load_images()[key]).resize((W, W), Image.LANCZOS))
            for order in ("RGB", "BGR"):
                gi = g if order == "RGB" else g[..., ::-1]
                if mname == "migan":
                    o = sess.run(None, {"image": np.ascontiguousarray(gi.transpose(2, 0, 1)[None]),
                                        "mask": np.where(hole, 0, 255).astype(np.uint8)[None, None]})[0][0]
                else:
                    o = sess.run(None, {"image": np.ascontiguousarray(gi.astype(np.float32).transpose(2, 0, 1)[None] / 255.0),
                                        "mask": hole.astype(np.float32)[None, None]})[0][0]
                o = o.transpose(1, 2, 0).astype(np.float32)
                errs[order].append(float(np.abs(o[:, 256:] - gi[:, 256:].astype(np.float32)).mean()))
        print(f"{mname} ground-truth channel order (mean abs error vs original on hidden half, 4 photos): "
              f"fed RGB {np.mean(errs['RGB']):.1f}  fed BGR {np.mean(errs['BGR']):.1f}")

    print("== accepted spatial sizes")
    sm = make_session("migan", "cpu")
    for hh, ww in ((512, 512), (256, 384), (500, 500), (1024, 1024)):
        try:
            r = sm.run(None, {"image": np.zeros((1, 3, hh, ww), np.uint8), "mask": np.full((1, 1, hh, ww), 255, np.uint8)})[0]
            print("migan", hh, ww, "ok", r.shape)
        except Exception as e:
            print("migan", hh, ww, "FAIL", str(e)[:100])
    sl = make_session("lama_fp32", "cpu")
    try:
        sl.run(None, {"image": np.zeros((1, 3, 256, 256), np.float32), "mask": np.zeros((1, 1, 256, 256), np.float32)})
        print("lama 256x256 ok")
    except Exception as e:
        print("lama 256x256 REJECTED", str(e).splitlines()[0][:100])


# --------------------------------------------------------------------------- plan 12 probe
def sha256_of(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gt_case():
    """A real photo (astronaut, 512x512) with a hidden right strip and a hidden interior square:
    the hidden pixels have a ground truth, so every convention can be scored against it."""
    import skimage.data as sd

    g = np.ascontiguousarray(sd.astronaut()[..., :3])
    hole = np.zeros((W, W), bool)
    hole[:, 320:] = True
    hole[100:230, 60:190] = True
    return g, ~hole


def probe12(only=None, quick=False):
    import onnx
    import onnxruntime as ort

    g, known = gt_case()
    garb = np.where(known[..., None], g, np.uint8([0, 255, 0]))  # hole = green garbage, never the truth
    print("providers available:", ort.get_available_providers(), "onnxruntime", ort.__version__)
    for name, c in P12.items():
        if only and name not in only:
            continue
        path = os.path.join(SPIKE, c["path"])
        print(f"\n===== {name}  {c['path']}")
        print("bytes", os.path.getsize(path), "sha256", sha256_of(path))
        data = os.path.splitext(path)[0] + ".data"
        if os.path.exists(data):
            print("external data", os.path.basename(data), os.path.getsize(data), sha256_of(data))
        m = onnx.load(path, load_external_data=False)
        print("opset", [(o.domain or "ai.onnx", o.version) for o in m.opset_import], "producer", m.producer_name, m.producer_version)
        t = time.time()
        gm = Gen(name, "cpu")
        print(f"cpu session create {time.time() - t:.2f}s")
        for i in gm.sess.get_inputs():
            print("  input ", i.name, i.type, i.shape)
        for o in gm.sess.get_outputs():
            print("  output", o.name, o.type, o.shape)
        # convention search: input range x mask polarity x hole content; output scored in three scales
        print("  convention search (MAE vs ground truth in 0..255; known = visible part, hole = hidden part)")
        rows = []
        for rng in ((0, 1), (-1, 1), (0, 255)):
            for hole1 in (True, False):
                for fill in (None, 0, 128, 255):
                    g2 = Gen(name, "cpu", dict(inp=rng, hole1=hole1), share=gm)
                    try:
                        o = g2.raw(garb, known, fill)
                    except Exception as e:
                        print("   ", rng, "hole=1" if hole1 else "hole=0", "FAILED", str(e).splitlines()[0][:80])
                        continue
                    for orng in ((0, 1), (-1, 1), (0, 255)):
                        q = to255(o, orng)
                        ek = float(np.abs(q[known] - g[known]).mean())
                        eh = float(np.abs(q[~known] - g[~known]).mean())
                        rows.append((ek + eh, rng, hole1, fill, orng, ek, eh, float(o.min()), float(o.max())))
        rows.sort(key=lambda r: r[0])
        for r in rows[:8]:
            print(f"    best: in {r[1]} {'hole=1' if r[2] else 'known=1'} fill={'garbage' if r[3] is None else r[3]} out {r[4]}  known MAE {r[5]:.2f}  hole MAE {r[6]:.1f}  raw out {r[7]:.2f}..{r[8]:.2f}")
        worst = rows[-1]
        print(f"    worst: in {worst[1]} {'hole=1' if worst[2] else 'known=1'} out {worst[4]} known MAE {worst[5]:.1f} hole MAE {worst[6]:.1f}")
        # compare to the plausible-but-wrong runner up of the best input convention
        b = rows[0]
        same_in = [r for r in rows if r[1] == b[1] and r[2] == b[2] and r[3] == b[3] and r[4] == b[4]]
        print("    chosen:", b[1], "hole=1" if b[2] else "known=1", "out", b[4])
        # hole content sensitivity with the chosen convention
        gb = Gen(name, "cpu", dict(inp=b[1], hole1=b[2], out=b[4], fill=b[3]), share=gm)
        o1 = gb.raw(garb, known, None)
        o2 = gb.raw(garb, known, 0)
        o3 = gb.raw(garb, known, 255)
        print("    hole content sensitivity (max abs diff of raw outputs, garbage vs black vs white):",
              round(float(np.abs(o1 - o2).max()), 4), round(float(np.abs(o1 - o3).max()), 4), "(output units)")
        o1 = gb.raw(garb, known, b[3])
        q = to255(o1, b[4])
        print("    known pixels exactly preserved after rounding:", bool(np.array_equal(q[known].astype(np.uint8), g[known])),
              " max known abs err", int(np.abs(q[known] - g[known]).max()))
        Image.fromarray(np.concatenate([g, np.where(known[..., None], g, 0).astype(np.uint8), q.astype(np.uint8)], 1)).save(
            os.path.join(SPIKE, "out", "plan12", f"probe_{name}.png"))
        if quick:
            continue
        # non-512 sizes (the app feeds 512)
        shp = gm.sess.get_inputs()[0].shape
        dyn = not all(isinstance(v, int) for v in shp)
        print("  spatial dims dynamic in graph:", dyn)
        for hh, ww in ((256, 256), (384, 512), (640, 640), (500, 500)):
            try:
                gi = np.asarray(Image.fromarray(g).resize((ww, hh)))
                kk = np.ones((hh, ww), bool)
                kk[:, ww * 5 // 8 :] = False
                r = gb.raw(gi, kk, b[3])
                print(f"    {hh}x{ww}: ran, out {r.shape}")
            except Exception as e:
                print(f"    {hh}x{ww}: REJECTED {str(e).splitlines()[0][:110]}")
        # timing
        rng_ = np.random.default_rng(0)
        img = rng_.integers(0, 255, (W, W, 3), dtype=np.uint8)
        kn = np.zeros((W, W), bool)
        kn[:, :256] = True
        for prov in ("cpu", "dml"):
            try:
                t = time.time()
                mm = Gen(name, prov, dict(inp=b[1], hole1=b[2], out=b[4], fill=b[3]))
                load = time.time() - t
                t = time.time()
                mm(img, kn)
                first = time.time() - t
                mm(img, kn)
                ts = []
                for _ in range(5):
                    t = time.time()
                    mm(img, kn)
                    ts.append(time.time() - t)
                line = f"  timing {prov}: session create {load:.2f}s  first run {first:.2f}s  mean {np.mean(ts)*1000:.0f} ms (min {np.min(ts)*1000:.0f}, max {np.max(ts)*1000:.0f})"
                if prov == "dml":  # is the DirectML output correct? compare with CPU on the photo
                    od = mm.raw(garb, known, b[3])
                    d = np.abs(to255(od, b[4]) - q)
                    line += f"  | DML vs CPU on photo: mean abs diff {d.mean():.2f}, max {d.max():.0f}"
                    ed = float(np.abs(to255(od, b[4])[~known] - g[~known]).mean())
                    line += f", DML hole MAE {ed:.1f}"
                print(line)
            except Exception as e:
                print(f"  timing {prov}: FAILED {str(e).splitlines()[0][:200]}")


# --------------------------------------------------------------------------- timing
def timing():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (W, W, 3), dtype=np.uint8)
    known = np.zeros((W, W), bool)
    known[:, :256] = True
    for name in ("migan", "lama"):
        for prov in ("cpu", "dml"):
            try:
                t = time.time()
                m = Model(name, prov)
                load = time.time() - t
                m(img, known)
                m(img, known)  # warm up
                ts = []
                for _ in range(5):
                    t = time.time()
                    m(img, known)
                    ts.append(time.time() - t)
                print(f"timing {name} {prov}: load {load:.2f}s  run mean {np.mean(ts)*1000:.0f} ms  "
                      f"min {np.min(ts)*1000:.0f} ms  max {np.max(ts)*1000:.0f} ms")
            except Exception as e:  # LaMa on DirectML fails
                print(f"timing {name} {prov}: FAILED {str(e).splitlines()[0][:160]}")


# --------------------------------------------------------------------------- contact sheets
def font(sz):
    for f in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def tile(arr, width, label):
    im = Image.fromarray(arr)
    sc = width / im.width
    im = im.resize((width, max(1, int(im.height * sc))), Image.LANCZOS)
    pad = 22
    out = Image.new("RGB", (im.width, im.height + pad), (30, 30, 30))
    out.paste(im, (0, pad))
    ImageDraw.Draw(out).text((4, 3), label, fill=(255, 255, 255), font=font(14))
    return out


def sheet(path, rows, cols, tiles, tw):
    """tiles[row][col] -> PIL; draw a grid."""
    gap = 6
    heights = [max(tl.height for tl in r) for r in tiles]
    sheet_im = Image.new("RGB", (cols * tw + (cols + 1) * gap, sum(heights) + (rows + 1) * gap), (90, 90, 90))
    y = gap
    for r in range(rows):
        for c in range(cols):
            sheet_im.paste(tiles[r][c], (gap + c * (tw + gap), y))
        y += heights[r] + gap
    sheet_im.save(path)
    print("wrote", path, sheet_im.size)


def sheets(provider_for=None, only=None):
    provider_for = provider_for or {"migan": "dml", "lama": "cpu"}
    models = {n: Model(n, p) for n, p in provider_for.items()}
    images = load_images()
    log = []
    for iname, src in images.items():
        if only and iname not in only:
            continue
        for mode in ("right", "all4"):
            rows = []
            for mg in (64, 128, 256):
                margins = (0, 0, mg, 0) if mode == "right" else (mg, mg, mg, mg)
                base = np.pad(src, ((margins[1], margins[3]), (margins[0], margins[2]), (0, 0)), mode="reflect")
                tw = 460
                row = [tile(base, tw, f"{iname} +{mg} {mode}: mirror prefill")]
                for mn, m in models.items():
                    st = {}
                    t = time.time()
                    out = outpaint(src, margins, m, 1, st)
                    dt = time.time() - t
                    Image.fromarray(out).save(os.path.join(OUT, f"{iname}_{mode}_{mg}_{mn}.png"))
                    row.append(tile(out, tw, f"{mn} ({m.provider}) {dt:.1f}s, {st.get('windows', 0)} win"))
                    log.append((iname, mode, mg, mn, st.get("windows", 0), round(st.get("min_known_frac", 1), 3), st.get("relaxed", 0), round(dt, 1)))
                rows.append(row)
            sheet(os.path.join(OUT, f"sheet_{iname}_{mode}.png"), 3, 1 + len(models), rows, tw)
    with open(os.path.join(OUT, "sheets_log.txt"), "a") as f:  # image mode margin model windows min_known relaxed seconds
        for r in log:
            f.write("\t".join(map(str, r)) + "\n")
            print(*r)


# --------------------------------------------------------------------------- plan 12 sheets
def load_images12():
    """The six images of the first spike plus two illustration-like stand-ins that ship inside
    scikit-image (no downloads): `logo` (flat-colour logo on white) and `horse` (black silhouette on
    white). They are NOT anime or manga; no such image exists in the allowed offline set."""
    import skimage.data as sd

    imgs = load_images()
    logo = np.asarray(sd.logo()).astype(np.float32)  # 500x500 RGBA, transparent background
    a = logo[..., 3:4] / 255.0
    imgs["logo_flat"] = np.ascontiguousarray((logo[..., :3] * a + 255 * (1 - a)).round().astype(np.uint8))
    horse = np.asarray(sd.horse()).astype(bool)  # 328x400 silhouette; True = horse
    imgs["horse_silhouette"] = np.ascontiguousarray(np.stack([np.where(horse, 0, 255).astype(np.uint8)] * 3, -1))
    return imgs


def sheets12(only=None, margins_list=(64, 192), modes=("right", "all4")):
    out12 = os.path.join(OUT, "plan12")
    os.makedirs(out12, exist_ok=True)
    names = ["lama", "migan"] + list(P12)
    models = {}
    for n in names:
        if n == "lama":
            models[n] = Model("lama", "cpu")
        elif n == "migan":
            models[n] = Model("migan", "dml")
        else:
            models[n] = Gen(n, P12[n].get("prov", "cpu"))
    groups = {"a": ["lama", "migan", "lama_manga", "cv_lama"], "b": ["lama", "qc_lama", "aot", "qc_aot"]}
    images = load_images12()
    log = open(os.path.join(out12, "sheets12_log.txt"), "a")
    for iname, src in images.items():
        if only and iname not in only:
            continue
        for mode in modes:
            res = {}
            for mg in margins_list:
                mgn = (0, 0, mg, 0) if mode == "right" else (mg, mg, mg, mg)
                for mn in names:
                    f = os.path.join(out12, f"{iname}_{mode}_{mg}_{mn}.png")
                    st = {}
                    t = time.time()
                    o = outpaint(src, mgn, models[mn], 1, st)
                    dt = time.time() - t
                    Image.fromarray(o).save(f)
                    res[(mg, mn)] = (o, dt, st.get("windows", 0))
                    log.write("\t".join(map(str, (iname, mode, mg, mn, st.get("windows", 0), st.get("relaxed", 0), round(dt, 1)))) + "\n")
                    log.flush()
            for gk, gn in groups.items():
                rows = []
                tw = 400
                for mg in margins_list:
                    mgn = (0, 0, mg, 0) if mode == "right" else (mg, mg, mg, mg)
                    base = np.pad(src, ((mgn[1], mgn[3]), (mgn[0], mgn[2]), (0, 0)), mode="reflect")
                    row = [tile(base, tw, f"{iname} +{mg} {mode}: mirror prefill")]
                    for mn in gn:
                        o, dt, nw = res[(mg, mn)]
                        row.append(tile(o, tw, f"{mn} ({models[mn].provider}) {dt:.1f}s {nw}w"))
                    rows.append(row)
                sheet(os.path.join(out12, f"sheet_{iname}_{mode}_{gk}.png"), len(rows), 1 + len(gn), rows, tw)


# --------------------------------------------------------------------------- k experiment
def big_photo(width=3000):
    """No 3000 px photo exists in the allowed offline set. Lanczos-upsample a bundled photo
    (so fine detail is soft; the experiment still shows context/resampling behaviour)."""
    import skimage.data as sd

    im = Image.fromarray(sd.rocket()).resize((width, int(sd.rocket().shape[0] * width / sd.rocket().shape[1])), Image.LANCZOS)
    return np.asarray(im)


def kexp(provider_for=None):
    provider_for = provider_for or {"migan": "dml", "lama": "cpu"}
    src = big_photo()
    h, w = src.shape[:2]
    print("k-experiment source", src.shape)
    rows_log = []
    for mn, p in provider_for.items():
        m = Model(mn, p)
        rows = []
        for mg in (192, 384, 768):
            row = []
            for k in (1, 2, 4):
                st = {}
                t = time.time()
                out = outpaint(src, (0, 0, mg, 0), m, k, st)
                dt = time.time() - t
                Image.fromarray(out).save(os.path.join(OUT, f"k_{mn}_m{mg}_k{k}.png"))
                # crop around the seam, same physical window for all k: 1400 px wide x 1000 high
                cx0 = max(0, w - 700)
                crop = out[h // 2 - 500 : h // 2 + 500, cx0 : w + min(mg, 700)]
                row.append(tile(crop, 480, f"{mn} margin {mg} k={k} ({st['windows']} win, {dt:.0f}s)"))
                rows_log.append((mn, mg, k, st["windows"], round(dt, 1)))
            rows.append(row)
        sheet(os.path.join(OUT, f"sheet_k_{mn}.png"), 3, 3, rows, 480)
    for r in rows_log:
        print(*r)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("probe12", "conv12"):
        probe12(sys.argv[2:] or None, quick=(cmd == "conv12"))
        sys.exit(0)
    if cmd == "sheets12":
        sheets12(sys.argv[2:] or None)
        sys.exit(0)
    if cmd in ("probe", "all"):
        probe()
    if cmd in ("timing", "all"):
        timing()
    if cmd in ("sheets", "all"):
        sheets(only=sys.argv[2:] or None)
    if cmd in ("kexp", "all"):
        kexp()
