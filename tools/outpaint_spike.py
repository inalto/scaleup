#!/usr/bin/env python
"""Throwaway feasibility spike for the canvas-extension (outpainting) feature.

Not shipped. It exists to produce the measurements in docs/OUTPAINT_SPIKE.md:
  probe   tensor layouts, value ranges, mask polarity, channel order (by experiment)
  timing  one 512 window on CPU and on the DirectML provider
  sheets  contact sheets: image x margin (right / all four sides) x model
  kexp    context-factor experiment on a ~3000 px photo
  all     probe + timing + sheets + kexp

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
        return np.round(o[:, sl].reshape(-1, 3).mean(0), 1).tolist()

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
    print("migan mask 1 instead of 255, mean over hole:", np.round(o1[0].transpose(1, 2, 0)[:, 256:].reshape(-1, 3).mean(0), 1).tolist())
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
    if cmd in ("probe", "all"):
        probe()
    if cmd in ("timing", "all"):
        timing()
    if cmd in ("sheets", "all"):
        sheets(only=sys.argv[2:] or None)
    if cmd in ("kexp", "all"):
        kexp()
