#!/usr/bin/env python3
"""ThereminQ Holoqubed: storage and simulator precision of the 4x4 qubit tile format.

Every linear map in the transformer layers (q, k, v, o, gate, up, down for Qwen / Llama) is cut into
4x4 tiles. A tile is 16 weights = a 4-qubit real state: an fp16 norm plus 15 angles, prepared by
15 RY + 14 CNOT (the same gate list as holoqubed_atlas.amplitude_prep_ops). Each experiment rewrites
the weights through one storage or simulation scheme, then measures perplexity, top-1 agreement and
KL against the unmodified model on WikiText-2.

Experiments (--experiments, comma separated):
  fp16         plain fp16 weights and fp16 tiles (15 fp16 Gray-code angles + fp16 norm)
  gray         Gray-code circuit angles, uniform int-b code per angle slot
  tree         Mottonen tree node angles (7 splits in [0, pi/2], 8 signed leaves), uniform and Lloyd-Max
  int          ordinary symmetric int-b weights with an fp16 scale per group (baselines)
  statevector  simulator precision: amplitudes and RY coefficients rounded after every gate
               (fp32, fp16, bf16, fp8 e4m3, fp8 e5m2)
  qformat      quantum format (qneuron SiLU gating) with exact tiles and with fp16 tiles

Tile decoding is emulated in NumPy (batched 4-qubit statevectors, little endian as in Qrack).
"""
import argparse
import json
import math
import os
import sys
import time

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holoqubed_atlas as H  # noqa: E402

SCHED = H._tile_schedule(4, [0, 1, 2, 3])
FMT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
       "fp8 e4m3": getattr(torch, "float8_e4m3fn", None), "fp8 e5m2": getattr(torch, "float8_e5m2", None)}


# ---------------------------------------------------------------- tiles

def to_tiles(W):
    o, i = W.shape
    return W.reshape(o // 4, 4, i // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)


def from_tiles(T, shape):
    o, i = shape
    return T.reshape(o // 4, i // 4, 4, 4).transpose(0, 2, 1, 3).reshape(o, i)


def _round(x, fmt):
    if fmt is None:
        return x
    return torch.from_numpy(x).float().to(FMT[fmt]).float().numpy().astype(np.float64)


def decode_circuit(theta, norm, fmt=None):
    """Run the preparation circuit on |0000> for every tile; optionally round to a float format after each gate."""
    n = theta.shape[0]
    psi = np.zeros((n, 16))
    psi[:, 0] = 1.0
    idx = np.arange(16)
    for op in SCHED:
        if op[0] == "ry":
            q, th = op[1], theta[:, op[2]].astype(np.float64)
            v = psi.reshape(n, 2 ** (3 - q), 2, 2 ** q)
            c = _round(np.cos(th / 2), fmt)[:, None, None]
            s = _round(np.sin(th / 2), fmt)[:, None, None]
            a0, a1 = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
            v[:, :, 0, :] = c * a0 - s * a1
            v[:, :, 1, :] = s * a0 + c * a1
            psi = _round(psi, fmt)
        else:
            psi = psi[:, np.where((idx >> op[1]) & 1, idx ^ (1 << op[2]), idx)]
    return psi * norm.astype(np.float64)[:, None]


def gray_angles(W):
    T = to_tiles(W)
    nrm = np.linalg.norm(T, axis=1)
    keep = nrm > 0
    th = np.zeros((T.shape[0], 15))
    th[keep] = H._batched_prep_angles(T[keep] / nrm[keep, None], 4)
    return th, nrm


def encode_tree(T):
    a, b = T[:, 0::2], T[:, 1::2]
    phi, r = np.arctan2(b, a), np.hypot(a, b)
    betas = []
    while r.shape[1] > 1:
        r0, r1 = r[:, 0::2], r[:, 1::2]
        betas.append(np.arctan2(r1, r0))
        r = np.hypot(r0, r1)
    return r[:, 0], betas[::-1], phi


def decode_tree(nrm, betas, phi):
    r = nrm[:, None]
    for be in betas:
        r = np.stack([r * np.cos(be), r * np.sin(be)], -1).reshape(r.shape[0], -1)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], -1).reshape(r.shape[0], -1)


def _lloyd(x, k, iters=25):
    x = np.sort(x[np.random.default_rng(0).choice(x.size, min(x.size, 200000), replace=False)])
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        idx = np.searchsorted((c[1:] + c[:-1]) / 2, x)
        n = np.bincount(idx, None, k)
        c = np.where(n > 0, np.bincount(idx, x, k) / np.maximum(n, 1), c)
    return np.sort(c)


def _q_uniform(x, bits, lo, hi, circular=False):
    if circular:
        d = 2 * np.pi / 2 ** bits
        return np.round(x / d) * d
    d = (hi - lo) / (2 ** bits - 1)
    return lo + np.round((x - lo) / d) * d


def _q_lloyd(x, bits):
    c = _lloyd(x.ravel(), 2 ** bits)
    return c[np.searchsorted((c[1:] + c[:-1]) / 2, x)]


# ---------------------------------------------------------------- schemes (W float64 -> W float64)

def s_identity(W):
    return W


def s_fp16(W):
    return W.astype(np.float16).astype(np.float64)


def s_fp16_tiles(W):
    th, nrm = gray_angles(W)
    return from_tiles(decode_circuit(th.astype(np.float16), nrm.astype(np.float16)), W.shape)


def s_gray(bits):
    def f(W):
        th, nrm = gray_angles(W)
        lo, hi = th.min(0), th.max(0)
        step = np.where(hi > lo, (hi - lo) / (2 ** bits - 1), 1.0)
        thq = lo + np.round((th - lo) / step) * step
        return from_tiles(decode_circuit(thq, nrm.astype(np.float16)), W.shape)
    return f


def s_tree(bi, bl, lloyd):
    def f(W):
        nrm, betas, phi = encode_tree(to_tiles(W))
        if lloyd:
            betas, phi = [_q_lloyd(be, bi) for be in betas], _q_lloyd(phi, bl)
        else:
            betas = [_q_uniform(be, bi, 0, np.pi / 2) for be in betas]
            phi = _q_uniform(phi, bl, -np.pi, np.pi, circular=True)
        return from_tiles(decode_tree(nrm.astype(np.float16).astype(np.float64), betas, phi), W.shape)
    return f


def s_int(bits, group):
    def f(W):
        o, i = W.shape
        Wg = W.reshape(o, i // group, group)
        qm = 2 ** (bits - 1) - 1
        s = (np.abs(Wg).max(-1, keepdims=True) / qm).astype(np.float16).astype(np.float64)
        s[s == 0] = 1
        return (np.clip(np.round(Wg / s), -qm, qm) * s).reshape(o, i)
    return f


def s_state(fmt):
    def f(W):
        th, nrm = gray_angles(W)
        return from_tiles(decode_circuit(th, nrm, fmt), W.shape)
    return f


def experiment_rows(names):
    rows = []
    if "fp16" in names:
        rows += [("source weights", 16, s_identity, False),
                 ("plain fp16 weights", 16, s_fp16, False),
                 ("fp16 tiles (15 fp16 Gray angles + fp16 norm)", 16, s_fp16_tiles, False)]
    if "gray" in names:
        rows += [(f"Gray angles, int{b} + fp16 norm", (15 * b + 16) / 16, s_gray(b), False) for b in (8, 6, 4)]
    if "tree" in names:
        for bi, bl in ((6, 8), (5, 7), (4, 6), (3, 5), (2, 4)):
            bpw = (7 * bi + 8 * bl + 16) / 16
            rows.append((f"tree, uniform, int{bi} splits / int{bl} leaves", bpw, s_tree(bi, bl, False), False))
            rows.append((f"tree, Lloyd-Max, int{bi} splits / int{bl} leaves", bpw, s_tree(bi, bl, True), False))
    if "int" in names:
        rows += [(f"int{b} weights, fp16 scale per {g}", b + 16 / g, s_int(b, g), False)
                 for b, g in ((8, 16), (8, 128), (6, 16), (4, 16), (4, 128))]
    if "statevector" in names:
        rows += [(f"statevector {f}", 16, s_state(f), False) for f in FMT if FMT[f] is not None]
    if "qformat" in names:
        rows += [("quantum format, exact tiles", 16, s_identity, True),
                 ("quantum format, fp16 tiles", 16, s_fp16_tiles, True)]
    return rows


# ---------------------------------------------------------------- evaluation

def tile_linears(model, n_layers):
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else None
    if layers is None:
        raise SystemExit("expected a Llama / Qwen style model with model.model.layers")
    out = []
    for lay in list(layers)[:n_layers or None]:
        for m in lay.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.shape[0] % 4 == 0 and m.weight.shape[1] % 128 == 0:
                out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--eval", default="wiki_test.txt")
    ap.add_argument("--calib", default="wiki_calib.txt", help="calibration text for the qformat experiment")
    ap.add_argument("--experiments", default="fp16,gray,tree,int,statevector,qformat")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--window_len", type=int, default=256)
    ap.add_argument("--layers", type=int, default=0, help="rewrite only the first N layers (0 = all)")
    ap.add_argument("--cosines", type=int, default=8)
    ap.add_argument("--calib_tokens", type=int, default=1024)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    tok, model = H.load_model(a.model_id)
    torch.set_num_threads(os.cpu_count() or 1)
    lins = tile_linears(model, a.layers)
    orig = [m.weight.detach().to(torch.bfloat16).clone() for m in lins]
    nw = sum(o.numel() for o in orig)
    exact_bf16 = all(torch.equal(m.weight.detach(), o.to(m.weight.dtype)) for m, o in zip(lins, orig))
    if not exact_bf16:
        print("note: weights are not bf16-exact; the reference keeps the loaded weights", flush=True)
        orig = [m.weight.detach().clone() for m in lins]
    ids = tok(open(a.eval).read(), return_tensors="pt").input_ids
    windows = [ids[:, j * a.window_len:(j + 1) * a.window_len] for j in range(a.windows)]
    calib = open(a.calib).read() if "qformat" in a.experiments else ""

    def restore():
        with torch.no_grad():
            for m, o in zip(lins, orig):
                m.weight.data.copy_(o.to(m.weight.dtype))

    with torch.no_grad():
        ref = [model(w).logits[0, :-1].float() for w in windows]
    print(f"{len(lins)} linear maps, {nw / 1e6:.1f}M weights, {a.windows} x {a.window_len} tokens", flush=True)
    print("\n| Scheme | Bits/weight | Size | Rel. weight error | PPL | Top-1 vs source | KL vs source |\n|---|---|---|---|---|---|---|", flush=True)
    results = []
    for label, bpw, fn, qformat in experiment_rows(a.experiments.split(",")):
        t0 = time.time()
        err = []
        with torch.no_grad():
            for m, o in zip(lins, orig):
                W = o.double().numpy()
                Wq = fn(W)
                err.append(float(np.linalg.norm(Wq - W) / max(np.linalg.norm(W), 1e-30)))
                m.weight.data.copy_(torch.from_numpy(Wq).to(m.weight.dtype))
        keep = None
        if qformat:
            mlps, params = H.qformat_calibrate(model, tok, calib, a.calib_tokens, a.cosines)
            keep = [r["owner"].forward for r in mlps]
            for i, r in enumerate(mlps):
                r["owner"].forward = H._qformat_mlp(r, params[i], a.cosines)
        nll = kl = hits = cnt = 0.0
        with torch.no_grad():
            for w, lg_r in zip(windows, ref):
                lg = model(w).logits[0, :-1].float()
                tgt = w[0, 1:]
                nll += float(F.cross_entropy(lg, tgt, reduction="sum"))
                hits += float((lg.argmax(-1) == lg_r.argmax(-1)).sum())
                lp_r, lp = F.log_softmax(lg_r, -1), F.log_softmax(lg, -1)
                kl += float((lp_r.exp() * (lp_r - lp)).sum())
                cnt += tgt.numel()
                del lg, lp, lp_r
        if keep:
            for r, f in zip(mlps, keep):
                r["owner"].forward = f
        restore()
        row = dict(scheme=label, bits_per_weight=bpw, size_mb=nw * bpw / 8 / 1e6, rel_weight_error=float(np.mean(err)),
                   ppl=math.exp(nll / cnt), top1=hits / cnt, kl=max(0.0, kl / cnt), seconds=time.time() - t0)   # KL >= 0: clamp float noise
        results.append(row)
        print(f"| {label} | {bpw:.3f} | {row['size_mb']:.0f} MB | {row['rel_weight_error']:.2e} | {row['ppl']:.2f} | "
              f"{row['top1'] * 100:.2f}% | {row['kl']:.4f} |", flush=True)
    if a.json:
        json.dump(dict(model_id=a.model_id, weights=nw, rows=results), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
