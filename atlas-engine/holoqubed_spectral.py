#!/usr/bin/env python3
"""ThereminQ Holoqubed Spectral: how far can a dense model's spectra flatten before quality drops?

Every linear map in the transformer layers becomes W = U diag(s) V^H. U and V^H are unitary and exact as
circuits; the diagonal is the only non-unitary part. Block encoded with s_max, one matrix-vector product
succeeds with probability ||W x||^2 / (s_max^2 ||x||^2), which is mean(s^2) / s_max^2 for isotropic input,
and these probabilities multiply along the 196 products of a token.

This probe freezes U and V^H (fp16) and trains only the singular values, s = s0 exp(a), on WikiText-2 text
by distillation against the unmodified model (the same modules with s = s0 reproduce it exactly):

    loss = KL(dense || model) + lambda * mean over matrices of log(s_max^2 / mean(s^2))

lambda ramps geometrically over the run. Every --eval_minutes it records perplexity, top-1 agreement and KL
against the original dense model, the spectral flatness per matrix type and the success probability of the
whole chain of products, measured with real activations. The result is the trade-off curve between chain
success and model quality. U and V^H stay fixed, so a full fine-tune could do better than this curve.

  python3 holoqubed_spectral.py --minutes 80 --lambda_start 0.001 --lambda_end 30
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holoqubed_atlas as H  # noqa: E402

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class _SVDMatmul(torch.autograd.Function):
    """y = ((x V^H^T) * s) U^T with U, V^H stored in fp16; saves only x and x V^H^T."""

    @staticmethod
    def forward(ctx, x, s, U16, Vh16):
        z = x @ Vh16.float().t()
        ctx.save_for_backward(z, s, U16, Vh16)
        return (z * s) @ U16.float().t()

    @staticmethod
    def backward(ctx, gy):
        z, s, U16, Vh16 = ctx.saved_tensors
        gzs = gy @ U16.float()
        gs = (gzs * z).reshape(-1, z.shape[-1]).sum(0)
        gx = (gzs * s) @ Vh16.float()
        return gx, gs, None, None


class SVDLinear(nn.Module):
    def __init__(self, lin: nn.Linear, kind: str):
        super().__init__()
        W = lin.weight.detach().double()
        U_, s, Vh = torch.linalg.svd(W, full_matrices=False)
        self.kind = kind
        self.register_buffer("U", U_.half(), persistent=False)
        self.register_buffer("Vh", Vh.half(), persistent=False)
        self.register_buffer("s0", s.float(), persistent=False)
        self.a = nn.Parameter(torch.zeros_like(s, dtype=torch.float32))
        self.bias = lin.bias
        self.track = False
        self.teacher = False                             # use the original singular values (exact dense model)
        self.stat = [0.0, 0]

    def s(self):
        return self.s0 if self.teacher else self.s0 * torch.exp(self.a)

    def forward(self, x):
        s = self.s()
        y = _SVDMatmul.apply(x, s, self.U, self.Vh)
        if self.track:                                   # real-activation success: ||Wx||^2 / (s_max^2 ||x||^2)
            with torch.no_grad():
                r = (y.float().pow(2).sum(-1) / (s.max() ** 2 * x.float().pow(2).sum(-1).clamp_min(1e-30))).reshape(-1)
                self.stat[0] += float(torch.log10(r.clamp_min(1e-30)).sum()); self.stat[1] += r.numel()
        return y if self.bias is None else y + self.bias

    def flat_penalty(self):
        s = self.s()
        return torch.log(s.max() ** 2 / (s ** 2).mean())


def _rss_gb():
    try:
        return [int(l.split()[1]) / 2 ** 20 for l in open("/proc/self/status") if l.startswith("VmRSS")][0]
    except (OSError, IndexError):
        return float("nan")


def _trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)          # hand freed SVD workspace back to the OS
    except OSError:
        pass


def convert(model, log=print):
    mods = []
    for li, lay in enumerate(model.model.layers):
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(lay, parent_name)
            for k in KINDS:
                if hasattr(parent, k):
                    m = SVDLinear(getattr(parent, k), k)
                    setattr(parent, k, m)
                    mods.append(m)
        _trim()
    log(f"{len(mods)} linear maps factored into U diag(s) V^H; {sum(m.a.numel() for m in mods):,} trainable singular values")
    return mods


def spectra(mods):
    out = {}
    for k in KINDS:
        ms = [m for m in mods if m.kind == k]
        with torch.no_grad():
            r = [float(m.s().max() / m.s().mean()) for m in ms]
            p = [float((m.s() ** 2).mean() / m.s().max() ** 2) for m in ms]
        out[k] = (float(np.median(r)), float(np.median(p)))
    with torch.no_grad():
        chain_iso = sum(float(torch.log10((m.s() ** 2).mean() / m.s().max() ** 2)) for m in mods)
    return out, chain_iso


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--train", default="wiki_calib.txt", help="training text (WikiText-2 train)")
    ap.add_argument("--eval", default="wiki_test.txt")
    ap.add_argument("--minutes", type=float, default=80)
    ap.add_argument("--eval_minutes", type=float, default=6)
    ap.add_argument("--lambda_start", type=float, default=1e-3)
    ap.add_argument("--lambda_end", type=float, default=30.0)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json", default="spectral.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    log = lambda s: print(s, flush=True)
    torch.manual_seed(0)

    tok, model = H.load_model(a.model_id)
    for p in model.parameters():
        p.requires_grad_(False)
    ev = tok(open(a.eval).read()[:12000], return_tensors="pt").input_ids
    windows = [ev[:, j * 256:(j + 1) * 256] for j in range(2)]
    with torch.no_grad():
        dense = [model(w).logits[0, :-1].float() for w in windows]
    ppl_dense = math.exp(sum(float(F.cross_entropy(lg, w[0, 1:], reduction="sum")) for w, lg in zip(windows, dense))
                         / sum(w.shape[1] - 1 for w in windows))
    mods = convert(model, log)
    train = tok(open(a.train).read(), return_tensors="pt").input_ids[0]
    log(f"training text {train.numel():,} tokens; dense PPL {ppl_dense:.2f}; RSS after factoring {_rss_gb():.2f} GB")
    opt = torch.optim.Adam([m.a for m in mods], lr=a.lr)

    def evaluate():
        for m in mods:
            m.track, m.stat = True, [0.0, 0]
        nll = kl = hits = cnt = 0.0
        with torch.no_grad():
            for w, lg_d in zip(windows, dense):
                lg = model(w).logits[0, :-1].float(); tgt = w[0, 1:]
                nll += float(F.cross_entropy(lg, tgt, reduction="sum")); hits += float((lg.argmax(-1) == lg_d.argmax(-1)).sum())
                lp_d, lp = F.log_softmax(lg_d, -1), F.log_softmax(lg, -1)
                kl += float((lp_d.exp() * (lp_d - lp)).sum()); cnt += tgt.numel()
        chain_real = sum(m.stat[0] / max(1, m.stat[1]) for m in mods)
        for m in mods:
            m.track = False
        return math.exp(nll / cnt), hits / cnt, max(0.0, kl / cnt), chain_real

    rows = []

    def record(step, lam, elapsed):
        ppl, top1, kl, chain_real = evaluate()
        sp, chain_iso = spectra(mods)
        r = dict(step=step, minutes=elapsed / 60, lam=lam, ppl=ppl, top1=top1, kl=kl, chain_log10_real=chain_real,
                 chain_log10_iso=chain_iso, spectra=sp)
        rows.append(r)
        log(f"| {step} | {elapsed / 60:.0f} | {lam:.3g} | {ppl:.2f} | {top1 * 100:.2f}% | {kl:.4f} | "
            f"{sp['q_proj'][0]:.2f} | {sp['up_proj'][0]:.2f} | {sp['down_proj'][0]:.2f} | "
            f"{np.median([v[1] for v in sp.values()]):.3f} | 10^{chain_real:.0f} |")
        json.dump(dict(args=vars(a), ppl_dense=ppl_dense, rows=rows), open(a.json, "w"), indent=1)

    log("\n| Step | Minutes | lambda | PPL | Top-1 vs dense | KL | s_max/s_mean q | up | down | "
        "Median success per product | Chain success, real activations |\n|---|---|---|---|---|---|---|---|---|---|---|")
    record(0, 0.0, 0.0)
    t0 = time.time(); last_ev = t0; step = 0
    g = torch.Generator().manual_seed(0)
    while True:
        elapsed = time.time() - t0
        if elapsed >= a.minutes * 60:
            break
        frac = elapsed / (a.minutes * 60)
        lam = a.lambda_start * (a.lambda_end / a.lambda_start) ** frac
        idx = torch.randint(0, train.numel() - a.seq - 1, (a.batch,), generator=g)
        x = torch.stack([train[i:i + a.seq] for i in idx])
        with torch.no_grad():
            for m in mods:
                m.teacher = True
            ht = model.model(x).last_hidden_state
            for m in mods:
                m.teacher = False
        h = model.model(x).last_hidden_state                            # chunked head: no full logits tensor
        ce = 0.0
        for c in range(0, h.shape[1], 64):
            with torch.no_grad():
                lt = F.log_softmax(model.lm_head(ht[:, c:c + 64]).float(), -1)
            ls = F.log_softmax(model.lm_head(h[:, c:c + 64]).float(), -1)
            ce = ce + (lt.exp() * (lt - ls)).sum()
        ce = ce / (h.shape[0] * h.shape[1])
        pen = torch.stack([m.flat_penalty() for m in mods]).mean()
        loss = ce + lam * pen
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 10 == 0:
            log(f"  step {step}: KL {ce.item():.4f}, penalty {pen.item():.3f}, lambda {lam:.3g}, {elapsed / 60:.1f} min, "
                f"RSS {_rss_gb():.2f} GB")
        if time.time() - last_ev >= a.eval_minutes * 60:
            record(step, lam, time.time() - t0); last_ev = time.time()
    record(step, lam, time.time() - t0)


if __name__ == "__main__":
    main()
