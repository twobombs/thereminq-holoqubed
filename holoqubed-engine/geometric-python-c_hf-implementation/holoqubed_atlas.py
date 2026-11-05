#!/usr/bin/env python3
"""
ThereminQ Holoqubed - Geometric Atlas Engine (single file)

Geometry decides, per token, which MLP neurons of a Hugging Face causal LM are
evaluated. Each MLP input is placed on an atlas of the activation manifold:

  chart     rank-r principal subspace of calibration MLP inputs (exact scores)
  regions   C Voronoi cells inside the chart (local context)
  spread    per region and neuron, the residual ellipsoid the chart misses
  curve     n-D Hilbert curve over region centroids (O(log C) region lookup)
  layout    neurons stored in Hilbert order of their firing centroid

A neuron is evaluated only when chart score plus ellipsoid spread can reach its
active range. Evaluated neurons use the full weights, exactly.

Subcommands
  data      fetch WikiText-2 calibration and evaluation text
  forge     build a .atlas routing dictionary from calibration text
  bench     divergence harness vs the dense model and a static pruning baseline
  generate  greedy text generation through the atlas
  serve     OpenAI-compatible REST API (models, completions, chat, streaming)
  selftest  OpenCL kernels vs NumPy reference on an existing atlas

Backends: dense (layout check), reference (NumPy), opencl (PyOpenCL, Rusticl or
any OpenCL 1.2+ device), static (context-free pruning baseline). No CUDA.

Requirements: numpy torch transformers; pyopencl for the opencl backend;
datasets for the data subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

ATLAS_FORMAT = "holoqubed-atlas/1"
SILU_MIN_X = -1.2784645
SILU_MIN_Y = 0.2784645


# =============================================================================
# 1. GEOMETRY
# =============================================================================

def hilbert_encode(coords: np.ndarray, bits: int) -> np.ndarray:
    """
    Map integer coordinates [n, k] (each in [0, 2**bits)) to Hilbert distances.
    Vectorised Skilling "AxesToTranspose" followed by bit interleaving.
    k * bits must be <= 64.
    """
    X = np.ascontiguousarray(coords, dtype=np.int64).copy()
    n, k = X.shape
    if k * bits > 64:
        raise ValueError(f"k*bits = {k * bits} exceeds 64-bit key space")
    M = 1 << (bits - 1)

    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(k):
            hi = (X[:, i] & Q) != 0
            X[hi, 0] ^= P
            lo = ~hi
            t = (X[lo, 0] ^ X[lo, i]) & P
            X[lo, 0] ^= t
            X[lo, i] ^= t
        Q >>= 1

    for i in range(1, k):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(n, dtype=np.int64)
    Q = M
    while Q > 1:
        t[(X[:, k - 1] & Q) != 0] ^= Q - 1
        Q >>= 1
    for i in range(k):
        X[:, i] ^= t

    h = np.zeros(n, dtype=np.uint64)
    one = np.uint64(1)
    for b in range(bits - 1, -1, -1):
        for i in range(k):
            h = (h << one) | ((X[:, i] >> b) & 1).astype(np.uint64)
    return h


def quantize(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, bits: int) -> np.ndarray:
    """Quantise float points into the integer Hilbert grid defined by [lo, hi]."""
    span = np.maximum(hi - lo, 1e-12)
    q = (points - lo) / span * ((1 << bits) - 1)
    return np.clip(np.rint(q), 0, (1 << bits) - 1).astype(np.int64)


def kmeans(X: np.ndarray, C: int, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd k-means. Returns centroids [C, r] and assignment [n]."""
    rng = np.random.default_rng(seed)
    C = min(C, X.shape[0])
    cent = X[rng.choice(X.shape[0], C, replace=False)].astype(np.float64)
    x2 = (X.astype(np.float64) ** 2).sum(1, keepdims=True)
    a = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(iters):
        d = x2 - 2.0 * X @ cent.T + (cent ** 2).sum(1)[None, :]
        a = d.argmin(1)
        for c in range(C):
            m = a == c
            if m.any():
                cent[c] = X[m].mean(0)
            else:
                cent[c] = X[rng.integers(X.shape[0])]
    return cent.astype(np.float32), a


def nearest_region(Z: np.ndarray, cent: np.ndarray) -> np.ndarray:
    d = (Z ** 2).sum(1, keepdims=True) - 2.0 * Z @ cent.T + (cent ** 2).sum(1)[None, :]
    return d.argmin(1)


def silu_maxabs(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """max |silu(g)| for g in [lo, hi]."""
    def silu(v):
        return v / (1.0 + np.exp(-np.clip(v, -60, 60)))
    v = np.maximum(np.abs(silu(lo)), np.abs(silu(hi)))
    inside = (lo < SILU_MIN_X) & (hi > SILU_MIN_X)
    return np.where(inside, np.maximum(v, SILU_MIN_Y), v)


def activate(kind: str, pre: np.ndarray, up: np.ndarray | None) -> np.ndarray:
    if kind in ("relu", "relu_glu"):
        h = np.maximum(pre, 0.0)
    elif kind == "gelu":
        h = 0.5 * pre * (1.0 + np.tanh(0.7978845608 * (pre + 0.044715 * pre ** 3)))
    else:
        h = pre / (1.0 + np.exp(-np.clip(pre, -60, 60)))
    return h * up if up is not None else h


# =============================================================================
# 2. MODEL DISCOVERY
# =============================================================================

def load_model(model_id: str, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 trust_remote_code=trust_remote_code).eval()
    return tok, model


def _linear(lin) -> tuple[np.ndarray, np.ndarray | None]:
    w = lin.weight.detach().float()
    if lin.__class__.__name__ == "Conv1D":  # GPT-2 stores [in, out]
        w = w.t()
    b = lin.bias.detach().float().numpy() if getattr(lin, "bias", None) is not None else None
    return w.contiguous().numpy(), b


def find_mlps(model) -> list[dict]:
    """
    One record per MLP: the module whose input is the MLP input (hook point)
    and the weight matrices in [out, in] layout.
      relu      OPT style fc1 -> ReLU -> fc2
      gelu      GPT-2 style c_fc -> GELU -> c_proj
      relu_glu  ReLU(gate) * up -> down   (ProSparse / ReluLLaMA style)
      silu_glu  SiLU(gate) * up -> down   (Llama / Qwen style)
    """
    cfg = model.config
    act = str(getattr(cfg, "hidden_act", getattr(cfg, "activation_function", ""))).lower()
    out = []
    for name, mod in model.named_modules():
        if hasattr(mod, "fc1") and hasattr(mod, "fc2") and hasattr(mod, "activation_fn"):
            out.append(dict(name=name, kind="relu" if "relu" in act else "gelu",
                            hook=mod.fc1, owner=mod, up=mod.fc1, down=mod.fc2, gate=None))
        elif hasattr(mod, "c_fc") and hasattr(mod, "c_proj") and mod.__class__.__name__.endswith("MLP"):
            out.append(dict(name=name, kind="gelu", hook=mod, owner=mod, up=mod.c_fc, down=mod.c_proj, gate=None))
        elif hasattr(mod, "gate_proj") and hasattr(mod, "up_proj") and hasattr(mod, "down_proj"):
            out.append(dict(name=name, kind="relu_glu" if "relu" in act else "silu_glu",
                            hook=mod, owner=mod, up=mod.up_proj, down=mod.down_proj, gate=mod.gate_proj))
    return out


# =============================================================================
# 3. FORGE
# =============================================================================

def capture_inputs(model, tok, mlps, text: str, n_tokens: int, chunk: int) -> list[np.ndarray]:
    caps: list[list[torch.Tensor]] = [[] for _ in mlps]

    def hook(i):
        def f(_mod, inp):
            caps[i].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).float())
        return f

    handles = [m["hook"].register_forward_pre_hook(hook(i)) for i, m in enumerate(mlps)]
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    with torch.no_grad():
        for s in range(0, ids.shape[1], chunk):
            model(ids[:, s:s + chunk])
    for h in handles:
        h.remove()
    print(f"Captured {ids.shape[1]} calibration tokens across {len(mlps)} MLP layers")
    return [torch.cat(c).numpy() for c in caps]


def forge_layer(rec: dict, X: np.ndarray, args) -> dict:
    kind = rec["kind"]
    if rec["gate"] is not None:
        W1, b1 = _linear(rec["gate"])
        Wu, bu = _linear(rec["up"])
    else:
        W1, b1 = _linear(rec["up"])
        Wu, bu = None, None
    W2, b2 = _linear(rec["down"])
    n, d = W1.shape
    b1 = b1 if b1 is not None else np.zeros(n, np.float32)
    if Wu is not None and bu is None:
        bu = np.zeros(n, np.float32)
    b2 = b2 if b2 is not None else np.zeros(W2.shape[0], np.float32)

    X = X.astype(np.float32)
    if X.shape[0] > args.max_calib_rows:
        X = X[np.random.default_rng(0).choice(X.shape[0], args.max_calib_rows, replace=False)]

    # --- the chart -------------------------------------------------------------
    mu = X.mean(0)
    D = X - mu
    r = min(args.rank, d - 1)
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    P = np.ascontiguousarray(Vt[:r].T)                   # [d, r]
    Z = D @ P
    R = D - Z @ P.T                                      # what the chart cannot see
    chart_energy = float((S[:r] ** 2).sum() / (S ** 2).sum())

    # --- the atlas -------------------------------------------------------------
    cent, assign = kmeans(Z, args.regions, iters=args.kmeans_iters)
    C = cent.shape[0]

    def region_spread(Wm: np.ndarray) -> np.ndarray:
        Rp = R @ Wm.T                                     # residual term per token and neuron
        glob = np.sqrt((Rp ** 2).mean(0))
        sig = np.empty((C, n), np.float32)
        for c in range(C):
            m = assign == c
            sig[c] = np.sqrt((Rp[m] ** 2).mean(0)) if m.sum() > 1 else glob
        return sig

    sig = region_spread(W1)
    sig_u = region_spread(Wu) if Wu is not None else None
    rn = np.linalg.norm(R, axis=1)
    rtyp = np.array([rn[assign == c].mean() if (assign == c).any() else rn.mean() for c in range(C)], np.float32)

    # --- calibration activations (layout + tolerance) -------------------------
    pre = X @ W1.T + b1
    upc = (X @ Wu.T + bu) if Wu is not None else None
    H = activate(kind, pre, upc)
    colnorm = np.linalg.norm(W2, axis=0).astype(np.float32)
    contrib = np.abs(H) * colnorm
    tau = float(contrib.mean())

    # --- the curve: Hilbert order of regions over leading chart axes ----------
    q = args.curve_dims
    lo = Z[:, :q].min(0)
    hi = Z[:, :q].max(0)
    rkeys = hilbert_encode(quantize(cent[:, :q], lo, hi, args.curve_bits), args.curve_bits)
    rorder = np.argsort(rkeys, kind="stable")

    # --- the layout: neurons sorted by Hilbert key of their firing centroid ---
    w = contrib / (contrib.sum(0, keepdims=True) + 1e-12)
    fire_centroid = (w.T @ Z[:, :q])                      # [n, q]
    dead = contrib.sum(0) <= 0
    nkeys = hilbert_encode(quantize(fire_centroid, lo, hi, args.curve_bits), args.curve_bits)
    nkeys = np.where(dead, np.iinfo(np.uint64).max, nkeys)
    perm = np.argsort(nkeys, kind="stable").astype(np.int64)

    def p(a):
        return np.ascontiguousarray(a[perm]) if a is not None else None

    W1p, b1p = p(W1), p(b1)
    out = dict(
        kind=kind, n=n, d_in=d, d_out=W2.shape[0], rank=r, regions=C, chart_energy=chart_energy,
        arrays=dict(
            perm=perm, W1=W1p, b1=b1p, W2T=np.ascontiguousarray(W2.T[perm]), b2=b2.astype(np.float32),
            mu=mu.astype(np.float32), P=P.astype(np.float32),
            base=(mu @ W1p.T + b1p).astype(np.float32), Kp=np.ascontiguousarray((W1p @ P).astype(np.float32)),
            cent=cent, sig=np.ascontiguousarray(sig[:, perm]), rtyp=rtyp, colnorm=colnorm[perm],
            importance=contrib.mean(0)[perm].astype(np.float32),
            curve_keys=rkeys[rorder], curve_regions=rorder.astype(np.int64),
            curve_lo=lo.astype(np.float32), curve_hi=hi.astype(np.float32),
            curve_bits=np.array(args.curve_bits), curve_dims=np.array(q), tau=np.array(tau, np.float32),
        ),
    )
    if Wu is not None:
        Wup, bup = p(Wu), p(bu)
        out["arrays"].update(
            Wu=Wup, bu=bup, base_u=(mu @ Wup.T + bup).astype(np.float32),
            Kp_u=np.ascontiguousarray((Wup @ P).astype(np.float32)),
            sig_u=np.ascontiguousarray(sig_u[:, perm]),
        )
    out["calib_active_frac"] = float((contrib > 0).mean()) if kind in ("relu", "relu_glu") else float("nan")
    return out


def cmd_forge(args):
    t0 = time.time()
    out = args.out or args.model_id.rstrip("/").split("/")[-1] + ".atlas"
    print(f"Igniting Atlas Forge: {args.model_id}")
    tok, model = load_model(args.model_id, args.trust_remote_code)
    mlps = find_mlps(model)
    if not mlps:
        raise SystemExit("No supported MLP blocks found in this model")
    text = open(args.calib, encoding="utf-8").read()
    caps = capture_inputs(model, tok, mlps, text, args.calib_tokens, args.chunk)

    os.makedirs(out, exist_ok=True)
    manifest = dict(model_id=args.model_id, format=ATLAS_FORMAT, calib_tokens=int(caps[0].shape[0]),
                    rank=args.rank, regions=args.regions, curve_dims=args.curve_dims, curve_bits=args.curve_bits,
                    layers=[])
    print(f"{'layer':<34}{'kind':<10}{'neurons':>8}{'chart energy':>14}{'calib active':>14}")
    for i, (rec, X) in enumerate(zip(mlps, caps)):
        L = forge_layer(rec, X, args)
        files = {}
        for key, arr in L["arrays"].items():
            fn = f"L{i:03d}.{key}.npy"
            np.save(os.path.join(out, fn), arr, allow_pickle=False)
            files[key] = fn
        manifest["layers"].append(dict(index=i, module=rec["name"], kind=L["kind"], n=L["n"], d_in=L["d_in"],
                                       d_out=L["d_out"], rank=L["rank"], regions=L["regions"],
                                       chart_energy=L["chart_energy"], calib_active_frac=L["calib_active_frac"],
                                       files=files))
        print(f"{rec['name']:<34}{L['kind']:<10}{L['n']:>8}{L['chart_energy']*100:>13.1f}%"
              f"{L['calib_active_frac']*100:>13.1f}%")
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Atlas written to {out} in {time.time() - t0:.1f}s")


# =============================================================================
# 4. ATLAS DICTIONARY, QUERY PLANNER, REFERENCE ROUTER
# =============================================================================

class ChartPlanner:
    """
    CPU query planner. Places each token on the atlas:
      Z      chart coordinates  (x - mu) @ P            [T, r]
      region Voronoi region in the chart                 [T]
      scale  how far the token sits outside its region's
             typical residual radius (>= 1)              [T]
    Region lookup is exact (C distances) or by Hilbert curve: the token's key
    is binary searched among region keys and only a window of neighbouring
    regions is compared exactly.
    """

    def __init__(self, A: dict, lookup: str = "exact", window: int = 8):
        self.mu, self.P, self.cent, self.rtyp = A["mu"], A["P"], A["cent"], A["rtyp"]
        self.lookup, self.window = lookup, window
        self.curve_keys, self.curve_regions = A["curve_keys"], A["curve_regions"]
        self.curve_lo, self.curve_hi = A["curve_lo"], A["curve_hi"]
        self.curve_bits, self.curve_dims = int(A["curve_bits"]), int(A["curve_dims"])

    def region_by_curve(self, Z: np.ndarray) -> np.ndarray:
        q = quantize(Z[:, : self.curve_dims], self.curve_lo, self.curve_hi, self.curve_bits)
        keys = hilbert_encode(q, self.curve_bits)
        pos = np.searchsorted(self.curve_keys, keys)
        C = len(self.curve_keys)
        out = np.empty(Z.shape[0], dtype=np.int64)
        for t in range(Z.shape[0]):
            cand = self.curve_regions[max(0, pos[t] - self.window):min(C, pos[t] + self.window)]
            out[t] = cand[((self.cent[cand] - Z[t]) ** 2).sum(1).argmin()]
        return out

    def plan(self, X: np.ndarray):
        D = X - self.mu
        Z = D @ self.P
        res = np.linalg.norm(D - Z @ self.P.T, axis=1)
        region = self.region_by_curve(Z) if self.lookup == "curve" else nearest_region(Z, self.cent)
        scale = np.maximum(res / self.rtyp[region], 1.0).astype(np.float32)
        return Z.astype(np.float32), region.astype(np.int32), scale


class AtlasLayer:
    """
    Routing test per token t and neuron j (region c = region(t)):
      chart score   s = base_j + Z_t . Kp_j
      spread        e = z * sig[c, j] * scale_t
      relu, relu_glu  route if s + e > 0
      gelu            route if s + e > gelu_floor
      silu_glu        route if max|silu([s-e, s+e])| * (|u_s| + e_u) * |W2_j| > kappa * tau
    """

    def __init__(self, meta: dict, arrays: dict, lookup: str = "exact", window: int = 8):
        self.meta, self.kind, self.A = meta, meta["kind"], arrays
        self.glu = "Wu" in arrays
        self.planner = ChartPlanner(arrays, lookup=lookup, window=window)

    def route(self, X: np.ndarray, z: float, kappa: float = 1.0, gelu_floor: float = -3.0):
        A = self.A
        Z, region, scale = self.planner.plan(X)
        s = A["base"] + Z @ A["Kp"].T
        e = z * A["sig"][region] * scale[:, None]
        if self.kind in ("relu", "relu_glu"):
            sel = (s + e) > 0.0
        elif self.kind == "gelu":
            sel = (s + e) > gelu_floor
        else:
            su = A["base_u"] + Z @ A["Kp_u"].T
            eu = z * A["sig_u"][region] * scale[:, None]
            sel = silu_maxabs(s - e, s + e) * (np.abs(su) + eu) * A["colnorm"] > kappa * float(A["tau"])
        return sel, (Z, region, scale)

    def dense(self, X: np.ndarray) -> np.ndarray:
        A = self.A
        up = (X @ A["Wu"].T + A["bu"]) if self.glu else None
        return activate(self.kind, X @ A["W1"].T + A["b1"], up) @ A["W2T"] + A["b2"]

    def forward(self, X: np.ndarray, z: float, kappa: float = 1.0, gelu_floor: float = -3.0):
        """Reference sparse path: only routed neurons contribute. Returns (Y, sel, active, plan)."""
        sel, plan = self.route(X, z, kappa, gelu_floor)
        return self.forward_mask(X, sel) + (plan,)

    def forward_mask(self, X: np.ndarray, sel: np.ndarray):
        """Evaluate only neurons flagged in sel [T, n]. Returns (Y, sel, active) where
        active [T, n] marks routed neurons with non-zero activation (W2T rows read)."""
        A = self.A
        out = np.empty((X.shape[0], A["W2T"].shape[1]), np.float32)
        active = np.zeros(sel.shape, bool)
        for t in range(X.shape[0]):
            idx = np.nonzero(sel[t])[0]
            pre = A["W1"][idx] @ X[t] + A["b1"][idx]
            up = (A["Wu"][idx] @ X[t] + A["bu"][idx]) if self.glu else None
            h = activate(self.kind, pre, up)
            active[t, idx[h != 0]] = True
            out[t] = h @ A["W2T"][idx] + A["b2"]
        return out, sel, active


class AtlasDictionary:
    def __init__(self, path: str, lookup: str = "exact", window: int = 8, mmap: bool = True):
        with open(os.path.join(path, "manifest.json")) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != ATLAS_FORMAT:
            raise SystemExit(f"{path}: unsupported atlas format {self.manifest.get('format')}")
        self.path = path
        self.layers: list[AtlasLayer] = []
        for meta in self.manifest["layers"]:
            arrays = {}
            for key, fn in meta["files"].items():
                a = np.load(os.path.join(path, fn), mmap_mode="r" if mmap else None, allow_pickle=False)
                arrays[key] = a if a.ndim else a[()]
            self.layers.append(AtlasLayer(meta, arrays, lookup=lookup, window=window))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i: int) -> AtlasLayer:
        return self.layers[i]


# =============================================================================
# 5. OPENCL ENGINE
# =============================================================================
#   CPU query planner  chart coordinates, region, residual scale
#   atlas_route_up     per (token, neuron): chart score + region ellipsoid; only
#                      survivors compute their full dot product; non-zero
#                      results are compacted into a per-token active list
#   atlas_down         per (token, output dim): sum over the active list only;
#                      W2 is stored neuron-major so skipped rows are never read

KIND_ID = {"relu": 0, "gelu": 1, "relu_glu": 2, "silu_glu": 3}

KERNELS = r"""
#define SILU_MIN_X (-1.2784645f)
#define SILU_MIN_Y (0.2784645f)

inline float silu_f(float v) { return v / (1.0f + exp(-v)); }

__kernel void atlas_route_up(
    const int T, const int n, const int d, const int r,
    __global const float* X, __global const float* Z,
    __global const int* region, __global const float* scale,
    __global const float* base, __global const float* Kp, __global const float* sig,
    __global const float* W1, __global const float* b1,
#if GLU
    __global const float* base_u, __global const float* Kp_u, __global const float* sig_u,
    __global const float* Wu, __global const float* bu, __global const float* colnorm,
#endif
    const float zf, const float thr,
    __global int* cnt, __global int* act_idx, __global float* act_val, __global int* nrouted)
{
    const int j = get_global_id(0);
    const int t = get_global_id(1);
    if (j >= n || t >= T) return;

    /* 1. chart score: exact projection inside the principal chart */
    const __global float* zt = Z + (size_t)t * r;
    const __global float* kp = Kp + (size_t)j * r;
    float s = base[j];
    for (int k = 0; k < r; k++) s += zt[k] * kp[k];

    /* 2. region ellipsoid spread along this neuron's direction */
    const size_t rc = (size_t)region[t] * n + j;
    const float e = zf * sig[rc] * scale[t];

#if KIND == 3
    const __global float* kpu = Kp_u + (size_t)j * r;
    float su = base_u[j];
    for (int k = 0; k < r; k++) su += zt[k] * kpu[k];
    const float eu = zf * sig_u[rc] * scale[t];
    const float lo = s - e, hi = s + e;
    float m = fmax(fabs(silu_f(lo)), fabs(silu_f(hi)));
    if (lo < SILU_MIN_X && hi > SILU_MIN_X) m = fmax(m, SILU_MIN_Y);
    const int pass = (m * (fabs(su) + eu) * colnorm[j]) > thr;
#else
    const int pass = (s + e) > thr;
#endif
    if (!pass) return;
    atomic_inc(&nrouted[t]);

    /* 3. exact evaluation of the routed neuron */
    const __global float* x = X + (size_t)t * d;
    const __global float* w = W1 + (size_t)j * d;
    float pre = b1[j];
    for (int k = 0; k < d; k++) pre += x[k] * w[k];

    float h;
#if KIND == 0
    h = fmax(pre, 0.0f);
#elif KIND == 1
    h = 0.5f * pre * (1.0f + tanh(0.7978845608f * (pre + 0.044715f * pre * pre * pre)));
#else
    h = (KIND == 2) ? fmax(pre, 0.0f) : silu_f(pre);
    if (h != 0.0f) {
        const __global float* wu = Wu + (size_t)j * d;
        float u = bu[j];
        for (int k = 0; k < d; k++) u += x[k] * wu[k];
        h *= u;
    }
#endif
    if (h != 0.0f) {
        const int slot = atomic_inc(&cnt[t]);
        act_idx[(size_t)t * n + slot] = j;
        act_val[(size_t)t * n + slot] = h;
    }
}

__kernel void atlas_down(
    const int T, const int n, const int dout,
    __global const int* cnt, __global const int* act_idx, __global const float* act_val,
    __global const float* W2T, __global const float* b2, __global float* Y)
{
    const int i = get_global_id(0);
    const int t = get_global_id(1);
    if (i >= dout || t >= T) return;
    const int c = cnt[t];
    const __global int* ix = act_idx + (size_t)t * n;
    const __global float* v = act_val + (size_t)t * n;
    float acc = b2[i];
    for (int k = 0; k < c; k++) acc += v[k] * W2T[(size_t)ix[k] * dout + i];
    Y[(size_t)t * dout + i] = acc;
}
"""


def _require_cl():
    try:
        import pyopencl as cl
    except ImportError as e:
        raise SystemExit("pyopencl is required for the opencl backend: pip install pyopencl") from e
    return cl


class _LayerCL:
    STATIC = ["base", "Kp", "sig", "W1", "b1", "W2T", "b2"]
    STATIC_GLU = ["base_u", "Kp_u", "sig_u", "Wu", "bu", "colnorm"]

    def __init__(self, cl, ctx, queue, layer: AtlasLayer, host_dictionary: bool):
        self.cl, self.ctx, self.queue, self.layer = cl, ctx, queue, layer
        A = layer.A
        self.n, self.d = A["W1"].shape
        self.dout = A["W2T"].shape[1]
        self.r = A["Kp"].shape[1]
        self.glu = layer.glu
        mf = cl.mem_flags
        self.buf = {}
        for key in self.STATIC + (self.STATIC_GLU if self.glu else []):
            arr = np.ascontiguousarray(A[key], dtype=np.float32)
            if host_dictionary:
                # Zero-copy: the dictionary stays in system RAM, the device reads it over PCIe
                if not arr.flags.writeable:
                    arr = arr.copy()
                self.buf[key] = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=arr)
            else:
                self.buf[key] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)
        opts = [f"-DKIND={KIND_ID[layer.kind]}", f"-DGLU={1 if self.glu else 0}"]
        prg = cl.Program(ctx, KERNELS).build(options=opts)
        self.k_up = cl.Kernel(prg, "atlas_route_up")
        self.k_down = cl.Kernel(prg, "atlas_down")

    def forward(self, X: np.ndarray, z: float, thr: float):
        cl, ctx, q = self.cl, self.ctx, self.queue
        mf = cl.mem_flags
        T = X.shape[0]
        Z, region, scale = self.layer.planner.plan(X)
        X = np.ascontiguousarray(X, dtype=np.float32)
        bX = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        bZ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Z))
        bR = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=region)
        bS = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=scale)
        cnt = np.zeros(T, np.int32)
        nrouted = np.zeros(T, np.int32)
        bC = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cnt)
        bN = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nrouted)
        bI = cl.Buffer(ctx, mf.READ_WRITE, size=T * self.n * 4)
        bV = cl.Buffer(ctx, mf.READ_WRITE, size=T * self.n * 4)
        bY = cl.Buffer(ctx, mf.WRITE_ONLY, size=T * self.dout * 4)
        B = self.buf
        args = [np.int32(T), np.int32(self.n), np.int32(self.d), np.int32(self.r), bX, bZ, bR, bS,
                B["base"], B["Kp"], B["sig"], B["W1"], B["b1"]]
        if self.glu:
            args += [B["base_u"], B["Kp_u"], B["sig_u"], B["Wu"], B["bu"], B["colnorm"]]
        args += [np.float32(z), np.float32(thr), bC, bI, bV, bN]
        self.k_up.set_args(*args)
        cl.enqueue_nd_range_kernel(q, self.k_up, (self.n, T), None)
        self.k_down.set_args(np.int32(T), np.int32(self.n), np.int32(self.dout), bC, bI, bV, B["W2T"], B["b2"], bY)
        cl.enqueue_nd_range_kernel(q, self.k_down, (self.dout, T), None)
        Y = np.empty((T, self.dout), np.float32)
        cl.enqueue_copy(q, Y, bY)
        cl.enqueue_copy(q, cnt, bC)
        cl.enqueue_copy(q, nrouted, bN)
        q.finish()
        return Y, nrouted, cnt


class AtlasCLEngine:
    """Layers are spread round-robin over the available OpenCL devices (GPUs first)."""

    def __init__(self, dictionary: AtlasDictionary, max_devices: int | None = None, host_dictionary: bool = False):
        cl = _require_cl()
        gpus, others = [], []
        for p in cl.get_platforms():
            for d in p.get_devices():
                (gpus if d.type & cl.device_type.GPU else others).append(d)
        devs = gpus or others
        if not devs:
            raise SystemExit("No OpenCL devices found")
        self.devices = devs[:max_devices] if max_devices else devs
        ctxs = [cl.Context([d]) for d in self.devices]
        queues = [cl.CommandQueue(c) for c in ctxs]
        self.layers = []
        for i, layer in enumerate(dictionary.layers):
            k = i % len(self.devices)
            self.layers.append(_LayerCL(cl, ctxs[k], queues[k], layer, host_dictionary))
        names = ", ".join(d.name.strip() for d in self.devices)
        print(f"Atlas OpenCL engine: {len(self.layers)} layers on {len(self.devices)} device(s): {names}")

    def forward(self, i: int, X: np.ndarray, z: float, kappa: float = 1.0, gelu_floor: float = -3.0):
        L = self.layers[i]
        kind = L.layer.kind
        if kind in ("relu", "relu_glu"):
            thr = 0.0
        elif kind == "gelu":
            thr = gelu_floor
        else:
            thr = kappa * float(L.layer.A["tau"])
        return L.forward(X, z, thr)


# =============================================================================
# 6. HUGGING FACE INTEGRATION
# =============================================================================

class AtlasRuntime:
    """Executes one atlas layer on the selected backend and keeps routing statistics."""

    def __init__(self, dictionary: AtlasDictionary, backend: str = "reference", engine=None, z: float = 3.0,
                 kappa: float = 1.0, gelu_floor: float = -3.0, static_frac: float = 0.5):
        self.dict, self.backend, self.engine = dictionary, backend, engine
        self.z, self.kappa, self.gelu_floor, self.static_frac = z, kappa, gelu_floor, static_frac
        self.capture_masks = False
        self.reset()

    def reset(self):
        self.routed = self.nonzero = self.slots = 0
        self.masks: list[tuple[int, np.ndarray, np.ndarray]] = []

    @property
    def routed_frac(self):
        return self.routed / max(1, self.slots)

    @property
    def nonzero_frac(self):
        return self.nonzero / max(1, self.slots)

    def run(self, i: int, X: np.ndarray) -> np.ndarray:
        layer = self.dict[i]
        n = layer.A["W1"].shape[0]
        if self.backend == "dense":
            Y = layer.dense(X)
            routed = nnz = np.full(X.shape[0], n)
        elif self.backend == "reference":
            Y, sel, active, _ = layer.forward(X, self.z, self.kappa, self.gelu_floor)
            routed, nnz = sel.sum(1), active.sum(1)
            if self.capture_masks:
                self.masks.append((i, sel, active))
        elif self.backend == "static":
            keep = np.zeros(n, bool)
            keep[np.argsort(-np.asarray(layer.A["importance"]))[:max(1, int(round(self.static_frac * n)))]] = True
            Y, sel, active = layer.forward_mask(X, np.broadcast_to(keep, (X.shape[0], n)))
            routed, nnz = sel.sum(1), active.sum(1)
        elif self.backend == "opencl":
            Y, routed, nnz = self.engine.forward(i, X, self.z, self.kappa, self.gelu_floor)
        else:
            raise ValueError(self.backend)
        self.routed += int(np.sum(routed))
        self.nonzero += int(np.sum(nnz))
        self.slots += X.shape[0] * n
        return Y


class AtlasMLP(nn.Module):
    def __init__(self, index: int, runtime: AtlasRuntime):
        super().__init__()
        self.index, self.runtime = index, runtime

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        shp = hidden_states.shape
        X = hidden_states.detach().reshape(-1, shp[-1]).float().cpu().numpy()
        Y = self.runtime.run(self.index, np.ascontiguousarray(X))
        return torch.from_numpy(Y).reshape(*shp[:-1], Y.shape[-1]).to(hidden_states.dtype)


def inject_atlas(model, runtime: AtlasRuntime) -> int:
    mlps = find_mlps(model)
    layers = runtime.dict.manifest["layers"]
    if len(mlps) != len(layers):
        raise SystemExit(f"Model has {len(mlps)} MLPs, atlas has {len(layers)}")
    for i, (rec, meta) in enumerate(zip(mlps, layers)):
        if rec["name"] != meta["module"]:
            raise SystemExit(f"Layer {i}: model module {rec['name']} != atlas module {meta['module']}")
        owner = rec["owner"]
        if hasattr(owner, "fc1") and hasattr(owner, "fc2"):
            # OPT: fc1 -> activation_fn -> fc2 are called inline by the decoder layer
            owner.fc1 = AtlasMLP(i, runtime)
            owner.activation_fn = nn.Identity()
            owner.fc2 = nn.Identity()
        else:
            parent_name, attr = rec["name"].rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attr, AtlasMLP(i, runtime))
    return len(mlps)


def attach_atlas(model, args, backend: str | None = None, z: float = 3.5):
    """Load the atlas named in args, build the backend and swap the model's MLPs."""
    backend = backend or args.backend
    dic = AtlasDictionary(args.atlas, lookup=args.lookup)
    engine = AtlasCLEngine(dic, max_devices=args.gpus, host_dictionary=args.host_dictionary) \
        if backend == "opencl" else None
    rt = AtlasRuntime(dic, backend=backend, engine=engine, z=z, kappa=args.kappa)
    return dic, rt, inject_atlas(model, rt)


# =============================================================================
# 7. DATA
# =============================================================================

def cmd_data(args):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("The data subcommand needs datasets: pip install datasets")
    os.makedirs(args.out_dir, exist_ok=True)
    for split, fn, n in [("train", "wiki_calib.txt", args.calib_rows), ("test", "wiki_test.txt", None)]:
        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
        rows = rows if n is None else rows[:n]
        path = os.path.join(args.out_dir, fn)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(r for r in rows if r.strip()))
        print(f"{path}: {os.path.getsize(path):,} bytes")


# =============================================================================
# 8. BENCH
# =============================================================================

def _eval_windows(model, windows):
    logits, nll = [], []
    with torch.no_grad():
        for w in windows:
            lg = model(w).logits[0, :-1].float()
            logits.append(lg)
            nll.append(F.cross_entropy(lg, w[0, 1:], reduction="none"))
    return torch.cat(logits), torch.cat(nll)


def _compare(dense_logits, logits, nll):
    lp_d = F.log_softmax(dense_logits, -1)
    lp_r = F.log_softmax(logits, -1)
    kl = (lp_d.exp() * (lp_d - lp_r)).sum(-1).mean().item()
    top1 = (dense_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
    return float(torch.exp(nll.mean())), top1, kl


def _mlp_work(dic: AtlasDictionary, routed_frac: float, nonzero_frac: float) -> float:
    """Multiply-add count relative to dense MLP, router (chart + scores) included."""
    num = den = 0.0
    for L in dic.layers:
        n, d = L.A["W1"].shape
        dout = L.A["W2T"].shape[1]
        r = L.A["Kp"].shape[1]
        glu = 2 if L.glu else 1
        num += d * r + n * r * glu + routed_frac * n * d * glu + nonzero_frac * n * dout
        den += n * d * glu + n * dout
    return num / den


def _block_locality(dic: AtlasDictionary, masks, block_bytes: int) -> dict:
    """Mean DMA blocks touched per token and layer: W1 rows of routed neurons and
    W2T rows of active neurons, in Hilbert layout vs original neuron order."""
    acc = {"w1_h": [], "w1_o": [], "w2_h": [], "w2_o": []}
    for i, sel, active in masks:
        L = dic[i]
        row1, row2 = L.A["W1"].shape[1] * 4, L.A["W2T"].shape[1] * 4
        perm = np.asarray(L.A["perm"])
        for t in range(sel.shape[0]):
            j, a = np.nonzero(sel[t])[0], np.nonzero(active[t])[0]
            acc["w1_h"].append(np.unique(j * row1 // block_bytes).size)
            acc["w1_o"].append(np.unique(perm[j] * row1 // block_bytes).size)
            acc["w2_h"].append(np.unique(a * row2 // block_bytes).size)
            acc["w2_o"].append(np.unique(perm[a] * row2 // block_bytes).size)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cmd_bench(args):
    """
    Per configuration: perplexity, top-1 agreement and KL(dense || routed) vs the
    dense model, routed and non-zero neuron fractions, MLP work (router included)
    and DMA blocks touched per token (Hilbert layout vs original order). Static
    rows keep the globally most important neurons for every token, the best
    that context-free weight pruning can do at that fraction.
    """
    tok, model = load_model(args.model_id, args.trust_remote_code)
    ids = tok(open(args.eval, encoding="utf-8").read()[: args.windows * args.window_len * 12],
              return_tensors="pt").input_ids
    windows = [ids[:, k * args.window_len:(k + 1) * args.window_len] for k in range(args.windows)]
    dense_logits, dense_nll = _eval_windows(model, windows)
    rows = [dict(config="dense (HF)", ppl=float(torch.exp(dense_nll.mean())), top1=1.0, kl=0.0,
                 routed=1.0, nonzero=float("nan"), work=1.0, blocks=None)]

    dic, rt, _ = attach_atlas(model, args, backend="dense")
    engine = AtlasCLEngine(dic, max_devices=args.gpus, host_dictionary=args.host_dictionary) \
        if args.backend == "opencl" else None
    rt.engine = engine

    def run(label, **kw):
        for k, v in kw.items():
            setattr(rt, k, v)
        rt.reset()
        t0 = time.time()
        lg, nll = _eval_windows(model, windows)
        ppl, top1, kl = _compare(dense_logits, lg, nll)
        row = dict(config=label, ppl=ppl, top1=top1, kl=kl, routed=rt.routed_frac, nonzero=rt.nonzero_frac,
                   work=_mlp_work(dic, rt.routed_frac, rt.nonzero_frac) if rt.backend != "dense" else 1.0,
                   blocks=None, seconds=time.time() - t0)
        if rt.capture_masks and rt.masks:
            row["blocks"] = _block_locality(dic, rt.masks, args.block_kb * 1024)
        rows.append(row)
        print(f"{label:<22} ppl {ppl:8.3f}  top1 {top1*100:6.2f}%  KL {kl:.4f}  routed {row['routed']*100:5.1f}%"
              f"  work {row['work']*100:5.1f}%  ({row['seconds']:.0f}s)", flush=True)

    run("atlas dense layout", backend="dense")
    for z in [float(v) for v in args.z.split(",") if v]:
        run(f"geometric z={z}", backend=args.backend, z=z, capture_masks=args.backend == "reference")
    rt.capture_masks = False
    for f in [float(v) for v in args.static.split(",") if v]:
        run(f"static {int(f*100)}%", backend="static", static_frac=f)

    kb = args.block_kb
    print(f"\n| Config | PPL | Top-1 vs dense | KL | Routed | Non-zero | MLP work "
          f"| W1 {kb}K blocks (Hilbert / orig) | W2T {kb}K blocks (Hilbert / orig) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b = r["blocks"]
        w1 = f"{b['w1_h']:.1f} / {b['w1_o']:.1f}" if b else ""
        w2 = f"{b['w2_h']:.1f} / {b['w2_o']:.1f}" if b else ""
        nz = "" if np.isnan(r["nonzero"]) else f"{r['nonzero']*100:.1f}%"
        print(f"| {r['config']} | {r['ppl']:.2f} | {r['top1']*100:.2f}% | {r['kl']:.4f} | {r['routed']*100:.1f}% "
              f"| {nz} | {r['work']*100:.1f}% | {w1} | {w2} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(model=args.model_id, atlas=args.atlas, backend=args.backend, lookup=args.lookup,
                           tokens=args.windows * args.window_len, rows=rows), f, indent=1)


# =============================================================================
# 9. GENERATE
# =============================================================================

def _generate(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    ids = tok(prompt, return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True), time.time() - t0


def cmd_generate(args):
    tok, model = load_model(args.model_id, args.trust_remote_code)
    if args.compare:
        text, dt = _generate(model, tok, args.prompt, args.max_new_tokens)
        print(f"--- [Dense] ({dt:.1f}s) ---\n{text}\n")
    _, rt, n = attach_atlas(model, args, z=args.z)
    text, dt = _generate(model, tok, args.prompt, args.max_new_tokens)
    print(f"--- [Atlas: {args.backend}, z={args.z}, lookup={args.lookup}, {n} MLP layers] ({dt:.1f}s) ---\n{text}\n")
    print(f"Routed neurons  : {rt.routed_frac * 100:.1f}% of (token, neuron) pairs")
    print(f"Non-zero neurons: {rt.nonzero_frac * 100:.1f}%")


# =============================================================================
# 10. SELFTEST
# =============================================================================

def cmd_selftest(args):
    """OpenCL kernels vs the NumPy reference on synthetic inputs around each layer's chart mean."""
    dic = AtlasDictionary(args.atlas, lookup=args.lookup)
    engine = AtlasCLEngine(dic, max_devices=args.gpus, host_dictionary=args.host_dictionary)
    rng = np.random.default_rng(0)
    idx = [int(v) for v in args.layers.split(",")] if args.layers else range(len(dic))
    ok = True
    print(f"{'layer':>5} {'kind':<10}{'max rel diff':>14}{'routed match':>14}{'non-zero match':>16}")
    for i in idx:
        L = dic[i]
        mu = np.asarray(L.A["mu"])
        X = (mu + rng.standard_normal((args.tokens, mu.size)).astype(np.float32) * args.noise).astype(np.float32)
        yr, sel, act, _ = L.forward(X, args.z, args.kappa)
        yc, nr, nz = engine.forward(i, X, args.z, args.kappa)
        rel = float(np.abs(yr - yc).max() / (np.abs(yr).max() + 1e-12))
        rm, nm = bool((sel.sum(1) == nr).all()), bool((act.sum(1) == nz).all())
        ok &= rm and nm and rel < 1e-4
        print(f"{i:>5} {L.kind:<10}{rel:>14.2e}{str(rm):>14}{str(nm):>16}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# =============================================================================
# 11. OPENAI-COMPATIBLE API SERVER
# =============================================================================
#   GET  /health                    no auth
#   GET  /v1/models[/{id}]
#   POST /v1/completions            single string prompt, echo
#   POST /v1/chat/completions       tokenizer chat template, or a User/Assistant
#                                   transcript with automatic turn stop
#   stream: true                    Server-Sent Events, data: [DONE]
# Request extensions: atlas_z, atlas_kappa. Response extension: usage.atlas.
# Generation is serialised by a lock; concurrent requests queue.

class ApiError(Exception):
    def __init__(self, status: int, message: str, etype: str = "invalid_request_error", param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.etype, self.param = status, message, etype, param


class _StopOnStrings(StoppingCriteria):
    def __init__(self, tok, prompt_len: int, stops: list[str]):
        self.tok, self.prompt_len, self.stops = tok, prompt_len, stops

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self.stops:
            return False
        text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        return any(s in text for s in self.stops)


def _cut_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    idx = [text.find(s) for s in stops if s and s in text]
    return (text[: min(idx)], True) if idx else (text, False)


class AtlasService:
    def __init__(self, args):
        self.args = args
        self.name = args.served_name or args.model_id.rstrip("/").split("/")[-1] + "-atlas"
        self.created = int(time.time())
        self.tok, self.model = load_model(args.model_id, args.trust_remote_code)
        _, self.rt, self.n_layers = attach_atlas(self.model, args, z=args.z)
        self.lock = threading.Lock()
        self.ctx_len = getattr(self.model.config, "max_position_embeddings", 2048)
        print(f"Serving '{self.name}' ({args.model_id}, {self.n_layers} atlas MLP layers, backend {args.backend})")

    def chat_prompt(self, messages: list[dict]) -> tuple[str, list[str]]:
        """Returns the prompt and any stop strings the prompt format implies."""
        if not isinstance(messages, list) or not messages:
            raise ApiError(400, "'messages' must be a non-empty array", param="messages")
        norm = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):  # content parts: keep the text parts
                content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
            norm.append({"role": m.get("role", "user"), "content": content or ""})
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=True), []
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in norm]
        return "\n".join(lines) + "\nAssistant:", ["\nUser:", "\nSystem:"]

    @staticmethod
    def _stops(body: dict, extra) -> list[str]:
        s = body.get("stop")
        user = [] if s is None else ([s] if isinstance(s, str) else [x for x in s if x][:4])
        return user + list(extra)

    def _encode(self, prompt: str):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        if ids.shape[1] >= self.ctx_len:
            raise ApiError(400, f"Prompt has {ids.shape[1]} tokens; context length is {self.ctx_len}",
                           param="messages", etype="context_length_exceeded")
        return ids

    def _gen_kwargs(self, body: dict, prompt_len: int, stops: list[str]) -> dict:
        max_tokens = int(body.get("max_completion_tokens", body.get("max_tokens")) or self.args.max_tokens_default)
        if max_tokens < 1:
            raise ApiError(400, "max_tokens must be >= 1", param="max_tokens")
        max_tokens = min(max_tokens, max(1, self.ctx_len - prompt_len))
        temperature = 1.0 if body.get("temperature") is None else float(body["temperature"])
        top_p = 1.0 if body.get("top_p") is None else float(body["top_p"])
        kw = dict(max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                  stopping_criteria=StoppingCriteriaList([_StopOnStrings(self.tok, prompt_len, stops)]))
        if temperature <= 0:
            kw["do_sample"] = False
        else:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        if body.get("seed") is not None:
            torch.manual_seed(int(body["seed"]))
        return kw

    def _configure(self, body: dict):
        self.rt.z = float(body.get("atlas_z", self.args.z))
        self.rt.kappa = float(body.get("atlas_kappa", self.args.kappa))
        self.rt.reset()

    def _finish(self, ids, out, kw, stopped: bool) -> tuple[str, dict]:
        gen = out[0, ids.shape[1]:]
        eos = self.tok.eos_token_id
        hit_eos = gen.numel() > 0 and eos is not None and int(gen[-1]) == eos
        finish = "stop" if (stopped or hit_eos or gen.numel() < kw["max_new_tokens"]) else "length"
        usage = dict(prompt_tokens=int(ids.shape[1]), completion_tokens=int(gen.numel()),
                     total_tokens=int(ids.shape[1] + gen.numel()),
                     atlas=dict(z=self.rt.z, routed_frac=round(self.rt.routed_frac, 4),
                                nonzero_frac=round(self.rt.nonzero_frac, 4)))
        return finish, usage

    def complete(self, prompt: str, body: dict, extra_stops=()) -> dict:
        stops = self._stops(body, extra_stops)
        ids = self._encode(prompt)
        with self.lock:
            self._configure(body)
            kw = self._gen_kwargs(body, ids.shape[1], stops)
            with torch.no_grad():
                out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kw)
            text, stopped = _cut_at_stop(self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), stops)
            finish, usage = self._finish(ids, out, kw, stopped)
        return dict(text=text, finish_reason=finish, usage=usage)

    def stream(self, prompt: str, body: dict, extra_stops=()):
        """Validates up front (errors return before the event stream opens), then yields
        text deltas; the final item is a dict with finish_reason and usage."""
        stops = self._stops(body, extra_stops)
        ids = self._encode(prompt)
        self._gen_kwargs(body, ids.shape[1], stops)
        return self._stream(ids, body, stops)

    def _stream(self, ids, body: dict, stops: list[str]):
        hold = max((len(s) for s in stops), default=0)
        with self.lock:
            self._configure(body)
            kw = self._gen_kwargs(body, ids.shape[1], stops)
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            result = {}

            def run():
                with torch.no_grad():
                    result["out"] = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                                                        streamer=streamer, **kw)

            th = threading.Thread(target=run, daemon=True)
            th.start()
            buf, sent, stopped = "", 0, False
            for piece in streamer:
                if stopped:
                    continue
                buf += piece
                cut, stopped = _cut_at_stop(buf, stops)
                safe = len(cut) if stopped else max(sent, len(buf) - hold)
                if safe > sent:
                    yield cut[sent:safe] if stopped else buf[sent:safe]
                    sent = safe
            th.join()
            if not stopped and len(buf) > sent:
                yield buf[sent:]
            finish, usage = self._finish(ids, result["out"], kw, stopped)
            yield dict(finish_reason=finish, usage=usage)


def _make_handler(svc: AtlasService, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "HoloqubedAtlas/1"

        def log_message(self, fmt, *a):
            print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % a}")

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, status: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _error(self, e: ApiError):
            self._json(e.status, {"error": {"message": e.message, "type": e.etype, "param": e.param,
                                            "code": e.etype if e.status != 401 else "invalid_api_key"}})

        def _auth(self):
            if api_key and self.headers.get("Authorization", "") != f"Bearer {api_key}":
                raise ApiError(401, "Invalid API key", etype="authentication_error")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                raise ApiError(400, "Request body is not valid JSON")
            if not isinstance(body, dict):
                raise ApiError(400, "Request body must be a JSON object")
            if body.get("n", 1) != 1:
                raise ApiError(400, "Only n=1 is supported", param="n")
            model = body.get("model")
            if model and model != svc.name:
                raise ApiError(404, f"The model '{model}' does not exist; this server serves '{svc.name}'",
                               param="model", etype="model_not_found")
            return body

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj):
            payload = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()

        def _model_obj(self):
            return {"id": svc.name, "object": "model", "created": svc.created, "owned_by": "thereminq"}

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            try:
                path = self.path.split("?")[0].rstrip("/")
                if path == "/health":
                    return self._json(200, {"status": "ok", "model": svc.name})
                self._auth()
                if path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [self._model_obj()]})
                if path == f"/v1/models/{svc.name}":
                    return self._json(200, self._model_obj())
                raise ApiError(404, f"Unknown path {self.path}", etype="not_found")
            except ApiError as e:
                self._error(e)

        def do_POST(self):
            try:
                path = self.path.split("?")[0].rstrip("/")
                self._auth()
                body = self._body()
                if path == "/v1/chat/completions":
                    prompt, fmt_stops = svc.chat_prompt(body.get("messages"))
                    return self._chat(prompt, body, fmt_stops)
                if path == "/v1/completions":
                    prompt = body.get("prompt", "")
                    if isinstance(prompt, list):
                        if len(prompt) != 1 or not isinstance(prompt[0], str):
                            raise ApiError(400, "Only a single string prompt is supported", param="prompt")
                        prompt = prompt[0]
                    if not isinstance(prompt, str) or not prompt:
                        raise ApiError(400, "'prompt' must be a non-empty string", param="prompt")
                    return self._completion(prompt, body)
                raise ApiError(404, f"Unknown path {self.path}", etype="not_found")
            except ApiError as e:
                self._error(e)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # keep the server alive, report as OpenAI style error
                self._error(ApiError(500, f"{type(e).__name__}: {e}", etype="server_error"))

        def _chat(self, prompt: str, body: dict, fmt_stops: list[str]):
            cid, now = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
            if not body.get("stream"):
                r = svc.complete(prompt, body, fmt_stops)
                return self._json(200, {
                    "id": cid, "object": "chat.completion", "created": now, "model": svc.name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": r["text"]},
                                 "logprobs": None, "finish_reason": r["finish_reason"]}],
                    "usage": r["usage"]})
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            gen = svc.stream(prompt, body, fmt_stops)
            self._sse_start()
            base = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": svc.name}
            self._sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                            "logprobs": None, "finish_reason": None}]})
            for item in gen:
                if isinstance(item, str):
                    self._sse({**base, "choices": [{"index": 0, "delta": {"content": item},
                                                    "logprobs": None, "finish_reason": None}]})
                else:
                    self._sse({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None,
                                                    "finish_reason": item["finish_reason"]}]})
                    if include_usage:
                        self._sse({**base, "choices": [], "usage": item["usage"]})
            self._sse("[DONE]")

        def _completion(self, prompt: str, body: dict):
            cid, now = f"cmpl-{uuid.uuid4().hex}", int(time.time())
            echo = prompt if body.get("echo") else ""
            if not body.get("stream"):
                r = svc.complete(prompt, body)
                return self._json(200, {
                    "id": cid, "object": "text_completion", "created": now, "model": svc.name,
                    "choices": [{"index": 0, "text": echo + r["text"], "logprobs": None,
                                 "finish_reason": r["finish_reason"]}],
                    "usage": r["usage"]})
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            gen = svc.stream(prompt, body)
            self._sse_start()
            base = {"id": cid, "object": "text_completion", "created": now, "model": svc.name}
            if echo:
                self._sse({**base, "choices": [{"index": 0, "text": echo, "logprobs": None, "finish_reason": None}]})
            for item in gen:
                if isinstance(item, str):
                    self._sse({**base, "choices": [{"index": 0, "text": item, "logprobs": None,
                                                    "finish_reason": None}]})
                else:
                    self._sse({**base, "choices": [{"index": 0, "text": "", "logprobs": None,
                                                    "finish_reason": item["finish_reason"]}]})
                    if include_usage:
                        self._sse({**base, "choices": [], "usage": item["usage"]})
            self._sse("[DONE]")

    return Handler


def cmd_serve(args):
    svc = AtlasService(args)
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(svc, args.api_key))
    server.daemon_threads = True
    print(f"OpenAI-compatible API on http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# =============================================================================
# 12. CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="holoqubed_atlas.py",
                                 description="ThereminQ Holoqubed geometric Atlas engine",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Run '<subcommand> -h' for the options of each subcommand.")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="subcommand")

    model_p = argparse.ArgumentParser(add_help=False)
    model_p.add_argument("--model_id", required=True, help="Hugging Face model id or local path")
    model_p.add_argument("--trust_remote_code", action="store_true")

    atlas_p = argparse.ArgumentParser(add_help=False)
    atlas_p.add_argument("--atlas", required=True, help=".atlas directory from 'forge'")
    atlas_p.add_argument("--kappa", type=float, default=1.0, help="Tolerance for SiLU gated units")
    atlas_p.add_argument("--lookup", choices=["exact", "curve"], default="exact",
                         help="Region lookup: exact distances or Hilbert curve binary search")
    atlas_p.add_argument("--gpus", type=int, default=None, help="Max OpenCL devices")
    atlas_p.add_argument("--host_dictionary", action="store_true",
                         help="Keep the atlas in system RAM (zero-copy) instead of uploading to VRAM")

    p = sub.add_parser("data", help="Fetch WikiText-2 calibration and evaluation text")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--calib_rows", type=int, default=4000, help="Rows of the train split used for calibration")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("forge", parents=[model_p], help="Build a .atlas routing dictionary")
    p.add_argument("--calib", required=True, help="Plain text file used for calibration")
    p.add_argument("--out", default=None, help="Output directory (default <model>.atlas)")
    p.add_argument("--calib_tokens", type=int, default=8192)
    p.add_argument("--max_calib_rows", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--rank", type=int, default=128, help="Chart rank r")
    p.add_argument("--regions", type=int, default=64, help="Voronoi regions C")
    p.add_argument("--kmeans_iters", type=int, default=25)
    p.add_argument("--curve_dims", type=int, default=4, help="Leading chart axes used by the Hilbert curve")
    p.add_argument("--curve_bits", type=int, default=12)
    p.set_defaults(func=cmd_forge)

    p = sub.add_parser("bench", parents=[model_p, atlas_p], help="Divergence harness vs dense and static pruning")
    p.add_argument("--eval", required=True, help="Plain text file for evaluation")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--window_len", type=int, default=256)
    p.add_argument("--backend", choices=["reference", "opencl"], default="reference")
    p.add_argument("--z", type=str, default="2.0,2.5,3.0,3.5,4.0", help="Comma separated z values")
    p.add_argument("--static", type=str, default="0.3,0.5", help="Comma separated static fractions")
    p.add_argument("--block_kb", type=int, default=64, help="DMA block size for the locality metric")
    p.add_argument("--json", type=str, default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("generate", parents=[model_p, atlas_p], help="Greedy generation through the atlas")
    p.add_argument("--prompt", default="there is no spoon")
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--backend", choices=["opencl", "reference", "dense"], default="opencl")
    p.add_argument("--z", type=float, default=3.5, help="Ellipsoid width in sigmas (higher = safer, more work)")
    p.add_argument("--compare", action="store_true", help="Also run the dense model")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("serve", parents=[model_p, atlas_p], help="OpenAI-compatible REST API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api_key", default=None, help="Require 'Authorization: Bearer <key>' when set")
    p.add_argument("--served_name", default=None, help="Model id reported by the API (default <model>-atlas)")
    p.add_argument("--backend", choices=["opencl", "reference", "dense"], default="opencl")
    p.add_argument("--z", type=float, default=3.5)
    p.add_argument("--max_tokens_default", type=int, default=256)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("selftest", parents=[atlas_p], help="OpenCL kernels vs NumPy reference")
    p.add_argument("--layers", default="", help="Comma separated layer indices (default all)")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--z", type=float, default=3.0)
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
