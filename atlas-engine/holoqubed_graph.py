#!/usr/bin/env python3
"""
ThereminQ Holoqubed - Geometry and Topology Graphs
Python-native (matplotlib) figures of the geometric Atlas engine and its PyQrack circuits.
Every figure is drawn from real data: the .atlas arrays, the engine's own gate lists and,
optionally, real token activations.

Figures
  geometry   per MLP layer: chart plane with Voronoi regions, the Hilbert curve through the
             regions and real tokens; neuron keys coloured by Hilbert layout position; the
             residual-ellipsoid spread matrix (regions x neurons)
  model      layer stack: attention and MLP blocks, tile counts, chart energy, Loom links,
             the layer run in PyQrack
  tile       the 4x4 qubit tile circuit: 15 RY + 14 CNOT preparation, query, amplitude readout
             (executed in PyQrack when available and checked against NumPy)
  qneuron    qneuron circuit (cosine qubits + small-angle up qubit) and SiLU reconstruction
  patches    QrackBridge patch topology for landscape, landscape --signed and qneuron, with the
             seam fidelity vs bond budget of a real chart-key tile

Usage
  python3 holoqubed_graph.py --atlas opt-125m.atlas --layers 0,5,11 \
      --model_id facebook/opt-125m --text wiki_test.txt --out graphs
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):  # otherwise PyQrack falls back to its bundled library
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holoqubed_atlas as H  # noqa: E402

INK = "#1f2430"
MUTED = "#6b7280"
ACCENT = "#7c3aed"
QRACK = "#0e9f6e"
LOOM = "#d97706"
SEAM = "#dc2626"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, args, name):
    path = os.path.join(args.out, f"{name}.{args.format}")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


# =============================================================================
# 1. GEOMETRY
# =============================================================================

def fig_geometry(args, dic, li, Z_tokens=None):
    L = dic[li]
    A = L.A
    meta = L.meta
    cent = np.asarray(A["cent"], np.float64)
    order = np.asarray(A["curve_regions"])
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    Kp = np.asarray(A["Kp"], np.float64)
    imp = np.asarray(A["importance"], np.float64)
    sig = np.asarray(A["sig"], np.float64)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.25]))
    fig.suptitle(f"Atlas geometry, layer {li} ({meta['kind']}, {meta['n']} neurons, chart rank {meta['rank']}, "
                 f"{meta['regions']} regions, chart energy {meta['chart_energy']*100:.1f}%)", fontweight="bold", y=1.02)

    # --- A: chart plane, Voronoi regions, Hilbert curve, tokens
    ax = axs[0]
    pts = cent[:, :2] if Z_tokens is None else np.vstack([cent[:, :2], Z_tokens[:, :2]])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    lab = ((grid[:, None, :] - cent[None, :, :2]) ** 2).sum(-1).argmin(1)
    ax.imshow(rank[lab].reshape(gx.shape), origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
              cmap="twilight", alpha=0.35, aspect="auto", interpolation="nearest")
    if Z_tokens is not None:
        ax.scatter(Z_tokens[:, 0], Z_tokens[:, 1], s=3, c=INK, alpha=0.25, lw=0, label=f"{len(Z_tokens)} real tokens")
    path = cent[order, :2]
    for i in range(len(path) - 1):
        ax.plot(path[i:i + 2, 0], path[i:i + 2, 1], color=plt.cm.plasma(i / max(1, len(path) - 2)), lw=1.6)
    ax.scatter(cent[:, 0], cent[:, 1], s=26, c=rank, cmap="plasma", edgecolor="white", lw=0.6, zorder=3,
               label="region centroids (Hilbert order)")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("chart axis 0"); ax.set_ylabel("chart axis 1")
    ax.set_title("Chart plane: Voronoi regions and Hilbert region curve")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # --- B: neuron keys coloured by Hilbert layout position
    ax = axs[1]
    n = Kp.shape[0]
    s = 4 + 40 * (imp / (imp.max() + 1e-12)) ** 0.5
    sc = ax.scatter(Kp[:, 0], Kp[:, 1], s=s, c=np.arange(n), cmap="viridis", alpha=0.7, lw=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("stored row (Hilbert neuron layout)")
    ax.set_xlabel("neuron key on chart axis 0"); ax.set_ylabel("neuron key on chart axis 1")
    ax.set_title("Neuron keys Kp = W1 P (size = calibration importance)")

    # --- C: residual ellipsoid spread, regions (curve order) x neurons (layout order)
    ax = axs[2]
    S_ = sig[order]
    step = max(1, n // 768)
    im = ax.imshow(S_[:, ::step] + 1e-6, aspect="auto", cmap="magma", interpolation="nearest",
                   norm=LogNorm(vmin=max(np.percentile(S_, 1), 1e-6), vmax=S_.max()))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("residual spread sigma (log)")
    ax.set_xlabel(f"neuron (stored row{'' if step == 1 else f', every {step}th'})")
    ax.set_ylabel("region (Hilbert curve order)")
    ax.set_title("Residual ellipsoid spread per region and neuron")
    fig.tight_layout()
    return save(fig, args, f"geometry_layer{li:02d}")


# =============================================================================
# 2. MODEL TOPOLOGY
# =============================================================================

def attention_tiles(cfg, d):
    if cfg is None:
        return 4 * d * d // 16, "4 d^2 (MHA assumed)"
    hq = cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or d // hq
    hkv = getattr(cfg, "num_key_value_heads", None) or hq
    return (d * hq * hd + 2 * d * hkv * hd + hq * hd * d) // 16, f"q {hq}x{hd}, kv {hkv}x{hd}"


def fig_model(args, dic, cfg):
    layers = dic.manifest["layers"]
    nL = len(layers)
    d = layers[0]["d_in"]
    att_t, att_note = attention_tiles(cfg, d)
    rows = [(m["index"], m["kind"], m["n"], m["chart_energy"],
             (m["n"] * m["d_in"] * (3 if "glu" in m["kind"] else 2)) // 16) for m in layers]
    lm_t = (cfg.vocab_size * d // 16) if cfg is not None else None
    h = 0.42 * nL + 2.4
    fig, ax = plt.subplots(figsize=(13, h))
    ax.set_xlim(0, 13); ax.set_ylim(-1.6, nL + 0.9); ax.axis("off")
    model_id = dic.manifest.get("model_id", "")
    ax.text(0, nL + 0.55, f"Model topology: {model_id}, {nL} layers, hidden {d}", fontsize=12, fontweight="bold")
    ax.text(0, nL + 0.15, "each linear map = 4x4 qubit tiles (4 qubits, 15 RY + 14 CNOT to prepare); MLP gating = "
            "qneurons; norms, RoPE, softmax and residuals = classical Loom links", fontsize=8, color=MUTED)
    for (i, kind, n, ce, mlp_t) in rows:
        y = nL - 1 - i
        q = (i == args.qrack_layer)
        col = QRACK if q else ACCENT
        ax.add_patch(FancyBboxPatch((0.6, y + 0.08), 3.2, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, alpha=0.16 if not q else 0.32, ec=col, lw=1.2))
        ax.text(0.75, y + 0.39, f"attention  {att_t:,} tiles", va="center", fontsize=8)
        ax.add_patch(FancyBboxPatch((4.6, y + 0.08), 4.3, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, alpha=0.16 if not q else 0.32, ec=col, lw=1.2))
        ax.text(4.75, y + 0.39, f"MLP {kind}  {n} neurons  {mlp_t:,} tiles  chart {ce*100:.0f}%", va="center", fontsize=8)
        ax.annotate("", xy=(4.6, y + 0.39), xytext=(3.8, y + 0.39),
                    arrowprops=dict(arrowstyle="-|>", color=LOOM, lw=1.2))
        ax.text(0.05, y + 0.39, f"L{i:02d}", va="center", fontsize=8, color=MUTED)
        if q:
            ax.text(9.1, y + 0.39, "PyQrack: tiles + qneuron qubits", va="center", fontsize=8, color=QRACK, fontweight="bold")
        if i < nL - 1:
            ax.annotate("", xy=(2.2, y - 0.0), xytext=(2.2, y + 0.08),
                        arrowprops=dict(arrowstyle="-|>", color=LOOM, lw=1.0))
            ax.plot([8.9, 9.0, 9.0, 2.2], [y + 0.39, y + 0.39, y - 0.04, y - 0.04], color=LOOM, lw=0.8, alpha=0.6)
    total = nL * att_t + sum(r[4] for r in rows) + (lm_t or 0)
    ax.text(0, -0.55, f"attention tiles per layer: {att_t:,} ({att_note});  MLP tiles: "
            f"{sum(r[4] for r in rows):,};  LM head tiles: {lm_t:,}" if lm_t else
            f"attention tiles per layer: {att_t:,} ({att_note});  MLP tiles: {sum(r[4] for r in rows):,}", fontsize=8)
    ax.text(0, -1.0, f"total {total:,} tiles = {4 * total:,} qubits if all resident; "
            f"orange = Loom (classical expectation hand-off)", fontsize=8, color=MUTED)
    return save(fig, args, "topology_model")


# =============================================================================
# 3. CIRCUIT DRAWING
# =============================================================================

def draw_circuit(ax, wires, ops, title, x0=0.6, dx=0.55):
    """ops: ('ry', q, angle, label), ('h', q), ('cx', c, t), ('mcx', [c], t), ('acx', c, t) anti-control,
    ('mcry', [c], t), ('barrier', text), ('meas', [q], text)."""
    nW = len(wires)

    def span(op):
        if op[0] in ("ry", "h"):
            return [op[1]]
        if op[0] in ("cx", "acx"):
            return [op[1], op[2]]
        if op[0] in ("mcx", "mcry"):
            return list(op[1]) + [op[2]]
        if op[0] == "meas":
            return list(op[1])
        return list(range(nW))

    # column packing: a gate takes the first column free on every wire its vertical line crosses
    last, colidx = [-1] * nW, []
    for op in ops:
        s = span(op)
        cover = range(min(s), max(s) + 1)
        col = max(last) + 1 if op[0] == "barrier" else max(last[w] for w in cover) + 1
        for w in (range(nW) if op[0] == "barrier" else cover):
            last[w] = col
        colidx.append(col)
    cols = [x0 + c * dx for c in colidx]
    xmax = x0 + (max(colidx) + 1) * dx + 0.2
    for i, name in enumerate(wires):
        ax.plot([x0 - 0.3, xmax], [nW - 1 - i] * 2, color=MUTED, lw=0.9, zorder=0)
        ax.text(x0 - 0.4, nW - 1 - i, name, ha="right", va="center", fontsize=8)
    for op, cx in zip(ops, cols):
        if op[0] in ("ry", "h"):
            y = nW - 1 - op[1]
            fc = "#ede9fe" if op[0] == "ry" else "#dbeafe"
            ax.add_patch(Rectangle((cx - 0.22, y - 0.22), 0.44, 0.44, fc=fc, ec=ACCENT if op[0] == "ry" else "#2563eb", lw=0.9))
            ax.text(cx, y + 0.02, "RY" if op[0] == "ry" else "H", ha="center", va="center", fontsize=6.5)
            if op[0] == "ry" and len(op) > 3 and op[3]:
                ax.text(cx, y + 0.27, op[3], ha="center", va="bottom", fontsize=5.5, color=MUTED)
        elif op[0] == "cx":
            yc, yt = nW - 1 - op[1], nW - 1 - op[2]
            ax.plot([cx, cx], [yc, yt], color=INK, lw=0.9)
            ax.add_patch(Circle((cx, yc), 0.07, color=INK))
            ax.add_patch(Circle((cx, yt), 0.16, fc="white", ec=INK, lw=0.9))
            ax.plot([cx - 0.16, cx + 0.16], [yt, yt], color=INK, lw=0.9)
            ax.plot([cx, cx], [yt - 0.16, yt + 0.16], color=INK, lw=0.9)
        elif op[0] in ("mcx", "acx"):
            ctrls = op[1] if op[0] == "mcx" else [op[1]]
            t = op[2]
            ys = [nW - 1 - c for c in ctrls]
            yt = nW - 1 - t
            ax.plot([cx, cx], [min(ys + [yt]), max(ys + [yt])], color=INK, lw=0.9)
            for yy in ys:
                if op[0] == "acx":
                    ax.add_patch(Circle((cx, yy), 0.08, fc="white", ec=INK, lw=0.9))
                else:
                    ax.add_patch(Circle((cx, yy), 0.07, color=INK))
            ax.add_patch(Circle((cx, yt), 0.16, fc="white", ec=INK, lw=0.9))
            ax.plot([cx - 0.16, cx + 0.16], [yt, yt], color=INK, lw=0.9)
            ax.plot([cx, cx], [yt - 0.16, yt + 0.16], color=INK, lw=0.9)
        elif op[0] == "mcry":
            ys = [nW - 1 - c for c in op[1]]
            yt = nW - 1 - op[2]
            for yy in ys:                                       # control line stops at the box edge
                ax.plot([cx, cx], [yy, yt + (0.22 if yy > yt else -0.22)], color=INK, lw=0.9)
                ax.add_patch(Circle((cx, yy), 0.07, color=INK))
            ax.add_patch(Rectangle((cx - 0.22, yt - 0.22), 0.44, 0.44, fc="#ede9fe", ec=ACCENT, lw=0.9, zorder=3))
            ax.text(cx, yt, "RY", ha="center", va="center", fontsize=6.5, zorder=4)
        elif op[0] == "barrier":
            ax.axvline(cx, color=SEAM, ls=":", lw=1.0)
            ax.text(cx, nW - 0.35, op[1], ha="center", va="bottom", fontsize=7, color=SEAM)
        elif op[0] == "meas":
            for q in op[1]:
                y = nW - 1 - q
                ax.add_patch(Rectangle((cx - 0.24, y - 0.22), 0.48, 0.44, fc="#fef3c7", ec=LOOM, lw=0.9))
                ax.text(cx, y, op[2], ha="center", va="center", fontsize=6)
    ax.set_xlim(x0 - 1.6, xmax + 0.2)
    ax.set_ylim(-1.3, nW + 0.1)
    ax.axis("off")
    ax.set_title(title, loc="left")


def _tile_and_input(dic, li, x_token=None):
    W1 = np.asarray(dic[li].A["W1"], np.float64)
    T = W1[:4, :4]
    x = x_token[:4] if x_token is not None else np.random.default_rng(0).standard_normal(4)
    return T, x


def fig_tile(args, dic, x_token):
    li = args.layers_list[0]
    T, x = _tile_and_input(dic, li, x_token)
    unit = (T / np.linalg.norm(T)).reshape(-1)                 # index = row * 4 + col
    prep = H.amplitude_prep_ops(unit, [0, 1, 2, 3])
    q_ops = H.amplitude_prep_ops(x / np.linalg.norm(x), [0, 1])
    ops = [("ry", o[1], o[2], f"{o[2]:+.2f}") if o[0] == "ry" else ("cx", o[1], o[2]) for o in prep]
    ops.append(("barrier", "query"))
    ops += [("ry", o[1], -o[2], f"{-o[2]:+.2f}") if o[0] == "ry" else ("cx", o[1], o[2]) for o in reversed(q_ops)]
    ops.append(("meas", [0, 1, 2, 3], "amp"))
    check = ""
    try:
        from pyqrack import QrackSimulator
        sim = QrackSimulator(4)
        H.apply_ops(sim, prep)
        ov = np.vdot(unit, np.asarray(sim.out_ket()))
        H.apply_ops(sim, q_ops, inverse=True)
        amp = np.asarray(sim.out_ket()).reshape(4, 4)[:, 0] * np.conj(ov) / abs(ov)
        y = amp.real * np.linalg.norm(T) * np.linalg.norm(x)
        check = f"PyQrack W x = {np.array2string(y, precision=3)}  vs NumPy {np.array2string(T @ x, precision=3)}  " \
                f"(max diff {np.abs(y - T @ x).max():.1e})"
    except Exception as e:  # pyqrack optional for drawing
        check = f"PyQrack check skipped ({type(e).__name__})"
    fig, ax = plt.subplots(figsize=(19, 3.6))
    draw_circuit(ax, ["col bit 0", "col bit 1", "row bit 0", "row bit 1"], ops,
                 f"4x4 qubit tile, layer {li} W1[0:4, 0:4]: preparation (15 RY + 14 CNOT) then query with the input "
                 f"slice (3 RY + 2 CNOT, inverted); amplitude <row, 0| = (W x)_row / (|W| |x|)")
    ax.text(0.0, -1.15, check, fontsize=8, color=QRACK, transform=ax.transData)
    return save(fig, args, "circuit_tile")


def fig_qneuron(args):
    M = args.cosines
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0])
    wires = [f"gate m={m}" for m in range(1, M + 1)] + ["up"]
    ops = [("ry", m - 1, 0.0, f"{m}pi(x+1)/2") for m in range(1, M + 1)]
    ops.append(("ry", M, 0.0, "pi/2 - eps u"))
    ops.append(("meas", list(range(M + 1)), "<Z>"))
    draw_circuit(ax, wires, ops, f"qneuron: {M} gate cosine qubits + 1 up qubit (all independent, one layer); "
                 f"h = (c0 + sum c_m <Z_m>) <Z_u> / eps", dx=1.6)
    ax2 = fig.add_subplot(gs[1])
    G = 4.0
    xg = np.linspace(-1, 1, 1024)
    target = G * xg / (1 + np.exp(-G * xg))
    ax2.plot(xg * G, target, color=INK, lw=2.2, label="SiLU(g)")
    for Mm, col in ((2, "#f59e0b"), (4, "#10b981"), (8, ACCENT)):
        B = np.stack([np.cos(k * np.pi * (xg + 1) / 2) for k in range(Mm + 1)], 1)
        c = np.linalg.lstsq(B, target, rcond=None)[0]
        fit = B @ c
        ax2.plot(xg * G, fit, color=col, lw=1.3, ls="--",
                 label=f"{Mm + 1} cosine qubits (max err {np.abs(fit - target).max():.3f})")
    ax2.set_xlabel("gate pre-activation g (range G = 4)")
    ax2.set_ylabel("activation")
    ax2.set_title("SiLU rebuilt from qubit expectations cos(m theta)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return save(fig, args, "circuit_qneuron")


def fig_signed(args):
    D = 4
    wires = ["chart 0", "chart 1", "ancilla", "neuron 0", "neuron 1", "neuron 2"]
    ops = [("barrier", "|W'> loaded"), ("h", 2)]
    zq = H.amplitude_prep_ops(np.array([0.6, -0.3, 0.5, 0.55]) / np.linalg.norm([0.6, -0.3, 0.5, 0.55]), [0, 1])
    for o in reversed(zq):
        ops.append(("mcry", [2], o[1]) if o[0] == "ry" else ("mcx", [2, o[1]], o[2]))
    ops.append(("barrier", "anc=0: |ref> -> |0>"))
    ops += [("acx", 2, 0), ("acx", 2, 1)]
    ops.append(("h", 2))
    ops.append(("meas", [0, 1, 2, 3, 4, 5], "shot"))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    draw_circuit(ax, wires, ops, "landscape --signed: Hadamard test in patch 0 (controlled U_z^dagger on anc=1, "
                 "anti-controlled X on anc=0); P(anc=0, j) - P(anc=1, j) = beta s_j / (|W|^2 |z|)", dx=0.7)
    ax.axhspan(2.5, 5.5, xmin=0.0, xmax=1.0, color=QRACK, alpha=0.05)
    ax.text(0.0, 5.2, "patch 0", fontsize=7, color=QRACK)
    ax.text(0.0, 2.2, "patch 1", fontsize=7, color=ACCENT)
    return save(fig, args, "circuit_signed")


# =============================================================================
# 4. PATCH TOPOLOGY
# =============================================================================

def draw_chain(ax, patches, bonds, title, couplings=()):
    """patches: list of (name, [(qubit label, colour)]); bonds: per seam label; couplings: ((p, i), (p2, j))."""
    x = 0.0
    centers = {}
    w_q, gap = 0.55, 1.6
    for p, (name, qubits) in enumerate(patches):
        width = max(2.0, w_q * len(qubits) + 0.4)
        ax.add_patch(FancyBboxPatch((x, 0), width, 1.6, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=ACCENT, alpha=0.07, ec=ACCENT, lw=1.2))
        ax.text(x + width / 2, 1.85, name, ha="center", fontsize=8, fontweight="bold")
        for i, (lab, col) in enumerate(qubits):
            cx = x + 0.45 + i * w_q
            ax.add_patch(Circle((cx, 0.8), 0.19, fc=col, ec="white", lw=0.8))
            ax.text(cx, 0.32, lab, ha="center", fontsize=6, rotation=90, va="top")
            centers[(p, i)] = (cx, 0.8)
        if p < len(patches) - 1:
            sx = x + width
            ax.plot([sx + 0.1, sx + gap / 2, sx + gap - 0.1], [1.25, 0.8, 1.25], color=SEAM, lw=1.2)
            ax.plot([sx + 0.1, sx + gap / 2, sx + gap - 0.1], [0.35, 0.8, 0.35], color=SEAM, lw=1.2)
            ax.text(sx + gap / 2, -0.25, bonds[p], ha="center", fontsize=7, color=SEAM)
        x += width + gap
    for a, b in couplings:
        (x1, y1), (x2, y2) = centers[a], centers[b]
        xm = (x1 + x2) / 2
        ax.plot(np.linspace(x1, x2, 40), y1 + 0.2 + 0.45 * np.sin(np.linspace(0, np.pi, 40)),
                color=LOOM, lw=0.6, alpha=0.55)
    ax.set_xlim(-0.3, x - gap + 0.3)
    ax.set_ylim(-0.7, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left")


def fig_patches(args, dic):
    li = args.layers_list[0]
    Kp = np.asarray(dic[li].A["Kp"], np.float64)
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], hspace=0.55)
    C, Q, B, K = "#a78bfa", "#34d399", "#f87171", args.inputs
    # landscape: weight tile, 2 patches
    ax = fig.add_subplot(gs[0, 0])
    qf, qn = int(math.log2(args.tile_dims)), int(math.log2(args.tile_neurons))
    draw_chain(ax, [("patch 0: chart register", [(f"c{i}", C) for i in range(qf)]),
                    ("patch 1: neuron register", [(f"n{i}", Q) for i in range(qn)])],
               [f"seam: max_bonds b -> rank 2^b of the tile (SVD)"],
               f"landscape: {args.tile_neurons} x {args.tile_dims} chart-key tile as one state "
               f"(prep: {2 ** (qf + qn + 1) - 3} RY/CNOT gates)",
               couplings=[((0, i), (1, j)) for i in range(0, qf, 2) for j in range(0, qn, 3)])
    # landscape --signed
    ax = fig.add_subplot(gs[1, 0])
    draw_chain(ax, [("patch 0: chart + ancilla", [(f"c{i}", C) for i in range(qf)] + [("anc", B)]),
                    ("patch 1: neurons", [(f"n{i}", Q) for i in range(qf + 1)])],
               ["query gates never cross the seam"],
               "landscape --signed: Hadamard test, columns = chart dims + base + reference beta")
    # qneuron landscape tensor
    ax = fig.add_subplot(gs[2, 0])
    N = K * (args.patches - 1)
    groups = [("patch 0: K input qubits (chart sign bits)", [(f"in{i}", C) for i in range(K)])]
    per = N // (args.patches - 1)
    for p in range(args.patches - 1):
        groups.append((f"patch {p + 1}: neuron qubits", [(f"N{p * per + i}", Q) for i in range(per)]))
    draw_chain(ax, groups, [f"bond <= 2^{K} (exact at max_bonds = K)"] * (args.patches - 1),
               f"qneuron landscape: {K} inputs, {N} neurons, single-control RY per input bit and neuron",
               couplings=[((0, i), (1 + (j // per), j % per)) for i in range(0, K, 3) for j in range(0, N, 5)])
    # seam fidelity vs bond budget for a real tile
    ax = fig.add_subplot(gs[:, 1])
    for R, D, col in ((64, 64, ACCENT), (128, 64, QRACK), (256, 128, LOOM)):
        if R <= Kp.shape[0] and D <= Kp.shape[1]:
            s2 = np.linalg.svd(Kp[:R, :D], compute_uv=False) ** 2
            bmax = int(math.log2(min(R, D)))
            b = np.arange(0, bmax + 1)
            ax.plot(b, [s2[: 2 ** k].sum() / s2.sum() for k in b], "o-", color=col, label=f"{R} x {D} tile")
    ax.set_xlabel("max_bonds b (seam keeps 2^b Schmidt components)")
    ax.set_ylabel("seam fidelity = retained SVD energy")
    ax.set_title(f"Seam fidelity vs bond budget, layer {li} chart keys")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.text(0.98, 0.03, "QrackBridge greedy truncation equals this curve exactly\n(measured: fidelity_estimate = SVD "
            "energy to 4 decimals)", transform=ax.transAxes, fontsize=7, color=MUTED, ha="right")
    return save(fig, args, "topology_patches")


# =============================================================================
# 5. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Holoqubed geometry and topology graphs (matplotlib)")
    ap.add_argument("--atlas", required=True, help=".atlas directory from holoqubed_atlas.py forge")
    ap.add_argument("--layers", default="0", help="Comma separated MLP layers for geometry figures")
    ap.add_argument("--figures", default="geometry,model,tile,qneuron,signed,patches")
    ap.add_argument("--model_id", default=None, help="Optional: real tokens and model config (attention, LM head)")
    ap.add_argument("--text", default=None, help="Optional: text for the real tokens")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--qrack_layer", type=int, default=-1, help="Layer to mark as running in PyQrack")
    ap.add_argument("--inputs", type=int, default=8, help="qneuron landscape inputs K")
    ap.add_argument("--patches", type=int, default=3, help="qneuron landscape patches")
    ap.add_argument("--tile_neurons", type=int, default=128)
    ap.add_argument("--tile_dims", type=int, default=64)
    ap.add_argument("--cosines", type=int, default=8, help="qneuron gate cosine qubits")
    ap.add_argument("--out", default="holoqubed_graphs")
    ap.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()
    args.layers_list = [int(v) for v in args.layers.split(",") if v]
    figs = set(args.figures.split(","))
    os.makedirs(args.out, exist_ok=True)
    dic = H.AtlasDictionary(args.atlas)
    cfg, Ztok, x_token = None, {}, None
    if args.model_id:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
        if args.text and figs & {"geometry", "tile"}:
            tok, model = H.load_model(args.model_id, args.trust_remote_code)
            caps = H.capture_inputs(model, tok, H.find_mlps(model), open(args.text, encoding="utf-8").read(),
                                    args.tokens, 256)
            for li in args.layers_list:
                Ztok[li] = dic[li].planner.plan(caps[li])[0]
            x_token = caps[args.layers_list[0]][-1].astype(np.float64)
    written = []
    if "geometry" in figs:
        for li in args.layers_list:
            written.append(fig_geometry(args, dic, li, Ztok.get(li)))
    if "model" in figs:
        written.append(fig_model(args, dic, cfg))
    if "tile" in figs:
        written.append(fig_tile(args, dic, x_token))
    if "qneuron" in figs:
        written.append(fig_qneuron(args))
    if "signed" in figs:
        written.append(fig_signed(args))
    if "patches" in figs:
        written.append(fig_patches(args, dic))
    print(f"{len(written)} figures in {args.out}")


if __name__ == "__main__":
    main()
