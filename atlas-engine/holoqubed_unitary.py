#!/usr/bin/env python3
"""ThereminQ Holoqubed Unitary: a natively unitary language model, trained on real text.

Sequential Born machine at character level. One register of n = m + v qubits: m memory qubits and v
vocabulary qubits, interleaved so the nearest-neighbour circuit mixes them. Per character:

  1. apply U: one learned brick-wall circuit of 2-qubit blocks (15 angles + 3 CNOTs per block),
     the same circuit at every step
  2. measure the vocabulary register: P(next char = k) = || (I x <k|) U |psi> ||^2  (Born rule)
  3. the measurement leaves the register in |k>, which is the input of the next step, and the
     memory in the renormalized conditional state (the only nonlinearity)

Generation is one shot per character: apply U, measure v qubits, repeat. Training uses exact complex
statevectors with teacher forcing (loss = -log P(true char)).

Backends:
  torch    CPU, autograd, checkpointed per step
  opencl   data parallel over --devices OpenCL devices, one worker process per device. Custom kernels
           for the gates, the measurement and its gradient; the backward pass uncomputes every gate
           with its adjoint (no stored intermediate states) and accumulates the block gradients on
           the device. The host sums the block gradients, maps them to the angles and steps Adam.

Long runs: --hours, checkpoints every --ckpt_minutes, --resume, evaluation every --eval_minutes
(best checkpoint kept), SIGINT / SIGTERM stop cleanly after the current chunk.

Baselines on the same characters and test text: interpolated n-grams, an LSTM with the same parameter
count, a large LSTM and a small causal transformer (CPU). After training, the model is replayed in
QrackSimulator with real gates and mid-circuit measurements, and compared with torch step by step.

  python3 holoqubed_unitary.py --minutes 120                                   # CPU, 2 hours in total
  python3 holoqubed_unitary.py --backend opencl --devices 6 --hours 8 \\
      --dataset wikitext-103-raw-v1 --mem_qubits 9 --depth 24 --batch 64       # 6 GPUs, 8 hours
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from collections import Counter

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LN2 = math.log(2)


# =============================================================================
# 1. DATA
# =============================================================================

def load_text(path: str, split: str, dataset: str) -> str:
    if path and os.path.exists(path):
        return open(path, encoding="utf-8").read()
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", dataset, split=split)
    text = "".join(ds["text"])
    if path:
        open(path, "w", encoding="utf-8").write(text)
    return text


def _codepoints(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)


class CharVocab:
    def __init__(self, text: str, size: int):
        cps = _codepoints(text)
        counts = np.bincount(cps)
        order = np.argsort(-counts, kind="stable")[: size - 1]
        common = [chr(int(c)) for c in order if counts[c] > 0]
        self.chars = common + ["\x00"] * (size - len(common))          # last id = unknown
        self.unk = size - 1
        self.index = {c: i for i, c in enumerate(common)}
        self.lut_max = int(cps.max()) if cps.size else 0

    def encode_np(self, text: str) -> np.ndarray:
        cps = _codepoints(text)
        lut = np.full(max(self.lut_max, int(cps.max()) if cps.size else 0) + 1, self.unk, dtype=np.int32)
        for c, i in self.index.items():
            if ord(c) < lut.size:
                lut[ord(c)] = i
        return lut[cps]

    def encode(self, text: str) -> torch.Tensor:
        return torch.from_numpy(self.encode_np(text).astype(np.int64))

    def decode(self, ids) -> str:
        return "".join(self.chars[i] if i != self.unk else "?" for i in ids)


# =============================================================================
# 2. UNITARY BORN MACHINE
# =============================================================================

def _u3(t, p, l):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    one = torch.ones_like(t)
    return torch.stack([torch.stack([c * one, -torch.polar(one, l) * s], -1),
                        torch.stack([torch.polar(one, p) * s, torch.polar(one, p + l) * c], -1)], -2)


def _ry(t):
    c, s = torch.cos(t / 2).to(torch.complex64), torch.sin(t / 2).to(torch.complex64)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rz(t):
    one, z = torch.ones_like(t), torch.zeros_like(t)
    e0, e1 = torch.polar(one, -t / 2), torch.polar(one, t / 2)
    return torch.stack([torch.stack([e0, torch.complex(z, z)], -1), torch.stack([torch.complex(z, z), e1], -1)], -2)


def _kron(hi, lo):  # 4x4 basis index = 2 * b_hi + b_lo
    return torch.einsum("gab,gcd->gacbd", hi, lo).reshape(hi.shape[0], 4, 4)


_I2 = torch.eye(2, dtype=torch.complex64)
_CX_LO_HI = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.complex64)  # control lo, target hi
_CX_HI_LO = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64)  # control hi, target lo


def block_ops(th: np.ndarray, q: int) -> list[tuple]:
    """Gate list of one 2-qubit block on (lo = q, hi = q + 1), in application order (for Qrack)."""
    lo, hi = q, q + 1
    return [("u3", hi, th[0:3]), ("u3", lo, th[3:6]), ("cx", lo, hi), ("ry", hi, th[6]), ("rz", lo, th[7]),
            ("cx", hi, lo), ("ry", hi, th[8]), ("cx", lo, hi), ("u3", hi, th[9:12]), ("u3", lo, th[12:15])]


class UnitaryBorn(nn.Module):
    def __init__(self, mem: int, voc: int, depth: int, init: float = 0.3, seed: int = 0):
        super().__init__()
        if mem < voc:
            raise SystemExit("--mem_qubits must be >= --vocab_qubits (vocabulary qubits are interleaved)")
        self.m, self.v, self.n = mem, voc, mem + voc
        self.depth = depth
        self.vq = [2 * j + 1 for j in range(voc)]                      # vocabulary qubit positions
        self.mq = [q for q in range(self.n) if q not in self.vq]       # memory qubit positions
        self.pairs = [q for L in range(depth) for q in range(L % 2, self.n - 1, 2)]
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(init * torch.randn(len(self.pairs), 15, generator=g))
        idx = np.zeros((2 ** mem, 2 ** voc), dtype=np.int64)
        for a in range(2 ** mem):
            fa = sum(((a >> j) & 1) << self.mq[j] for j in range(mem))
            for k in range(2 ** voc):
                idx[a, k] = fa + sum(((k >> j) & 1) << self.vq[j] for j in range(voc))
        self.register_buffer("idx", torch.from_numpy(idx.reshape(-1)), persistent=False)
        inv = np.empty_like(idx.reshape(-1)); inv[idx.reshape(-1)] = np.arange(idx.size)
        self.register_buffer("inv", torch.from_numpy(inv), persistent=False)

    def n_params(self) -> int:
        return self.theta.numel()

    def blocks(self) -> torch.Tensor:
        t = self.theta
        ops = [_kron(_u3(t[:, 0], t[:, 1], t[:, 2]), _u3(t[:, 3], t[:, 4], t[:, 5])),
               _CX_LO_HI.expand(len(t), 4, 4),
               _kron(_ry(t[:, 6]), _rz(t[:, 7])),
               _CX_HI_LO.expand(len(t), 4, 4),
               _kron(_ry(t[:, 8]), _I2.expand(len(t), 2, 2)),
               _CX_LO_HI.expand(len(t), 4, 4),
               _kron(_u3(t[:, 9], t[:, 10], t[:, 11]), _u3(t[:, 12], t[:, 13], t[:, 14]))]
        M = ops[0]
        for o in ops[1:]:
            M = o @ M
        return M

    def evolve(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        B, n = psi.shape[0], self.n
        for g, q in enumerate(self.pairs):
            v = psi.reshape(B, 2 ** (n - q - 2), 4, 2 ** q)
            psi = torch.einsum("ij,bajc->baic", M[g].to(psi.dtype), v).reshape(B, -1)
        return psi

    def probs(self, psi: torch.Tensor):
        S = psi[:, self.idx].reshape(psi.shape[0], 2 ** self.m, 2 ** self.v)
        return S, (S.real ** 2 + S.imag ** 2).sum(1)

    def project(self, S: torch.Tensor, p: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B = S.shape[0]
        pk = p.gather(1, k[:, None]).clamp_min(1e-30)
        a = S[torch.arange(B), :, k] / torch.sqrt(pk)
        onehot = F.one_hot(k, 2 ** self.v).to(S.dtype)
        return (a[:, :, None] * onehot[:, None, :]).reshape(B, -1)[:, self.inv]

    def init_state(self, first: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
        B = first.shape[0]
        psi = torch.zeros(B, 2 ** self.n, dtype=dtype)
        psi[torch.arange(B), self.idx[first]] = 1.0           # memory |0>, vocabulary |first char>
        return psi

    def step(self, psi, M, target):
        S, p = self.probs(self.evolve(psi, M))
        nll = -torch.log(p.gather(1, target[:, None]).squeeze(1).clamp_min(1e-30))
        return nll, self.project(S, p, target)


# =============================================================================
# 3. OPENCL DATA-PARALLEL BACKEND
# =============================================================================

CL_SRC = r"""
inline float2 cmul(float2 a, float2 b) { return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline float2 cmulc(float2 a, float2 b) { return (float2)(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y); }  /* a * conj(b) */

/* y = M x on qubits (q, q + 1) of every stream; 4x4 basis index = 2 b_{q+1} + b_q */
__kernel void apply4(__global float2* psi, __global const float2* M, const int g, const int q, const int n) {
    const long gid = get_global_id(0);
    const long N4 = 1L << (n - 2);
    const long b = gid / N4, r = gid % N4;
    const long low = r & ((1L << q) - 1), high = r >> q;
    const long base = (b << n) + (high << (q + 2)) + low, st = 1L << q;
    __global const float2* m = M + g * 16;
    float2 x0 = psi[base], x1 = psi[base + st], x2 = psi[base + 2 * st], x3 = psi[base + 3 * st];
    psi[base]          = cmul(m[0], x0)  + cmul(m[1], x1)  + cmul(m[2], x2)  + cmul(m[3], x3);
    psi[base + st]     = cmul(m[4], x0)  + cmul(m[5], x1)  + cmul(m[6], x2)  + cmul(m[7], x3);
    psi[base + 2 * st] = cmul(m[8], x0)  + cmul(m[9], x1)  + cmul(m[10], x2) + cmul(m[11], x3);
    psi[base + 3 * st] = cmul(m[12], x0) + cmul(m[13], x1) + cmul(m[14], x2) + cmul(m[15], x3);
}

/* backward through y = M x: uncompute x = M^H y, propagate lam_x = M^H lam_y, accumulate
   dL/dM[i][j] += lam_y[i] conj(x[j]) per work-group (summed over groups by reduce16) */
__kernel void back4(__global float2* phi, __global float2* lam, __global const float2* MH, const int g, const int q,
                    const int n, __global float2* partial, __local float2* red) {
    const long gid = get_global_id(0);
    const int lid = get_local_id(0), ls = get_local_size(0), grp = get_group_id(0), ngrp = get_num_groups(0);
    const long N4 = 1L << (n - 2);
    const long b = gid / N4, r = gid % N4;
    const long low = r & ((1L << q) - 1), high = r >> q;
    const long base = (b << n) + (high << (q + 2)) + low, st = 1L << q;
    __global const float2* m = MH + g * 16;
    float2 y[4], l[4], x[4], li[4];
    for (int i = 0; i < 4; i++) { y[i] = phi[base + i * st]; l[i] = lam[base + i * st]; }
    for (int i = 0; i < 4; i++) {
        x[i]  = cmul(m[4 * i], y[0]) + cmul(m[4 * i + 1], y[1]) + cmul(m[4 * i + 2], y[2]) + cmul(m[4 * i + 3], y[3]);
        li[i] = cmul(m[4 * i], l[0]) + cmul(m[4 * i + 1], l[1]) + cmul(m[4 * i + 2], l[2]) + cmul(m[4 * i + 3], l[3]);
    }
    for (int i = 0; i < 4; i++) { phi[base + i * st] = x[i]; lam[base + i * st] = li[i]; }
    for (int e = 0; e < 16; e++) red[e * ls + lid] = cmulc(l[e >> 2], x[e & 3]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls >> 1; s > 0; s >>= 1) {
        if (lid < s) for (int e = 0; e < 16; e++) red[e * ls + lid] += red[e * ls + lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < 16) partial[((long)g * ngrp + grp) * 16 + lid] += red[lid * ls];
}

__kernel void reduce16(__global const float2* partial, __global float2* out, const int ngrp) {
    const int gid = get_global_id(0), g = gid / 16, e = gid % 16;
    float2 s = (float2)(0.0f, 0.0f);
    for (int i = 0; i < ngrp; i++) s += partial[((long)g * ngrp + i) * 16 + e];
    out[gid] = s;
}

/* Born probabilities of the vocabulary register: p[b][k] = sum_a |phi[b][idx[a][k]]|^2 */
__kernel void probs(__global const float2* phi, __global const int* idx, __global float* p, const int n, const int m, const int v) {
    const int gid = get_global_id(0), K = 1 << v, b = gid / K, k = gid % K;
    float s = 0.0f;
    for (int a = 0; a < (1 << m); a++) { float2 z = phi[((long)b << n) + idx[a * K + k]]; s += z.x * z.x + z.y * z.y; }
    p[gid] = s;
}

__kernel void pick_char(__global const float* p, __global const int* tgt, __global float* pk, __global float* nll,
                     const int t, const int B, const int v) {
    const int b = get_global_id(0);
    const float x = p[b * (1 << v) + tgt[t * B + b]];
    pk[t * B + b] = x;
    nll[b] += -log(fmax(x, 1e-30f));
}

/* project the memory onto the measured character and renormalize; dst is zero-filled beforehand */
__kernel void project(__global const float2* phi, __global float2* dst, const long dst_off, __global const int* idx,
                      __global const int* tgt, __global const float* pk, const int t, const int B, const int n, const int v) {
    const int gid = get_global_id(0), M = get_global_size(0) / B, b = gid / M, a = gid % M, K = 1 << v;
    const long i = ((long)b << n) + idx[a * K + tgt[t * B + b]];
    dst[dst_off + i] = phi[i] * rsqrt(fmax(pk[t * B + b], 1e-30f));
}

__kernel void init_state(__global float2* dst, __global const int* idx, __global const int* first, const int n) {
    const int b = get_global_id(0);
    dst[((long)b << n) + idx[first[b]]] = (float2)(1.0f, 0.0f);
}

/* gradient of the measurement step: a = phi column k*, r = sqrt(p_k*):
   d(-log p)/da = -2 a / p;  d(<lam', a / r>)/da = lam' / r - a Re<lam', a> / r^3 */
__kernel void mdot(__global const float2* phi, __global const float2* lamp, __global const int* idx, __global const int* tgt,
                   __global float* dot, const int t, const int B, const int n, const int m, const int v) {
    const int b = get_global_id(0), K = 1 << v, k = tgt[t * B + b];
    float s = 0.0f;
    for (int a = 0; a < (1 << m); a++) {
        const long i = ((long)b << n) + idx[a * K + k];
        s += lamp[i].x * phi[i].x + lamp[i].y * phi[i].y;
    }
    dot[b] = s;
}

__kernel void mgrad(__global const float2* phi, __global const float2* lamp, __global float2* lam, __global const int* idx,
                    __global const int* tgt, __global const float* pk, __global const float* dot, const int t, const int B,
                    const int n, const int v) {
    const int gid = get_global_id(0), M = get_global_size(0) / B, b = gid / M, a = gid % M, K = 1 << v;
    const long i = ((long)b << n) + idx[a * K + tgt[t * B + b]];
    const float p = fmax(pk[t * B + b], 1e-30f), r = sqrt(p);
    const float2 z = phi[i];
    lam[i] = z * (-2.0f / p) + lamp[i] / r - z * (dot[b] / (r * r * r));
}
"""


def list_cl_devices(kind: str, platform: str = "") -> list[tuple]:
    import pyopencl as cl
    out = []
    for pi, p in enumerate(cl.get_platforms()):
        if platform and platform.lower() not in p.name.lower():
            continue
        for di, d in enumerate(p.get_devices()):
            is_gpu = bool(d.type & cl.device_type.GPU)
            if kind == "all" or (kind == "gpu" and is_gpu) or (kind == "cpu" and not is_gpu):
                out.append((pi, di, d.name.strip(), d.global_mem_size))
    return out


class _CLWorker:
    """One OpenCL device: holds its streams, the per-step state store and the kernels."""

    def __init__(self, spec, cfg, streams_path, rows, eval_rows):
        import pyopencl as cl
        self.cl = cl
        dev = cl.get_platforms()[spec[0]].get_devices()[spec[1]]
        self.name = dev.name.strip()
        self.ctx = cl.Context([dev])
        self.q = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, CL_SRC).build()
        self.k = {n: cl.Kernel(self.prg, n) for n in
                  ("apply4", "back4", "reduce16", "probs", "pick_char", "project", "init_state", "mdot", "mgrad")}
        self.n, self.m, self.v, self.T = cfg["n"], cfg["m"], cfg["v"], cfg["T"]
        self.pairs = cfg["pairs"]
        self.G = len(self.pairs)
        self.X = np.ascontiguousarray(np.load(os.path.join(streams_path, "train.npy"), mmap_mode="r")[rows[0]:rows[1]])
        ev = np.load(os.path.join(streams_path, "eval.npy"), mmap_mode="r")
        self.Xe = np.ascontiguousarray(ev[eval_rows[0]:eval_rows[1]]) if eval_rows[1] > eval_rows[0] else None
        self.B = self.X.shape[0]
        self.N, self.K, self.Mm = 1 << self.n, 1 << self.v, 1 << self.m
        self.ls = 64
        self.ngrp = self.B * (self.N >> 2) // self.ls
        mf = cl.mem_flags
        f2 = 8
        self.idx = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cfg["idx"].astype(np.int32))
        self.store = cl.Buffer(self.ctx, mf.READ_WRITE, (self.T + 1) * self.B * self.N * f2)
        self.phi = cl.Buffer(self.ctx, mf.READ_WRITE, self.B * self.N * f2)
        self.lam = cl.Buffer(self.ctx, mf.READ_WRITE, self.B * self.N * f2)
        self.lamp = cl.Buffer(self.ctx, mf.READ_WRITE, self.B * self.N * f2)
        self.p = cl.Buffer(self.ctx, mf.READ_WRITE, self.B * self.K * 4)
        self.pk = cl.Buffer(self.ctx, mf.READ_WRITE, self.T * self.B * 4)
        self.nll = cl.Buffer(self.ctx, mf.READ_WRITE, self.B * 4)
        self.dot = cl.Buffer(self.ctx, mf.READ_WRITE, self.B * 4)
        self.tgt = cl.Buffer(self.ctx, mf.READ_ONLY, self.T * self.B * 4)
        self.first = cl.Buffer(self.ctx, mf.READ_ONLY, self.B * 4)
        self.Mb = cl.Buffer(self.ctx, mf.READ_ONLY, self.G * 16 * f2)
        self.MHb = cl.Buffer(self.ctx, mf.READ_ONLY, self.G * 16 * f2)
        self.partial = cl.Buffer(self.ctx, mf.READ_WRITE, self.G * self.ngrp * 16 * f2)
        self.gout = cl.Buffer(self.ctx, mf.READ_WRITE, self.G * 16 * f2)
        self.red = cl.LocalMemory(16 * self.ls * f2)
        self.k["apply4"].set_arg(4, np.int32(self.n))
        self.bytes = (self.T + 4) * self.B * self.N * f2 + self.G * self.ngrp * 16 * f2

    # -- primitives
    def _run(self, name, gsize, lsize, *args):
        k = self.k[name]
        k.set_args(*args)
        self.cl.enqueue_nd_range_kernel(self.q, k, (gsize,), None if lsize is None else (lsize,))

    def _zero(self, buf, offset=0, size=None):
        self.cl.enqueue_fill_buffer(self.q, buf, np.float32(0), offset, size if size is not None else buf.size - offset)

    def _evolve(self, buf, B):
        k = self.k["apply4"]
        k.set_arg(0, buf); k.set_arg(1, self.Mb)
        for g, q in enumerate(self.pairs):
            k.set_arg(2, np.int32(g)); k.set_arg(3, np.int32(q))
            self.cl.enqueue_nd_range_kernel(self.q, k, (B * (self.N >> 2),), (self.ls,))

    def _init(self, dst, first, B):
        self.cl.enqueue_copy(self.q, self.first, np.ascontiguousarray(first.astype(np.int32)))
        self._zero(dst, 0, B * self.N * 8)
        self._run("init_state", B, None, dst, self.idx, self.first, np.int32(self.n))

    def set_blocks(self, M, MH):
        self.cl.enqueue_copy(self.q, self.Mb, np.ascontiguousarray(M.astype(np.complex64)))
        self.cl.enqueue_copy(self.q, self.MHb, np.ascontiguousarray(MH.astype(np.complex64)))

    # -- one truncated-BPTT chunk
    def chunk(self, M, MH, pos, reset):
        cl, B, N, T = self.cl, self.B, self.N, self.T
        sb = B * N * 8
        self.set_blocks(M, MH)
        if reset:
            self._init(self.store, self.X[:, pos], B)
        cl.enqueue_copy(self.q, self.tgt, np.ascontiguousarray(self.X[:, pos + 1:pos + T + 1].T.astype(np.int32)))
        self._zero(self.partial); self._zero(self.nll)
        for t in range(T):                                                   # forward
            cl.enqueue_copy(self.q, self.phi, self.store, src_offset=t * sb, byte_count=sb)
            self._evolve(self.phi, B)
            self._run("probs", B * self.K, None, self.phi, self.idx, self.p, np.int32(self.n), np.int32(self.m), np.int32(self.v))
            self._run("pick_char", B, None, self.p, self.tgt, self.pk, self.nll, np.int32(t), np.int32(B), np.int32(self.v))
            self._zero(self.store, (t + 1) * sb, sb)
            self._run("project", B * self.Mm, None, self.phi, self.store, np.int64((t + 1) * B * N), self.idx, self.tgt,
                      self.pk, np.int32(t), np.int32(B), np.int32(self.n), np.int32(self.v))
        self._zero(self.lamp)
        kb = self.k["back4"]
        for t in range(T - 1, -1, -1):                                       # backward
            cl.enqueue_copy(self.q, self.phi, self.store, src_offset=t * sb, byte_count=sb)
            self._evolve(self.phi, B)
            self._run("mdot", B, None, self.phi, self.lamp, self.idx, self.tgt, self.dot, np.int32(t), np.int32(B),
                      np.int32(self.n), np.int32(self.m), np.int32(self.v))
            self._zero(self.lam)
            self._run("mgrad", B * self.Mm, None, self.phi, self.lamp, self.lam, self.idx, self.tgt, self.pk, self.dot,
                      np.int32(t), np.int32(B), np.int32(self.n), np.int32(self.v))
            kb.set_args(self.phi, self.lam, self.MHb, np.int32(0), np.int32(0), np.int32(self.n), self.partial, self.red)
            for g in range(self.G - 1, -1, -1):
                kb.set_arg(3, np.int32(g)); kb.set_arg(4, np.int32(self.pairs[g]))
                cl.enqueue_nd_range_kernel(self.q, kb, (B * (N >> 2),), (self.ls,))
            self.lam, self.lamp = self.lamp, self.lam
        self._run("reduce16", self.G * 16, None, self.partial, self.gout, np.int32(self.ngrp))
        g = np.empty(self.G * 16, np.complex64); nll = np.empty(B, np.float32)
        cl.enqueue_copy(self.q, g, self.gout); cl.enqueue_copy(self.q, nll, self.nll)
        cl.enqueue_copy(self.q, self.store, self.store, src_offset=T * sb, dst_offset=0, byte_count=sb)  # carry the state
        self.q.finish()
        return g.astype(np.complex128), float(nll.astype(np.float64).sum()), B * T

    # -- forward-only evaluation on the worker's test streams
    def evaluate(self, M, MH):
        if self.Xe is None:
            return 0.0, 0
        cl = self.cl
        self.set_blocks(M, MH)
        Be, Le = self.Xe.shape[0], self.Xe.shape[1] - 1
        sb = Be * self.N * 8
        psi = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, sb)
        tg = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                       hostbuf=np.ascontiguousarray(self.Xe[:, 1:].T.astype(np.int32)))
        pk = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, Le * Be * 4)
        self._init(psi, self.Xe[:, 0], Be)
        self._zero(self.nll)
        for t in range(Le):
            cl.enqueue_copy(self.q, self.phi, psi, byte_count=sb)
            self._evolve(self.phi, Be)
            self._run("probs", Be * self.K, None, self.phi, self.idx, self.p, np.int32(self.n), np.int32(self.m), np.int32(self.v))
            self._run("pick_char", Be, None, self.p, tg, pk, self.nll, np.int32(t), np.int32(Be), np.int32(self.v))
            self._zero(psi, 0, sb)
            self._run("project", Be * self.Mm, None, self.phi, psi, np.int64(0), self.idx, tg, pk, np.int32(t),
                      np.int32(Be), np.int32(self.n), np.int32(self.v))
        nll = np.empty(self.B, np.float32)
        cl.enqueue_copy(self.q, nll, self.nll); self.q.finish()
        return float(nll[:Be].astype(np.float64).sum()), Be * Le


def _cl_worker_main(conn, spec, cfg, streams_path, rows, eval_rows):
    signal.signal(signal.SIGINT, signal.SIG_IGN)                 # the host decides when to stop
    try:
        w = _CLWorker(spec, cfg, streams_path, rows, eval_rows)
        conn.send(("ready", w.name, w.bytes))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        msg = conn.recv()
        try:
            if msg[0] == "chunk":
                conn.send(w.chunk(*msg[1:]))
            elif msg[0] == "eval":
                conn.send(w.evaluate(*msg[1:]))
            elif msg[0] == "stop":
                break
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))


class CLTrainer:
    """Data-parallel training of a UnitaryBorn model over several OpenCL devices."""

    def __init__(self, model, train_ids: np.ndarray, eval_ids: np.ndarray, a, log=print):
        self.model, self.a, self.log = model, a, log
        if model.n < 8:
            raise SystemExit("the OpenCL backend needs at least 8 qubits")
        devs = list_cl_devices(a.device_type, a.platform)
        if not devs:
            raise SystemExit(f"no OpenCL devices of type {a.device_type}")
        if len(devs) < a.devices and not a.oversubscribe:
            raise SystemExit(f"{len(devs)} OpenCL device(s) of type {a.device_type} found, {a.devices} requested "
                             "(use --oversubscribe to share devices)")
        specs = [devs[i % len(devs)] for i in range(a.devices)]
        D, B = a.devices, a.batch
        self.D, self.B, self.T = D, B, a.bptt
        Btot = D * B
        self.L = len(train_ids) // Btot
        dt = np.uint8 if model.v <= 8 else np.int16
        E = a.eval_streams
        Le = (len(eval_ids) - 1) // E
        ev = np.stack([eval_ids[i * Le:i * Le + Le + 1] for i in range(E)]).astype(dt)
        self.tmpdir = tempfile.mkdtemp(prefix="holoqubed_unitary_")      # memory-mapped by the workers
        np.save(os.path.join(self.tmpdir, "train.npy"), train_ids[: self.L * Btot].reshape(Btot, self.L).astype(dt))
        np.save(os.path.join(self.tmpdir, "eval.npy"), ev)
        cfg = dict(n=model.n, m=model.m, v=model.v, T=a.bptt, pairs=list(model.pairs), idx=model.idx.numpy())
        eval_split = np.array_split(np.arange(E), D)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for d in range(D):
            pc, cc = ctx.Pipe()
            er = (int(eval_split[d][0]), int(eval_split[d][-1]) + 1) if len(eval_split[d]) else (0, 0)
            p = ctx.Process(target=_cl_worker_main, args=(cc, specs[d][:2], cfg, self.tmpdir, (d * B, (d + 1) * B), er),
                            daemon=True)
            p.start()
            self.conns.append(pc); self.procs.append(p)
        for d, c in enumerate(self.conns):
            r = c.recv()
            if r[0] != "ready":
                raise SystemExit(f"OpenCL worker {d}: {r[1]}")
            log(f"  worker {d}: {r[1]}, {r[2] / 2 ** 30:.2f} GiB device memory, streams {d * B} to {(d + 1) * B - 1}")
        self.eval_chars = E * Le

    def _blocks(self):
        M = self.model.blocks()
        Mc = M.detach().numpy()
        Mn = Mc.reshape(-1, 16)
        MH = np.ascontiguousarray(np.conj(np.swapaxes(Mc, -1, -2))).reshape(-1, 16)
        return M, Mn, MH

    def _gather(self):
        out = []
        for d, c in enumerate(self.conns):
            r = c.recv()
            if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str) and r[0] == "error":
                raise SystemExit(f"OpenCL worker {d}: {r[1]}")
            out.append(r)
        return out

    def gradient(self, pos, reset):
        """Mean loss and dL/dtheta for one chunk (theta.grad is set)."""
        M, Mn, MH = self._blocks()
        for c in self.conns:
            c.send(("chunk", Mn, MH, pos, reset))
        res = self._gather()
        g = sum(r[0] for r in res); nll = sum(r[1] for r in res); cnt = sum(r[2] for r in res)
        self.model.theta.grad = None
        M.backward(torch.from_numpy((g / cnt).astype(np.complex64)).reshape(M.shape))
        return nll / cnt, cnt

    def evaluate(self):
        with torch.no_grad():
            _, Mn, MH = self._blocks()
        for c in self.conns:
            c.send(("eval", Mn, MH))
        res = self._gather()
        return sum(r[0] for r in res), sum(r[1] for r in res)

    def close(self):
        for c in self.conns:
            try:
                c.send(("stop",))
            except OSError:
                pass
        for p in self.procs:
            p.join(timeout=10)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def train_unitary_cl(model, trainer: CLTrainer, a, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    planned = a.train_minutes * 60
    state = dict(pos=0, elapsed=0.0, seen=0, chunks=0, hist=[], evals=[], best=float("inf"))
    if a.resume:
        ck = torch.load(a.resume, weights_only=False)
        model.theta.data.copy_(ck["theta"])
        opt.load_state_dict(ck["opt"])
        state.update({k: ck[k] for k in state if k in ck})
        planned = ck.get("planned", planned) if a.keep_schedule else planned
        log(f"  resumed {a.resume}: {state['seen'] / 1e6:.2f}M chars, {state['elapsed'] / 3600:.2f} h, "
            f"stream position {state['pos']} (memory restarts from |0>)")
    stop = {"flag": False}

    def handler(sig, frame):
        stop["flag"] = True
        log(f"  signal {sig}: stopping after this chunk")
    old = [signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)]

    def save(path, extra=None):
        d = dict(theta=model.theta.detach().clone(), opt=opt.state_dict(), planned=planned, mem=model.m, voc=model.v,
                 depth=model.depth, **state)
        if extra:
            d.update(extra)
        torch.save(d, path + ".tmp"); os.replace(path + ".tmp", path)

    t0 = time.time() - state["elapsed"]
    last_log = last_ck = last_ev = time.time()
    reset = True
    win_chars, win_t = 0, time.time()
    while not stop["flag"]:
        state["elapsed"] = time.time() - t0
        if state["elapsed"] >= planned:
            break
        frac = state["elapsed"] / planned
        for pg in opt.param_groups:
            pg["lr"] = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if state["pos"] + trainer.T + 1 > trainer.L:
            state["pos"] = 0; reset = True
        loss, cnt = trainer.gradient(state["pos"], reset)
        reset = False
        opt.step()
        state["pos"] += trainer.T; state["seen"] += cnt; state["chunks"] += 1
        state["hist"].append(loss / LN2)
        win_chars += cnt
        now = time.time()
        if now - last_log >= a.log_seconds:
            k = max(1, min(len(state["hist"]), 50))
            log(f"  unitary/opencl: {state['seen'] / 1e6:.2f}M chars, train BPC {np.mean(state['hist'][-k:]):.3f}, "
                f"{win_chars / (now - win_t):,.0f} chars/s, {state['elapsed'] / 3600:.2f} of {planned / 3600:.2f} h")
            last_log, win_chars, win_t = now, 0, now
        if a.eval_minutes and now - last_ev >= a.eval_minutes * 60:
            nll, n = trainer.evaluate()
            bpc = nll / n / LN2
            state["evals"].append((state["seen"], bpc))
            log(f"  eval: test BPC {bpc:.3f} on {n:,} chars")
            if bpc < state["best"]:
                state["best"] = bpc
                save(a.save + ".best")
            last_ev = now
        if a.save and now - last_ck >= a.ckpt_minutes * 60:
            save(a.save); last_ck = now
    for s, h in zip((signal.SIGINT, signal.SIGTERM), old):
        signal.signal(s, h)
    if a.save:
        save(a.save)
    return state, stop["flag"]


# =============================================================================
# 4. BASELINES
# =============================================================================

class LSTMLM(nn.Module):
    def __init__(self, V, e, h):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(V, e), nn.LSTM(e, h, batch_first=True), nn.Linear(h, V)

    def forward(self, x, hc=None):
        y, hc = self.rnn(self.emb(x), hc)
        return self.out(y), hc


class TinyTransformer(nn.Module):
    def __init__(self, V, d=128, layers=2, heads=4, ctx=128):
        super().__init__()
        self.ctx = ctx
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        h = self.enc(self.emb(x) + self.pos(torch.arange(T))[None], mask=mask, is_causal=True)
        return self.out(self.ln(h))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def lstm_with_params(V, target):
    best = None
    for k in range(1, 64):
        p = n_params(LSTMLM(V, k, k))
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (k, p)
    return best[0]


def train_lstm(model, train, test, minutes, lr, batch=64, bptt=128, log=print, label="lstm"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    L = (train.numel() - 1) // batch
    data = train[: L * batch + 1]
    X = data[:-1].reshape(batch, L); Y = data[1:].reshape(batch, L)
    t0, pos, hc, seen, steps = time.time(), 0, None, 0, 0
    while time.time() - t0 < minutes * 60:
        if pos + bptt > L:
            pos, hc = 0, None
        x, y = X[:, pos:pos + bptt], Y[:, pos:pos + bptt]
        logits, hc = model(x, hc)
        hc = tuple(h.detach() for h in hc)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pos += bptt; seen += x.numel(); steps += 1
        if steps % 200 == 0:
            log(f"  {label}: {seen / 1e6:.2f}M chars, train BPC {loss.item() / LN2:.3f}")
    return seen


@torch.no_grad()
def eval_lstm(model, test):
    logits, _ = model(test[None, :-1])
    return F.cross_entropy(logits[0], test[1:], reduction="sum").item(), test.numel() - 1


def train_transformer(model, train, test, minutes, lr=1e-3, batch=32, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ctx = model.ctx
    t0, seen, steps = time.time(), 0, 0
    g = torch.Generator().manual_seed(0)
    while time.time() - t0 < minutes * 60:
        frac = (time.time() - t0) / (minutes * 60)
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, (steps + 1) / 200) * (0.1 + 0.9 * (1 - frac))
        s = torch.randint(0, train.numel() - ctx - 1, (batch,), generator=g)
        x = torch.stack([train[i:i + ctx] for i in s]); y = torch.stack([train[i + 1:i + ctx + 1] for i in s])
        loss = F.cross_entropy(model(x).reshape(-1, model.out.out_features), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen += x.numel(); steps += 1
        if steps % 200 == 0:
            log(f"  transformer: {seen / 1e6:.2f}M chars, train BPC {loss.item() / LN2:.3f}")
    return seen


@torch.no_grad()
def eval_transformer(model, test):
    ctx, half = model.ctx, model.ctx // 2
    nll, cnt = 0.0, 0
    lg = model(test[None, :ctx])[0]
    nll += F.cross_entropy(lg[:-1], test[1:ctx], reduction="sum").item(); cnt += ctx - 1
    s = half
    while s + ctx <= test.numel() - 1:            # stride half a context, score the new half
        lg = model(test[None, s:s + ctx])[0]
        nll += F.cross_entropy(lg[half:], test[s + half + 1:s + ctx + 1], reduction="sum").item()
        cnt += ctx - half; s += half
    return nll, cnt


def ngram_eval(train_ids, test_ids, V, order, d=0.75):
    tr = "".join(chr(256 + i) for i in train_ids.tolist())
    te = "".join(chr(256 + i) for i in test_ids.tolist())
    counts, ctx_tot, ctx_types = [], [], []
    for n in range(1, order + 1):
        c = Counter(tr[i:i + n] for i in range(len(tr) - n + 1))
        tot, typ = Counter(), Counter()
        for g, k in c.items():
            tot[g[:-1]] += k; typ[g[:-1]] += 1
        counts.append(c); ctx_tot.append(tot); ctx_types.append(typ)
    nll = 0.0
    for i in range(1, len(te)):
        p = (counts[0].get(te[i], 0) + 1) / (len(tr) + V)                 # add-one unigram
        for n in range(2, order + 1):
            if i - n + 1 < 0:
                break
            h = te[i - n + 1:i]
            tot = ctx_tot[n - 1].get(h, 0)
            if tot == 0:
                continue
            p = max(counts[n - 1].get(h + te[i], 0) - d, 0) / tot + d * ctx_types[n - 1][h] / tot * p
        nll -= math.log(p)
    return nll, len(te) - 1


# =============================================================================
# 5. TORCH (CPU) TRAINING AND EVALUATION OF THE UNITARY MODEL
# =============================================================================

def train_unitary(model, train, minutes, batch, bptt, lr, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    L = train.numel() // batch
    streams = train[: L * batch].reshape(batch, L)
    psi = model.init_state(streams[:, 0])
    t0, pos, seen, steps, hist = time.time(), 0, 0, 0, []
    while time.time() - t0 < minutes * 60:
        frac = (time.time() - t0) / (minutes * 60)
        for pg in opt.param_groups:
            pg["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if pos + bptt + 1 > L:
            pos = 0; psi = model.init_state(streams[:, 0])
        M = model.blocks()
        total = 0.0
        for t in range(bptt):
            nll, psi = checkpoint(model.step, psi, M, streams[:, pos + t + 1], use_reentrant=False)
            total = total + nll.mean()
        loss = total / bptt
        opt.zero_grad(); loss.backward(); opt.step()
        psi = psi.detach()
        pos += bptt; seen += batch * bptt; steps += 1
        hist.append(loss.item() / LN2)
        if steps % 25 == 0:
            log(f"  unitary: {seen / 1e3:.0f}k chars, train BPC {np.mean(hist[-25:]):.3f}, "
                f"{(time.time() - t0) / 60:.1f} min")
    return seen, hist


@torch.no_grad()
def eval_unitary(model, test, streams=32):
    L = (test.numel() - 1) // streams
    S = test[: L * streams + 1]
    X = torch.stack([S[i * L:(i + 1) * L + 1] for i in range(streams)])
    psi = model.init_state(X[:, 0])
    M = model.blocks()
    nll = 0.0
    for t in range(L):
        n, psi = model.step(psi, M, X[:, t + 1])
        nll += n.sum().item()
    return nll, L * streams


def gradcheck_cl(a, log=print):
    """Compare the OpenCL gradient with torch autograd on one chunk of a small model."""
    torch.manual_seed(0)
    model = UnitaryBorn(4, 4, 3)
    rng = np.random.default_rng(0)
    B, T = 4, 5
    ids = rng.integers(0, 16, size=B * a.devices * 200).astype(np.int64)
    a2 = argparse.Namespace(**vars(a)); a2.batch, a2.bptt, a2.eval_streams = B, T, 2
    tr = CLTrainer(model, ids, ids[:401], a2, log)
    got = []
    for c in range(2):                                    # two chunks: the second uses the carried state
        loss_cl, _ = tr.gradient(c * T, c == 0)
        got.append((loss_cl, model.theta.grad.clone()))
    tr.close()
    Btot, L = B * a.devices, len(ids) // (B * a.devices)
    X = torch.from_numpy(ids[: Btot * L].reshape(Btot, L))
    psi = model.init_state(X[:, 0])
    rel = 0.0
    for c in range(2):
        model.theta.grad = None
        M = model.blocks(); total = 0.0
        for t in range(T):
            nll, psi = model.step(psi, M, X[:, c * T + t + 1]); total = total + nll.sum()
        loss = total / (Btot * T); loss.backward(); psi = psi.detach()
        r = float((got[c][1] - model.theta.grad).norm() / model.theta.grad.norm())
        rel = max(rel, r)
        log(f"gradient check, chunk {c + 1}: loss OpenCL {got[c][0]:.6f} vs torch {loss.item():.6f}; "
            f"relative gradient difference {r:.1e}")
    return rel


# =============================================================================
# 6. PYQRACK REPLAY
# =============================================================================

def _u3_np(t, p, l):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return [c, -np.exp(1j * l) * s, np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]


def replay_in_qrack(model, vocab, prompt, n_chars, log=print):
    from pyqrack import QrackSimulator
    th = model.theta.detach().double().numpy()
    ops = [op for g, q in enumerate(model.pairs) for op in block_ops(th[g], q)]
    sim = QrackSimulator(model.n)
    first = vocab.encode(prompt[0])
    for j in range(model.v):
        if (int(first[0]) >> j) & 1:
            sim.x(model.vq[j])
    psi = model.init_state(first, dtype=torch.complex128)
    M = model.blocks().detach().to(torch.complex128)
    idx = model.idx.numpy()
    forced = vocab.encode(prompt[1:]).tolist()
    out, diffs, t0 = [int(first[0])], [], time.time()
    for step in range(len(forced) + n_chars):
        for op in ops:
            if op[0] == "u3":
                sim.mtrx(_u3_np(*op[2]), op[1])
            elif op[0] == "ry":
                c, s = math.cos(op[2] / 2), math.sin(op[2] / 2)
                sim.mtrx([c, -s, s, c], op[1])
            elif op[0] == "rz":
                sim.mtrx([np.exp(-0.5j * op[2]), 0, 0, np.exp(0.5j * op[2])], op[1])
            else:
                sim.mcx([op[1]], op[2])
        pq = np.asarray(sim.out_probs())[idx].reshape(2 ** model.m, 2 ** model.v).sum(0)
        with torch.no_grad():
            S, pt = model.probs(model.evolve(psi, M))
        diffs.append(float(np.abs(pq - pt[0].numpy()).max()))
        if step < len(forced):
            k = forced[step]                                        # prompt: post-select the known char
            for j in range(model.v):
                sim.force_m(model.vq[j], bool((k >> j) & 1))
        else:
            k = 0                                                   # free running: one shot, Qrack measures
            for j in range(model.v):
                if sim.m(model.vq[j]):
                    k |= 1 << j
        with torch.no_grad():
            psi = model.project(S, pt, torch.tensor([k]))
        out.append(k)
    return dict(text=vocab.decode(out), prompt=prompt, chars=n_chars, max_prob_diff=max(diffs), gates_per_char=len(ops),
                measured_qubits_per_char=model.v, shots_per_char=1, qubits=model.n, seconds=time.time() - t0)


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=120, help="total budget when --hours is not given (55%% unitary)")
    ap.add_argument("--hours", type=float, default=0, help="unitary training time; baselines get --baseline_minutes")
    ap.add_argument("--baseline_minutes", type=float, default=40, help="CPU baselines when --hours is given")
    ap.add_argument("--dataset", default="wikitext-2-raw-v1", help="Salesforce/wikitext config, e.g. wikitext-103-raw-v1")
    ap.add_argument("--train", default="", help="training text file (default: <dataset>_train.txt, downloaded if missing)")
    ap.add_argument("--test", default="", help="test text file (default: <dataset>_test.txt)")
    ap.add_argument("--eval_chars", type=int, default=65536)
    ap.add_argument("--baseline_chars", type=int, default=0, help="cap the baselines' training text (0 = all)")
    ap.add_argument("--mem_qubits", type=int, default=7)
    ap.add_argument("--vocab_qubits", type=int, default=7, help="vocabulary = 2^v characters (last one = unknown)")
    ap.add_argument("--depth", type=int, default=12, help="brick-wall layers of 2-qubit blocks per character")
    ap.add_argument("--batch", type=int, default=32, help="streams (per device with --backend opencl)")
    ap.add_argument("--bptt", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--backend", choices=["torch", "opencl"], default="torch")
    ap.add_argument("--devices", type=int, default=1, help="OpenCL devices (one worker process each)")
    ap.add_argument("--device_type", choices=["gpu", "all", "cpu"], default="gpu")
    ap.add_argument("--platform", default="", help="only OpenCL platforms whose name contains this (e.g. rusticl, AMD)")
    ap.add_argument("--list_devices", action="store_true", help="list OpenCL devices and exit")
    ap.add_argument("--oversubscribe", action="store_true", help="allow more workers than devices (testing)")
    ap.add_argument("--eval_streams", type=int, default=32)
    ap.add_argument("--eval_minutes", type=float, default=30, help="periodic test evaluation (opencl backend)")
    ap.add_argument("--ckpt_minutes", type=float, default=10)
    ap.add_argument("--log_seconds", type=float, default=60)
    ap.add_argument("--resume", default="", help="checkpoint to continue from (opencl backend)")
    ap.add_argument("--keep_schedule", action="store_true", help="on resume keep the original planned duration")
    ap.add_argument("--gradcheck", action="store_true", help="check the OpenCL gradient against torch and exit")
    ap.add_argument("--baselines", default="ngram,lstm_equal,lstm,transformer", help="comma list, or none")
    ap.add_argument("--prompt", default="The ")
    ap.add_argument("--replay_chars", type=int, default=160)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="unitary_results.json")
    ap.add_argument("--save", default="unitary_model.pt")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    log = lambda s: print(s, flush=True)

    if a.list_devices:
        import pyopencl as cl
        for pi, p in enumerate(cl.get_platforms()):
            for di, d in enumerate(p.get_devices()):
                log(f"platform {pi} {p.name.strip()!r}, device {di}: {d.name.strip()}, "
                    f"{cl.device_type.to_string(d.type)}, {d.global_mem_size / 2 ** 30:.1f} GiB")
        return
    if a.gradcheck:
        a.backend = "opencl"
        rel = gradcheck_cl(a, log)
        sys.exit(0 if rel < 1e-3 else 1)

    tag = a.dataset.replace("-raw-v1", "").replace("-v1", "")
    train_text = load_text(a.train or f"{tag}_train.txt", "train", a.dataset)
    test_text = load_text(a.test or f"{tag}_test.txt", "test", a.dataset)
    V = 2 ** a.vocab_qubits
    vocab = CharVocab(train_text, V)
    train_np = vocab.encode_np(train_text)
    test_np = vocab.encode_np(test_text)[: a.eval_chars + 1]
    del train_text
    log(f"{a.dataset}: train {train_np.size:,} chars, test {test_np.size - 1:,} chars, vocabulary {V} "
        f"(unknown: {(train_np == vocab.unk).mean() * 100:.2f}% train, {(test_np == vocab.unk).mean() * 100:.2f}% test)")

    base = [b for b in a.baselines.split(",") if b and b != "none"]
    share = {"lstm_equal": 0.07, "lstm": 0.13, "transformer": 0.13}
    if a.hours:
        a.train_minutes = a.hours * 60
        tot = sum(share[b] for b in base if b in share) or 1
        bmin = {b: a.baseline_minutes * share[b] / tot for b in share}
    else:
        a.train_minutes = a.minutes * (0.88 - sum(share[b] for b in base if b in share))
        bmin = {b: a.minutes * share[b] for b in share}
    results = []

    def record(name, params, nll_cnt, seen, minutes, state):
        nll, cnt = nll_cnt
        bpc = nll / cnt / LN2
        r = dict(model=name, params=params, state=state, train_chars=seen, train_minutes=minutes,
                 bpc=bpc, char_ppl=math.exp(nll / cnt))
        results.append(r)
        log(f"== {name}: BPC {bpc:.3f}, char PPL {r['char_ppl']:.2f}, {params:,} params, {seen / 1e6:.2f}M chars")

    # unitary model
    model = UnitaryBorn(a.mem_qubits, a.vocab_qubits, a.depth)
    state_txt = f"{2 ** a.mem_qubits} complex amplitudes ({a.mem_qubits} qubits) after each measurement"
    log(f"\nunitary Born machine: {model.n} qubits ({a.mem_qubits} memory + {a.vocab_qubits} vocabulary), "
        f"depth {a.depth}, {len(model.pairs)} blocks, {model.n_params():,} angles, backend {a.backend}, "
        f"training {a.train_minutes:.0f} min")
    extra = {}
    if a.backend == "opencl":
        tr = CLTrainer(model, train_np, test_np, a, log)
        log(f"  {a.devices} workers x {a.batch} streams = {a.devices * a.batch} streams, bptt {a.bptt}, "
            f"stream length {tr.L:,} chars")
        st, interrupted = train_unitary_cl(model, tr, a, log)
        nll, cnt = tr.evaluate()
        tr.close()
        seen, hist = st["seen"], st["hist"]
        extra = dict(evals=st["evals"], interrupted=interrupted, elapsed_h=st["elapsed"] / 3600)
        name = f"unitary Born machine ({a.devices} OpenCL devices)"
    else:
        train_t = torch.from_numpy(train_np.astype(np.int64))
        seen, hist = train_unitary(model, train_t, a.train_minutes, a.batch, a.bptt, a.lr, log)
        if a.save:
            torch.save(dict(theta=model.theta.detach(), mem=a.mem_qubits, voc=a.vocab_qubits, depth=a.depth,
                            chars=vocab.chars), a.save)
        nll, cnt = eval_unitary(model, torch.from_numpy(test_np.astype(np.int64)), a.eval_streams)
        name = "unitary Born machine"
    record(name, model.n_params(), (nll, cnt), seen, a.train_minutes, state_txt)

    # baselines (CPU)
    train = torch.from_numpy((train_np[: a.baseline_chars] if a.baseline_chars else train_np).astype(np.int64))
    test = torch.from_numpy(test_np.astype(np.int64))
    if "ngram" in base:
        for order in (1, 3, 5):
            t0 = time.time()
            nc = ngram_eval(train, test, V, order)
            label = "1-gram (add-one)" if order == 1 else f"{order}-gram (interpolated, discount 0.75)"
            record(label, 0, nc, train.numel(), (time.time() - t0) / 60, "none" if order == 1 else f"last {order - 1} chars")
    if "lstm_equal" in base:
        k = lstm_with_params(V, model.n_params())
        m = LSTMLM(V, k, k)
        log(f"\nLSTM with equal parameters: embedding {k}, hidden {k}, {n_params(m):,} params")
        s = train_lstm(m, train, test, bmin["lstm_equal"], lr=1e-2, log=log, label="lstm_equal")
        record("LSTM, equal parameters", n_params(m), eval_lstm(m, test), s, bmin["lstm_equal"], f"{k} + {k} reals")
    if "lstm" in base:
        m = LSTMLM(V, 64, 256)
        log(f"\nLSTM large: embedding 64, hidden 256, {n_params(m):,} params")
        s = train_lstm(m, train, test, bmin["lstm"], lr=3e-3, log=log, label="lstm")
        record("LSTM, hidden 256", n_params(m), eval_lstm(m, test), s, bmin["lstm"], "256 + 256 reals")
    if "transformer" in base:
        m = TinyTransformer(V)
        log(f"\ntransformer: 2 layers, d 128, 4 heads, context 128, {n_params(m):,} params")
        s = train_transformer(m, train, test, bmin["transformer"], log=log)
        record("transformer, 2 layers d 128", n_params(m), eval_transformer(m, test), s, bmin["transformer"],
               "context 128 chars")

    # replay
    log("\nPyQrack replay: real gates, mid-circuit measurement of the vocabulary register, one shot per char")
    rep = replay_in_qrack(model, vocab, a.prompt, a.replay_chars, log=log)
    log(f"sample: {rep['text']!r}")
    log(f"max |P_qrack - P_torch| over {len(a.prompt) - 1 + a.replay_chars} steps: {rep['max_prob_diff']:.1e}; "
        f"{rep['gates_per_char']} gates and 1 shot ({rep['measured_qubits_per_char']} measured qubits) per char")

    log("\n| Model | Parameters | Recurrent state | Training chars | BPC | Char perplexity |\n|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: r["bpc"]):
        log(f"| {r['model']} | {r['params']:,} | {r['state']} | {r['train_chars'] / 1e6:.2f}M | {r['bpc']:.3f} | {r['char_ppl']:.2f} |")
    json.dump(dict(args=vars(a), results=results, replay=rep, unitary_train_bpc=hist, **extra), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
