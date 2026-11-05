#!/usr/bin/env python3
"""ThereminQ Holoqubed Live: interactive generation with realtime performance and geometry.

A local web dashboard served by the Python standard library (no extra dependencies, no CDN). Type a
prompt, choose the backend and z, and watch each generated token arrive with:

  tokens/s, ms per token, prefill time, MLP work, MLP time, process RSS and CPU
  per-layer heatmap over tokens: routed neurons, MLP work, MLP time or MLP error vs dense
  the chart plane of one layer: Voronoi region centroids, the Hilbert curve through them,
  and the token's path across the regions
  run history, so backends and z values can be compared side by side

Backends: torch (the original MLP modules), reference (atlas, NumPy), opencl (atlas, OpenCL kernels).
Without --atlas only torch runs and the geometry panels stay empty.

  python3 holoqubed_live.py --model_id facebook/opt-125m --atlas opt-125m.atlas
  open http://127.0.0.1:8765
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
if os.path.exists(QRACK_LIB_PATH):
    os.environ.setdefault("PYQRACK_SHARED_LIB_PATH", QRACK_LIB_PATH)

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holoqubed_atlas as H  # noqa: E402


# =============================================================================
# 1. INSTRUMENTED RUNTIME
# =============================================================================

def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


class LiveRuntime(H.AtlasRuntime):
    """AtlasRuntime that also runs the original torch MLPs and records per-layer stats of every step."""

    def __init__(self, dictionary, torch_mlps, **kw):
        super().__init__(dictionary, **kw)
        self.torch_mlps = torch_mlps
        self.measure_error = False
        self.step_stats = None
        n = len(dictionary)
        self.cost = []
        for L in dictionary.layers:
            nn_, d = L.A["W1"].shape
            dout = L.A["W2T"].shape[1]
            r = L.A["Kp"].shape[1]
            glu = 2 if L.glu else 1
            self.cost.append((nn_, d, dout, r, glu))
        self.n_layers = n

    def begin_step(self):
        n = self.n_layers
        self.step_stats = dict(routed=[0.0] * n, nonzero=[0.0] * n, work=[0.0] * n, ms=[0.0] * n,
                               err=[None] * n, region=[0] * n, xy=[[0.0, 0.0]] * n, scale=[1.0] * n)

    def run(self, i: int, X: np.ndarray) -> np.ndarray:
        layer = self.dict[i]
        t0 = time.perf_counter()
        if self.backend == "torch":
            with torch.no_grad():
                Y = self.torch_mlps[i](torch.from_numpy(X)).numpy()
            n = layer.A["W1"].shape[0]
            self.routed += X.shape[0] * n
            self.nonzero += X.shape[0] * n
            self.slots += X.shape[0] * n
            rf = nf = 1.0
        else:
            r0, z0, s0 = self.routed, self.nonzero, self.slots
            Y = super().run(i, X)
            ds = max(1, self.slots - s0)
            rf, nf = (self.routed - r0) / ds, (self.nonzero - z0) / ds
        ms = (time.perf_counter() - t0) * 1e3
        st = self.step_stats
        if st is not None:
            nn_, d, dout, r, glu = self.cost[i]
            if self.backend == "torch":
                work = 1.0
            else:
                work = (d * r + nn_ * r * glu + rf * nn_ * d * glu + nf * nn_ * dout) / (nn_ * d * glu + nn_ * dout)
            Z, region, scale = layer.planner.plan(X[-1:])
            st["routed"][i], st["nonzero"][i], st["work"][i], st["ms"][i] = rf, nf, work, ms
            st["region"][i], st["xy"][i], st["scale"][i] = int(region[0]), [float(Z[0, 0]), float(Z[0, 1])], float(scale[0])
            if self.measure_error and self.backend != "torch":
                Yd = layer.dense(X[-1:])
                st["err"][i] = float(np.linalg.norm(Y[-1] - Yd[0]) / max(np.linalg.norm(Yd[0]), 1e-12))
            elif self.measure_error:
                st["err"][i] = 0.0
        return Y


def _torch_mlp_callables(model):
    """Callables that reproduce each original MLP on its hook input, captured before injection."""
    out = []
    for rec in H.find_mlps(model):
        owner = rec["owner"]
        if hasattr(owner, "fc1") and hasattr(owner, "fc2"):
            fc1, act, fc2 = owner.fc1, owner.activation_fn, owner.fc2
            out.append(lambda x, fc1=fc1, act=act, fc2=fc2: fc2(act(fc1(x))))
        else:
            out.append(model.get_submodule(rec["name"]))
    return out


# =============================================================================
# 2. SESSION
# =============================================================================

class LiveSession:
    def __init__(self, args):
        self.args = args
        self.tok, self.model = H.load_model(args.model_id, args.trust_remote_code)
        if args.threads:
            torch.set_num_threads(args.threads)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.dic = self.rt = None
        self.engine = None
        self.backends = ["torch"]
        if args.atlas:
            torch_mlps = _torch_mlp_callables(self.model)
            self.dic = H.AtlasDictionary(args.atlas, lookup=args.lookup)
            self.rt = LiveRuntime(self.dic, torch_mlps, backend="torch", z=3.5, kappa=args.kappa)
            H.inject_atlas(self.model, self.rt)
            self.backends += ["reference", "opencl"]
            if not args.no_warmup:
                t0 = time.perf_counter()
                for L in self.dic.layers:  # page the memory-mapped atlas in, so the first prefill is not a disk read
                    for arr in L.A.values():
                        if isinstance(arr, np.ndarray) and arr.size:
                            float(np.asarray(arr).ravel()[::1024].sum())
                print(f"atlas paged in: {time.perf_counter() - t0:.1f} s", flush=True)
        self.history = []

    def info(self) -> dict:
        cfg = self.model.config
        out = dict(model=self.args.model_id, backends=self.backends, layers=0, atlas=None,
                   params=sum(p.numel() for p in self.model.parameters()),
                   threads=torch.get_num_threads(), history=self.history)
        if self.dic is not None:
            layers = []
            for L in self.dic.layers:
                A = L.A
                cent = np.asarray(A["cent"])[:, :2]
                layers.append(dict(kind=L.kind, neurons=int(A["W1"].shape[0]), dim=int(A["W1"].shape[1]),
                                   rank=int(A["Kp"].shape[1]), regions=int(cent.shape[0]),
                                   cent=np.round(cent, 4).tolist(),
                                   curve=[int(c) for c in np.asarray(A["curve_regions"])]))
            out.update(layers=len(layers), atlas=dict(path=self.args.atlas, layers=layers))
        else:
            out["layers"] = getattr(cfg, "num_hidden_layers", 0)
        return out

    def _ensure_engine(self):
        if self.engine is None:
            cl = H._require_cl()
            devs = [d for p in cl.get_platforms() for d in p.get_devices()]
            cpu_only = bool(devs) and not any(d.type & cl.device_type.GPU for d in devs)
            # a CPU OpenCL device shares system RAM: use the mapped atlas in place instead of a second copy
            host = self.args.host_dictionary or cpu_only
            self.engine = H.AtlasCLEngine(self.dic, max_devices=self.args.gpus, host_dictionary=host)
            print(f"OpenCL dictionary: {'zero-copy host memory' if host else 'uploaded to device memory'}", flush=True)
            self.rt.engine = self.engine
            self.engine_name = ", ".join(d.name.strip() for d in self.engine.devices)

    def generate(self, body: dict, emit):
        prompt = str(body.get("prompt") or "")
        if not prompt:
            raise ValueError("prompt is empty")
        backend = body.get("backend", "torch")
        if backend not in self.backends:
            raise ValueError(f"backend {backend} not available (have {', '.join(self.backends)})")
        max_new = max(1, min(int(body.get("max_new_tokens", 32)), 2048))
        temperature = float(body.get("temperature", 0.0))
        top_k = int(body.get("top_k", 40))
        seed = body.get("seed")
        z = float(body.get("z", 3.5))
        if self.rt is not None:
            if backend == "opencl":
                self._ensure_engine()
            self.rt.backend, self.rt.z = backend, z
            self.rt.measure_error = bool(body.get("measure_error", False))
            self.rt.reset()
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        ids = self.tok(prompt, return_tensors="pt").input_ids
        emit(dict(type="start", backend=backend, z=z, prompt_tokens=int(ids.shape[1]),
                  device=getattr(self, "engine_name", "CPU") if backend == "opencl" else "CPU"))
        past, cur, generated = None, ids, []
        prev_text = ""
        eos = self.tok.eos_token_id
        decode_s = 0.0
        prefill_ms = 0.0
        works = []
        self.stop.clear()
        wall0 = time.perf_counter()
        for step in range(max_new):
            if self.stop.is_set():
                break
            if self.rt is not None:
                self.rt.begin_step()
            c0, t0 = os.times(), time.perf_counter()
            with torch.no_grad():
                out = self.model(input_ids=cur, past_key_values=past, use_cache=True)
            dt = time.perf_counter() - t0
            c1 = os.times()
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            lp = F.log_softmax(logits, -1)
            if temperature > 0:
                k = max(1, min(top_k, logits.numel()))
                v, ix = torch.topk(logits / temperature, k)
                nxt = int(ix[torch.multinomial(F.softmax(v, -1), 1, generator=gen)])
            else:
                nxt = int(logits.argmax())
            ent = float(-(lp.exp() * lp).sum())
            prob = float(lp[nxt].exp())
            cpu = 100.0 * ((c1.user - c0.user) + (c1.system - c0.system)) / max(dt, 1e-9)
            ev = dict(type="step", step=step, prefill=step == 0, tokens_in=int(cur.shape[1]),
                      ms=dt * 1e3, cpu_pct=cpu, rss_mb=_rss_mb(), prob=prob, entropy=ent)
            if self.rt is not None:
                st = self.rt.step_stats
                num = den = 0.0
                for i, (nn_, d, dout, r, glu) in enumerate(self.rt.cost):
                    dense = nn_ * d * glu + nn_ * dout
                    num += st["work"][i] * dense
                    den += dense
                ev.update(layers=st, work=num / den, mlp_ms=float(sum(st["ms"])))
                if step > 0:
                    works.append(num / den)
            if step == 0:
                prefill_ms = dt * 1e3
            else:
                decode_s += dt
            generated.append(nxt)
            text = self.tok.decode(generated, skip_special_tokens=True)
            ev.update(token_id=nxt, text=text[len(prev_text):], token=self.tok.decode([nxt]))
            prev_text = text
            stop_now = len(generated) >= max_new or (eos is not None and nxt == eos)
            emit(ev)
            if stop_now:
                break
            cur = torch.tensor([[nxt]])
        n_dec = max(0, len(generated) - 1)
        run = dict(backend=backend, z=z if backend != "torch" else None, prompt_tokens=int(ids.shape[1]),
                   tokens=len(generated), prefill_ms=prefill_ms,
                   decode_tps=(n_dec / decode_s) if decode_s > 0 else float("nan"),
                   work=float(np.mean(works)) if works else (1.0 if backend == "torch" else float("nan")),
                   wall_s=time.perf_counter() - wall0, stopped=self.stop.is_set(),
                   when=time.strftime("%H:%M:%S"))
        self.history.append(run)
        emit(dict(type="done", run=run, text=prev_text))


# =============================================================================
# 3. HTTP SERVER
# =============================================================================

def _make_handler(sess: LiveSession):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "HoloqubedLive/1"

        def log_message(self, fmt, *a):
            if self.args_verbose:
                print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % a}")

        args_verbose = sess.args.verbose

        def _send(self, status, ctype, data: bytes):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, obj):
            self._send(status, "application/json", json.dumps(obj, allow_nan=False, default=_clean).encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/api/info":
                self._json(200, _sanitize(sess.info()))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "body is not valid JSON"})
            if self.path == "/api/stop":
                sess.stop.set()
                return self._json(200, {"ok": True})
            if self.path != "/api/generate":
                return self._json(404, {"error": "not found"})
            if not sess.lock.acquire(blocking=False):
                return self._json(409, {"error": "a generation is already running"})
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def emit(obj):
                    self.wfile.write(f"data: {json.dumps(_sanitize(obj), allow_nan=False)}\n\n".encode())
                    self.wfile.flush()
                try:
                    sess.generate(body, emit)
                except (BrokenPipeError, ConnectionResetError):
                    sess.stop.set()
                except Exception as e:  # report to the page, keep serving
                    try:
                        emit(dict(type="error", message=f"{type(e).__name__}: {e}"))
                    except OSError:
                        pass
            finally:
                sess.lock.release()

    return Handler


def _clean(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _sanitize(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _sanitize(o.item())
    return o


# =============================================================================
# 4. PAGE
# =============================================================================

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Holoqubed Live</title>
<style>
:root{color-scheme:light;--bg:#f4f4f2;--surface:#fcfcfb;--line:#e3e2de;--grid:#ecebe7;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;
--s1:#2a78d6;--s1soft:#86b6ef;--s2:#eb6834;--seq0:#cde2fb;--seq1:#0d366b;--warn:#c98500;--bad:#e34948}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#2c2c2a;--grid:#242422;
--ink:#f0efec;--ink2:#c3c2b7;--ink3:#8a8984;--s1:#3987e5;--s1soft:#256abf;--s2:#d95926;--seq0:#1d2a3a;--seq1:#9ec5f4;--warn:#c98500;--bad:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#2c2c2a;--grid:#242422;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#8a8984;
--s1:#3987e5;--s1soft:#256abf;--s2:#d95926;--seq0:#1d2a3a;--seq1:#9ec5f4;--warn:#c98500;--bad:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--surface)}
header h1{font-size:17px;margin:0;font-weight:650}header .meta{color:var(--ink2);font-size:13px;flex:1;min-width:0;overflow-wrap:anywhere}
main{padding:16px 20px;display:grid;gap:14px;grid-template-columns:repeat(12,1fr);max-width:1500px;margin:0 auto}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.card h2{font-size:13px;font-weight:600;margin:0 0 8px;color:var(--ink2)}
.c12{grid-column:span 12}.c8{grid-column:span 8}.c6{grid-column:span 6}.c4{grid-column:span 4}
@media (max-width:1000px){.c8,.c6,.c4{grid-column:span 12}}
textarea{width:100%;min-height:64px;resize:vertical;font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
.row{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end;margin-top:10px}
label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--ink2)}
label.inline{flex-direction:row;align-items:center;gap:6px;color:var(--ink)}
select,input[type=number]{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);min-width:90px}
button{font:inherit;font-weight:600;padding:7px 16px;border-radius:8px;border:1px solid var(--s1);background:var(--s1);color:#fff;cursor:pointer}
button.ghost{background:transparent;color:var(--ink);border-color:var(--line)}button:disabled{opacity:.45;cursor:default}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.kpi{border:1px solid var(--line);border-radius:9px;padding:8px 10px}.kpi .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--ink2)}.kpi .s{font-size:11px;color:var(--ink3)}
#out{white-space:pre-wrap;word-break:break-word;min-height:72px;max-height:260px;overflow:auto;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
#out .p{color:var(--ink3)}#out .t{border-radius:3px;padding:1px 0;cursor:default}
canvas{width:100%;display:block}.cv{position:relative}
#tip{position:fixed;pointer-events:none;z-index:9;background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:6px 9px;font-size:12px;
box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;white-space:nowrap}#tip b{font-weight:600}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin-top:6px;align-items:center}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.ramp{display:inline-block;width:90px;height:8px;border-radius:4px;vertical-align:middle;margin:0 6px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
#status{font-size:12px;color:var(--ink2)}#status.err{color:var(--bad)}
.note{font-size:12px;color:var(--ink3)}
</style></head><body>
<header><h1>Holoqubed Live</h1><span class="meta" id="meta">loading</span>
<button class="ghost" id="theme" title="Toggle light and dark">Theme</button></header>
<main>
<section class="card c12">
<textarea id="prompt">The capital of the Netherlands is</textarea>
<div class="row">
<label>Backend<select id="backend"></select></label>
<label>z (ellipsoid width)<input id="z" type="number" value="3.5" step="0.25" min="0"></label>
<label>Max new tokens<input id="maxnew" type="number" value="48" min="1" max="2048"></label>
<label>Temperature<input id="temp" type="number" value="0" step="0.1" min="0"></label>
<label>Seed<input id="seed" type="number" value="0"></label>
<label class="inline" title="Runs the dense MLP of every layer next to the atlas; step latency includes that extra pass, per-layer MLP time does not"><input id="err" type="checkbox"> MLP error vs dense (slower)</label>
<span style="flex:1"></span>
<span id="status"></span>
<button id="run">Generate</button><button class="ghost" id="stop" disabled>Stop</button>
</div></section>

<section class="card c12"><div class="kpis">
<div class="kpi"><div class="l">Decode speed</div><div class="v" id="k_tps">-</div><div class="s">tokens/s, this run</div></div>
<div class="kpi"><div class="l">Last token</div><div class="v" id="k_ms">-</div><div class="s">ms per token</div></div>
<div class="kpi"><div class="l">Prefill</div><div class="v" id="k_pre">-</div><div class="s" id="k_pre_s">ms</div></div>
<div class="kpi"><div class="l">MLP work</div><div class="v" id="k_work">-</div><div class="s">of dense multiply-adds</div></div>
<div class="kpi"><div class="l">MLP time</div><div class="v" id="k_mlp">-</div><div class="s">share of last token</div></div>
<div class="kpi"><div class="l">CPU</div><div class="v" id="k_cpu">-</div><div class="s" id="k_cpu_s">process, last token</div></div>
<div class="kpi"><div class="l">Memory</div><div class="v" id="k_rss">-</div><div class="s">resident MB</div></div>
</div></section>

<section class="card c12"><h2>Output <span class="note">token tint = MLP work of that step (light = little, dark = dense)</span></h2><div id="out"></div></section>

<section class="card c6"><h2>Latency per token (ms)</h2><div class="cv"><canvas id="c_ms" height="200"></canvas></div></section>
<section class="card c6"><h2 id="h_work">MLP work per token (% of dense)</h2><div class="cv"><canvas id="c_work" height="200"></canvas></div></section>

<section class="card c8"><h2>Layers over tokens
<select id="metric" style="margin-left:8px;min-width:0;padding:2px 6px">
<option value="routed">routed neurons</option><option value="work">MLP work</option><option value="ms">MLP time</option><option value="err">MLP error vs dense</option></select></h2>
<div class="cv"><canvas id="c_heat" height="300"></canvas></div>
<div class="legend"><span>low</span><span class="ramp" id="ramp"></span><span>high</span><span id="heat_range" class="note"></span></div></section>
<section class="card c4"><h2>Per layer, last token</h2><div class="cv"><canvas id="c_bars" height="300"></canvas></div></section>

<section class="card c6"><h2>Chart plane, layer <select id="layer" style="min-width:0;padding:2px 6px"></select></h2>
<div class="cv"><canvas id="c_geo" height="360"></canvas></div>
<div class="legend"><span><span class="sw" style="background:var(--ink3)"></span>region centroid</span>
<span><span class="sw" style="background:var(--grid);border:1px solid var(--ink3)"></span>Hilbert curve order</span>
<span><span class="sw" style="background:var(--s1)"></span>token path (faded = older)</span><span><span class="sw" style="background:var(--s2)"></span>current region</span></div>
<div class="note">Chart axes 1 and 2 (largest variance directions of the layer's MLP inputs), each axis scaled to the 2 to 98% range; hollow dots lie outside it and are drawn at the edge. Regions are assigned in all chart dimensions, so the current token and its region's centroid need not sit together in this 2-D view.</div></section>
<section class="card c6"><h2>Regions visited, layer <span id="layer2"></span></h2><div class="cv"><canvas id="c_reg" height="360"></canvas></div>
<div class="note">Region id per token, regions ordered along the Hilbert curve; dot size = distance outside the region's typical radius.</div></section>

<section class="card c12"><h2>Run history</h2><table id="hist"><thead><tr><th>Time</th><th>Backend</th><th>z</th><th>Prompt tokens</th><th>Tokens</th>
<th>Prefill ms</th><th>Decode tokens/s</th><th>MLP work</th><th>Wall s</th></tr></thead><tbody></tbody></table></section>
</main><div id="tip"></div>
<script>
const $=s=>document.querySelector(s);let INFO=null,RUN=null,busy=false,ctl=null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function hex(h){h=h.replace('#','');return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]}
function ramp(t){t=Math.max(0,Math.min(1,t));const a=hex(css('--seq0')),b=hex(css('--seq1'));return`rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`}
function fmt(v,d=1){return v==null||!isFinite(v)?'-':v.toFixed(d)}
const tip=$('#tip');function showTip(e,html){tip.innerHTML=html;tip.style.display='block';const w=tip.offsetWidth,h=tip.offsetHeight;
let x=e.clientX+14,y=e.clientY+14;if(x+w>innerWidth-8)x=e.clientX-w-14;if(y+h>innerHeight-8)y=e.clientY-h-14;tip.style.left=x+'px';tip.style.top=y+'px'}
function hideTip(){tip.style.display='none'}
function setup(cv){const r=window.devicePixelRatio||1,w=cv.clientWidth,h=+(cv.dataset.h||(cv.dataset.h=cv.getAttribute('height')));cv.width=w*r;cv.height=h*r;cv.style.height=h+'px';
const g=cv.getContext('2d');g.setTransform(r,0,0,r,0,0);g.clearRect(0,0,w,h);g.font='11px system-ui,sans-serif';return[g,w,h]}
function nice(max){if(!(max>0))return 1;const p=Math.pow(10,Math.floor(Math.log10(max))),m=max/p;return(m<=1?1:m<=2?2:m<=2.5?2.5:m<=5?5:10)*p}
function qtl(a,q){const b=a.slice().sort((x,y)=>x-y);if(!b.length)return 0;const i=(b.length-1)*q,lo=Math.floor(i);return b[lo]+(b[Math.min(lo+1,b.length-1)]-b[lo])*(i-lo)}

// ---------- line chart with crosshair
function lineChart(cv,xs,ys,o){const[g,w,h]=setup(cv);const L=44,R=10,T=8,B=24,pw=w-L-R,ph=h-T-B;cv._hit=null;
const n=ys.length,ymax=nice(Math.max(o.minMax||0,...ys.filter(v=>v!=null)));
g.strokeStyle=css('--grid');g.fillStyle=css('--ink3');g.lineWidth=1;g.textAlign='right';g.textBaseline='middle';
for(let i=0;i<=4;i++){const y=T+ph-ph*i/4;g.beginPath();g.moveTo(L,y+.5);g.lineTo(L+pw,y+.5);g.stroke();g.fillText(fmt(ymax*i/4,ymax<10?1:0),L-6,y)}
g.textAlign='center';g.textBaseline='top';const X=i=>L+(n<2?pw/2:pw*i/(n-1));
const step=Math.max(1,Math.ceil(n/8));for(let i=0;i<n;i+=step)g.fillText(xs[i],X(i),T+ph+6);
if(!n){g.fillStyle=css('--ink3');g.fillText('no data yet',L+pw/2,T+ph/2);return}
g.strokeStyle=css('--s1');g.lineWidth=2;g.lineJoin='round';g.beginPath();let first=true;
ys.forEach((v,i)=>{if(v==null){first=true;return}const y=T+ph-ph*v/ymax;first?g.moveTo(X(i),y):g.lineTo(X(i),y);first=false});g.stroke();
if(n<=60){g.fillStyle=css('--s1');ys.forEach((v,i)=>{if(v==null)return;g.beginPath();g.arc(X(i),T+ph-ph*v/ymax,2.5,0,7);g.fill()})}
cv._hit=e=>{const b=cv.getBoundingClientRect(),mx=e.clientX-b.left;const i=Math.round((mx-L)/(n<2?1:pw/(n-1)));if(i<0||i>=n||ys[i]==null)return hideTip();
lineChart(cv,xs,ys,o);const[g2]=[cv.getContext('2d')];g2.strokeStyle=css('--ink3');g2.lineWidth=1;g2.beginPath();g2.moveTo(X(i)+.5,T);g2.lineTo(X(i)+.5,T+ph);g2.stroke();
g2.fillStyle=css('--s1');g2.strokeStyle=css('--surface');g2.lineWidth=2;g2.beginPath();g2.arc(X(i),T+ph-ph*ys[i]/ymax,5,0,7);g2.fill();g2.stroke();
showTip(e,o.tip(i))}}

// ---------- heatmap layers x tokens
function heat(cv,M,o){const[g,w,h]=setup(cv);const L=34,R=6,T=4,B=22,pw=w-L-R,ph=h-T-B;cv._hit=null;const nl=M.length,nt=nl?M[0].length:0;
if(!nt){g.fillStyle=css('--ink3');g.textAlign='center';g.fillText(INFO&&INFO.atlas?'no data yet':'needs --atlas',w/2,h/2);return}
let lo=Infinity,hi=-Infinity;M.forEach(r=>r.forEach(v=>{if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}));if(o.zero)lo=0;if(!(hi>lo))hi=lo+1e-9;
const cw=pw/nt,ch=ph/nl,gap=(cw>4&&ch>4)?1:0;
for(let l=0;l<nl;l++)for(let t=0;t<nt;t++){const v=M[l][t];g.fillStyle=v==null?css('--grid'):ramp((v-lo)/(hi-lo));g.fillRect(L+t*cw,T+l*ch,Math.max(.5,cw-gap),Math.max(.5,ch-gap))}
g.fillStyle=css('--ink3');g.textAlign='right';g.textBaseline='middle';const ls=Math.max(1,Math.ceil(nl/10));for(let l=0;l<nl;l+=ls)g.fillText('L'+l,L-5,T+l*ch+ch/2);
g.textAlign='center';g.textBaseline='top';const ts=Math.max(1,Math.ceil(nt/10));for(let t=0;t<nt;t+=ts)g.fillText(t===0?'pre':t,L+t*cw+cw/2,T+ph+5);
$('#heat_range').textContent=`${o.f(lo)} to ${o.f(hi)}`;
cv._hit=e=>{const b=cv.getBoundingClientRect(),t=Math.floor((e.clientX-b.left-L)/cw),l=Math.floor((e.clientY-b.top-T)/ch);
if(t<0||t>=nt||l<0||l>=nl)return hideTip();showTip(e,`<b>layer ${l}</b>, ${t===0?'prefill':'token '+t}<br>${o.name}: <b>${o.f(M[l][t])}</b>${RUN.tok[t]!=null?'<br>token '+esc(JSON.stringify(RUN.tok[t])):''}`)}}

// ---------- bars per layer
function bars(cv,vals,o){const[g,w,h]=setup(cv);const L=50,R=6,T=6,B=22,pw=w-L-R,ph=h-T-B;cv._hit=null;const n=vals.length;
if(!n){g.fillStyle=css('--ink3');g.textAlign='center';g.fillText(INFO&&INFO.atlas?'no data yet':'needs --atlas',w/2,h/2);return}
const max=nice(Math.max(o.minMax||0,...vals.map(v=>v||0)));g.strokeStyle=css('--grid');g.fillStyle=css('--ink3');g.textAlign='right';g.textBaseline='middle';
for(let i=0;i<=4;i++){const y=T+ph-ph*i/4;g.beginPath();g.moveTo(L,y+.5);g.lineTo(L+pw,y+.5);g.stroke();g.fillText((o.axis||o.f)(max*i/4),L-5,y)}
const bw=pw/n,gap=Math.min(2,bw*.25);g.fillStyle=css('--s1');
vals.forEach((v,i)=>{if(v==null)return;const bh=ph*v/max,x=L+i*bw+gap/2,y=T+ph-bh,ww=bw-gap,r=Math.min(4,ww/2,bh);
g.beginPath();g.moveTo(x,T+ph);g.lineTo(x,y+r);g.quadraticCurveTo(x,y,x+r,y);g.lineTo(x+ww-r,y);g.quadraticCurveTo(x+ww,y,x+ww,y+r);g.lineTo(x+ww,T+ph);g.fill()});
g.fillStyle=css('--ink3');g.textAlign='center';g.textBaseline='top';const s=Math.max(1,Math.ceil(n/8));for(let i=0;i<n;i+=s)g.fillText('L'+i,L+i*bw+bw/2,T+ph+5);
cv._hit=e=>{const b=cv.getBoundingClientRect(),i=Math.floor((e.clientX-b.left-L)/bw);if(i<0||i>=n)return hideTip();showTip(e,`<b>layer ${i}</b><br>${o.name}: <b>${o.f(vals[i])}</b>`)}}

// ---------- chart plane
function geo(cv,li){const[g,w,h]=setup(cv);cv._hit=null;if(!INFO||!INFO.atlas){g.fillStyle=css('--ink3');g.textAlign='center';g.fillText('needs --atlas',w/2,h/2);return}
const A=INFO.atlas.layers[li],C=A.cent,trail=RUN?RUN.xy.map(s=>s[li]):[],reg=RUN?RUN.region.map(s=>s[li]):[];
const pts=C.concat(trail),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let x0=qtl(xs,.02),x1=qtl(xs,.98),y0=qtl(ys,.02),y1=qtl(ys,.98);
const mx_=(x1-x0)*.08||1,my_=(y1-y0)*.08||1;x0-=mx_;x1+=mx_;y0-=my_;y1+=my_;const pad=14,sx=(w-2*pad)/(x1-x0),sy=(h-2*pad)/(y1-y0);
const out=p=>p[0]<x0||p[0]>x1||p[1]<y0||p[1]>y1;
const P=p=>[pad+(Math.min(x1,Math.max(x0,p[0]))-x0)*sx,h-(pad+(Math.min(y1,Math.max(y0,p[1]))-y0)*sy)];
g.strokeStyle=css('--grid');g.lineWidth=1.5;g.beginPath();A.curve.forEach((c,i)=>{const[x,y]=P(C[c]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
const cur=reg.length?reg[reg.length-1]:-1;
C.forEach((c,i)=>{const[x,y]=P(c);g.beginPath();g.arc(x,y,3,0,7);if(out(c)){g.strokeStyle=css('--ink3');g.lineWidth=1.5;g.stroke()}else{g.fillStyle=css('--ink3');g.fill()}});
if(cur>=0){const[x,y]=P(C[cur]);g.strokeStyle=css('--s2');g.lineWidth=2;g.beginPath();g.arc(x,y,9,0,7);g.stroke()}
if(trail.length){const nT=trail.length,age=i=>nT<2?1:.15+.85*i/(nT-1);g.strokeStyle=css('--s1');g.lineWidth=1.5;
for(let i=1;i<nT;i++){const[a,b_]=[P(trail[i-1]),P(trail[i])];g.globalAlpha=.6*age(i);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b_[0],b_[1]);g.stroke()}
trail.forEach((p,i)=>{const[x,y]=P(p),last=i===nT-1;g.globalAlpha=last?1:age(i);g.fillStyle=css('--s1');g.strokeStyle=css('--surface');g.lineWidth=2;g.beginPath();g.arc(x,y,last?6:3.5,0,7);g.fill();if(last)g.stroke()});g.globalAlpha=1}
cv._hit=e=>{const b=cv.getBoundingClientRect(),mx=e.clientX-b.left,my=e.clientY-b.top;let best=null,bd=144;
trail.forEach((p,i)=>{const[x,y]=P(p),d=(x-mx)**2+(y-my)**2;if(d<bd){bd=d;best=['t',i]}});
C.forEach((p,i)=>{const[x,y]=P(p),d=(x-mx)**2+(y-my)**2;if(d<bd){bd=d;best=['c',i]}});
if(!best)return hideTip();if(best[0]==='c'){const pos=A.curve.indexOf(best[1]);showTip(e,`<b>region ${best[1]}</b><br>Hilbert position ${pos} of ${C.length}${out(C[best[1]])?'<br>outside the view, drawn at the edge':''}`)}
else{const i=best[1];showTip(e,`<b>${i===0?'prefill (last prompt token)':'token '+i}</b> ${RUN.tok[i]!=null?esc(JSON.stringify(RUN.tok[i])):''}<br>region ${reg[i]}, scale ${fmt(RUN.scale[i][li],2)}`)}}}

// ---------- regions over tokens (dot strip)
function regions(cv,li){const[g,w,h]=setup(cv);cv._hit=null;if(!INFO||!INFO.atlas||!RUN||!RUN.region.length){g.fillStyle=css('--ink3');g.textAlign='center';
g.fillText(INFO&&INFO.atlas?'no data yet':'needs --atlas',w/2,h/2);return}
const A=INFO.atlas.layers[li],pos={};A.curve.forEach((c,i)=>pos[c]=i);const C=A.cent.length,n=RUN.region.length;
const L=40,R=10,T=8,B=22,pw=w-L-R,ph=h-T-B;g.strokeStyle=css('--grid');g.fillStyle=css('--ink3');g.textAlign='right';g.textBaseline='middle';
for(let i=0;i<=4;i++){const y=T+ph-ph*i/4;g.beginPath();g.moveTo(L,y+.5);g.lineTo(L+pw,y+.5);g.stroke();g.fillText(Math.round((C-1)*i/4),L-5,y)}
const X=i=>L+(n<2?pw/2:pw*i/(n-1)),Y=r=>T+ph-ph*pos[r]/(C-1||1);
g.strokeStyle=css('--s1soft');g.lineWidth=1.5;g.beginPath();RUN.region.forEach((s,i)=>{const x=X(i),y=Y(s[li]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
RUN.region.forEach((s,i)=>{const sc=RUN.scale[i][li];g.fillStyle=css('--s1');g.strokeStyle=css('--surface');g.lineWidth=2;g.beginPath();
g.arc(X(i),Y(s[li]),Math.min(9,3+2.5*Math.log2(Math.max(1,sc))),0,7);g.fill();g.stroke()});
g.fillStyle=css('--ink3');g.textAlign='center';g.textBaseline='top';const ts=Math.max(1,Math.ceil(n/10));for(let i=0;i<n;i+=ts)g.fillText(i===0?'pre':i,X(i),T+ph+5);
cv._hit=e=>{const b=cv.getBoundingClientRect(),i=Math.round((e.clientX-b.left-L)/(n<2?1:pw/(n-1)));if(i<0||i>=n)return hideTip();
showTip(e,`<b>${i===0?'prefill':'token '+i}</b> ${RUN.tok[i]!=null?esc(JSON.stringify(RUN.tok[i])):''}<br>region ${RUN.region[i][li]} (curve position ${pos[RUN.region[i][li]]}), scale ${fmt(RUN.scale[i][li],2)}`)}}

function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
const pct0=v=>Math.round(100*v)+'%';
const METRIC={routed:{name:'routed neurons',f:v=>v==null?'-':(100*v).toFixed(1)+'%',axis:pct0,zero:true},work:{name:'MLP work',f:v=>v==null?'-':(100*v).toFixed(1)+'%',axis:pct0,zero:true},
ms:{name:'MLP time',f:v=>v==null?'-':v.toFixed(2)+' ms',zero:true},err:{name:'MLP error vs dense',f:v=>v==null?'-':v.toExponential(1),zero:true}};

function render(){if(!RUN)RUN=blank();const xs=RUN.ms.map((_,i)=>i===0?'pre':String(i));
lineChart($('#c_ms'),xs,RUN.ms,{tip:i=>`<b>${i===0?'prefill ('+RUN.tin[0]+' tokens)':'token '+i}</b> ${RUN.tok[i]!=null?esc(JSON.stringify(RUN.tok[i])):''}<br>${fmt(RUN.ms[i],1)} ms, CPU ${fmt(RUN.cpu[i],0)}%`});
lineChart($('#c_work'),xs,RUN.work.map(v=>v==null?null:100*v),{minMax:100,tip:i=>`<b>${i===0?'prefill':'token '+i}</b><br>MLP work ${fmt(100*RUN.work[i],1)}%`});
const m=$('#metric').value,mt=METRIC[m];const nl=INFO&&INFO.atlas?INFO.layers:0;const M=[];
for(let l=0;l<nl;l++)M.push(RUN.layers.map(s=>s?s[m][l]:null));heat($('#c_heat'),M,mt);
bars($('#c_bars'),RUN.layers.length?RUN.layers[RUN.layers.length-1][m]:[],{name:mt.name,f:mt.f,axis:mt.axis,minMax:(m==='routed'||m==='work')?1:0});
const li=+$('#layer').value||0;$('#layer2').textContent=li;geo($('#c_geo'),li);regions($('#c_reg'),li);
$('#ramp').style.background=`linear-gradient(90deg,${ramp(0)},${ramp(1)})`}
function blank(){return{ms:[],work:[],cpu:[],tin:[],tok:[],layers:[],xy:[],region:[],scale:[],dec:0,decN:0}}

function kpis(ev){$('#k_ms').textContent=fmt(ev.ms,ev.ms<100?1:0);$('#k_cpu').textContent=fmt(ev.cpu_pct,0)+'%';$('#k_rss').textContent=fmt(ev.rss_mb,0);
$('#k_work').textContent=ev.work!=null?fmt(100*ev.work,1)+'%':'-';$('#k_mlp').textContent=ev.mlp_ms!=null?fmt(100*ev.mlp_ms/ev.ms,0)+'%':'-';
if(ev.prefill){$('#k_pre').textContent=fmt(ev.ms,0);$('#k_pre_s').textContent=`ms for ${ev.tokens_in} prompt tokens`}
else{RUN.dec+=ev.ms;RUN.decN++;$('#k_tps').textContent=fmt(1000*RUN.decN/RUN.dec,2)}}

function addToken(ev){if(ev.text==null)return;const s=document.createElement('span');s.className='t';s.textContent=ev.text;
const w=ev.work!=null?ev.work:1;s.style.background=ramp(.08+.55*w);if(w>.6)s.style.color='#fff';
s.title=`step ${ev.step}: ${fmt(ev.ms,1)} ms, p=${fmt(ev.prob,3)}, entropy ${fmt(ev.entropy,2)}${ev.work!=null?', MLP work '+fmt(100*ev.work,1)+'%':''}`;$('#out').appendChild(s);
$('#out').scrollTop=1e9}

function history(rows){const tb=$('#hist tbody');tb.innerHTML='';rows.slice().reverse().forEach(r=>{const tr=document.createElement('tr');
[r.when,r.backend+(r.stopped?' (stopped)':''),r.z==null?'-':r.z,r.prompt_tokens,r.tokens,fmt(r.prefill_ms,0),fmt(r.decode_tps,2),r.work==null?'-':fmt(100*r.work,1)+'%',fmt(r.wall_s,1)]
.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});tb.appendChild(tr)})}

function status(t,err){const s=$('#status');s.textContent=t;s.className=err?'err':''}
async function run(){if(busy)return;busy=true;$('#run').disabled=true;$('#stop').disabled=false;RUN=blank();$('#out').innerHTML='';
const p=document.createElement('span');p.className='p';p.textContent=$('#prompt').value;$('#out').appendChild(p);['#k_tps','#k_ms','#k_pre','#k_work','#k_mlp','#k_cpu'].forEach(k=>$(k).textContent='-');
render();status('running');const body={prompt:$('#prompt').value,backend:$('#backend').value,z:+$('#z').value,max_new_tokens:+$('#maxnew').value,
temperature:+$('#temp').value,seed:+$('#seed').value,measure_error:$('#err').checked};
try{ctl=new AbortController();const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctl.signal});
if(!r.ok){const j=await r.json().catch(()=>({error:r.statusText}));throw new Error(j.error||r.statusText)}
const rd=r.body.getReader(),dec=new TextDecoder();let buf='',pending=false;
const flush=()=>{pending=false;render()};
for(;;){const{value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let k;
while((k=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,k);buf=buf.slice(k+2);if(!line.startsWith('data: '))continue;const ev=JSON.parse(line.slice(6));
if(ev.type==='start')status(`running on ${ev.backend}${ev.backend!=='torch'?' (z '+ev.z+')':''}, ${ev.device}, ${ev.prompt_tokens} prompt tokens`);
else if(ev.type==='step'){RUN.ms.push(ev.ms);RUN.work.push(ev.work);RUN.cpu.push(ev.cpu_pct);RUN.tin.push(ev.tokens_in);RUN.tok.push(ev.token!=null?ev.token:null);
if(ev.layers){RUN.layers.push(ev.layers);RUN.xy.push(ev.layers.xy);RUN.region.push(ev.layers.region);RUN.scale.push(ev.layers.scale)}
kpis(ev);addToken(ev);if(!pending){pending=true;requestAnimationFrame(flush)}}
else if(ev.type==='done'){INFO.history.push(ev.run);history(INFO.history);status(`done: ${ev.run.tokens} tokens, ${fmt(ev.run.decode_tps,2)} tokens/s`)}
else if(ev.type==='error')status(ev.message,true)}}}
catch(e){if(e.name!=='AbortError')status(e.message,true)}
busy=false;$('#run').disabled=false;$('#stop').disabled=true;render()}

$('#run').onclick=run;$('#stop').onclick=()=>fetch('/api/stop',{method:'POST',body:'{}'});
$('#prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))run()});
['#metric','#layer'].forEach(s=>$(s).onchange=render);
$('#backend').onchange=()=>{$('#z').disabled=$('#backend').value==='torch'};
$('#theme').onclick=()=>{const d=document.documentElement,cur=d.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');d.dataset.theme=cur==='dark'?'light':'dark';render()};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
document.addEventListener('mousemove',e=>{const cv=e.target.closest&&e.target.closest('canvas');if(cv&&cv._hit)cv._hit(e);else hideTip()});
document.addEventListener('mouseleave',hideTip);window.addEventListener('resize',render);
fetch('/api/info').then(r=>r.json()).then(j=>{INFO=j;const shortName=s=>s.includes('/')&&s.startsWith('/')?s.split('/').filter(Boolean).slice(-2).join('/'):s;$('#meta').title=j.model;$('#meta').textContent=`${shortName(j.model)}, ${(j.params/1e6).toFixed(0)}M parameters, ${j.threads} threads`+
(j.atlas?`, atlas ${shortName(j.atlas.path)} (${j.layers} layers, ${j.atlas.layers[0].regions} regions, rank ${j.atlas.layers[0].rank})`:', no atlas');
j.backends.forEach(b=>{const o=document.createElement('option');o.value=b;o.textContent={torch:'torch (dense)',reference:'atlas, NumPy',opencl:'atlas, OpenCL'}[b];$('#backend').appendChild(o)});
if(j.backends.includes('reference'))$('#backend').value='reference';$('#backend').onchange();
for(let l=0;l<(j.atlas?j.layers:0);l++){const o=document.createElement('option');o.value=l;o.textContent=l;$('#layer').appendChild(o)}
if(!j.atlas){$('#err').disabled=true;$('#metric').disabled=true}history(j.history);RUN=blank();render()}).catch(e=>status('info: '+e.message,true));
</script></body></html>
"""


# =============================================================================
# 5. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--atlas", default=None, help=".atlas directory from 'forge' (enables the atlas backends and geometry)")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--lookup", choices=["exact", "curve"], default="exact")
    ap.add_argument("--gpus", type=int, default=None, help="Max OpenCL devices")
    ap.add_argument("--host_dictionary", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    ap.add_argument("--no_warmup", action="store_true", help="skip paging the atlas into memory at startup")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--verbose", action="store_true", help="log HTTP requests")
    args = ap.parse_args()
    sess = LiveSession(args)
    srv = ThreadingHTTPServer((args.host, args.port), _make_handler(sess))
    srv.daemon_threads = True
    print(f"Holoqubed Live: http://{args.host}:{args.port}  ({args.model_id}, backends: {', '.join(sess.backends)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
