# Geometric Atlas Implementation

Holoqubed variant in which geometry takes part in the computation. The shape of the MLP input cloud decides, per token, which neurons are evaluated. Neurons whose pre-activation cannot plausibly become positive given where the token sits on the activation manifold are never read.

One Python file, `holoqubed_atlas.py`, plus PyOpenCL. No CUDA, no C++ build step.

| Requirement | Needed for |
|---|---|
| `numpy`, `torch`, `transformers` | everything |
| `pyopencl` | `--backend opencl`, `selftest` |
| `datasets` | `data` |

---

## Geometric Objects

| Object | Built from | Stored as | Used for | Runtime cost per token |
|---|---|---|---|---|
| Chart | Mean `mu` and top-`r` principal axes `P` of calibration MLP inputs | `mu`, `P` | Exact low-rank coordinates `Z = (x - mu) P` | `d * r` |
| Neuron keys in chart | `Kp = W1 P`, `base = mu W1 + b1` | `Kp`, `base` | Chart score `s = base + Z Kp` | `n * r` |
| Atlas regions | k-means of chart coordinates (Voronoi cells) | `cent` | Local context for the residual bound | `C * r` (exact) or `log C` (curve) |
| Residual ellipsoid | Per region, RMS of `(x - mu - Z P^T) . w_j` over calibration tokens | `sig [C, n]`, `rtyp [C]` | Spread `e = z * sig[region, j] * scale` | lookup |
| Region curve | n-D Hilbert keys of region centroids over the leading chart axes | `curve_keys`, `curve_regions` | O(log C) region lookup by binary search plus a small exact window | `log C + window * r` |
| Neuron layout | n-D Hilbert key of each neuron's firing centroid in the chart | `perm` (rows reordered) | Neurons that fire together share DMA blocks | none |

`scale = max(1, |x_res| / rtyp[region])` widens the ellipsoid for tokens that sit further from their chart than calibration tokens did.

---

## Routing Rule

For token `t` in region `c` and neuron `j`:

| MLP kind | Example models | Neuron is evaluated if | Skipped neuron means |
|---|---|---|---|
| `relu` | OPT | `s_tj + e_tj > 0` | pre-activation would need a residual excursion beyond `z` sigma of its region ellipsoid to leave zero |
| `relu_glu` | ProSparse, ReluLLaMA, TurboSparse | gate `s_tj + e_tj > 0` | same, on the gate |
| `silu_glu` | Llama, Qwen | `max abs silu([s-e, s+e]) * (abs(u_s) + e_u) * norm(W2_j) > kappa * tau` | interval bound on the product is below `kappa` times the mean neuron contribution |
| `gelu` | GPT-2 | `s_tj + e_tj > gelu_floor` | GELU input stays below the floor |

Evaluated neurons are computed exactly with the full weights. The approximation is only in which neurons are skipped.

---

## Layout of `holoqubed_atlas.py`

| Section | Contents |
|---|---|
| 1. Geometry | n-D Hilbert encoder (Skilling), quantiser, k-means, SiLU interval bound, activations |
| 2. Model discovery | `load_model`, `find_mlps` (OPT, GPT-2, Llama/Qwen style blocks) |
| 3. Forge | Calibration capture, chart, regions, ellipsoid spreads, curves, neuron layout, `.atlas` writer |
| 4. Dictionary | `ChartPlanner` (CPU query planner), `AtlasLayer` (router + NumPy reference), `AtlasDictionary` (mmap) |
| 5. OpenCL engine | Kernels `atlas_route_up` and `atlas_down`, multi-device round robin, optional zero-copy host dictionary |
| 6. HF integration | `AtlasRuntime` (backends `dense`, `reference`, `opencl`, `static`), `AtlasMLP`, `inject_atlas` |
| 7 to 11. Subcommands | `data`, `bench`, `generate`, `selftest`, `serve` |
| 12. CLI | argparse subcommands with shared model and atlas options |

---

## Usage

| Step | Command |
|---|---|
| Fetch WikiText-2 | `python3 holoqubed_atlas.py data` |
| Forge | `python3 holoqubed_atlas.py forge --model_id facebook/opt-125m --calib wiki_calib.txt --out opt-125m.atlas` |
| Validate kernels | `python3 holoqubed_atlas.py selftest --atlas opt-125m.atlas` |
| Benchmark | `python3 holoqubed_atlas.py bench --model_id facebook/opt-125m --atlas opt-125m.atlas --eval wiki_test.txt --z 3.0,3.5,4.0` |
| Benchmark on OpenCL | add `--backend opencl` |
| Generate | `python3 holoqubed_atlas.py generate --model_id facebook/opt-125m --atlas opt-125m.atlas --prompt "there is no spoon" --compare` |
| Serve | `python3 holoqubed_atlas.py serve --model_id facebook/opt-125m --atlas opt-125m.atlas --port 8000` |
| Curve lookup | add `--lookup curve` (bench, generate, serve, selftest) |
| Zero-copy dictionary in RAM | add `--host_dictionary` (generate, serve, selftest) |

`python3 holoqubed_atlas.py <subcommand> -h` lists every option.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `--rank` (forge) | 128 | Chart rank `r`. Higher: tighter scores, more router cost |
| `--regions` (forge) | 64 | Voronoi regions `C` |
| `--curve_dims` / `--curve_bits` (forge) | 4 / 12 | Hilbert curve over the leading chart axes |
| `--calib_tokens` (forge) | 8192 | Calibration tokens per layer |
| `--z` | 3.5 (generate, serve), list (bench) | Ellipsoid width in sigmas. Higher: safer, more work |
| `--kappa` | 1.0 | Tolerance for `silu_glu` |
| `--lookup` | exact | `exact` (C distances) or `curve` (Hilbert binary search) |

---

## OpenAI-Compatible API

```bash
python3 holoqubed_atlas.py serve --model_id facebook/opt-125m --atlas opt-125m.atlas --port 8000 --api_key sk-local
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
r = c.chat.completions.create(model="opt-125m-atlas", messages=[{"role": "user", "content": "Hi"}])
```

| Endpoint | Notes |
|---|---|
| `GET /health` | No auth |
| `GET /v1/models`, `GET /v1/models/{id}` | One model, id `<model>-atlas` or `--served_name` |
| `POST /v1/completions` | Single string prompt, `echo` supported |
| `POST /v1/chat/completions` | Uses the tokenizer chat template; models without one get a `User:` / `Assistant:` transcript with automatic stop at the next turn |
| `stream: true` | Server-Sent Events, OpenAI chunk format, `data: [DONE]` terminator, `stream_options.include_usage` |

| Request field | Supported |
|---|---|
| `max_tokens`, `max_completion_tokens` | yes, clipped to the context length |
| `temperature` | yes, `0` means greedy |
| `top_p`, `seed` | yes |
| `stop` | up to 4 strings, also enforced while streaming |
| `n` | `1` only |
| `logprobs`, `tools`, `response_format` | not implemented |
| `atlas_z`, `atlas_kappa` | per-request routing override (send via `extra_body` in the SDK) |

Every response carries `usage.atlas = {z, routed_frac, nonzero_frac}` for that request.

| Server flag | Default | Effect |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8000` | Bind address |
| `--api_key` | none | Require `Authorization: Bearer <key>` |
| `--served_name` | `<model>-atlas` | Model id reported to clients |
| `--backend` | `opencl` | `opencl`, `reference` or `dense` |
| `--max_tokens_default` | 256 | When the request sets no limit |
| `--z`, `--kappa`, `--lookup`, `--gpus`, `--host_dictionary` | | As for `generate` |

Errors use the OpenAI error object (`401` bad key, `404` unknown model or path, `400` invalid request or context overflow, `500` server error). Generation is serialised by a lock: concurrent requests queue, there is no batching. Reasoning models (Qwen3) return their `<think>` block inside `content`.

### API Validation

Run with the official `openai` Python SDK against OPT-125m (OpenCL backend) and Qwen3-0.6B.

| Check | Result |
|---|---|
| `models.list`, `completions.create`, `chat.completions.create` | pass |
| Streamed chat text equals non-streamed text (greedy) | identical |
| `stop` in non-streamed and streamed completions | identical cut, `finish_reason: stop` |
| Same `seed` with sampling | repeatable |
| `atlas_z: 2.0` override | routed fraction 21% vs 48% at default z=3.5 |
| Bad key / unknown model / empty messages / `n=2` | 401 / 404 / 400 / 400 |
| Qwen3-0.6B chat template | "The capital of the Netherlands is Amsterdam.", stops at end of turn |

---

## `.atlas` Format

A directory with `manifest.json` and one `.npy` file per array per layer (`L000.W1.npy`, ...), uncompressed so every array can be memory mapped.

| Array | Shape | Notes |
|---|---|---|
| `W1`, `b1` | `[n, d]`, `[n]` | Up (or gate) projection, rows in Hilbert layout |
| `Wu`, `bu` | `[n, d]`, `[n]` | GLU up projection (GLU kinds only) |
| `W2T`, `b2` | `[n, d_out]`, `[d_out]` | Down projection, neuron-major |
| `perm` | `[n]` | Stored row to original neuron index |
| `mu`, `P` | `[d]`, `[d, r]` | Chart |
| `base`, `Kp` (`base_u`, `Kp_u`) | `[n]`, `[n, r]` | Neuron keys in the chart |
| `cent`, `rtyp` | `[C, r]`, `[C]` | Regions and typical residual radius |
| `sig` (`sig_u`) | `[C, n]` | Region residual spread per neuron |
| `curve_*` | | Region Hilbert curve |
| `colnorm`, `importance`, `tau` | `[n]`, `[n]`, scalar | Down column norms, calibration importance, mean contribution |

The atlas stores the full weights plus `C * n` spreads; for OPT-125m that is 250 MB in FP32 (model itself 500 MB in FP32 including embeddings).

---

## Measured Results

WikiText-2 raw: 8192 calibration tokens from `train`, 1024 evaluation tokens from `test` (4 windows of 256). Rank 128, 64 regions, exact lookup unless noted. Attention, embeddings and LM head stay dense.

**MLP work** counts multiply-adds of the routed MLP relative to the dense MLP, including the chart projection and chart scores (about 10% of dense at rank 128).

### OPT-125m (ReLU, 12 layers, 3072 neurons)

| Config | PPL | Top-1 vs dense | KL | Routed | Non-zero | MLP work |
|---|---|---|---|---|---|---|
| Dense (HF) | 48.37 | 100.00% | 0 | 100% | | 100% |
| Atlas, dense layout | 48.37 | 100.00% | 0 | 100% | 100% | 100% |
| Geometric z=2.0 | 51.76 | 86.47% | 0.0685 | 19.9% | 6.0% | 23.4% |
| Geometric z=2.5 | 49.22 | 93.53% | 0.0185 | 27.4% | 6.1% | 27.2% |
| Geometric z=3.0 | 48.81 | 96.86% | 0.0059 | 36.4% | 6.2% | 31.7% |
| Geometric z=3.5 | 48.55 | 99.12% | 0.0015 | 46.2% | 6.2% | 36.6% |
| Geometric z=4.0 | 48.39 | 99.22% | 0.0004 | 55.7% | 6.2% | 41.4% |
| Static top 30% | 159.00 | 42.16% | 1.3701 | 30.0% | 4.6% | 27.7% |
| Static top 50% | 76.94 | 60.20% | 0.6102 | 50.0% | 5.4% | 38.1% |

Static rows keep the neurons with the highest mean calibration contribution for every token, which is the best that context-free weight pruning can do at that fraction. At equal work, geometric routing is near lossless where static pruning is not.

#### Variants on OPT-125m

| Variant | z | PPL | Top-1 | KL | MLP work |
|---|---|---|---|---|---|
| Rank 128, exact lookup | 3.5 | 48.55 | 99.12% | 0.0015 | 36.6% |
| Rank 128, Hilbert curve lookup | 3.5 | 48.69 | 98.82% | 0.0025 | 36.9% |
| Rank 64, exact lookup | 3.5 | 48.83 | 97.84% | 0.0043 | 36.2% |
| Rank 64, exact lookup | 4.0 | 48.53 | 99.02% | 0.0013 | 41.2% |

#### Memory Locality of the Hilbert Neuron Layout

Mean 16 KB blocks touched per token and layer (576 blocks per matrix), z=3.5.

| Matrix | Rows read | Hilbert layout | Original order | Reduction |
|---|---|---|---|---|
| `W1` | routed neurons (46.2%) | 458.7 | 499.9 | 8% |
| `W2T` | non-zero neurons (6.2%) | 130.0 | 157.5 | 17% |

### OPT-350m (ReLU, 24 layers, 4096 neurons)

| Config | PPL | Top-1 vs dense | KL | Routed | Non-zero | MLP work |
|---|---|---|---|---|---|---|
| Dense (HF) | 35.81 | 100.00% | 0 | 100% | | 100% |
| Atlas, dense layout | 35.81 | 100.00% | 0 | 100% | 100% | 100% |
| Geometric z=2.5 | 36.69 | 92.45% | 0.0247 | 52.9% | 9.0% | 38.8% |
| Geometric z=3.0 | 36.28 | 95.69% | 0.0113 | 64.9% | 9.0% | 44.8% |
| Geometric z=3.5 | 36.04 | 97.84% | 0.0049 | 74.0% | 9.0% | 49.4% |
| Geometric z=4.0 | 36.00 | 98.73% | 0.0033 | 80.5% | 9.0% | 52.6% |
| Static top 50% | 120.48 | 40.98% | 1.5125 | 50.0% | 6.5% | 36.0% |

The chart captures less of the input cloud here (59 to 73% energy at rank 128 of 1024, vs 63 to 74% of 768 for OPT-125m), so ellipsoids are wider and more neurons are routed for the same `z`. Mean 64 KB blocks of `W2T` read per token at z=3.5: 155.1 in Hilbert layout vs 184.9 in original order (16% fewer).

### Qwen3-0.6B (SiLU GLU, 28 layers, 3072 neurons)

4096 calibration tokens, 512 evaluation tokens, z=3.0. For `silu_glu` every routed neuron is non-zero, so routed equals non-zero.

| Config | PPL | Top-1 vs dense | KL | Routed | MLP work |
|---|---|---|---|---|---|
| Dense (HF) | 28.02 | 100.00% | 0 | 100% | 100% |
| Atlas, dense layout | 28.02 | 100.00% | 0 | 100% | 100% |
| Geometric kappa=1.0 | 28.03 | 99.61% | 0.0001 | 99.2% | 108.9% |
| Geometric kappa=2.0 | 27.68 | 93.73% | 0.0122 | 90.4% | 100.1% |
| Geometric kappa=4.0 | 33.19 | 73.14% | 0.3119 | 71.0% | 80.7% |

No net saving: the router (two chart projections for gate and up) costs about 10%, and the interval bound only becomes selective once quality degrades. A SwiGLU model needs ReLU-fication of the gate (`relu_glu`) before this variant pays off.

### Kernel Validation

| Check | Result |
|---|---|
| `atlas_route_up` + `atlas_down` vs NumPy reference, routed count per token | identical |
| Same, non-zero count per token | identical |
| Same, max relative output difference | 1e-6 |
| End-to-end OPT-125m metrics, `--backend opencl` vs `reference` | identical to 2 decimals |
| Greedy generation, OPT-125m, curve lookup, host dictionary | token-identical to dense |
| `selftest`, all 12 OPT-125m layers and Qwen3-0.6B layers 0, 13, 27 | PASS, max relative difference 2.8e-6 |

Device used for validation: PoCL CPU OpenCL. Wall-clock numbers from that device are not representative and are not reported.

---

## Model Support

| Kind | Status | Notes |
|---|---|---|
| `relu` (OPT) | Supported, effective | 2 to 11% of neurons fire per token; geometry recovers them reliably |
| `relu_glu` | Supported | Same exact-zero structure on the gate as `relu` |
| `silu_glu` (Llama, Qwen) | Supported, little gain | Dense activations; an oracle keeping the top 50% of neurons per token still has 10% relative MLP output error on Qwen3-0.6B |
| `gelu` (GPT-2) | Runs, not useful | MLP outputs rely on cancellation across many neurons; an oracle keeping the top 30% per token still has 88% relative MLP output error |

---

## Known Limits

| Area | Current state |
|---|---|
| Buffers | Per-call OpenCL buffer allocation for activations; static weights are pinned once |
| Precision | FP32 storage and math; no `half2` path yet |
| Determinism | Active list is compacted with atomics, so summation order (not the set) varies between runs |
| Planner | Chart projection and region lookup run on the CPU from host activations |
| Scope | MLP blocks only |
| Calibration | Atlas quality depends on calibration text resembling the deployment distribution |

---

## Relation to the Phase Variants

The `hilberspace`, `gpt2` and `qwen35` variants encode weights as `A cos(row+col)` and `A sin(row+col)` and recombine them as `w_r cos + w_i sin`, which returns `A` exactly; the runtime phase then scales a whole layer by `cos(phi)`. Geometry there changes storage order and a global gain, not which computation happens. In this variant the geometry of the input determines the computation itself.
