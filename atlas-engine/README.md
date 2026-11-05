# Geometric Atlas Implementation

Holoqubed variant in which geometry takes part in the computation. The shape of the MLP input cloud decides, per token, which neurons are evaluated. Neurons whose pre-activation cannot plausibly become positive given where the token sits on the activation manifold are never read.

One engine file, `holoqubed_atlas.py`, plus PyOpenCL; `holoqubed_graph.py` draws its geometry, circuits and topology; `holoqubed_precision.py` measures tile storage and simulator precision. No CUDA, no C++ build step.

| Requirement | Needed for |
|---|---|
| `numpy`, `torch`, `transformers` | everything |
| `pyopencl` | `--backend opencl`, `selftest` |
| `datasets` | `data` |
| `matplotlib` | `holoqubed_graph.py` |
| `pyqrack` | `landscape`, `qneuron`, the tile check in `holoqubed_graph.py` (uses `/usr/local/lib/qrack/libqrack_pinvoke.so` when present, else the pip-bundled library) |

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
| 7 to 10. Subcommands | `data`, `bench`, `generate`, `selftest` |
| 11. QrackBridge landscape | Real-amplitude state preparation (RY + CNOT), tile placement, exact and shot-based queries |
| 12. QrackNeuron landscape | `qneuron`: neuron embedding, QrackBridge landscape tensor, amplitude amplification |
| 13. Quantum format | `qconvert`: 4x4 qubit tiles, qneuron gating, PyQrack tile and qneuron workers |
| 14. API server | `serve` |
| 15. CLI | argparse subcommands with shared model and atlas options |

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
| QrackBridge landscape | `python3 holoqubed_atlas.py landscape --atlas opt-125m.atlas --layers 5 --model_id facebook/opt-125m --eval wiki_test.txt` |
| QrackNeuron landscape | `python3 holoqubed_atlas.py qneuron --atlas opt-125m.atlas --model_id facebook/opt-125m --text wiki_calib.txt --layer 5` |
| Quantum format | `python3 holoqubed_atlas.py qconvert --model_id Qwen/Qwen3-0.6B --calib wiki_calib.txt --eval wiki_test.txt --qrack_layer 14` |
| Everything for Qwen3-0.6B | `./qwen_holoqubed.sh` |
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

## QrackBridge Weight Landscape

Measurement tool, not an inference path. `landscape` places tiles of chart keys into QrackBridge patches and queries them through PyQrack, to measure how much of a weight tile a patch seam keeps for a given bond budget.

```bash
python3 holoqubed_atlas.py landscape --atlas opt-125m.atlas --layers 0,5,11 \
    --tile_neurons 64 --tile_dims 64 --bonds 2,3,4,5,6 --shots 1024,8192 \
    --model_id facebook/opt-125m --eval wiki_test.txt
```

### Construction

| Element | Realisation |
|---|---|
| Tile | `W = Kp[offset : offset + tile_neurons, : tile_dims]`: chart keys of consecutive (Hilbert-layout) neurons on the leading chart axes |
| Encoding | `sum W[j,d] \|j>\|d> / norm(W)` over `log2(tile_neurons) + log2(tile_dims)` qubits |
| Patches | 2-patch QrackBridge: chart qubits fill patch 0, neuron qubits patch 1. The Schmidt decomposition across the cut is the SVD of `W`; `max_bonds = b` caps the rank at `2^b` |
| Placement | Real-amplitude state-preparation tree: uniformly controlled RY in Gray-code form (angles via a fast Walsh-Hadamard transform), using only RY and single-control CNOT, which QrackBridge allows across patches. Run inside `deferred()` so the seam is truncated once, after loading. QrackBridge has no `in_ket` |
| Query | `U_z^dagger` for the token's chart vector `z` on patch 0 (in-patch gates). Amplitude `<j, 0\|` is then `(W z)_j / (norm(W) norm(z))`, the tile's chart score without `base` |
| Exact readout | Dense `out_ket` contraction (QrackBridge's testing path) |
| Native readout | `measure_shots`, post-selected on the chart register returning to `\|0>`; counts estimate `(W z)_j^2`, so magnitudes only (see `--signed` below for signs) |
| Library | `QRACK_LIB_PATH = /usr/local/lib/qrack/libqrack_pinvoke.so` when present, else PyQrack's bundled library |

Tile shape is constrained by the equal 2-patch split: `tile_neurons` is `tile_dims` or `2 * tile_dims`.

### Measured: OPT-125m, 64 neurons x 64 chart dims (12 qubits)

16 real WikiText-2 test tokens per layer. Score error is relative to `W z` in NumPy; top-8 recall compares the 8 largest `|score|` neurons.

| Layer | Bonds | Rank | Fidelity | SVD energy | Bridge estimate | Score error | Sign agreement | Top-8 recall | Exact query ms |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 4 | 0.3704 | 0.3704 | 0.3704 | 0.868 | 71.7% | 37.5% | 2.6 |
| 0 | 4 | 16 | 0.7416 | 0.7416 | 0.7416 | 0.515 | 85.7% | 60.9% | 3.6 |
| 0 | 5 | 32 | 0.9338 | 0.9338 | 0.9338 | 0.262 | 91.8% | 78.1% | 5.0 |
| 0 | 6 | 64 | 1.0000 | 1.0000 | 1.0000 | 0.0005 | 100.0% | 100.0% | 8.0 |
| 5 | 2 | 4 | 0.3307 | 0.3307 | 0.3307 | 0.923 | 65.6% | 28.9% | 4.3 |
| 5 | 4 | 16 | 0.7321 | 0.7321 | 0.7321 | 0.536 | 82.0% | 53.1% | 3.5 |
| 5 | 5 | 32 | 0.9330 | 0.9330 | 0.9330 | 0.257 | 90.7% | 79.7% | 4.8 |
| 5 | 6 | 64 | 1.0000 | 1.0000 | 1.0000 | 0.0007 | 100.0% | 100.0% | 8.2 |
| 11 | 2 | 4 | 0.3653 | 0.3653 | 0.3653 | 0.838 | 77.2% | 46.9% | 2.5 |
| 11 | 4 | 16 | 0.7761 | 0.7761 | 0.7761 | 0.453 | 84.9% | 68.0% | 3.5 |
| 11 | 5 | 32 | 0.9545 | 0.9545 | 0.9545 | 0.216 | 92.1% | 79.7% | 4.9 |
| 11 | 6 | 64 | 1.0000 | 1.0000 | 1.0000 | 0.0007 | 99.9% | 99.2% | 8.2 |

Load: 8,189 gates in 1.9 to 2.1 s per tile.

| Layer | Shots | Accepted shots | Top-8 recall (magnitude) | Query ms |
|---|---|---|---|---|
| 0 | 1024 | 1.46% | 50.0% | 9.7 |
| 0 | 8192 | 1.70% | 69.5% | 15.2 |
| 5 | 1024 | 1.24% | 43.8% | 9.9 |
| 5 | 8192 | 1.51% | 67.2% | 12.7 |
| 11 | 1024 | 1.29% | 39.8% | 9.5 |
| 11 | 8192 | 1.53% | 71.9% | 13.0 |

### Measured: OPT-125m layer 5, 256 neurons x 128 chart dims (15 qubits)

8 real tokens. Load: 65,533 gates in 71 to 74 s.

| Bonds | Rank | Fidelity | SVD energy | Score error | Sign agreement | Top-32 recall | Exact query ms |
|---|---|---|---|---|---|---|---|
| 4 | 16 | 0.4133 | 0.4133 | 0.819 | 73.6% | 44.5% | 10.3 |
| 5 | 32 | 0.6232 | 0.6232 | 0.634 | 80.3% | 52.0% | 17.0 |
| 6 | 64 | 0.8544 | 0.8544 | 0.369 | 88.0% | 70.7% | 32.2 |
| 7 | 128 | 1.0000 | 1.0000 | 0.0027 | 99.9% | 99.6% | 51.9 |

8192 shots at full bonds: 0.74% accepted, top-32 magnitude recall 48.8%, 73.1 ms per query.

### Reading the Results

| Observation | Value |
|---|---|
| Seam truncation vs optimal rank-`2^b` SVD (Eckart-Young) | identical to 4 decimals in every row; `fidelity_estimate` reports it exactly |
| Bond budget for exact tile scores | full rank: chart keys have no low-rank shortcut (rank 32 of 64 keeps 93 to 95% energy, 22 to 26% score error) |
| Exact query vs NumPy `W z` | 64x64: 5 to 8 ms vs 1.5 us; 256x128: 52 ms vs 5.7 us |
| Native readout acceptance | about `1 / tile_dims` (1.2 to 1.7% at 64 dims, 0.7% at 128 dims), and signs are lost |
| Load cost growth | 8x the gates (12 to 15 qubits) took 37x the time; a full-layer `Kp` tile (19 qubits) extrapolates to hours |

Device for these numbers: the 2 vCPU validation VM, Qrack on its CPU engine (the bundled library found no OpenCL platform there).

### Signed Readout (`--signed`)

Hadamard test with one ancilla in patch 0, so the shot path returns signed scores for a ReLU sign test.

```bash
python3 holoqubed_atlas.py landscape --signed --atlas opt-125m.atlas --layers 0,5,11 \
    --tile_neurons 128 --tile_dims 64 --bonds 3,4,5,6 --shots 1024,8192,65536 \
    --model_id facebook/opt-125m --eval wiki_test.txt
```

| Element | Realisation |
|---|---|
| Tile columns | `tile_dims - 2` chart dims, a `base` column (query coordinate 1, so the score is `base_j + Kp_j . z`), a constant reference column `beta` (`--no_base` drops the base column) |
| `beta` | `ref_scale * norm(tile without reference) / sqrt(tile_neurons)`, default `ref_scale = 1` |
| Qubits | chart register and ancilla in patch 0, neurons in patch 1; requires `tile_neurons = 2 * tile_dims` or `4 * tile_dims` |
| Query | `H(anc)`; anc = 1 branch: controlled `U_z^dagger` (RY with 1 control, CNOT with 2 controls, all in patch 0); anc = 0 branch: anti-controlled X maps the reference column to `\|0>`; `H(anc)` |
| Readout | post-select chart register `\|0>`: `P(anc=0, j) - P(anc=1, j) = beta * s_j / (norm(W)^2 norm(z))`, sign and magnitude from one shot batch |
| Seam | the query never crosses it; the ancilla is in a product state during placement |

Positive recall and precision treat `s_j > 0` as the ReLU "fires" class, the quantity a routing sign test must not miss. Plain sign agreement is dominated by the negative majority and is shown for reference only.

#### OPT-125m, 128 neurons x 64 columns (13 data qubits + ancilla), 16 real tokens per layer

Exact readout (dense `out_ket` of the Hadamard-test state):

| Layer | Bonds | Rank | Fidelity | SVD energy | Signed score error | Positive scores | Positive recall | Positive precision | Query ms |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 8 | 0.8164 | 0.8164 | 0.387 | 8.8% | 37.9% | 75.3% | 3.7 |
| 0 | 5 | 32 | 0.9564 | 0.9565 | 0.154 | 8.8% | 81.3% | 85.3% | 9.3 |
| 0 | 6 | 64 | 1.0000 | 1.0000 | 0.0008 | 8.8% | 99.2% | 100.0% | 18.6 |
| 5 | 3 | 8 | 0.9272 | 0.9272 | 0.205 | 0.5% | 71.9% | 71.2% | 3.6 |
| 5 | 5 | 32 | 0.9827 | 0.9827 | 0.101 | 0.5% | 84.4% | 80.2% | 10.2 |
| 5 | 6 | 64 | 1.0000 | 1.0000 | 0.0007 | 0.5% | 100.0% | 100.0% | 16.7 |
| 11 | 3 | 8 | 0.9344 | 0.9344 | 0.334 | 17.3% | 54.0% | 67.4% | 3.7 |
| 11 | 5 | 32 | 0.9870 | 0.9870 | 0.151 | 17.3% | 80.2% | 87.4% | 12.3 |
| 11 | 6 | 64 | 1.0000 | 1.0000 | 0.0010 | 17.3% | 99.2% | 100.0% | 17.4 |

Native shot readout at full bonds:

| Layer | Shots | Accepted shots | Positive recall | Positive precision | Mass-weighted sign agreement | Signed score error | Query ms |
|---|---|---|---|---|---|---|---|
| 0 | 1024 | 28.2% | 51.9% | 23.0% | 70.7% | 1.859 | 18.5 |
| 0 | 8192 | 27.5% | 56.5% | 30.5% | 93.5% | 0.603 | 22.7 |
| 0 | 65536 | 26.6% | 76.0% | 70.4% | 99.2% | 0.204 | 48.6 |
| 5 | 1024 | 28.0% | 56.2% | 0.0% | 70.6% | 1.839 | 18.3 |
| 5 | 8192 | 27.3% | 75.0% | 4.0% | 95.4% | 0.585 | 25.9 |
| 5 | 65536 | 26.5% | 87.5% | 47.1% | 99.7% | 0.202 | 45.6 |
| 11 | 1024 | 27.3% | 38.3% | 33.3% | 60.1% | 3.496 | 18.4 |
| 11 | 8192 | 26.5% | 49.5% | 40.9% | 84.8% | 1.155 | 24.0 |
| 11 | 65536 | 25.4% | 76.9% | 73.5% | 97.5% | 0.381 | 47.2 |

Load: 16,381 gates in 5.4 to 5.9 s per tile.

#### Variants

| Tile | Score | Readout | Positive scores | Positive recall | Positive precision | Signed score error | Query ms |
|---|---|---|---|---|---|---|---|
| L5, 128 x 64 | `Kp . z` (`--no_base`) | exact | 43.7% | 100.0% | 100.0% | 0.0012 | 16.2 |
| L5, 128 x 64 | `Kp . z` (`--no_base`) | 65536 shots | 43.7% | 85.9% | 88.1% | 0.353 | 48.1 |
| L5, 256 x 128 (15 data qubits + ancilla) | with base | exact, rank 32 | 3.1% | 47.4% | 78.1% | 0.196 | 38.9 |
| L5, 256 x 128 | with base | exact, full rank | 3.1% | 100.0% | 100.0% | 0.0018 | 114.7 |
| L5, 256 x 128 | with base | 8192 shots | 3.1% | 42.9% | 9.7% | 1.017 | 117.2 |
| L5, 256 x 128 | with base | 65536 shots | 3.1% | 52.7% | 41.0% | 0.328 | 153.3 |

256 x 128 load: 65,533 gates in 77 s.

#### Reading the Signed Results

| Observation | Value |
|---|---|
| Signed reconstruction, exact readout, full bonds | score error 0.0007 to 0.0018, positive recall 99.2 to 100% |
| Reference column effect on acceptance | 25 to 28% of shots kept, vs 0.7 to 1.7% for the unsigned readout |
| Shot noise | signed score error falls as `1 / sqrt(shots)` (1.84, 0.59, 0.20 for 1K, 8K, 64K shots on L5) |
| 65536 shots, 128 x 64 | 76 to 88% of positive scores found, precision 47 to 74% |
| Comparison | the ellipsoid router misses about 0.3% of truly active neurons at z = 3.0 (rank 128, full input); the shot sign test at 65536 shots misses 12 to 24% of positive tile scores |
| Shots for ~0.02 score error | extrapolates to about 6.5M per query per tile |
| Untested knob | `--ref_scale` trades acceptance against interference contrast; only `1.0` is measured |

---

## QrackNeuron Landscape

Measurement tool, not an inference path. `qneuron` turns MLP neurons into qubits driven by chart inputs, projects them together as one landscape tensor over QrackBridge patches, and iterates that landscape with amplitude amplification.

```bash
python3 holoqubed_atlas.py qneuron --atlas opt-125m.atlas --model_id facebook/opt-125m \
    --text wiki_calib.txt --layer 5 --inputs 8 --mode embed,landscape,iterate
```

### Construction

| Element | Realisation |
|---|---|
| Inputs | `K` qubits: sign bits of the `K` leading chart axes, `z_i = (2 p_i - 1) sigma_i` with `sigma_i` the axis spread over real tokens |
| Neuron score on input pattern `p` | `s_j(p) = base_j + sum_i Kp[j,i] sigma_i (2 p_i - 1)`, linear in the input bits |
| Angle | `theta_j(p) = alpha s_j(p)`, `alpha = (pi / 2) / max abs(s)` keeps the map monotone |
| `embed`: QrackNeuron | table of `2^K` angles, `NeuronActivationFn.Sigmoid`, `predict(r=True)`: `P(fire \| p) = (1 + sin theta) / 2` |
| `embed`: circuit | `RY(pi/2 + alpha c_j)` plus one single-control `RY(2 alpha Kp[j,i] sigma_i)` per input bit: `K + 1` rotations |
| `landscape` | inputs in patch 0 (real token pattern distribution, or `--uniform`), neurons in the other patches of a QrackBridge chain, joined only by single-control rotations. Seam Schmidt rank is bounded by `2^K`, so `max_bonds = K` is exact |
| Landscape readout | `pauli_expectation`: `P(fire) = (1 - <Z_j>) / 2` per neuron, `<Z_a Z_b>` per pair, co-firing covariance `<Z_a Z_b> - <Z_a><Z_b>` |
| `iterate` | amplitude amplification on one neuron: state preparation = inputs then `QrackNeuron.predict(r=False)` (coherent `RY(theta_p)`), inverse via `unpredict`, oracle = `Z` on the neuron qubit, reflection about `\|0...0>` |
| Oracle tables | `linear` (`pi u`, `u` = normalised score), `hard` (`pi` on the top `--top` fraction), `soft` (ramp from the top `3 x --top` up) |
| Patch rule | patch 0 must hold exactly the `K` inputs: `neurons = K * (patches - 1)` |

### Measured: OPT-125m, real WikiText-2 tokens (2048 per layer)

#### Embedding

| Layer | K | Neurons tested | QrackNeuron vs closed form | Circuit vs QrackNeuron | QrackNeuron per pattern | Circuit per pattern |
|---|---|---|---|---|---|---|
| 5 | 8 | 4 | 1.6e-7 to 3.0e-7 | 5.5e-6 to 4.7e-5 | 63 to 64 ms (256 angles) | 143 to 148 us (9 rotations) |
| 11 | 6 | 4 | 1.9e-7 | 8.9e-7 to 4.9e-6 | 1.8 to 2.1 ms (64 angles) | 115 to 129 us (7 rotations) |

#### Landscape tensor

| Layer | Qubits (patches) | max_bonds | Fidelity estimate | Max error P(fire) | Max error `<ZZ>` | Build s | Pair readout s |
|---|---|---|---|---|---|---|---|
| 5 | 24 (8, 8, 8): 8 inputs, 16 neurons | 2 | 0.9115 | 6.2e-3 | 3.1e-2 | 0.1 | 0.1 (120 pairs) |
| 5 | | 4 | 0.9938 | 7.4e-4 | 2.6e-3 | 0.9 | 0.8 |
| 5 | | 6 | 0.9999 | 1.1e-5 | 3.0e-5 | 24.0 | 31.1 |
| 5 | | 8 (exact) | 1.0000 | 3.0e-6 | 6.1e-6 | 308.9 | 1385.8 |
| 11 | 18 (6, 6, 6): 6 inputs, 12 neurons | 2 | 0.9577 | 6.4e-3 | 2.0e-2 | 0.0 | 0.0 (66 pairs) |
| 11 | | 4 | 0.9990 | 2.2e-4 | 7.2e-4 | 0.2 | 0.2 |
| 11 | | 6 (exact) | 1.0000 | 5.5e-7 | 3.7e-6 | 1.8 | 8.0 |

Default `--bonds 2,4,6`: the exact `K = 8` landscape costs about 28 minutes on the validation VM.

| Layer | Landscape P(fire) vs real pre-activation > 0 rate (correlation over neurons) | Strongest co-firing pair (stored rows) | Covariance |
|---|---|---|---|
| 5 (K = 8) | 0.753 | 628, 629 | +0.180 |
| 11 (K = 6) | 0.889 | 72, 32 | +0.151 |

The strongest layer 5 pair are adjacent rows of the Hilbert neuron layout.

#### Iterating: amplitude amplification, layer 5 neuron 1100 (K = 8, 256 patterns, top 5%)

| Input landscape | Oracle | P(fire) at 0 iterations | Best iterations | P(fire) | P(top 5% patterns): start, best |
|---|---|---|---|---|---|
| uniform | linear | 0.500 | none | 0.500 | 0.051, 0.051 |
| uniform | hard | 0.051 | 3 | 1.000 | 0.051, 1.000 |
| uniform | soft | 0.043 | 3 | 0.988 | 0.051, 0.741 |
| real patterns | linear | 0.569 | 2 | 0.817 | 0.091, 0.127 |
| real patterns | hard | 0.091 | 2 | 0.998 | 0.091, 0.998 |
| real patterns | soft | 0.070 | 2 | 0.948 | 0.091, 0.750 |

Layer 11 neuron 65 (K = 6, uniform): hard 0.063 to 0.961 and soft 0.064 to 0.955 after 3 iterations. Every measured `P(fire)` matches `sin^2((2m + 1) asin(sqrt(p0)))` to 3 decimals.

### Reading the Results

| Observation | Value |
|---|---|
| Neuron embedding | exact: table, closed form and `K + 1`-rotation circuit agree to 5e-5 or better |
| Circuit vs table cost | 9 rotations: 0.15 ms per pattern; 256-angle QrackNeuron: 64 ms per pattern |
| Landscape bond budget | `max_bonds = K - 2` already gives fidelity 0.9999 (layer 5, K = 8) and 0.9990 (layer 11, K = 6) |
| Build cost per extra input qubit | about 10x (0.03, 1.8, 24, 309 s for K = 4, 6, 7, 8 at exact bonds); patch engines hold data plus bond qubits of both seams |
| Linear oracle | fires on about half the patterns, so amplification gains nothing; amplification needs a rare firing set |
| Threshold oracles | `hard` and `soft` tables are built from classically computed scores, so finding their top patterns is circular here; the speedup argument only holds on hardware with large `K` |
| Input resolution | sign bits only: landscape firing rates track real rates with correlation 0.75 to 0.89 |

Device: the 2 vCPU validation VM, Qrack on its CPU engine.

---

## Quantum Format (`qconvert`)

The whole model in quantum format, for gated SiLU MLPs (Llama / Qwen style). Measurement and proof tool: in simulation it reproduces the model but is far slower than dense.

| Part | Quantum form | Emulation |
|---|---|---|
| Every linear map | 4x4 qubit tiles: 16 weights as a 4-qubit state, prepared from 15 RY + 14 CNOT; query = inverse preparation of the 4-value input slice; amplitude `<row, 0\|` = `(W x)_row / (\|W\| \|x\|)` | exact matmul |
| SiLU gating | qneurons: `--cosines` gate qubits, `<Z> = cos(m pi (x + 1) / 2)`, cosine series fitted to SiLU on a dense grid; one up qubit `RY(pi/2 - eps u)`, `<Z> = sin(eps u)`; `h = f(g) sin(eps u) / eps` | same formula |
| RMSNorm, RoPE, softmax, residuals | classical (Loom) | classical |

```bash
python3 holoqubed_atlas.py qconvert --model_id Qwen/Qwen3-0.6B --calib wiki_calib.txt --eval wiki_test.txt \
    --out qwen3.qformat --qrack_layer 14 --max_new_tokens 6
```

| Step | Output |
|---|---|
| calibrate | per layer: gate range `G`, cosine coefficients `c`, up scale `eps`; `--out` writes them as `.npy` + `manifest.json` (`holoqubed-qformat/1`) |
| `--eval` | perplexity, top-1 agreement and KL of the quantum format vs dense |
| generate | greedy generation with every layer emulated in quantum format |
| `--qrack_layer L` | layer L's q, k, v, o, gate, up, down through PyQrack `QrackSimulator` tile workers and its gating through qneuron qubits; tokens and logits compared with the emulation |

| Option | Default | Effect |
|---|---|---|
| `--cosines` | 8 | gate cosine qubits per neuron (9 terms with the constant) |
| `--calib_tokens` | 1024 | calibration tokens |
| `--windows`, `--window_len` | 2, 256 | evaluation windows |
| `--workers` | 2 | QrackSimulator worker processes for `--qrack_layer` |

Implementation details that matter for exactness: each tile's global phase is read once after preparation and divided out (`QrackSimulator.out_ket()` carries an arbitrary global phase), and the qneuron qubit is reset before every feature (reusing one qubit for rotate, read, unrotate cycles drifts).

### Measured: Qwen3-0.6B

Quantum-format MLPs in all 28 layers (exact expectations), WikiText-2, 512 evaluation tokens:

| Gate qubits + up qubit per neuron | Mean MLP error | PPL (dense 28.02) | Top-1 vs dense | KL vs dense |
|---|---|---|---|---|
| 1 + 1 (single qubit per path) | 0.51 | 28,820 | 8.0% | 7.25 |
| 5 + 1 | 0.192 | 34.39 | 69.4% | 0.288 |
| 9 + 1 (default) | 0.050 | 27.69 | 89.6% | 0.043 |
| 13 + 1 | 0.023 | 28.67 | 89.0% | 0.032 |
| 17 + 1 | 0.016 | 28.36 | 89.8% | 0.026 |

Layer 14 genuinely in PyQrack, the other 27 layers emulated, prompt "The capital of the Netherlands is", 6 greedy tokens:

| Check | Result |
|---|---|
| Output | "the city of Amsterdam. The", token-identical to the emulation at all 6 steps |
| Max logit difference | 6.1e-5 (largest logit 19.38) |
| Tiles | 983,040 tiles, 279,183,360 gates, max relative error 2.7e-5 |
| qneurons | 912,384 qubit calls, max relative error 7.3e-6 |
| Time | 2,634 s on 2 vCPU (dense model: 4.24 tokens/s) |

Whole-model estimate on this VM: 37,249,024 tiles per token at about 0.46 ms per tile cycle, about 4.7 hours per token. Storage is unchanged: 15 angles + 1 norm per 16 weights.

## Precision (`holoqubed_precision.py`)

Storage precision of the tile angles and simulator precision of the statevector, against ordinary weight quantization. Each scheme rewrites every linear map in the transformer layers; perplexity, top-1 and KL are measured against the unmodified model. Tile decoding is emulated in NumPy with the engine's own gate list.

```bash
python3 holoqubed_precision.py --model_id Qwen/Qwen3-0.6B --eval wiki_test.txt --calib wiki_calib.txt \
    --experiments fp16,gray,tree,int,statevector,qformat --json precision.json
```

| Experiment | Rows |
|---|---|
| `fp16` | source, plain fp16 weights, fp16 tiles (15 fp16 Gray-code angles + fp16 norm) |
| `gray` | Gray-code circuit angles, uniform int8 / int6 / int4 per angle slot |
| `tree` | Mottonen tree node angles: 7 splits in [0, pi/2], 8 signed leaves on the circle; uniform and Lloyd-Max codes |
| `int` | symmetric int8 / int6 / int4 weights, fp16 scale per 16 or 128 |
| `statevector` | amplitudes and RY coefficients rounded to fp32, fp16, bf16, fp8 e4m3, fp8 e5m2 after every gate |
| `qformat` | qneuron SiLU gating with exact and with fp16 tiles |

| Option | Default | Effect |
|---|---|---|
| `--layers` | 0 (all) | rewrite only the first N layers |
| `--windows`, `--window_len` | 2, 256 | evaluation windows |
| `--cosines`, `--calib_tokens` | 8, 1024 | qformat gating |

Tree angles map exactly onto the circuit's Gray-code angles (FWHT); tree decode and circuit decode agree to 3e-15.

### Measured: Qwen3-0.6B, all 196 linear maps, 512 tokens

Source file 1.50 GB bf16: MLP 528 MB, attention 352 MB, embedding 311 MB, lm_head 311 MB. Sizes below cover the 440M layer weights.

| Scheme | Bits/weight | Size | Rel. weight error | PPL | Top-1 | KL |
|---|---|---|---|---|---|---|
| bf16 source | 16 | 881 MB | 0 | 28.02 | 100.00% | 0.0000 |
| plain fp16 weights | 16 | 881 MB | 1.2e-8 | 28.02 | 100.00% | 0.0000 |
| fp16 tiles | 16 | 881 MB | 5.2e-4 | 28.03 | 100.00% | 0.0000 |
| int8 weights, scale per 16 | 9.00 | 495 MB | 4.8e-3 | 27.88 | 98.43% | 0.0010 |
| Gray angles, int8 | 8.50 | 468 MB | 1.7e-2 | 28.20 | 94.31% | 0.0142 |
| int8 weights, scale per 128 | 8.13 | 447 MB | 6.8e-3 | 28.06 | 96.47% | 0.0037 |
| tree, uniform, 6 / 8 | 7.63 | 420 MB | 1.4e-2 | 28.18 | 95.88% | 0.0054 |
| tree, Lloyd-Max, 6 / 8 | 7.63 | 420 MB | 1.6e-2 | 28.37 | 91.76% | 0.0158 |
| int6 weights, scale per 16 | 7.00 | 385 MB | 2.0e-2 | 28.37 | 92.35% | 0.0125 |
| tree, uniform, 5 / 7 | 6.69 | 368 MB | 2.9e-2 | 28.63 | 92.35% | 0.0243 |
| tree, Lloyd-Max, 5 / 7 | 6.69 | 368 MB | 2.8e-2 | 28.63 | 90.20% | 0.0399 |
| Gray angles, int6 | 6.63 | 365 MB | 6.7e-2 | 31.65 | 80.59% | 0.1835 |
| tree, uniform, 4 / 6 | 5.75 | 317 MB | 6.0e-2 | 29.79 | 85.49% | 0.0796 |
| tree, Lloyd-Max, 4 / 6 | 5.75 | 317 MB | 5.1e-2 | 31.28 | 82.75% | 0.1263 |
| int4 weights, scale per 16 | 5.00 | 275 MB | 8.7e-2 | 31.81 | 77.06% | 0.2217 |
| tree, uniform, 3 / 5 | 4.81 | 265 MB | 1.3e-1 | 35.61 | 71.76% | 0.3233 |
| tree, Lloyd-Max, 3 / 5 | 4.81 | 265 MB | 9.8e-2 | 38.42 | 65.88% | 0.4718 |
| Gray angles, int4 | 4.75 | 261 MB | 2.8e-1 | 3002 | 14.90% | 5.1142 |
| int4 weights, scale per 128 | 4.13 | 227 MB | 1.2e-1 | 37.25 | 66.47% | 0.4082 |
| tree, uniform, 2 / 4 | 3.88 | 213 MB | 2.9e-1 | 367 | 28.43% | 2.7328 |
| tree, Lloyd-Max, 2 / 4 | 3.88 | 213 MB | 1.9e-1 | 405 | 34.12% | 2.9525 |

| Statevector format | Mantissa bits | Rel. weight error | PPL | Top-1 | KL |
|---|---|---|---|---|---|
| fp32 (`FPPOW=5`, pip build) | 23 | 1.2e-7 | 28.02 | 100.00% | 0.0000 |
| fp16 (`FPPOW=4`) | 10 | 1.0e-3 | 28.02 | 99.61% | 0.0000 |
| bf16 | 7 | 8.1e-3 | 28.02 | 98.04% | 0.0016 |
| fp8 e4m3 | 3 | 1.3e-1 | 38.29 | 69.41% | 0.3716 |
| fp8 e5m2 | 2 | 2.4e-1 | 88.96 | 46.27% | 1.4176 |

| Quantum format (9 + 1 qubits per neuron) | PPL | Top-1 | KL |
|---|---|---|---|
| exact tiles | 27.70 | 89.41% | 0.0432 |
| fp16 tiles | 27.70 | 89.61% | 0.0432 |

| Finding | Evidence |
|---|---|
| fp16 storage and fp16 simulation are free | fp16 tiles and fp16 statevector: KL 0.0000 |
| Gray-code angles quantize badly | leaf angles span about 10 rad; one root error moves all 16 weights |
| Tree angles fix most of it | error per level is equal, total about step / sqrt(3); int4-class KL 5.11 to 0.32 |
| Ordinary quantization still wins by about half a bit per weight | tree 6 / 8 at 7.63 bits sits between int6 per 16 and int8 per 128 |
| Lloyd-Max lowers L2 error but raises KL | codes crowd the dense middle and coarsen the tails that carry large weights |
| The qneuron gating, not tile precision, sets quantum-format quality | exact and fp16 tiles give the same KL 0.0432 |
| fp8 is not a simulator format | e4m3 statevector KL 0.37, close to int4 per 128; Qrack's minimum is `FPPOW=4` (`cmake/FpMath.cmake`) |

Practical split: store tree angles at 6 / 8 bits (420 MB layer file that loads straight into the tile circuits), simulate at fp16 (`cmake -DFPPOW=4`, OpenCL needs `cl_khr_fp16`) or fp32.

## `qwen_holoqubed.sh`

Downloads Qwen3-0.6B, converts it both ways and runs it through `holoqubed_atlas.py`.

| Stage | Commands |
|---|---|
| 1. dependencies | `pip install torch transformers numpy pyopencl pyqrack datasets huggingface_hub` |
| 2. download | `huggingface_hub.snapshot_download` into `$WORK/model` |
| 3. data | `data` (WikiText-2) |
| 4. geometric atlas | `forge` (4096 calibration tokens), `selftest` (layers 0, 14, 27), `bench` (z, 2 windows), `generate` (OpenCL, `--compare`) |
| 5. quantum format | `qconvert` with `--eval`, `--out qwen3.qformat`, `--qrack_layer` |

| Variable | Default | Effect |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-0.6B` | any gated SiLU model for stage 5; any supported model for stage 4 |
| `WORK` | `./qwen3-holoqubed` | working directory |
| `PROMPT` | "The capital of the Netherlands is" | generation prompt |
| `Z` | 3.0 | atlas ellipsoid width |
| `QRACK_LAYER` | 14 | layer run in PyQrack; `-1` skips it |
| `QRACK_TOKENS` | 6 | tokens generated in the PyQrack run (about 7 minutes per token on 2 vCPU) |
| `WORKERS` | CPU count | QrackSimulator worker processes |
| `SKIP_PIP` | 0 | `1` skips stage 1 |

---

## Graphs (`holoqubed_graph.py`)

Matplotlib figures drawn from real data: the `.atlas` arrays, the engine's own gate lists (`amplitude_prep_ops`) and, with `--model_id` and `--text`, real token activations.

```bash
python3 holoqubed_graph.py --atlas opt-125m.atlas --layers 0,5,11 \
    --model_id facebook/opt-125m --text wiki_test.txt --qrack_layer 5 --out graphs
```

| Figure | File | Shows |
|---|---|---|
| Atlas geometry | `geometry_layerNN` | chart plane with Voronoi regions, Hilbert curve through region centroids, real tokens; neuron keys coloured by Hilbert layout row; residual spread matrix (regions in curve order x neurons) |
| Model topology | `topology_model` | every layer's attention and MLP block with tile counts and chart energy, Loom links, the layer marked as running in PyQrack, totals (tiles, resident qubits) |
| Tile circuit | `circuit_tile` | 4x4 tile: 15 RY + 14 CNOT preparation, inverted 2-qubit query, amplitude readout; executed in PyQrack and checked against NumPy |
| qneuron circuit | `circuit_qneuron` | parallel cosine qubits and the small-angle up qubit; SiLU rebuilt from 3, 5, 9 cosine qubits |
| Signed readout | `circuit_signed` | Hadamard test inside patch 0: controlled query, anti-controlled reference flip |
| Patch topology | `topology_patches` | QrackBridge chains for `landscape`, `landscape --signed` and `qneuron`; seam fidelity vs bond budget for real chart-key tiles |

| Option | Default | Effect |
|---|---|---|
| `--figures` | all | comma separated: `geometry,model,tile,qneuron,signed,patches` |
| `--layers` | `0` | layers for geometry figures (first one also feeds tile and patches) |
| `--qrack_layer` | none | layer highlighted as running in PyQrack |
| `--inputs`, `--patches`, `--tile_neurons`, `--tile_dims`, `--cosines` | 8, 3, 128, 64, 8 | sizes drawn in the circuit and patch figures |
| `--format`, `--dpi` | png, 150 | also `svg`, `pdf` |

Circuits are drawn with column packing: gates on disjoint wires share a column, and each wire keeps its gate order.

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
