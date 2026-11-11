# Geometric Atlas Implementation

Holoqubed variant in which geometry takes part in the computation. The shape of the MLP input cloud decides, per token, which neurons are evaluated. Neurons whose pre-activation cannot plausibly become positive given where the token sits on the activation manifold are never read.

One engine file, `holoqubed_atlas.py`, plus PyOpenCL; `holoqubed_graph.py` draws its geometry, circuits and topology; `holoqubed_live.py` is the interactive dashboard; `holoqubed_precision.py` measures tile storage and simulator precision. No CUDA, no C++ build step.

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
| Zero-copy dictionary in RAM | add `--host_dictionary` (generate, serve, selftest); the atlas is mapped copy-on-write, so OpenCL uses the file pages in place (Qwen3-0.6B on a CPU device: 234 MB instead of 1,375 MB anonymous memory) |

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

### Adaptive Tile Storage (`qconvert --tile_bits`)

Stores every linear map of the transformer layers smaller than the dense model, and runs the rest of `qconvert` (qneuron calibration, evaluation, generation, PyQrack layer) on the stored tiles. What is stored are the scaled weights B = W diag(w), where w_j is the rms of input column j on calibration text and the columns are sorted by w. Uniform absolute precision on B puts the bits where the inputs are large; a query divides the input by w before it is loaded, W x = B (x / w).

```bash
python3 holoqubed_atlas.py qconvert --model_id Qwen/Qwen3-0.6B --calib wiki_calib.txt --eval wiki_test.txt \
    --tile_bits 4 --tile_codec scalar --tiles_out qwen3.qtiles
```

| Option | Default | Effect |
|---|---|---|
| `--tile_bits` | 0 (off) | target bits per weight of the stored codes (entropy) |
| `--tile_codec` | `scalar` | `scalar`: round(B / step), entropy coded per band of 32 columns; tiles and preparation angles are derived at load. `angles`: per 4x4 tile a log-coded norm and the 15 Mottonen tree angles, circuit ready |
| `--tiles_out` | `<out>/tiles` | directory of the tile file |
| `--tile_calib_tokens` | 4096 | calibration tokens for the input scales |

| File | Content |
|---|---|
| `manifest.json` | format `holoqubed-qtiles/1`, codec, per matrix: shape, step or level offset, stream sizes |
| `scales.npz` | input scales per matrix, fp16 (0.6 MB for Qwen3-0.6B) |
| `tiles.bin` | entropy-coded streams (bz2 or LZMA per stream, whichever is smaller) |

| Angles codec detail | Rule |
|---|---|
| Precision of a tile | level b = rint(offset + log2 of its quantized norm): no side information; tiles below level 0 are dropped |
| Bits per angle | leaves b + 2, split levels b + 2 / b + 1 / b (root first) |
| Norm | 1/16 octave log code, plus b - 3 refinement bits above level 3 |

`qtiles_load(path, model)` decodes a tile directory into a loaded model.

Measured: Qwen3-0.6B, all 196 linear maps (440M weights, 881 MB at bf16), 510 test tokens, dense activation:

| Storage | File size | Bits/weight, file / entropy | PPL (dense 28.02) | Top-1 vs dense | KL |
|---|---|---|---|---|---|
| bf16 source | 881 MB | 16 | 28.02 | 100.00% | 0.0000 |
| int8 weights, scale per 128 | 447 MB | 8.13 | 28.06 | 96.47% | 0.0037 |
| tree angles 6 / 8, fixed precision | 420 MB | 7.63 | 28.18 | 95.88% | 0.0054 |
| **scalar, 6 bits** | 350 MB | 6.36 / 6.00 | 28.16 | 96.67% | 0.0041 |
| **angles, 6 bits** | 335 MB | 6.08 / 6.00 | 28.00 | 96.27% | 0.0063 |
| **scalar, 5 bits** | 285 MB | 5.19 / 5.00 | 28.74 | 91.96% | 0.0177 |
| **scalar, 4 bits** | 229 MB | 4.17 / 4.00 | 29.64 | 88.24% | 0.0581 |
| **angles, 4 bits** | 227 MB | 4.12 / 4.00 | 29.58 | 81.37% | 0.1101 |
| int4 weights, scale per 128 | 227 MB | 4.13 | 37.25 | 66.47% | 0.4082 |
| **scalar, 3 bits** | 179 MB | 3.24 / 3.00 | 32.65 | 70.39% | 0.2609 |
| **angles, 3 bits** | 172 MB | 3.12 / 2.99 | 38.01 | 64.31% | 0.4917 |
| **scalar, 2 bits** | 128 MB | 2.32 / 2.00 | 120.84 | 37.45% | 1.6515 |

| Full `qconvert` on the 4-bit scalar tiles | PPL | Top-1 vs dense | KL |
|---|---|---|---|
| adaptive tiles, dense activation | 29.61 | 88.04% | 0.0581 |
| quantum format (qneuron gating) on the stored tiles | 29.53 | 84.12% | 0.0928 |

Encoding takes about 3 minutes and decoding 25 to 45 seconds on 2 vCPU. Embeddings and the output head (622 MB at bf16) are not tiled: the whole model goes from 1.50 GB to about 0.85 GB at 4 bits.

What was measured on the way (probes on Qwen3-0.6B):

| Question | Result |
|---|---|
| Entropy coding of the fixed 6 / 8 tree angles | 7.62 to 6.75 bits per weight at unchanged quality |
| Input scales per column vs one step per matrix, 4 bits | KL 0.059 vs 0.179 |
| Full rotation into the activation eigenbasis instead of scales | KL 0.059, needs 705 MB of rotations: no gain |
| Seam between tile index and tile content (bond 4 / 8 / 12 of 16) | energy 0.26 / 0.51 / 0.76: no shared structure, also after reordering neurons |
| Seam between rows and columns (rank truncation) at 25% size | loses 8 to 58% of the output energy; 2 to 28% with input scaling; quantization at that size loses 0.06 to 0.4% |
| Angle storage vs scalar storage at equal size | angles cost about 0.6 bits per weight at 4 bits, about 0.2 at 6 |
| Compressor overhead over the code entropy | 3 to 6% (bz2 / LZMA from the standard library) |

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

Downloads Qwen3-0.6B, converts it both ways, runs every tool on it and ends in the live dashboard.

| Stage | Commands | Output |
|---|---|---|
| 1. dependencies | `pip install torch transformers numpy pyopencl pyqrack datasets huggingface_hub matplotlib` | |
| 2. download | `huggingface_hub.snapshot_download` | `$WORK/model` |
| 3. data | `data` (WikiText-2) | `wiki_calib.txt`, `wiki_test.txt` |
| 4. geometric atlas | `forge` (4096 calibration tokens, skipped if present), `selftest` (layers 0, 14, 27), `bench` (z, 2 windows), `generate` (OpenCL, `--compare`) | `qwen3.atlas`, `bench_atlas.json` |
| 5. quantum format | `qconvert` with `--eval`, `--out`, `--qrack_layer` | `qwen3.qformat`, `qconvert.json` |
| 6. graphs | `holoqubed_graph.py` with real tokens | `graphs/` |
| 7. precision | `holoqubed_precision.py` | `precision.json` |
| 8. live dashboard | `holoqubed_live.py` (runs until Ctrl+C) | `http://$LIVE_HOST:$LIVE_PORT` |

| Variable | Default | Effect |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-0.6B` | any gated SiLU model for stages 5 and 7; any supported model for stages 4, 6, 8 |
| `WORK` | `./qwen3-holoqubed` | working directory |
| `STAGES` | `1 2 3 4 5 6 7 8` | stages to run; a stage whose inputs are missing stops with the stage to run first |
| `PROMPT` | "The capital of the Netherlands is" | generation prompt |
| `Z` | 3.0 | atlas ellipsoid width |
| `QRACK_LAYER` | 14 | layer run in PyQrack (stage 5) and marked in the topology graph; `-1` skips the PyQrack run |
| `QRACK_TOKENS` | 6 | tokens generated in the PyQrack run (about 7 minutes per token on 2 vCPU) |
| `TILE_BITS`, `TILE_CODEC` | 0 (off), `scalar` | stage 5 stores the linear maps as adaptive tiles at this many bits per weight and runs on them |
| `WORKERS` | CPU count | QrackSimulator worker processes |
| `GRAPH_LAYERS` | `0,14,27` | layers drawn in the geometry figures |
| `PRECISION_EXPERIMENTS` | all six | rows of stage 7 |
| `PRECISION_LAYERS` | 0 (all) | layers rewritten in stage 7; all 28 take about 70 minutes on 2 vCPU |
| `LIVE_HOST`, `LIVE_PORT` | 127.0.0.1, 8765 | dashboard address |
| `SKIP_PIP` | 0 | `1` skips stage 1 |

| Run | Command |
|---|---|
| everything | `bash qwen_holoqubed.sh` |
| quick, no PyQrack layer, one precision layer | `QRACK_LAYER=-1 PRECISION_LAYERS=1 bash qwen_holoqubed.sh` |
| dashboard only, after a full run | `STAGES=8 bash qwen_holoqubed.sh` |
| dashboard for another machine | `STAGES=8 LIVE_HOST=0.0.0.0 bash qwen_holoqubed.sh` |

Measured fresh run on the 2 vCPU VM (`QRACK_TOKENS=2 PRECISION_LAYERS=4`, everything else default):

| Stage | Time | Result |
|---|---|---|
| 1. dependencies | 5 s | already installed |
| 2. download | 2 min 16 s | 7 files |
| 3. data | 31 s | 1.2 MB calibration, 1.3 MB evaluation text |
| 4a. forge | 2 min 01 s | 28 layers, 64 regions, rank 128 |
| 4b. selftest | 15 s | OpenCL kernels match NumPy on layers 0, 14, 27 |
| 4c. bench | 1 min 43 s | z 3.0: PPL 28.03 vs 28.02, top-1 99.61%, MLP work 108.9% |
| 4d. generate | 36 s | OpenCL and dense outputs compared |
| 5. qconvert | 13 min 47 s | PPL 27.70, top-1 89.41%; layer 14 in PyQrack token-identical, max logit difference 6.0e-5, 125,829,120 gates in 782 s |
| 6. graphs | 29 s | 8 figures |
| 7. precision | 8 min 42 s | 28 rows on 4 layers |
| total to dashboard | 30 min 25 s | |
| 8. dashboard | | torch 5.26, atlas NumPy 2.98, atlas OpenCL 1.54 decode tokens/s (40 tokens, PoCL CPU device) |

Same run with `TILE_BITS=4` (fresh work folder, all 8 stages): 23 min 13 s to the dashboard. Stage 5 takes 11 min 20 s: it writes `qwen3.qtiles` (229 MB), reloads it and runs on the stored tiles; layer 14 in PyQrack stays token-identical to the emulation (max logit difference 7.2e-5, 425 s). Stages 6 to 8 use the dense model in `$WORK/model`.

---

## Unitary Model (`holoqubed_unitary.py`)

A natively unitary language model trained on real text, instead of a converted one. Sequential Born machine at character level: one register of memory qubits interleaved with vocabulary qubits; per character one learned brick-wall circuit of 2-qubit blocks (15 angles + 3 CNOTs each, the same circuit at every step), then a measurement of the vocabulary register. The measurement samples the next character (Born rule, one shot), applies the only nonlinearity (project and renormalize the memory) and leaves the register in |k>, which is the next step's input. Trained in torch on exact complex statevectors with teacher forcing (loss = -log P(true char)), checkpointed per step; replayed in QrackSimulator with real gates and mid-circuit measurements.

```bash
python3 holoqubed_unitary.py --minutes 120                                  # CPU (torch), 2 hours in total
python3 holoqubed_unitary.py --list_devices                                 # OpenCL platforms and devices
python3 holoqubed_unitary.py --gradcheck --devices 6                        # OpenCL gradient vs torch autograd
python3 holoqubed_unitary.py --backend opencl --devices 6 --hours 8 \
    --dataset wikitext-103-raw-v1 --batch 64 --baseline_minutes 40          # 6 OpenCL GPUs, 8 hours
python3 holoqubed_unitary.py --backend opencl --devices 6 --hours 16 \
    --dataset wikitext-103-raw-v1 --batch 64 --resume unitary_model.pt       # continue to 16 hours in total
```

| Option | Default | Effect |
|---|---|---|
| `--minutes` | 120 | total training budget: 55% unitary model, 7% equal-parameter LSTM, 13% LSTM, 13% transformer, rest evaluation |
| `--mem_qubits`, `--vocab_qubits` | 7, 7 | register size; vocabulary = 2^v most frequent characters (last one = unknown) |
| `--depth` | 12 | brick-wall layers per character (78 blocks, 1,170 angles at 14 qubits) |
| `--batch`, `--bptt`, `--lr` | 32, 32, 0.03 | parallel streams, backpropagation steps, Adam learning rate (cosine decay) |
| `--baselines` | `ngram,lstm_equal,lstm,transformer` | 1-, 3-, 5-gram, LSTM with the unitary model's parameter count, LSTM hidden 256, 2-layer transformer |
| `--train`, `--test` | `<dataset>_train.txt`, `<dataset>_test.txt` (e.g. `wikitext-2_train.txt`) | downloaded from `--dataset` when missing |
| `--prompt`, `--replay_chars` | "The ", 160 | PyQrack replay: prompt post-selected, then free sampling |
| `--backend` | `torch` | `torch` (CPU autograd) or `opencl` (data parallel, custom kernels) |
| `--devices`, `--device_type`, `--platform` | 1, `gpu`, any | OpenCL workers (one process per device), device type, platform name filter (e.g. `rusticl`, `AMD`) |
| `--hours` | 0 | unitary training time (total, including a resumed part); baselines then get `--baseline_minutes` (40) |
| `--dataset` | `wikitext-2-raw-v1` | any `Salesforce/wikitext` config, e.g. `wikitext-103-raw-v1` (about 540M characters) |
| `--eval_minutes`, `--ckpt_minutes`, `--log_seconds` | 30, 10, 60 | periodic test evaluation (best kept as `<save>.best`), checkpoint, progress line |
| `--resume`, `--keep_schedule` | | continue from a checkpoint; keep its planned duration instead of the new `--hours` |
| `--baseline_chars` | 0 (all) | cap the CPU baselines' training text on large datasets |
| `--oversubscribe` | off | more workers than devices (testing on one device) |

### OpenCL backend

| Part | Implementation |
|---|---|
| Parallelism | one worker process per device (no shared interpreter lock), each with `--batch` streams; the host sums the block gradients of all workers, maps them to the angles through the block construction and steps Adam |
| Forward | `apply4` per 2-qubit block, `probs` (Born probabilities of the vocabulary register), `pick_char` (loss), `project` (measurement and renormalization) |
| Backward | `mdot` and `mgrad` (gradient of the measurement step), `back4` per block: uncomputes the block with its adjoint, propagates the gradient and accumulates dL/dM per work-group, so no intermediate states are stored; `reduce16` sums the work-groups |
| Device memory | (bptt + 4) x batch x 2^n x 8 bytes + blocks x groups x 128 bytes; 14 qubits, batch 64: about 0.35 GiB |
| Kernel launches | bptt x (3 x blocks + 10) per chunk per device |
| Streams | written once as `.npy`, memory-mapped by the workers |
| Stop and resume | SIGINT or SIGTERM ends after the current chunk and saves; `--resume` restarts the stream positions with the memory in \|0> |

| Check (PoCL CPU, 2 workers sharing one device) | Result |
|---|---|
| Gradient vs torch autograd, first chunk | relative difference 3.6e-6, loss 4.459702 both |
| Gradient vs torch autograd, second chunk (carried state) | relative difference 1.1e-6, loss 4.037255 both |
| Training with periodic evaluation and checkpoints | test BPC 6.97, then 6.80; best checkpoint written |
| Resume to a new total, then SIGINT | continued from 9.2k to 13.3k characters, saved and exited cleanly, workers stopped |
| PyQrack replay of the OpenCL-trained model | max probability difference 2.7e-7 |

Benchmark, torch backend vs OpenCL backend on PoCL, one worker on the 2 vCPU VM (same data, same initial angles, depth 12, bptt 32; first chunk excluded as warm-up):

| Qubits (memory + vocabulary) | Blocks | Streams | torch train chars/s | PoCL train chars/s | PoCL / torch | torch eval chars/s | PoCL eval chars/s | Eval NLL difference |
|---|---|---|---|---|---|---|---|---|
| 10 (5 + 5) | 54 | 16 | 572 | 916 | 1.60x | 3,271 | 3,488 | 1.9e-7 |
| 10 (5 + 5) | 54 | 64 | 1,303 | 1,118 | 0.86x | 3,166 | 3,835 | 1.0e-7 |
| 12 (6 + 6) | 66 | 16 | 286 | 236 | 0.83x | 1,144 | 1,189 | 1.3e-8 |
| 12 (6 + 6) | 66 | 64 | 306 | 249 | 0.81x | 1,141 | 1,165 | 4.5e-8 |
| 14 (7 + 7) | 78 | 16 | 71 | 54 | 0.76x | 347 | 317 | 7.4e-8 |
| 14 (7 + 7) | 78 | 64 | 74 | 54 | 0.73x | 318 | 316 | 2.9e-7 |
| 16 (9 + 7) | 90 | 16 | 17 | 12 | 0.69x | 79 | 74 | 3.1e-9 |

Same model trained for 10 chunks on both backends (14 qubits, 16 streams): the losses agree to 1e-7 for 3 chunks, then drift (largest gap 0.12 BPC, angles 10% apart after 10 Adam steps). Two torch runs whose angles differ by 1e-6 drift the same way (largest gap 0.077 BPC, angles 7.8% apart): the training itself is that sensitive, the backends are equivalent.

On a CPU both backends are limited by memory bandwidth (characters per second barely change between 16 and 64 streams from 12 qubits up); PoCL is 0.7x to 0.86x torch, and only wins (1.6x) at 10 qubits with 16 streams, where torch's per-operation overhead dominates. The OpenCL backend is for GPUs, where the same kernels run on device memory bandwidth and the six workers run on six devices instead of two shared cores. On GPUs the cost per chunk is dominated by kernel launches, which grow with depth and bptt but not with the batch, so a larger `--batch` per device raises characters per second almost linearly until the device is busy.

### Measured: WikiText-2, 120 minutes on 2 vCPU

Test: first 65,536 characters of the test split; vocabulary 128 characters (0.06% unknown).

| Model | Parameters | Recurrent state | Training chars | BPC | Char perplexity |
|---|---|---|---|---|---|
| LSTM, hidden 256 | 370,816 | 256 + 256 reals | 84.76M | 1.982 | 3.95 |
| transformer, 2 layers d 128 | 446,080 | context 128 chars | 41.93M | 2.132 | 4.38 |
| 5-gram (interpolated, discount 0.75) | counts | last 4 chars | 10.89M | 2.141 | 4.41 |
| 3-gram (interpolated, discount 0.75) | counts | last 2 chars | 10.89M | 2.973 | 7.85 |
| LSTM, equal parameters | 1,312 | 4 + 4 reals | 621.30M | 3.403 | 10.57 |
| 1-gram (add-one) | counts | none | 10.89M | 4.608 | 24.39 |
| **unitary Born machine, 14 qubits** | 1,170 | 128 complex amplitudes after each measurement | 0.34M | **5.049** | **33.11** |

Same training characters as the unitary model (0.34M):

| Model | BPC | Char perplexity |
|---|---|---|
| LSTM, equal parameters, lr 0.03 | 3.855 | 14.47 |
| LSTM, equal parameters, lr 0.01 | 4.176 | 18.07 |
| 1-gram (add-one) | 4.614 | 24.49 |
| unitary Born machine | 5.049 | 33.11 |

| Unitary training | Value |
|---|---|
| Throughput | about 85 characters per second (equal-parameter LSTM: about 1.2M) |
| Train BPC per 5 minutes | 6.65, 5.89, 5.59, 5.30, 5.26, 5.21, 5.15, 5.14, 5.07, 5.04, 5.00, 5.01, 4.97 (still falling) |
| Uniform guess | 7.00 BPC |

| PyQrack replay | Value |
|---|---|
| Circuit per character | 780 gates on 14 qubits, 7 measured qubits, 1 shot |
| Max \|P_qrack - P_torch\| over 163 steps | 3.8e-5 (fp32 Qrack) |
| Sample after "The " | characters without words yet: BPC 5 leaves about 33 plausible characters per step |

| Finding | Evidence |
|---|---|
| The native design works mechanically | one shot per character, no readout of vectors, Qrack replay matches torch to 3.8e-5 |
| Pure unitary models are far behind on real text at this size | 5.05 BPC: behind the unigram (4.61) at equal data and the equal-parameter LSTM (3.86) |
| Classical simulation of the training is the bottleneck | 0.34M characters in 66 minutes, still improving; each step updates a 16,384-amplitude statevector per stream |
| Where to push | more depth and data (hardware or a faster simulator), a learned output basis change before measurement, measuring fewer than all vocabulary qubits per step |

---

## Spectral Flattening (`holoqubed_spectral.py`)

Every dense linear map converts exactly to W = U diag(s) V^H: U and V^H are unitary circuits, the diagonal is the only non-unitary part (one ancilla, one controlled rotation per singular value). Block encoded with s_max, a product succeeds with probability ||W x||^2 / (s_max^2 ||x||^2), and the probabilities multiply along the 196 products of a token. The probe freezes U and V^H (fp16) and trains only the singular values (s = s0 exp(a), 200,704 values for Qwen3-0.6B) by distillation against the unmodified model, with a penalty that pushes s_max / s_mean toward 1:

    loss = KL(dense || model) + lambda * mean over matrices of log(s_max^2 / mean(s^2))

```bash
python3 holoqubed_spectral.py --model_id Qwen/Qwen3-0.6B --minutes 80 --lambda_start 0.001 --lambda_end 30
```

| Option | Default | Effect |
|---|---|---|
| `--minutes`, `--eval_minutes` | 80, 6 | run length (lambda ramps geometrically over it), evaluation interval |
| `--lambda_start`, `--lambda_end` | 0.001, 30 | penalty weight at the start and the end |
| `--lr`, `--seq`, `--batch` | 0.005, 256, 1 | Adam on the log singular values, training window |
| `--train`, `--eval` | `wiki_calib.txt`, `wiki_test.txt` | distillation text, evaluation text (2 windows of 256 tokens) |

Conversions without training (all 196 maps):

| Conversion | PPL (dense 28.02) | Top-1 vs dense | KL |
|---|---|---|---|
| exact U diag(s) V^H (U, V^H in fp16) | 28.03 | 100.00% | 0.0000 |
| nearest unitary (polar factor times mean s) | 2.5e13 | 0.00% | 27.3 |
| spectrum flattened, s^0.5 | 8.1e10 | 1.96% | 21.5 |
| nearest unitary in attention only | 1.6e8 | 1.37% | 15.3 |
| nearest unitary in the MLP only | 1.9e10 | 0.20% | 20.3 |

| Matrix (start) | s_max / s_mean | Condition number | Success per product, isotropic input |
|---|---|---|---|
| q_proj | 5.5 | 48 | 0.051 |
| k_proj | 5.5 | 17,529 | 0.059 |
| v_proj | 3.2 | 8,768 | 0.146 |
| o_proj | 3.2 | 20 | 0.131 |
| gate_proj | 4.0 | 69 | 0.073 |
| up_proj | 2.4 | 46 | 0.208 |
| down_proj | 3.1 | 18 | 0.124 |

### Measured: 80 minutes on 2 vCPU, 575 steps

| Step | lambda | PPL | Top-1 vs dense | KL | s_max/s_mean q / up / down | Median success per product | Chain of 196 products, real activations |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 28.03 | 100.00% | 0.0000 | 5.50 / 2.36 / 3.05 | 0.124 | 10^-221 |
| 88 | 0.005 | 27.74 | 90.20% | 0.0343 | 5.54 / 2.35 / 3.03 | 0.126 | 10^-219 |
| 219 | 0.052 | 28.13 | 87.65% | 0.0437 | 5.39 / 2.29 / 2.97 | 0.131 | 10^-214 |
| 308 | 0.254 | 27.96 | 89.02% | 0.0550 | 4.93 / 2.13 / 2.65 | 0.168 | 10^-198 |
| 353 | 0.565 | 28.60 | 85.88% | 0.0741 | 4.44 / 1.98 / 2.32 | 0.219 | 10^-186 |
| 398 | 1.26 | 29.38 | 82.16% | 0.1212 | 3.64 / 1.85 / 2.06 | 0.241 | 10^-172 |
| 443 | 2.79 | 30.29 | 76.86% | 0.2018 | 3.14 / 1.77 / 1.91 | 0.262 | 10^-160 |
| 487 | 6.17 | 31.73 | 73.14% | 0.3345 | 2.95 / 1.72 / 1.81 | 0.268 | 10^-151 |
| 532 | 13.7 | 35.20 | 67.06% | 0.4841 | 2.77 / 1.70 / 1.77 | 0.306 | 10^-147 |
| 575 | 29.6 | 41.92 | 61.18% | 0.7203 | 2.91 / 1.75 / 1.78 | 0.288 | 10^-157 |

| Finding | Evidence |
|---|---|
| The conversion itself is exact | U diag(s) V^H reproduces the model (KL 0.0000); the spectrum carries the knowledge, a unitary replacement destroys it |
| Flattening buys orders of magnitude, not a usable chain | 10^-221 to 10^-147 at KL 0.48 and 67% top-1 |
| A usable chain needs near-unitary products | chain success 0.5 over 196 products needs 0.9965 per product; the run reached a median of about 0.3 |
| Quality falls before the spectra get close | s_max / s_mean stays at 1.7 to 2.9 while top-1 drops to 61% |
| Frozen singular vectors limit the probe | only s is trained; a full fine-tune can rotate U and V and may flatten further at equal quality |
| Training noise costs a little on its own | at negligible lambda the KL already reaches 0.02 to 0.04 (top-1 about 90%) from Adam steps on 200k values |

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

## Live Dashboard (`holoqubed_live.py`)

Interactive variant of the graphs: a local web page served by the Python standard library (no extra packages, no CDN, works offline). Prompts go in, tokens stream out, and every step's performance and geometry is drawn as it arrives.

```bash
python3 holoqubed_live.py --model_id facebook/opt-125m --atlas opt-125m.atlas
# open http://127.0.0.1:8765
```

| Input | Effect |
|---|---|
| Prompt | text to continue; Ctrl+Enter runs |
| Backend | `torch` (original MLP modules), `atlas, NumPy` (reference), `atlas, OpenCL` (kernels, engine built on first use) |
| z | ellipsoid width for the atlas backends |
| Max new tokens, temperature, seed | greedy at temperature 0, else top-40 sampling |
| MLP error vs dense | runs each layer's dense MLP next to the atlas and records the relative error; adds a dense pass to step latency |
| Stop | ends the run after the current token |

| Panel | Shows |
|---|---|
| KPI tiles | decode tokens/s, ms of the last token, prefill ms, MLP work (multiply-adds vs dense, router included), MLP share of step time, process CPU, resident memory |
| Output | generated text; each token tinted by the MLP work of its step, hover for ms, probability and entropy |
| Latency per token | ms per step, prefill first, crosshair tooltip |
| MLP work per token | % of dense multiply-adds per step |
| Layers over tokens | heatmap, selectable metric: routed neurons, MLP work, MLP time, MLP error vs dense |
| Per layer, last token | the same metric as bars |
| Chart plane | one layer: region centroids on chart axes 1 and 2, the Hilbert curve through them, the token path (older steps faded), the current region ringed |
| Regions visited | region per token in Hilbert curve order; dot size grows with distance outside the region's typical radius |
| Run history | backend, z, tokens, prefill, decode tokens/s, mean MLP work and wall time of every run, for side by side comparison |

| Option | Default | Effect |
|---|---|---|
| `--atlas` | none | enables the atlas backends and the geometry panels; without it only `torch` runs |
| `--host`, `--port` | 127.0.0.1, 8765 | listen address; use `--host 0.0.0.0` to watch from another machine |
| `--threads` | torch default | torch threads |
| `--gpus`, `--host_dictionary`, `--lookup`, `--kappa` | as `generate` | OpenCL device count, zero-copy atlas (automatic when every OpenCL device is a CPU), region lookup, SiLU tolerance |
| `--no_warmup` | off | skip paging the memory-mapped atlas in at startup (otherwise the first prefill includes the disk read) |

| Endpoint | Method | Body / result |
|---|---|---|
| `/` | GET | the page |
| `/api/info` | GET | model, backends, per-layer centroids (chart axes 1, 2) and Hilbert order, run history |
| `/api/generate` | POST | `{prompt, backend, z, max_new_tokens, temperature, seed, measure_error}`; server-sent events `start`, `step` (per token: ms, CPU, RSS, probability, entropy, per-layer routed / nonzero / work / ms / error / region / chart position / scale), `done` (run summary), `error` |
| `/api/stop` | POST | stops the running generation |

One generation runs at a time; a second request gets HTTP 409. In the `torch` backend the original MLPs run through the atlas hooks, so geometry is still recorded and the timing includes a small NumPy to torch conversion per layer.

Measured on the 2 vCPU VM, prompt "The capital of the Netherlands is":

| Model | Backend | Decode tokens/s | MLP work | Output |
|---|---|---|---|---|
| OPT-125m | torch | 24.3 | 100% | " the capital of the Netherlands.\n\n" |
| OPT-125m | atlas, NumPy, z 3.5 | 14.9 | 37.7% | identical |
| OPT-125m | atlas, OpenCL (PoCL CPU), z 3.5 | 10.6 | 37.7% | identical |
| Qwen3-0.6B | torch | 7.3 | 100% | " the city of Amsterdam." |
| Qwen3-0.6B | atlas, NumPy, z 3.0 | 3.3 | 109.6% | identical (dense SiLU activations, no saving) |

---

## Checks (`holoqubed_check.sh`)

Run before committing or bundling; about one minute on 2 vCPU.

| Check | What it runs |
|---|---|
| 1. syntax | `py_compile` on every Python script, `bash -n` on the shell scripts |
| 2. command lines | `--help` of every script |
| 3. unitary gradient | `holoqubed_unitary.py --gradcheck`: OpenCL gradient vs torch autograd over two chunks (`DEVICE_TYPE`, default `all`) |
| 4. atlas kernels | `selftest` on `ATLAS=path/to/model.atlas`, when set |

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
