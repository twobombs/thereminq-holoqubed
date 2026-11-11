#!/usr/bin/env bash
# ThereminQ Holoqubed: download Qwen3-0.6B, convert it, run it through holoqubed_atlas.py and finish in the
# live dashboard.
#
#   1. dependencies          torch, transformers, pyopencl, pyqrack, datasets, huggingface_hub, matplotlib
#   2. download              Qwen3-0.6B into $WORK/model (huggingface_hub.snapshot_download)
#   3. data                  WikiText-2 calibration and evaluation text
#   4. geometric atlas       forge -> OpenCL kernel selftest -> bench vs dense -> generate
#   5. quantum format        qconvert: 4x4 qubit tiles + qneuron gating, perplexity vs dense,
#                            generation, and one layer genuinely in PyQrack compared token for token
#   6. graphs                holoqubed_graph.py: atlas geometry, model topology, tile / qneuron / signed
#                            circuits, QrackBridge patch chains
#   7. precision             holoqubed_precision.py: fp16, Gray / tree angle and int-b storage, fp32 to fp8
#                            statevector, qneuron gating vs exact tiles
#   8. live dashboard        holoqubed_live.py on http://$LIVE_HOST:$LIVE_PORT (runs until Ctrl+C)
#
# Settings (environment variables):
#   MODEL_ID              Hugging Face id                          default Qwen/Qwen3-0.6B
#   WORK                  working directory                        default ./qwen3-holoqubed
#   STAGES                stages to run, space separated           default "1 2 3 4 5 6 7 8"
#   PROMPT                generation prompt                        default "The capital of the Netherlands is"
#   Z                     atlas ellipsoid width                    default 3.0
#   QRACK_LAYER           layer run in PyQrack, -1 to skip         default 14
#   QRACK_TOKENS          tokens generated in the PyQrack run      default 6 (about 7 minutes per token on 2 vCPU)
#   TILE_BITS, TILE_CODEC adaptive tile storage in stage 5         default 0 (off), scalar; e.g. TILE_BITS=4
#   WORKERS               QrackSimulator worker processes          default: number of CPUs
#   GRAPH_LAYERS          layers drawn in the geometry figures     default 0,14,27
#   PRECISION_EXPERIMENTS precision rows                           default fp16,gray,tree,int,statevector,qformat
#   PRECISION_LAYERS      layers rewritten, 0 = all                default 0 (about 70 minutes on 2 vCPU)
#   LIVE_HOST, LIVE_PORT  dashboard address                        default 127.0.0.1, 8765
#   SKIP_PIP              set to 1 to skip dependency installation (same as leaving 1 out of STAGES)
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
WORK="${WORK:-$PWD/qwen3-holoqubed}"
STAGES=" ${STAGES:-1 2 3 4 5 6 7 8} "
PROMPT="${PROMPT:-The capital of the Netherlands is}"
Z="${Z:-3.0}"
QRACK_LAYER="${QRACK_LAYER:-14}"
QRACK_TOKENS="${QRACK_TOKENS:-6}"
TILE_BITS="${TILE_BITS:-0}"
TILE_CODEC="${TILE_CODEC:-scalar}"
WORKERS="${WORKERS:-$(nproc)}"
GRAPH_LAYERS="${GRAPH_LAYERS:-0,14,27}"
PRECISION_EXPERIMENTS="${PRECISION_EXPERIMENTS:-fp16,gray,tree,int,statevector,qformat}"
PRECISION_LAYERS="${PRECISION_LAYERS:-0}"
LIVE_HOST="${LIVE_HOST:-127.0.0.1}"
LIVE_PORT="${LIVE_PORT:-8765}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$HERE/holoqubed_atlas.py"
MODEL="$WORK/model"

step() { printf '\n== %s ==\n' "$1"; }
want() { [[ "$STAGES" == *" $1 "* ]]; }
need() { [ -e "$1" ] || { echo "missing $1: run stage $2 first (STAGES=\"$2 ...\")" >&2; exit 1; }; }
mkdir -p "$WORK"
cd "$WORK"

if want 1 && [ "${SKIP_PIP:-0}" != "1" ]; then
    step "1. dependencies"
    pip install --break-system-packages -q torch transformers numpy pyopencl pyqrack datasets huggingface_hub matplotlib
fi

if want 2; then
    step "2. download $MODEL_ID"
    python3 - "$MODEL_ID" "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print("model files in", snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"]))
EOF
fi

if want 3; then
    step "3. calibration and evaluation text"
    [ -f wiki_calib.txt ] && [ -f wiki_test.txt ] || python3 "$ENGINE" data --out_dir .
fi

if want 4; then
    need "$MODEL/config.json" 2; need wiki_calib.txt 3
    step "4a. geometric atlas: forge"
    [ -f qwen3.atlas/manifest.json ] || python3 "$ENGINE" forge --model_id "$MODEL" --calib wiki_calib.txt \
        --calib_tokens 4096 --out qwen3.atlas

    step "4b. geometric atlas: OpenCL kernels vs NumPy reference"
    python3 "$ENGINE" selftest --atlas qwen3.atlas --layers 0,14,27

    step "4c. geometric atlas: divergence vs dense"
    python3 "$ENGINE" bench --model_id "$MODEL" --atlas qwen3.atlas --eval wiki_test.txt --windows 2 \
        --z "$Z" --static "" --json bench_atlas.json

    step "4d. geometric atlas: generation (OpenCL backend)"
    python3 "$ENGINE" generate --model_id "$MODEL" --atlas qwen3.atlas --backend opencl --z "$Z" \
        --prompt "$PROMPT" --max_new_tokens 24 --compare
fi

if want 5; then
    need "$MODEL/config.json" 2; need wiki_calib.txt 3
    step "5. quantum format: tiles + qneurons, perplexity, PyQrack layer $QRACK_LAYER"
    python3 "$ENGINE" qconvert --model_id "$MODEL" --calib wiki_calib.txt --eval wiki_test.txt --out qwen3.qformat \
        --prompt "$PROMPT" --max_new_tokens "$QRACK_TOKENS" --qrack_layer "$QRACK_LAYER" --workers "$WORKERS" \
        --tile_bits "$TILE_BITS" --tile_codec "$TILE_CODEC" --tiles_out qwen3.qtiles --json qconvert.json
fi

if want 6; then
    need qwen3.atlas/manifest.json 4; need wiki_test.txt 3
    step "6. graphs: geometry, topology, circuits, patches"
    python3 "$HERE/holoqubed_graph.py" --atlas qwen3.atlas --layers "$GRAPH_LAYERS" --model_id "$MODEL" \
        --text wiki_test.txt --qrack_layer "$QRACK_LAYER" --out graphs
fi

if want 7; then
    need "$MODEL/config.json" 2; need wiki_calib.txt 3
    step "7. precision: tile storage and simulator formats vs int quantization"
    python3 "$HERE/holoqubed_precision.py" --model_id "$MODEL" --eval wiki_test.txt --calib wiki_calib.txt \
        --experiments "$PRECISION_EXPERIMENTS" --layers "$PRECISION_LAYERS" --json precision.json
fi

step "done"
echo "atlas:          $WORK/qwen3.atlas"
echo "quantum format: $WORK/qwen3.qformat"
echo "graphs:         $WORK/graphs"
echo "results:        $WORK/bench_atlas.json, $WORK/qconvert.json, $WORK/precision.json"
echo "serve API:      python3 $ENGINE serve --model_id $MODEL --atlas $WORK/qwen3.atlas --z $Z --port 8000"

if want 8; then
    need "$MODEL/config.json" 2; need qwen3.atlas/manifest.json 4
    step "8. live dashboard: http://$LIVE_HOST:$LIVE_PORT (Ctrl+C to stop)"
    exec python3 "$HERE/holoqubed_live.py" --model_id "$MODEL" --atlas qwen3.atlas \
        --host "$LIVE_HOST" --port "$LIVE_PORT"
fi
