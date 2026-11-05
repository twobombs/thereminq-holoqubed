#!/usr/bin/env bash
# ThereminQ Holoqubed: download Qwen3-0.6B, convert it and run it through holoqubed_atlas.py.
#
#   1. dependencies          torch, transformers, pyopencl, pyqrack, datasets, huggingface_hub
#   2. download              Qwen3-0.6B into $WORK/model (huggingface_hub.snapshot_download)
#   3. data                  WikiText-2 calibration and evaluation text
#   4. geometric atlas       forge -> OpenCL kernel selftest -> bench vs dense -> generate
#   5. quantum format        qconvert: 4x4 qubit tiles + qneuron gating, perplexity vs dense,
#                            generation, and one layer genuinely in PyQrack compared token for token
#
# Settings (environment variables):
#   MODEL_ID      Hugging Face id                      default Qwen/Qwen3-0.6B
#   WORK          working directory                    default ./qwen3-holoqubed
#   PROMPT        generation prompt                    default "The capital of the Netherlands is"
#   Z             atlas ellipsoid width                default 3.0
#   QRACK_LAYER   layer run in PyQrack, -1 to skip     default 14
#   QRACK_TOKENS  tokens generated in the PyQrack run  default 6 (about 7 minutes per token on 2 vCPU)
#   WORKERS       QrackSimulator worker processes      default: number of CPUs
#   SKIP_PIP      set to 1 to skip dependency installation
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
WORK="${WORK:-$PWD/qwen3-holoqubed}"
PROMPT="${PROMPT:-The capital of the Netherlands is}"
Z="${Z:-3.0}"
QRACK_LAYER="${QRACK_LAYER:-14}"
QRACK_TOKENS="${QRACK_TOKENS:-6}"
WORKERS="${WORKERS:-$(nproc)}"
ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/holoqubed_atlas.py"
MODEL="$WORK/model"

step() { printf '\n== %s ==\n' "$1"; }
mkdir -p "$WORK"
cd "$WORK"

if [ "${SKIP_PIP:-0}" != "1" ]; then
    step "1. dependencies"
    pip install --break-system-packages -q torch transformers numpy pyopencl pyqrack datasets huggingface_hub
fi

step "2. download $MODEL_ID"
python3 - "$MODEL_ID" "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print("model files in", snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"]))
EOF

step "3. calibration and evaluation text"
[ -f wiki_calib.txt ] && [ -f wiki_test.txt ] || python3 "$ENGINE" data --out_dir .

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

step "5. quantum format: tiles + qneurons, perplexity, PyQrack layer $QRACK_LAYER"
python3 "$ENGINE" qconvert --model_id "$MODEL" --calib wiki_calib.txt --eval wiki_test.txt --out qwen3.qformat \
    --prompt "$PROMPT" --max_new_tokens "$QRACK_TOKENS" --qrack_layer "$QRACK_LAYER" --workers "$WORKERS" \
    --json qconvert.json

step "done"
echo "atlas:          $WORK/qwen3.atlas"
echo "quantum format: $WORK/qwen3.qformat"
echo "results:        $WORK/bench_atlas.json, $WORK/qconvert.json"
echo "serve:          python3 $ENGINE serve --model_id $MODEL --atlas $WORK/qwen3.atlas --z $Z --port 8000"
