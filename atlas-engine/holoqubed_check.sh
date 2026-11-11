#!/usr/bin/env bash
# ThereminQ Holoqubed: quick checks before committing or bundling (about one minute on 2 vCPU).
#
#   1. every Python script compiles, the shell scripts parse
#   2. every command line interface loads (--help)
#   3. OpenCL gradient of the unitary model matches torch autograd (two chunks)
#   4. tile codecs round trip exactly, including codes beyond 16 bits
#   5. optional: OpenCL atlas kernels match the NumPy reference      ATLAS=path/to/model.atlas
#
#   DEVICE_TYPE   OpenCL device type for check 3: gpu, cpu or all     default all
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
fail=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fail=1; }

echo "1. syntax"
for f in *.py; do python3 -m py_compile "$f" 2>/dev/null && ok "$f" || bad "$f"; done
for f in *.sh; do bash -n "$f" 2>/dev/null && ok "$f" || bad "$f"; done

echo "2. command lines"
for f in holoqubed_atlas.py holoqubed_graph.py holoqubed_live.py holoqubed_precision.py holoqubed_spectral.py holoqubed_unitary.py; do
    python3 "$f" --help >/dev/null 2>&1 && ok "$f --help" || bad "$f --help"
done

echo "3. unitary model: OpenCL gradient vs torch"
out="$(python3 holoqubed_unitary.py --gradcheck --devices 1 --device_type "${DEVICE_TYPE:-all}" 2>&1)" \
    && ok "$(printf '%s\n' "$out" | grep 'chunk 2' | sed 's/.*relative/relative/')" || bad "gradcheck"

echo "4. tile codecs: round trip with an outlier input column"
python3 - <<'PY' >/dev/null 2>&1 && ok "scalar and angles codecs" || bad "tile codecs"
import numpy as np, holoqubed_atlas as H
rng = np.random.default_rng(0)
W = rng.standard_normal((64, 128)) * 0.05
scale = np.abs(rng.standard_normal(128)) + 0.1
scale[7] = 5000.0                                        # codes beyond 16 bits
scale = scale.astype(np.float16).astype(np.float64)
order = np.argsort(-scale, kind="stable")
for bits in (3.0, 6.0):
    meta, st = H.qtiles_encode(W, scale, bits, "scalar")
    Wq = H.qtiles_decode(meta, st, scale)
    direct = np.empty_like(W); direct[:, order] = np.rint((W * scale)[:, order] / meta["step"]) * meta["step"]
    assert np.allclose(Wq * scale, direct, rtol=1e-12, atol=0), "scalar codec is not the stored quantization"
    assert max(meta["width"]) == (4 if bits > 4 else 2) and abs(meta["entropy_bits"] - bits) < 0.05, "rate or code width"
    meta, st = H.qtiles_encode(W, scale, bits, "angles")
    Wa = H.qtiles_decode(meta, st, scale)
    assert np.linalg.norm((Wa - W) * scale) / np.linalg.norm(W * scale) < (0.01 if bits < 4 else 0.002), "angles codec error"
PY

if [ -n "${ATLAS:-}" ]; then
    echo "5. atlas kernels vs NumPy"
    python3 holoqubed_atlas.py selftest --atlas "$ATLAS" >/dev/null 2>&1 && ok "selftest $ATLAS" || bad "selftest $ATLAS"
fi

[ "$fail" = 0 ] && echo "all checks passed" || { echo "checks failed"; exit 1; }
