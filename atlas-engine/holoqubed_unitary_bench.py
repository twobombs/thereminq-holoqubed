"""Benchmark holoqubed_unitary.py: torch CPU backend vs OpenCL backend (one worker, one device).

  python3 holoqubed_unitary_bench.py speed     # train and eval characters per second, 10 to 16 qubits
  python3 holoqubed_unitary_bench.py equiv     # same model, 10 chunks on both backends, losses compared

Needs wikitext-2_train.txt and wikitext-2_test.txt (written by holoqubed_unitary.py on first run).
"""
QRACK_LIB_PATH = "/usr/local/lib/qrack/libqrack_pinvoke.so"
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holoqubed_unitary as U
from torch.utils.checkpoint import checkpoint

LN2 = math.log(2)


def torch_chunks(model, X, B, T, n_chunks, lr=0.03):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    psi = model.init_state(X[:, 0]); losses, times = [], []
    for c in range(n_chunks):
        t0 = time.perf_counter()
        M = model.blocks(); total = 0.0
        for t in range(T):
            nll, psi = checkpoint(model.step, psi, M, X[:, c * T + t + 1], use_reentrant=False)
            total = total + nll.mean()
        loss = total / T
        opt.zero_grad(); loss.backward(); opt.step(); psi = psi.detach()
        times.append(time.perf_counter() - t0); losses.append(loss.item())
    return losses, times


def cl_chunks(model, ids, eval_ids, B, T, n_chunks, lr=0.03):
    a = argparse.Namespace(devices=1, device_type="all", platform="", oversubscribe=False, batch=B, bptt=T, eval_streams=8)
    tr = U.CLTrainer(model, ids, eval_ids, a, log=lambda s: None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, times = [], []
    for c in range(n_chunks):
        t0 = time.perf_counter()
        loss, _ = tr.gradient(c * T, c == 0)
        opt.step()
        times.append(time.perf_counter() - t0); losses.append(loss)
    return tr, losses, times


def eval_rates(model, tr, eval_ids, streams=8):
    # forward only, same test streams for both backends
    E, Le = streams, (len(eval_ids) - 1) // streams
    t0 = time.perf_counter(); nll_cl, n_cl = tr.evaluate(); t_cl = time.perf_counter() - t0
    X = torch.from_numpy(np.stack([eval_ids[i * Le:i * Le + Le + 1] for i in range(E)]).astype(np.int64))
    t0 = time.perf_counter()
    with torch.no_grad():
        psi = model.init_state(X[:, 0]); M = model.blocks(); nll = 0.0
        for t in range(Le):
            n, psi = model.step(psi, M, X[:, t + 1]); nll += n.sum().item()
    t_t = time.perf_counter() - t0
    return n_cl / t_cl, n_cl / t_t, abs(nll_cl - nll) / max(abs(nll), 1e-9)


def main():
    torch.set_num_threads(2)
    text = open("wikitext-2_train.txt", encoding="utf-8").read()
    test_text = open("wikitext-2_test.txt", encoding="utf-8").read()
    mode = sys.argv[1] if len(sys.argv) > 1 else "speed"
    if mode == "speed":
        print("| Qubits (memory + vocab) | Depth, blocks | Streams | torch train chars/s | PoCL train chars/s | PoCL / torch | "
              "torch eval chars/s | PoCL eval chars/s | eval NLL difference |\n|---|---|---|---|---|---|---|---|---|", flush=True)
        rows = []
        for (m, v) in ((5, 5), (6, 6), (7, 7), (9, 7)):
            vocab = U.CharVocab(text, 2 ** v)
            ids = vocab.encode_np(text); ev = vocab.encode_np(test_text)[: 8 * 512 + 1]
            for B in (16, 64):
                if m + v >= 16 and B > 16:
                    continue
                T = 32
                chunks = 3 if m + v <= 14 else 2
                L = len(ids) // B
                X = torch.from_numpy(ids[: L * B].reshape(B, L).astype(np.int64))
                torch.manual_seed(0); mt = U.UnitaryBorn(m, v, 12)
                _, tt = torch_chunks(mt, X, B, T, chunks + 1)
                torch.manual_seed(0); mc = U.UnitaryBorn(m, v, 12)
                tr, _, tc = cl_chunks(mc, ids, ev, B, T, chunks + 1)
                rt, rc = B * T / np.mean(tt[1:]), B * T / np.mean(tc[1:])      # first chunk is warm-up
                ecl, et, ed = eval_rates(mc, tr, ev)
                tr.close()
                r = dict(n=m + v, m=m, v=v, blocks=len(mt.pairs), B=B, torch=rt, pocl=rc, eval_torch=et, eval_pocl=ecl, eval_diff=ed)
                rows.append(r)
                print(f"| {m + v} ({m} + {v}) | 12, {len(mt.pairs)} | {B} | {rt:,.0f} | {rc:,.0f} | {rc / rt:.2f}x | {et:,.0f} | {ecl:,.0f} | {ed:.1e} |", flush=True)
        json.dump(rows, open("bench_speed.json", "w"), indent=1)
    else:
        m, v, B, T, K = 7, 7, 16, 32, 10
        vocab = U.CharVocab(text, 2 ** v); ids = vocab.encode_np(text); ev = vocab.encode_np(test_text)[:8 * 64 + 1]
        L = len(ids) // B; X = torch.from_numpy(ids[: L * B].reshape(B, L).astype(np.int64))
        torch.manual_seed(0); mt = U.UnitaryBorn(m, v, 12); lt, _ = torch_chunks(mt, X, B, T, K)
        torch.manual_seed(0); mc = U.UnitaryBorn(m, v, 12); tr, lc, _ = cl_chunks(mc, ids, ev, B, T, K); tr.close()
        print("| Chunk | torch loss (BPC) | PoCL loss (BPC) | difference |\n|---|---|---|---|")
        for i, (x, y) in enumerate(zip(lt, lc)):
            print(f"| {i + 1} | {x / LN2:.6f} | {y / LN2:.6f} | {abs(x - y) / LN2:.1e} |")
        print(f"angles after {K} Adam steps: max |theta_torch - theta_pocl| = {float((mt.theta - mc.theta).abs().max()):.1e}, "
              f"relative {float((mt.theta - mc.theta).norm() / mt.theta.norm()):.1e}")


if __name__ == "__main__":
    main()
