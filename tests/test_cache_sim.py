import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import cache_sim as cs  # noqa: E402
import make_fake_log as mk  # noqa: E402


def test_sanity():
    log = mk.make_log(T=400, L=8, E=32, k=4, seed=1)
    log["num_experts"] = int(log["num_experts"])
    topk = log["topk"].astype(np.int64)
    E = log["num_experts"]
    rows, _, first = cs.cache_eval(topk, E, log["is_gen"], log["seq_id"], fracs=(0.1, 0.3, 0.5, 1.0))  # first = max hit rate = 1 - distinct/accesses
    r = {(f, n): a for f, _, n, a, _ in rows}
    for f in (0.1, 0.3, 0.5, 1.0):
        assert r[(f, "Belady")] >= r[(f, "LRU")] - 1e-12
        assert r[(f, "Belady")] >= r[(f, "LFU")] - 1e-12
        if f < 1.0:  # static is pre-loaded (no cold misses), so only comparable below 100%
            assert r[(f, "Belady")] >= r[(f, "Static(oracle)")] - 1e-12
    for n in ("LRU", "LFU", "Belady"):
        assert abs(r[(1.0, n)] - first) < 1e-9
    # LRU/LFU hit rate should not shrink with capacity
    assert r[(0.5, "LRU")] >= r[(0.1, "LRU")]
    # report runs end to end
    assert "## e. Co-occurrence" in cs.report(log)


NEW = ("ARC", "2Q", "LRFU(1tok)", "LRFU(4tok)", "LRFU(16tok)", "S3-FIFO", "LRU/layer", "LRU+look")


def test_new_policies_sanity():
    log = mk.make_log(T=400, L=8, E=32, k=4, seed=2)
    topk = log["topk"].astype(np.int64)
    E = int(log["num_experts"])
    pred = log.get("pred")
    offs = log.get("pred_offsets")
    if pred is None:  # synthesize a noisy d=1 prediction [1,T,L,2k]
        rng = np.random.default_rng(0)
        T, L, k = topk.shape
        pred = rng.integers(0, E, size=(1, T, L, 2 * k))
        pred[0, :, :-1, :k] = topk[:, 1:]
        offs = np.array([1])
    fr = (0.1, 0.3, 0.5, 1.0)
    rows, _, first = cs.cache_eval(topk, E, log["is_gen"], log["seq_id"], fracs=fr, pred=pred, pred_offsets=offs)
    r = {(f, n): a for f, _, n, a, _ in rows}
    for n in NEW:
        assert (0.1, n) in r, n
        for f in fr:
            assert r[(f, "Belady")] >= r[(f, n)] - 1e-12, (n, f)
        assert abs(r[(1.0, n)] - first) < 1e-9, n
