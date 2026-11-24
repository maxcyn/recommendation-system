"""Is the gap between two models bigger than chance?

Two things could move a score by chance:
  1. which people happen to be in the data  -> resample people
  2. the random start of training (the seed) -> resample the seeds that were run

Each of 2,000 rounds redraws the people (with replacement) and the seeds of each model
(with replacement), and recomputes the gap. The middle 95% of those gaps is the interval.
The comparison is paired: both models are scored on the same redrawn people each round.

Usage: python -m steamrec.compare
Reads results/<run>_test_per_user.csv, writes results/comparison.json.
"""
import json

import numpy as np
import pandas as pd

from steamrec.data import ROOT

RESULTS = ROOT / "results"
RUNS = {
    "popularity": ["popularity"],
    "mf": [f"mf_final_seed{i}" for i in range(3)],
    "lightgcn": [f"lightgcn_final_seed{i}" for i in range(3)],
}
PAIRS = [("mf", "popularity"), ("lightgcn", "mf"), ("lightgcn", "popularity")]
METRICS = ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]
BANDS = [2, 5, 10, 20, 50, 100000]
ROUNDS = 2000


def load(model: str, metric: str):
    frames = [pd.read_csv(RESULTS / f"{run}_test_per_user.csv") for run in RUNS[model]]
    return np.stack([f[metric].to_numpy() for f in frames]), frames[0].n_known.to_numpy()


def gap_interval(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """a, b: [seeds, people]. Returns the gap a - b with a 95% interval."""
    n = a.shape[1]
    gaps = np.empty(ROUNDS)
    for r in range(ROUNDS):
        people = rng.integers(0, n, n)
        a_mean = a[rng.integers(0, len(a), len(a))].mean(0)
        b_mean = b[rng.integers(0, len(b), len(b))].mean(0)
        gaps[r] = a_mean[people].mean() - b_mean[people].mean()
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    gap = a.mean() - b.mean()
    same_seed = [float(a[i].mean() - b[i].mean()) for i in range(len(a))] if len(a) == len(b) else None
    return {
        "gap_per_seed_when_both_use_the_same_seed": same_seed,
        "a": float(a.mean()), "b": float(b.mean()), "gap": float(gap), "relative_gain": float(gap / b.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi), "interval_excludes_zero": bool(lo > 0 or hi < 0),
    }


def main():
    rng = np.random.default_rng(0)
    out = {"rounds": ROUNDS, "runs": RUNS, "per_model": {}, "pairs": {}}
    for model in RUNS:
        out["per_model"][model] = {}
        for metric in METRICS:
            scores, _ = load(model, metric)
            per_seed = scores.mean(1)
            out["per_model"][model][metric] = {
                "per_seed": [float(x) for x in per_seed],
                "mean": float(per_seed.mean()),
                "sd_across_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
                "min": float(per_seed.min()), "max": float(per_seed.max()),
            }
    for a_name, b_name in PAIRS:
        key = f"{a_name}_vs_{b_name}"
        out["pairs"][key] = {"overall": {}, "by_known_liked_games": {}}
        for metric in METRICS:
            a, known = load(a_name, metric)
            b, _ = load(b_name, metric)
            out["pairs"][key]["overall"][metric] = gap_interval(a, b, rng)
        for metric in ("recall@10", "ndcg@10"):
            a, known = load(a_name, metric)
            b, _ = load(b_name, metric)
            for lo, hi in zip(BANDS[:-1], BANDS[1:]):
                m = (known >= lo) & (known < hi)
                res = gap_interval(a[:, m], b[:, m], rng)
                res["people"] = int(m.sum())
                out["pairs"][key]["by_known_liked_games"].setdefault(f"[{lo}, {hi})", {})[metric] = res
    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=2))
    for model, d in out["per_model"].items():
        print(model, {m: (round(v["mean"], 4), None if v["sd_across_seeds"] is None else round(v["sd_across_seeds"], 5)) for m, v in d.items()})
    for key, d in out["pairs"].items():
        print("==", key)
        for metric, v in d["overall"].items():
            print(f"  {metric}: {v['a']:.4f} vs {v['b']:.4f}  gap {v['gap']:+.4f} ({v['relative_gain']:+.1%})  95% [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]")
        for band, dd in d["by_known_liked_games"].items():
            v = dd["recall@10"]
            print(f"  {band} n={v['people']}: recall@10 {v['a']:.4f} vs {v['b']:.4f} gap {v['gap']:+.4f} ({v['relative_gain']:+.1%}) 95% [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]")


if __name__ == "__main__":
    main()
