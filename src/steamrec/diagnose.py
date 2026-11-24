"""Diagnostics: what do matrix factorisation and LightGCN recommend, and how popular is it?

Usage:
  python -m steamrec.diagnose train mf|lightgcn [--seed 0]   # retrain with the tuned settings, save top-20 lists and positions
  python -m steamrec.diagnose analyse [--seed 0]             # write results/diagnostics_seed<seed>.json
"""
import argparse
import json

import numpy as np
import pandas as pd

from steamrec.data import ROOT
from steamrec.evaluate import FastRecall, _known_and_hidden, load_split, top_k_lists
from steamrec.models.popularity import Popularity

OUT = ROOT / "results"
BANDS = [2, 5, 10, 20, 50, 100000]
# games ranked by number of training likes: rank 1 = most liked
POP_EDGES = [0, 50, 200, 1000, 100000]
POP_LABELS = ["top 50", "51-200", "201-1000", "1001+"]


def train(name: str, seed: int, overrides: dict, tag: str):
    from steamrec.run import make_model

    parts, n_users, n_items = load_split()
    model = make_model(name, {"seed": seed, **overrides})
    model.fit(parts["train"], n_users, n_items, validate=FastRecall("val", k=20, device=model.device))
    known, _ = _known_and_hidden("test")
    lists = top_k_lists(model.score, known, n_users, 20)
    np.savez(OUT / f"diag_{tag or name}_seed{seed}.npz", lists=lists, pu=model._pu, pi=model._pi,
             raw_u=model.user_emb.detach().cpu().numpy(), raw_i=model.item_emb.detach().cpu().numpy())
    print("saved", tag or name, seed, model.settings(), "best val", model.best_val)


def analyse(seed: int, extra: str = None):
    parts, n_users, n_items = load_split()
    known, hidden = _known_and_hidden("test")
    train_likes = parts["train"].groupby("item_idx").size().reindex(range(n_items), fill_value=0).to_numpy()
    rank = np.empty(n_items, dtype=np.int64)
    rank[np.argsort(-train_likes, kind="stable")] = np.arange(1, n_items + 1)
    bucket = np.digitize(rank, POP_EDGES[1:-1], right=True)  # 0..3

    pop = Popularity().fit(parts["train"], n_users, n_items)
    lists = {"popularity": top_k_lists(pop.score, known, n_users, 20)}
    saved = {}
    for name in ("mf", "lightgcn") + (tuple(extra.split(",")) if extra else ()):
        saved[name] = np.load(OUT / f"diag_{name}_seed{seed}.npz")
        lists[name] = saved[name]["lists"]

    n_known = np.array([len(k) for k in known])
    n_hidden = np.array([len(h) for h in hidden])
    band = pd.cut(n_known, BANDS, right=False).astype(str)
    groups = {"all": np.ones(n_users, dtype=bool), **{b: band == b for b in pd.unique(band)}}

    hid_u = np.repeat(np.arange(n_users), n_hidden)
    hid_i = np.concatenate(hidden)
    known_u = np.repeat(np.arange(n_users), n_known)
    known_i = np.concatenate(known)

    out = {"seed": seed, "games_per_bucket": dict(zip(POP_LABELS, np.bincount(bucket, minlength=4).tolist())), "groups": {}}
    for g, mask in sorted(groups.items()):
        res = {"people": int(mask.sum())}
        hm, km = mask[hid_u], mask[known_u]
        res["hidden_games_share_by_bucket"] = _shares(bucket[hid_i[hm]])
        res["known_games_share_by_bucket"] = _shares(bucket[known_i[km]])
        res["hidden_games_median_training_likes"] = float(np.median(train_likes[hid_i[hm]]))
        for name, L in lists.items():
            top = L[mask, :10]
            # was each hidden game caught in its person's top 10?
            is_caught = (L[hid_u[hm], :10] == hid_i[hm][:, None]).any(1)
            m = {
                "recall@10": float(pd.Series(is_caught).groupby(hid_u[hm]).mean().mean()),
                "recall@20": float(pd.Series((L[hid_u[hm], :20] == hid_i[hm][:, None]).any(1)).groupby(hid_u[hm]).mean().mean()),
                "recommended_share_by_bucket": _shares(bucket[top].ravel()),
                "recommended_median_training_likes": float(np.median(train_likes[top])),
                "distinct_games_recommended": int(len(np.unique(top))),
                # of all hidden games in a bucket, the share caught (pooled over games, not averaged over people)
                "share_of_hidden_games_caught_by_bucket": {
                    POP_LABELS[b]: float(is_caught[bucket[hid_i[hm]] == b].mean()) for b in range(4)
                },
                "hits_per_person_by_bucket": {
                    POP_LABELS[b]: float(is_caught[bucket[hid_i[hm]] == b].sum() / mask.sum()) for b in range(4)
                },
            }
            res[name] = m
        a, b = lists["mf"][mask, :10], lists["lightgcn"][mask, :10]
        res["mean_games_shared_by_mf_and_lightgcn_top10"] = float(np.mean([len(np.intersect1d(x, y)) for x, y in zip(a, b)]))
        for name in ("mf", "lightgcn"):
            p = lists["popularity"][mask, :10]
            res[f"mean_games_shared_by_{name}_and_popularity_top10"] = float(
                np.mean([len(np.intersect1d(x, y)) for x, y in zip(lists[name][mask, :10], p)]))
        out["groups"][g] = res

    # how strongly does a game's position size track its popularity?
    logpop = pd.Series(np.log1p(train_likes))
    out["game_position_size_vs_popularity_spearman"] = {
        name: float(pd.Series(np.linalg.norm(saved[name]["pi"], axis=1)).rank().corr(logpop.rank())) for name in saved
    }
    # LightGCN: how much of a person's final position is their own starting numbers?
    lg = saved["lightgcn"]
    own = lg["raw_u"] / 3  # 2 layers: final = (start + round1 + round2) / 3
    share = np.linalg.norm(own, axis=1) / np.linalg.norm(lg["pu"], axis=1)
    out["lightgcn_own_part_over_final_position_size_median_by_band"] = {
        g: float(np.median(share[mask])) for g, mask in sorted(groups.items())
    }
    (OUT / f"diagnostics{'_' + extra.replace(',', '+') if extra else ''}_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _shares(b):
    c = np.bincount(b, minlength=4)
    return dict(zip(POP_LABELS, (c / c.sum()).round(4).tolist()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "analyse"])
    ap.add_argument("model", nargs="?", choices=["mf", "lightgcn", "mf_tags", "lightgcn_tags"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reg", type=float, default=None)
    ap.add_argument("--norm", default=None)
    ap.add_argument("--tag", default=None, help="train: name to save under; analyse: an extra saved model to include")
    a = ap.parse_args()
    overrides = {k: v for k, v in (("reg", a.reg), ("norm", a.norm)) if v is not None}
    train(a.model, a.seed, overrides, a.tag) if a.stage == "train" else analyse(a.seed, a.tag)
