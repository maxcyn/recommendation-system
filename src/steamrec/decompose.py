"""Which part of LightGCN pulls its recommendations towards popular games?

LightGCN's final position is the mean of three parts: the starting numbers ("own"), the result of
the first mixing round ("round 1") and of the second ("round 2"). This script rebuilds the three
parts from the saved seed-0 model (see diagnose.py), then re-ranks games using one part at a time
and reports, for each variant, recall and how popular the recommended games are.

Usage: python -m steamrec.decompose [--seed 0]   -> results/decompose_seed<seed>.json
"""
import argparse
import json

import numpy as np
import torch

from steamrec.diagnose import OUT, POP_EDGES, POP_LABELS
from steamrec.evaluate import _known_and_hidden, load_split, top_k_lists


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main(seed: int):
    parts, n_users, n_items = load_split()
    known, hidden = _known_and_hidden("test")
    train = parts["train"]
    train_likes = train.groupby("item_idx").size().reindex(range(n_items), fill_value=0).to_numpy()
    rank = np.empty(n_items, dtype=np.int64)
    rank[np.argsort(-train_likes, kind="stable")] = np.arange(1, n_items + 1)
    bucket = np.digitize(rank, POP_EDGES[1:-1], right=True)

    mf = np.load(OUT / f"diag_mf_seed{seed}.npz")
    lg = np.load(OUT / f"diag_lightgcn_seed{seed}.npz")

    # rebuild the mixing matrix exactly as LightGCN.build does
    u = train.user_idx.to_numpy()
    i = train.item_idx.to_numpy() + n_users
    rows, cols = np.concatenate([u, i]), np.concatenate([i, u])
    deg = np.bincount(rows, minlength=n_users + n_items).astype(np.float64)
    inv = np.zeros_like(deg)
    inv[deg > 0] = deg[deg > 0] ** -0.5
    n = n_users + n_items
    adj = torch.sparse_coo_tensor(torch.as_tensor(np.stack([rows, cols])),
                                  torch.as_tensor(inv[rows] * inv[cols], dtype=torch.float32), (n, n)).coalesce()
    e0 = torch.as_tensor(np.concatenate([lg["raw_u"], lg["raw_i"]]))
    e1 = torch.sparse.mm(adj, e0)
    e2 = torch.sparse.mm(adj, e1)
    e = [x.numpy() for x in (e0, e1, e2)]
    final = (e[0] + e[1] + e[2]) / 3
    rebuild_error = float(max(np.abs(final[:n_users] - lg["pu"]).max(), np.abs(final[n_users:] - lg["pi"]).max()))
    U = [x[:n_users] for x in e]
    I = [x[n_users:] for x in e]
    names = ["own", "round 1", "round 2"]

    variants = {
        "mf: full": (mf["pu"], mf["pi"]),
        "mf: game sizes removed": (mf["pu"], unit(mf["pi"])),
        "lightgcn: full": (lg["pu"], lg["pi"]),
        "lightgcn: game sizes removed": (lg["pu"], unit(lg["pi"])),
        "lightgcn: own numbers only, both sides": (U[0], I[0]),
    }
    for k in range(3):
        variants[f"lightgcn: person {names[k]} x full game"] = (U[k], lg["pi"])
    for k in range(3):
        variants[f"lightgcn: full person x game {names[k]}"] = (lg["pu"], I[k])

    n_known = np.array([len(k) for k in known])
    n_hidden = np.array([len(h) for h in hidden])
    groups = {"everyone": np.ones(n_users, dtype=bool), "4 known": n_known < 5,
              "5 to 19 known": (n_known >= 5) & (n_known < 20), "20 to 49 known": (n_known >= 20) & (n_known < 50),
              "50+ known": n_known >= 50}
    hid_u = np.repeat(np.arange(n_users), n_hidden)
    hid_i = np.concatenate(hidden)
    niche_hidden = bucket[hid_i] >= 2  # ranked 201 and below

    out = {"seed": seed, "rebuild_error": rebuild_error, "variants": {}}
    for name, (pu, pi) in variants.items():
        L = top_k_lists(lambda users: pu[users] @ pi.T, known, n_users, 10)
        caught = (L[hid_u] == hid_i[:, None]).any(1)
        res = {}
        for g, mask in groups.items():
            hm = mask[hid_u]
            per_person = np.bincount(hid_u[hm], weights=caught[hm], minlength=n_users)[mask] / n_hidden[mask]
            b = bucket[L[mask]].ravel()
            res[g] = {
                "recall@10": float(per_person.mean()),
                "slots_top50": float((b == 0).mean()),
                "slots_rank_201_plus": float((b >= 2).mean()),
                "hits_per_person_top50": float(caught[hm & (bucket[hid_i] == 0)].sum() / mask.sum()),
                "hits_per_person_rank_201_plus": float(caught[hm & niche_hidden].sum() / mask.sum()),
                "distinct_games": int(len(np.unique(L[mask]))),
            }
        out["variants"][name] = res

    # how large are game positions, by popularity? (median length, relative to the least popular bucket)
    sizes = {"mf": mf["pi"], "lightgcn full": lg["pi"], "lightgcn own": I[0], "lightgcn round 1": I[1], "lightgcn round 2": I[2]}
    out["median_game_position_size_by_bucket"] = {
        k: {POP_LABELS[b]: float(np.median(np.linalg.norm(v, axis=1)[bucket == b])) for b in range(4)} for k, v in sizes.items()
    }
    # how large is each part of a person's position, by group (median length)
    out["median_person_part_size_by_group"] = {
        g: {names[k]: float(np.median(np.linalg.norm(U[k][mask], axis=1))) for k in range(3)} for g, mask in groups.items()
    }
    (OUT / f"decompose_seed{seed}.json").write_text(json.dumps(out, indent=2))

    print("rebuild error", rebuild_error)
    for g in groups:
        print("\n==", g)
        for name, res in out["variants"].items():
            r = res[g]
            print(f"  {name:42s} recall {r['recall@10']:.4f}  top50 slots {r['slots_top50']:.3f}  201+ slots {r['slots_rank_201_plus']:.3f}"
                  f"  hits top50 {r['hits_per_person_top50']:.3f}  hits 201+ {r['hits_per_person_rank_201_plus']:.3f}  distinct {r['distinct_games']}")
    print(json.dumps(out["median_game_position_size_by_bucket"], indent=1))
    print(json.dumps(out["median_person_part_size_by_group"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
