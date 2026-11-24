"""Score a model on the hidden games, the same way for every model.

A model is any function `score(user_idx_batch) -> array [len(batch), n_games]`, higher = better.
For each person, games they are already known to like are removed, the top K are taken,
and the list is compared with their hidden games.
  - on validation: training games are removed
  - on test: training and validation games are removed
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from steamrec.data import PROCESSED
from steamrec.metrics import ndcg_at_k, recall_at_k, recall_capped_at_k

SPLIT_DIR = PROCESSED / "split"
KS = (10, 20)


@lru_cache(maxsize=1)
def load_split():
    parts = {name: pd.read_csv(SPLIT_DIR / f"{name}.csv") for name in ("train", "val", "test")}
    n_users = len(pd.read_csv(SPLIT_DIR / "user_map.csv"))
    n_items = len(pd.read_csv(SPLIT_DIR / "item_map.csv"))
    return parts, n_users, n_items


def _by_user(df: pd.DataFrame, n_users: int) -> list:
    out = [np.empty(0, dtype=np.int64)] * n_users
    for u, items in df.groupby("user_idx").item_idx:
        out[u] = items.to_numpy()
    return out


@lru_cache(maxsize=2)
def _known_and_hidden(on: str):
    parts, n_users, _ = load_split()
    known_df = parts["train"] if on == "val" else pd.concat([parts["train"], parts["val"]])
    return _by_user(known_df, n_users), _by_user(parts[on], n_users)


def top_k_lists(score, known: list, n_users: int, k: int, batch: int = 1024) -> np.ndarray:
    """Top-k game indices per person, best first, never including a known game."""
    out = np.empty((n_users, k), dtype=np.int64)
    for start in range(0, n_users, batch):
        users = np.arange(start, min(start + batch, n_users))
        s = np.array(score(users), dtype=np.float32)
        rows = np.repeat(np.arange(len(users)), [len(known[u]) for u in users])
        s[rows, np.concatenate([known[u] for u in users])] = -np.inf
        top = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1, kind="stable")
        out[users] = np.take_along_axis(top, order, axis=1)
    return out


class FastRecall:
    """Recall@k on the graphics card, for use during training only.

    Same definition as `evaluate_model` (known games removed, hits / hidden games, averaged
    over people). Final results always come from `evaluate_model`.
    """

    def __init__(self, on: str = "val", k: int = 20, device="cpu", batch: int = 4096):
        import torch

        self.torch, self.k, self.batch = torch, k, batch
        parts, self.n_users, self.n_items = load_split()
        known_df = parts["train"] if on == "val" else pd.concat([parts["train"], parts["val"]])

        def prep(df):
            df = df.sort_values("user_idx")
            u = df.user_idx.to_numpy()
            starts = np.searchsorted(u, np.arange(self.n_users + 1))
            return torch.as_tensor(u, device=device), torch.as_tensor(df.item_idx.to_numpy(), device=device), starts

        self.known = prep(known_df)
        self.hidden = prep(parts[on])
        self.n_hidden = torch.as_tensor(np.diff(self.hidden[2]), device=device, dtype=torch.float32)

    def __call__(self, pu, pi) -> float:
        torch = self.torch
        hits = torch.zeros(self.n_users, device=pu.device)
        for start in range(0, self.n_users, self.batch):
            end = min(start + self.batch, self.n_users)
            s = pu[start:end] @ pi.T
            ku, ki, ks = self.known
            a, b = ks[start], ks[end]
            s[ku[a:b] - start, ki[a:b]] = -torch.inf
            top = s.topk(self.k, dim=1).indices
            hu, hi, hs = self.hidden
            a, b = hs[start], hs[end]
            is_hidden = torch.zeros_like(s, dtype=torch.bool)
            is_hidden[hu[a:b] - start, hi[a:b]] = True
            hits[start:end] = is_hidden.gather(1, top).sum(1)
        return float((hits / self.n_hidden).mean())


def evaluate_model(score, on: str = "test", ks=KS) -> tuple[dict, pd.DataFrame]:
    """Return (summary, per-person table) for the chosen hidden set ("val" or "test")."""
    parts, n_users, n_items = load_split()
    known, hidden = _known_and_hidden(on)
    lists = top_k_lists(score, known, n_users, max(ks))

    rows = []
    for u in range(n_users):
        h = set(hidden[u].tolist())
        ranked = lists[u].tolist()
        row = {"user_idx": u, "n_known": len(known[u]), "n_hidden": len(h)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, h, k)
            row[f"recall_capped@{k}"] = recall_capped_at_k(ranked, h, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, h, k)
            row[f"hits@{k}"] = sum(1 for g in ranked[:k] if g in h)
        rows.append(row)
    per_user = pd.DataFrame(rows)

    train_likes = parts["train"].groupby("item_idx").size().reindex(range(n_items), fill_value=0).to_numpy()
    summary = {"evaluated_on": on, "users": n_users, "games": n_items}
    for k in ks:
        summary[f"recall@{k}"] = float(per_user[f"recall@{k}"].mean())
        summary[f"recall_capped@{k}"] = float(per_user[f"recall_capped@{k}"].mean())
        summary[f"ndcg@{k}"] = float(per_user[f"ndcg@{k}"].mean())
        summary[f"mean_hits@{k}"] = float(per_user[f"hits@{k}"].mean())
        summary[f"share_of_people_with_a_hit@{k}"] = float((per_user[f"hits@{k}"] > 0).mean())
        rec = lists[:, :k]
        summary[f"distinct_games_recommended@{k}"] = int(len(np.unique(rec)))
        summary[f"median_training_likes_of_recommended_game@{k}"] = float(np.median(train_likes[rec]))
    return summary, per_user
