"""Ranking metrics for one person's recommendation list.

`ranked` is the list of recommended game indices, best first, with games the person
already has in training removed. `hidden` is the set of that person's hidden liked games.
"""
import numpy as np


def recall_at_k(ranked, hidden, k: int) -> float:
    """Share of the hidden games that appear in the top k."""
    hidden = set(hidden)
    hits = sum(1 for g in ranked[:k] if g in hidden)
    return hits / len(hidden)


def recall_capped_at_k(ranked, hidden, k: int) -> float:
    """Hidden games caught, as a share of the most a list of k could catch.

    Same as recall for people with k or fewer hidden games. For people with more, a
    perfect list scores 1.0, which plain recall cannot reach.
    """
    hidden = set(hidden)
    hits = sum(1 for g in ranked[:k] if g in hidden)
    return hits / min(k, len(hidden))


def ndcg_at_k(ranked, hidden, k: int) -> float:
    """Like recall, but a hidden game near the top of the list counts for more.

    1.0 means the hidden games fill the top of the list; 0.0 means none are in the top k.
    """
    hidden = set(hidden)
    gains = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = sum(gains[i] for i, g in enumerate(ranked[:k]) if g in hidden)
    return float(dcg / gains[: min(k, len(hidden))].sum())


def evaluate(ranked_by_user: dict, hidden_by_user: dict, ks=(10, 20)) -> dict:
    """Average each metric over the people who have at least one hidden game."""
    users = [u for u, h in hidden_by_user.items() if len(h) > 0]
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(ranked_by_user[u], hidden_by_user[u], k) for u in users]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(ranked_by_user[u], hidden_by_user[u], k) for u in users]))
    out["users"] = len(users)
    return out
