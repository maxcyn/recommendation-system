"""Hide part of each person's liked games for testing and tuning (decision D3 in the README).

Per person with n liked games: round(20% of n) go to test and round(10% of n) to
validation, at least one each; the rest are training. The choice is random with a fixed seed.

Writes data/processed/split/{train,val,test}.csv, the id maps, and results/split_summary.json.
"""
import json

import numpy as np
import pandas as pd

from steamrec.data import PROCESSED, ROOT, load_liked

TEST_SHARE = 0.2
VAL_SHARE = 0.1
SEED = 42


def _n_hidden(n: np.ndarray, share: float) -> np.ndarray:
    return np.maximum(1, np.floor(n * share + 0.5).astype(int))


def split_pairs(liked: pd.DataFrame, test_share: float = TEST_SHARE, val_share: float = VAL_SHARE, seed: int = SEED) -> pd.DataFrame:
    """Add a `split` column (train/val/test) to the liked pairs."""
    out = liked.sort_values(["steam_id", "item_id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out["_key"] = rng.random(len(out))
    order = out.groupby("steam_id")["_key"].rank(method="first").to_numpy() - 1
    n = out.groupby("steam_id").item_id.transform("size").to_numpy()
    n_test = _n_hidden(n, test_share)
    n_val = _n_hidden(n, val_share)
    if (n - n_test - n_val < 1).any():
        raise ValueError("a person would have no training games; raise the minimum liked games")
    out["split"] = np.where(order < n_test, "test", np.where(order < n_test + n_val, "val", "train"))
    return out.drop(columns="_key")


def main():
    pairs = split_pairs(load_liked())
    user_ids = np.sort(pairs.steam_id.unique())
    item_ids = np.sort(pairs.item_id.unique())
    pairs["user_idx"] = np.searchsorted(user_ids, pairs.steam_id)
    pairs["item_idx"] = np.searchsorted(item_ids, pairs.item_id)

    out_dir = PROCESSED / "split"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame({"user_idx": range(len(user_ids)), "steam_id": user_ids}).to_csv(out_dir / "user_map.csv", index=False)
    pd.DataFrame({"item_idx": range(len(item_ids)), "item_id": item_ids}).to_csv(out_dir / "item_map.csv", index=False)
    parts = {name: pairs.loc[pairs.split == name, ["user_idx", "item_idx"]] for name in ("train", "val", "test")}
    for name, part in parts.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)

    train_per_item = parts["train"].groupby("item_idx").size().reindex(range(len(item_ids)), fill_value=0)
    per_user = pairs.groupby(["user_idx", "split"]).size().unstack(fill_value=0)
    summary = {
        "seed": SEED,
        "users": len(user_ids),
        "games": len(item_ids),
        "liked_pairs": len(pairs),
        "pairs": {k: len(v) for k, v in parts.items()},
        "share_of_pairs": {k: len(v) / len(pairs) for k, v in parts.items()},
        "per_user_median": {k: float(per_user[k].median()) for k in ("train", "val", "test")},
        "per_user_min": {k: int(per_user[k].min()) for k in ("train", "val", "test")},
        "users_with_exactly_one_test_game": int((per_user["test"] == 1).sum()),
        "share_of_possible_pairs_in_train": len(parts["train"]) / (len(user_ids) * len(item_ids)),
        "games_by_training_lines": {
            "0": int((train_per_item == 0).sum()),
            "1_to_4": int(((train_per_item >= 1) & (train_per_item <= 4)).sum()),
            "5_to_19": int(((train_per_item >= 5) & (train_per_item <= 19)).sum()),
            "20_to_99": int(((train_per_item >= 20) & (train_per_item <= 99)).sum()),
            "100_plus": int((train_per_item >= 100).sum()),
        },
        "hidden_pairs_whose_game_has_no_training_line": {
            k: int((parts[k].item_idx.map(train_per_item) == 0).sum()) for k in ("val", "test")
        },
        "hidden_pairs_whose_game_has_under_5_training_lines": {
            k: int((parts[k].item_idx.map(train_per_item) < 5).sum()) for k in ("val", "test")
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
