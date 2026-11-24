"""Game tags from the UCSD metadata file, for the models that use them.

`load_item_tags()` returns which tags each game in the split has. Games missing from the
metadata file, or listed there without tags, simply have no tag lines.
"""
import ast
import gzip

import numpy as np
import pandas as pd

from steamrec.data import ROOT
from steamrec.evaluate import SPLIT_DIR

RAW_GAMES = ROOT / "data" / "raw" / "ucsd" / "steam_games.json.gz"
CACHE = SPLIT_DIR / "item_tags.csv"
TAG_MAP = SPLIT_DIR / "tag_map.csv"


def build_item_tags() -> pd.DataFrame:
    item_map = pd.read_csv(SPLIT_DIR / "item_map.csv")
    idx_of = dict(zip(item_map.item_id.astype(str), item_map.item_idx))
    pairs = set()
    with gzip.open(RAW_GAMES, "rt", encoding="utf-8") as f:
        for line in f:
            rec = ast.literal_eval(line)
            item = idx_of.get(str(rec.get("id")))
            if item is not None:
                pairs.update((item, tag) for tag in rec.get("tags") or [])
    df = pd.DataFrame(sorted(pairs), columns=["item_idx", "tag"])
    tags = sorted(df.tag.unique())
    pd.DataFrame({"tag_idx": range(len(tags)), "tag": tags}).to_csv(TAG_MAP, index=False)
    df["tag_idx"] = df.tag.map({t: i for i, t in enumerate(tags)})
    df[["item_idx", "tag_idx"]].to_csv(CACHE, index=False)
    return df[["item_idx", "tag_idx"]]


def load_item_tags() -> tuple[np.ndarray, np.ndarray, int]:
    """(item_idx, tag_idx) for every game-tag line, and the number of different tags."""
    df = pd.read_csv(CACHE) if CACHE.exists() else build_item_tags()
    return df.item_idx.to_numpy(), df.tag_idx.to_numpy(), int(df.tag_idx.max()) + 1


def tag_average_matrix(item_idx: np.ndarray, tag_idx: np.ndarray, n_items: int, n_tags: int, device):
    """Sparse [n_items, n_tags] matrix M with M[i, t] = 1 / (number of tags game i has).

    `torch.sparse.mm(M, tag_rows)` is, for each game, the average of its tags' rows
    (all zeros for a game without tags).
    """
    import torch

    counts = np.bincount(item_idx, minlength=n_items).astype(np.float32)
    values = 1.0 / counts[item_idx]
    return torch.sparse_coo_tensor(
        torch.as_tensor(np.stack([item_idx, tag_idx])), torch.as_tensor(values), (n_items, n_tags)
    ).coalesce().to(device)


if __name__ == "__main__":
    df = build_item_tags()
    print(len(df), "game-tag lines;", df.item_idx.nunique(), "games with tags;", df.tag_idx.nunique(), "tags")
