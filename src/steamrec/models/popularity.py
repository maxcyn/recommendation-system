"""Popularity baseline: recommend everyone the games most people liked in training."""
import numpy as np
import pandas as pd


class Popularity:
    history: list = []
    best_val = None

    def settings(self) -> dict:
        return {}

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, validate=None) -> "Popularity":
        self.counts = train.groupby("item_idx").size().reindex(range(n_items), fill_value=0).to_numpy(dtype=np.float64)
        # tiny tie-break by index so the ranking is the same on every run
        self.counts = self.counts - np.arange(n_items) * 1e-9
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.counts, (len(users), 1))
