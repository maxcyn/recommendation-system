"""Build the "liked" pairs from the processed library table (decision D4 in the README).

A person liked a game if their playtime is at or above the game's median playtime
among people who played it (more than zero minutes), and is at least 30 minutes.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

PERCENTILE = 50
MIN_MINUTES = 30
MIN_LIKED_PER_USER = 5


def liked_pairs(user_games: pd.DataFrame, percentile: float = PERCENTILE, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """Return the (steam_id, item_id) rows that count as liked."""
    played = user_games[user_games.playtime_forever > 0]
    cutoff = played.groupby("item_id").playtime_forever.quantile(percentile / 100)
    keep = (played.playtime_forever >= played.item_id.map(cutoff)) & (played.playtime_forever >= min_minutes)
    return played.loc[keep, ["steam_id", "item_id"]].reset_index(drop=True)


def filter_users(liked: pd.DataFrame, min_liked: int = MIN_LIKED_PER_USER) -> pd.DataFrame:
    """Keep people with at least min_liked liked games. Games are never dropped."""
    n = liked.groupby("steam_id").item_id.transform("size")
    return liked[n >= min_liked].reset_index(drop=True)


def load_liked() -> pd.DataFrame:
    return filter_users(liked_pairs(pd.read_csv(PROCESSED / "user_games.csv")))
