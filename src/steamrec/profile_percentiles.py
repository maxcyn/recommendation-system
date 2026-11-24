"""Compare within-game percentile cut-offs (and optional minimum minutes) for the "liked" rule.

A rule is (q, min_minutes): liked if playtime > 0, playtime >= the game's q-th percentile
among people who played it, and playtime >= min_minutes.
Thumbs up is recommend == 1 in data/processed/reviews.csv.

Writes results/percentile_sweep.json.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "percentile_sweep.json"
RULES = [(30, 0), (40, 0), (45, 0), (50, 0), (60, 0), (40, 30), (50, 30), (60, 30), (40, 60), (40, 120), (0, 120)]


def main():
    ug = pd.read_csv(PROCESSED / "user_games.csv")
    users = pd.read_csv(PROCESSED / "users.csv", dtype={"user_id": str})
    reviews = pd.read_csv(PROCESSED / "reviews.csv", dtype={"user_id": str})

    played = ug[ug.playtime_forever > 0].copy()
    grp = played.groupby("item_id").playtime_forever
    played["players"] = grp.transform("size")
    for q in sorted({q for q, _ in RULES if q}):
        played[f"p{q}"] = played.item_id.map(grp.quantile(q / 100))

    rv = reviews.merge(users[["user_id", "steam_id"]], on="user_id", how="inner")
    assert rv.steam_id.dtype == "int64"
    rp = rv.merge(played, on=["steam_id", "item_id"], how="inner")
    up = rp.recommend == 1

    result = {"played_rows": len(played), "reviewed_played_pairs": len(rp), "thumbs_up_rate": float(up.mean())}
    for q, mn in RULES:
        def liked(df):
            ok = df.playtime_forever >= max(mn, 1)
            return ok & (df.playtime_forever >= df[f"p{q}"]) if q else ok

        lk = liked(played)
        per_user = lk.groupby(played.steam_id).sum()
        per_game = lk.groupby(played.item_id).mean()
        n_players = played.groupby("item_id").size()
        inflated = (per_game > 0.9) & (n_players >= 20)
        lr = liked(rp)
        result[f"p{q}_min{mn}"] = {
            "liked_rows": int(lk.sum()),
            "share_of_played_rows": float(lk.mean()),
            "median_liked_per_user_with_games": float(per_user.reindex(ug.steam_id.unique(), fill_value=0).median()),
            "users_with_5_plus": int((per_user >= 5).sum()),
            "users_with_10_plus": int((per_user >= 10).sum()),
            "games_20plus_players_over_90pct_liked": int(inflated.sum()),
            "liked_rows_in_those_games": int(lk[played.item_id.map(inflated)].sum()),
            "thumbs_up_rate_when_liked": float(up[lr].mean()),
            "thumbs_up_rate_when_not_liked": float(up[~lr].mean()),
            "share_of_thumbs_up_marked_liked": float(lr[up].mean()),
            "share_of_thumbs_down_marked_not_liked": float((~lr)[~up].mean()),
        }
    OUT.write_text(json.dumps(result, indent=2))
    cols = list(next(v for v in result.values() if isinstance(v, dict)))
    print(pd.DataFrame({k: v for k, v in result.items() if isinstance(v, dict)}).loc[cols].T.to_string())
    print({k: v for k, v in result.items() if not isinstance(v, dict)})

    # hybrid: thumbs overrides the p40 rule where a review exists
    rall = rv.merge(ug, on=["steam_id", "item_id"], how="left", indicator=True)
    in_lib = rall["_merge"] == "both"
    p40 = rp.playtime_forever >= rp.p40
    print("hybrid vs p40: liked->removed (thumbs down)", int((p40 & ~up).sum()),
          "| not liked->added (thumbs up, played)", int((~p40 & up).sum()),
          "| thumbs up, in library, zero minutes", int((in_lib & (rall.playtime_forever == 0) & (rall.recommend == 1)).sum()),
          "| thumbs up, not in library", int((~in_lib & (rall.recommend == 1)).sum()),
          "| reviewers with a played+reviewed pair", int(rp.steam_id.nunique()))


if __name__ == "__main__":
    main()
