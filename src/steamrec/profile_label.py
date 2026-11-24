"""Profile the candidate "liked" rules on the processed UCSD tables.

Rules compared (playtime_forever is in minutes):
  two_hours  : playtime >= 120
  percentile : playtime > 0 and playtime >= the game's 40th percentile,
               measured among people who played the game (playtime > 0)
  both       : two_hours and percentile

Writes results/label_profile.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "label_profile.json"
RULES = ["two_hours", "percentile", "both"]
MIN_LIKED = [1, 2, 5, 10, 20, 50]


def add_rules(ug: pd.DataFrame) -> pd.DataFrame:
    played = ug[ug.playtime_forever > 0]
    grp = played.groupby("item_id").playtime_forever
    per_game = pd.DataFrame({"players": grp.size(), "p40": grp.quantile(0.40)})
    ug = ug.join(per_game, on="item_id")
    ug["two_hours"] = ug.playtime_forever >= 120
    ug["percentile"] = (ug.playtime_forever > 0) & (ug.playtime_forever >= ug.p40)
    ug["both"] = ug.two_hours & ug.percentile
    return ug, per_game


def liked_per_user(ug: pd.DataFrame, n_users_total: int) -> dict:
    out = {}
    n_with_games = ug.steam_id.nunique()
    for rule in RULES:
        counts = ug.groupby("steam_id")[rule].sum()
        q = counts.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        out[rule] = {
            "liked_rows": int(ug[rule].sum()),
            "share_of_owned_rows": float(ug[rule].mean()),
            "share_of_played_rows": float(ug.loc[ug.playtime_forever > 0, rule].mean()),
            "mean_per_user_with_games": float(counts.mean()),
            "quantiles_per_user_with_games": {str(k): float(v) for k, v in q.items()},
            "users_with_at_least": {
                str(m): {
                    "users": int((counts >= m).sum()),
                    "share_of_users_with_games": float((counts >= m).sum() / n_with_games),
                    "share_of_all_users": float((counts >= m).sum() / n_users_total),
                    "share_of_liked_rows": float(counts[counts >= m].sum() / counts.sum()),
                }
                for m in MIN_LIKED
            },
        }
    return out


def thumbs_agreement(ug: pd.DataFrame, users: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    out = {"review_rows": len(reviews)}
    pair_counts = reviews.groupby(["user_id", "item_id"]).recommend.agg(["size", "nunique"])
    out["pairs"] = len(pair_counts)
    out["pairs_reviewed_more_than_once"] = int((pair_counts["size"] > 1).sum())
    out["pairs_with_conflicting_thumbs"] = int((pair_counts["nunique"] > 1).sum())
    # one row per pair; drop the few pairs whose reviews disagree
    ok = pair_counts[pair_counts["nunique"] == 1].index
    rv = reviews.set_index(["user_id", "item_id"]).loc[ok].reset_index().drop_duplicates(["user_id", "item_id"])
    out["pairs_used"] = len(rv)
    out["user_id_unique_in_users"] = bool(users.user_id.is_unique)

    # inner merge keeps steam_id as int64 (a left merge would turn it into float and corrupt the ids)
    n_pairs = len(rv)
    rv = rv.merge(users[["user_id", "steam_id"]], on="user_id", how="inner")
    assert rv.steam_id.dtype == "int64"
    out["pairs_without_user_match"] = n_pairs - len(rv)
    m = rv.merge(ug, on=["steam_id", "item_id"], how="left", indicator=True)
    assert len(m) == len(rv)
    in_lib = m["_merge"] == "both"
    out["pairs_not_in_library"] = {
        "n": int((~in_lib).sum()),
        "thumbs_up_rate": float(m.loc[~in_lib, "recommend"].mean()),
    }
    m = m[in_lib].copy()
    m[RULES] = m[RULES].astype(bool)
    out["pairs_in_library"] = {"n": len(m), "thumbs_up_rate": float(m.recommend.mean())}
    zero = m.playtime_forever == 0
    out["pairs_in_library_zero_minutes"] = {
        "n": int(zero.sum()),
        "thumbs_up_rate": float(m.loc[zero, "recommend"].mean()),
    }
    p = m[~zero].copy()
    out["pairs_played"] = {
        "n": len(p),
        "thumbs_up": int(p.recommend.sum()),
        "thumbs_down": int((p.recommend == 0).sum()),
        "thumbs_up_rate": float(p.recommend.mean()),
        "median_minutes_thumbs_up": float(p.loc[p.recommend == 1, "playtime_forever"].median()),
        "median_minutes_thumbs_down": float(p.loc[p.recommend == 0, "playtime_forever"].median()),
    }

    def table(df, rule):
        liked = df[rule].astype(bool)
        up = df.recommend == 1
        return {
            "liked_up": int((liked & up).sum()),
            "liked_down": int((liked & ~up).sum()),
            "notliked_up": int((~liked & up).sum()),
            "notliked_down": int((~liked & ~up).sum()),
            "share_marked_liked": float(liked.mean()),
            "thumbs_up_rate_when_liked": float(up[liked].mean()),
            "thumbs_up_rate_when_not_liked": float(up[~liked].mean()),
            "share_of_thumbs_up_marked_liked": float(liked[up].mean()),
            "share_of_thumbs_down_marked_not_liked": float((~liked)[~up].mean()),
            "balanced_agreement": float((liked[up].mean() + (~liked)[~up].mean()) / 2),
        }

    out["rules_on_played_pairs"] = {r: table(p, r) for r in RULES}
    # where the two rules disagree
    only_2h = p.two_hours & ~p.percentile
    only_pct = ~p.two_hours & p.percentile
    out["disagreements_on_played_pairs"] = {
        "two_hours_only": {"n": int(only_2h.sum()), "thumbs_up_rate": float(p.loc[only_2h, "recommend"].mean())},
        "percentile_only": {"n": int(only_pct.sum()), "thumbs_up_rate": float(p.loc[only_pct, "recommend"].mean())},
        "both": {"n": int(p.both.sum()), "thumbs_up_rate": float(p.loc[p.both, "recommend"].mean())},
        "neither": {
            "n": int((~p.two_hours & ~p.percentile).sum()),
            "thumbs_up_rate": float(p.loc[~p.two_hours & ~p.percentile, "recommend"].mean()),
        },
    }
    # thumbs-up rate by within-game playtime rank (rank among all players of the game)
    played_all = ug[ug.playtime_forever > 0]
    rank = played_all.groupby("item_id").playtime_forever.rank(pct=True)
    p = p.merge(
        played_all.assign(rank_pct=rank)[["steam_id", "item_id", "rank_pct"]],
        on=["steam_id", "item_id"],
        how="left",
    )
    bins = pd.cut(p.rank_pct, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    out["thumbs_up_by_within_game_rank"] = {
        str(k): {"n": int(v["size"]), "thumbs_up_rate": float(v["mean"])}
        for k, v in p.groupby(bins, observed=True).recommend.agg(["size", "mean"]).iterrows()
    }
    hbins = pd.cut(p.playtime_forever, [0, 30, 60, 120, 300, 600, 1800, 6000, np.inf])
    out["thumbs_up_by_minutes"] = {
        str(k): {"n": int(v["size"]), "thumbs_up_rate": float(v["mean"])}
        for k, v in p.groupby(hbins, observed=True).recommend.agg(["size", "mean"]).iterrows()
    }
    return out


def main():
    ug = pd.read_csv(PROCESSED / "user_games.csv")
    users = pd.read_csv(PROCESSED / "users.csv", dtype={"user_id": str})
    reviews = pd.read_csv(PROCESSED / "reviews.csv", dtype={"user_id": str})
    assert not ug.duplicated(["steam_id", "item_id"]).any()

    ug, per_game = add_rules(ug)
    played_rows = int((ug.playtime_forever > 0).sum())
    result = {
        "owned_rows": len(ug),
        "played_rows": played_rows,
        "users_total": len(users),
        "users_with_games": int(ug.steam_id.nunique()),
        "games_owned": int(ug.item_id.nunique()),
        "games_played_by_someone": len(per_game),
        "per_game_p40_minutes": {
            "quantiles_across_games": {
                str(k): float(v) for k, v in per_game.p40.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()
            },
            "games_with_p40_below_120": int((per_game.p40 < 120).sum()),
            "games_with_p40_below_30": int((per_game.p40 < 30).sum()),
            "share_of_played_rows_in_games_with_p40_below_120": float(
                per_game.loc[per_game.p40 < 120, "players"].sum() / played_rows
            ),
        },
        "players_per_game": {
            "quantiles": {str(k): float(v) for k, v in per_game.players.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()},
            **{
                f"games_with_fewer_than_{m}_players": {
                    "games": int((per_game.players < m).sum()),
                    "share_of_played_rows": float(per_game.loc[per_game.players < m, "players"].sum() / played_rows),
                }
                for m in (5, 10, 20, 50)
            },
        },
        "liked_per_user": liked_per_user(ug, len(users)),
        "thumbs_agreement": thumbs_agreement(ug, users, reviews),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
