"""Flatten the raw UCSD Australian Steam files into tables.

No label or filter is applied here beyond dropping repeated user records
and repeated (user, game) rows; both are counted in the printed summary.
"""
import ast
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

RAW = Path("data/raw/ucsd")
OUT = Path("data/processed")


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield ast.literal_eval(line)


def flatten_libraries():
    stats = Counter()
    seen_users, games = set(), {}
    with open(OUT / "users.csv", "w", newline="", encoding="utf-8") as fu, \
         open(OUT / "user_games.csv", "w", newline="", encoding="utf-8") as fg:
        wu, wg = csv.writer(fu), csv.writer(fg)
        wu.writerow(["steam_id", "user_id", "items_count"])
        wg.writerow(["steam_id", "item_id", "playtime_forever", "playtime_2weeks"])
        for r in records(RAW / "australian_users_items.json.gz"):
            stats["user_records"] += 1
            sid = r["steam_id"]
            if sid in seen_users:
                stats["repeated_user_records_dropped"] += 1
                continue
            seen_users.add(sid)
            items = r["items"]
            stats["items_count_mismatch"] += r["items_count"] != len(items)
            stats["users_with_empty_library"] += not items
            wu.writerow([sid, r["user_id"], len(items)])
            seen_items = set()
            for it in items:
                iid = it["item_id"]
                if iid in seen_items:
                    stats["repeated_user_game_rows_dropped"] += 1
                    continue
                seen_items.add(iid)
                games.setdefault(iid, it["item_name"])
                pf, p2 = it["playtime_forever"], it["playtime_2weeks"]
                stats["playtime_missing_or_negative"] += pf is None or pf < 0
                stats["never_played_rows"] += pf == 0
                stats["user_game_rows"] += 1
                wg.writerow([sid, iid, pf, p2])
    with open(OUT / "games.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "item_name"])
        w.writerows(sorted(games.items(), key=lambda kv: int(kv[0])))
    stats["users_kept"] = len(seen_users)
    stats["games"] = len(games)
    return stats


def flatten_reviews():
    stats = Counter()
    seen = set()
    with open(OUT / "reviews.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "recommend", "posted"])
        for r in records(RAW / "australian_user_reviews.json.gz"):
            stats["review_user_records"] += 1
            for rv in r["reviews"]:
                key = (r["user_id"], rv["item_id"])
                if key in seen:
                    stats["repeated_reviews_dropped"] += 1
                    continue
                seen.add(key)
                stats["review_rows"] += 1
                w.writerow([r["user_id"], rv["item_id"], int(rv["recommend"]), rv["posted"]])
    stats["review_users"] = len({u for u, _ in seen})
    return stats


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"libraries": dict(flatten_libraries()), "reviews": dict(flatten_reviews())}
    (OUT / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
