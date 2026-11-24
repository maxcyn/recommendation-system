import math

import pandas as pd
import pytest

from steamrec.data import filter_users, liked_pairs
from steamrec.metrics import evaluate, ndcg_at_k, recall_at_k, recall_capped_at_k
from steamrec.split import split_pairs


def test_liked_pairs_uses_median_among_players_and_floor():
    ug = pd.DataFrame(
        {
            "steam_id": [1, 2, 3, 4, 5, 1, 2, 3],
            "item_id": [10, 10, 10, 10, 10, 20, 20, 20],
            # game 10: players 40, 100, 200 (median 100); owners 4 and 5 never opened it
            # game 20: players 1, 1, 20 (median 1); everyone is at the median but under 30 minutes
            "playtime_forever": [40, 100, 200, 0, 0, 1, 1, 20],
        }
    )
    liked = liked_pairs(ug)
    assert set(map(tuple, liked.to_numpy())) == {(2, 10), (3, 10)}


def test_filter_users_keeps_all_games():
    liked = pd.DataFrame({"steam_id": [1] * 5 + [2] * 4, "item_id": [1, 2, 3, 4, 99, 1, 2, 3, 4]})
    kept = filter_users(liked, min_liked=5)
    assert set(kept.steam_id) == {1}
    assert 99 in set(kept.item_id)


def test_split_counts_and_no_overlap():
    liked = pd.DataFrame(
        {"steam_id": [1] * 5 + [2] * 10 + [3] * 23, "item_id": list(range(5)) + list(range(10)) + list(range(23))}
    )
    out = split_pairs(liked)
    counts = out.groupby(["steam_id", "split"]).size().unstack(fill_value=0)
    assert counts.loc[1].to_dict() == {"test": 1, "train": 3, "val": 1}
    assert counts.loc[2].to_dict() == {"test": 2, "train": 7, "val": 1}
    assert counts.loc[3].to_dict() == {"test": 5, "train": 16, "val": 2}
    assert len(out) == len(liked)
    assert not out.duplicated(["steam_id", "item_id"]).any()
    assert out.equals(split_pairs(liked))  # same seed, same split


def test_split_refuses_people_with_no_training_games():
    with pytest.raises(ValueError):
        split_pairs(pd.DataFrame({"steam_id": [1, 1], "item_id": [1, 2]}))


def test_recall_and_ndcg_hand_worked():
    ranked = [7, 3, 9, 1, 5]
    hidden = {3, 5, 8}
    assert recall_at_k(ranked, hidden, 5) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, hidden, 2) == pytest.approx(1 / 3)
    # hits at positions 2 and 5; best possible is three hits at positions 1, 2, 3
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, hidden, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([3, 5, 8], hidden, 3) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], hidden, 2) == 0.0
    # 15 hidden games, 3 of them in a list of 10: recall 3/15, capped recall 3/10
    many = set(range(100, 115))
    ranked10 = [100, 1, 101, 2, 3, 102, 4, 5, 6, 7]
    assert recall_at_k(ranked10, many, 10) == pytest.approx(3 / 15)
    assert recall_capped_at_k(ranked10, many, 10) == pytest.approx(3 / 10)
    assert recall_capped_at_k(ranked, hidden, 5) == pytest.approx(2 / 3)
    # one hidden game found at position 1 scores 1.0 even when k is larger
    assert ndcg_at_k([4, 1, 2], {4}, 3) == pytest.approx(1.0)


def test_evaluate_averages_over_people():
    res = evaluate({"a": [1, 2], "b": [3, 4]}, {"a": {1}, "b": {9}}, ks=(2,))
    assert res["recall@2"] == pytest.approx(0.5)
    assert res["users"] == 2
