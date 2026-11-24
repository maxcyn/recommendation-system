import numpy as np
import pandas as pd
import torch

from steamrec.models.mf_tags import MatrixFactorisationTags

# game 0 has tags {0, 1}, game 1 has tag {0}, game 2 has none
ITEM_TAGS = (np.array([0, 0, 1]), np.array([0, 1, 0]), 2)
TRAIN = pd.DataFrame({"user_idx": [0, 0, 1, 1], "item_idx": [0, 1, 1, 2]})


def make(**kwargs):
    model = MatrixFactorisationTags(item_tags=ITEM_TAGS, **kwargs)
    model.device = torch.device("cpu")
    return model


def test_game_position_is_own_row_plus_average_of_tag_rows():
    model = make(dim=2)
    model.build(TRAIN, n_users=2, n_items=3)
    item = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    tag = torch.tensor([[10.0, 20.0], [30.0, 60.0]])
    with torch.no_grad():
        model.item_emb.copy_(item)
        model.tag_emb.copy_(tag)

    pu, pi, raw_u, raw_i = model.positions()
    # game 0: own + mean of both tags; game 1: own + tag 0; game 2: own only
    expected = torch.tensor([[1.0 + 20.0, 2.0 + 40.0], [3.0 + 10.0, 4.0 + 20.0], [5.0, 6.0]])
    assert torch.equal(pi, expected)
    assert pu is model.user_emb and raw_u is model.user_emb

    assert raw_i.shape == (3, 4)
    assert torch.equal(raw_i[:, :2], item)
    assert torch.equal(raw_i[:, 2:], torch.tensor([[20.0, 40.0], [10.0, 20.0], [0.0, 0.0]]))


def test_training_changes_tag_rows():
    model = make(dim=4, max_epochs=2, eval_every=1, batch_size=4)
    torch.manual_seed(model.seed)
    model.build(TRAIN, n_users=2, n_items=3)
    initial = model.tag_emb.detach().clone()

    model.fit(TRAIN, n_users=2, n_items=3, validate=None)
    # fit() rebuilds from the same seed, so the tag rows started from `initial`
    assert model.tag_emb.shape == initial.shape
    assert not torch.allclose(model.tag_emb, initial)
    assert model.score(np.array([0])).shape == (1, 3)


def test_state_dict_round_trip_restores_tag_rows():
    model = make(dim=4)
    model.build(TRAIN, n_users=2, n_items=3)
    assert set(model.state_dict()) == {"user_emb", "item_emb", "tag_emb"}
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert not torch.equal(model.tag_emb, saved["tag_emb"])

    model.load_state_dict(saved)
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k])
