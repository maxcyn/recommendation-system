import numpy as np
import pandas as pd
import torch

from steamrec.models.lightgcn import _SymmetricMM
from steamrec.models.lightgcn_tags import LightGCNTags

# people A, B; games X, Y; tag T. Node order A, B, X, Y, T.
TRAIN = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 0]})
ITEM_TAGS = (np.array([0]), np.array([0]), 1)
N_USERS, N_ITEMS = 2, 2


def expected_adj():
    a = torch.zeros(5, 5)
    for r, c, w in [(0, 2, 6 ** -0.5), (0, 3, 2 ** -0.5), (1, 2, 3 ** -0.5), (2, 4, 3 ** -0.5)]:
        a[r, c] = a[c, r] = w
    return a


def built(**kwargs):
    model = LightGCNTags(item_tags=ITEM_TAGS, **kwargs)
    model.device = torch.device("cpu")
    model.n_users, model.n_items = N_USERS, N_ITEMS
    torch.manual_seed(0)
    model.build(TRAIN, N_USERS, N_ITEMS)
    return model


def test_connection_table_matches_hand_example():
    model = built(dim=3)
    assert model.n_tags == 1
    assert torch.allclose(model.adj.to_dense(), expected_adj(), atol=1e-6)


def test_positions_one_layer():
    dim = 3
    model = built(dim=dim, layers=1)
    users = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    items = torch.tensor([[0.5, -2.0, 1.0], [4.0, 1.0, -3.0]])
    tags = torch.tensor([[2.0, -1.0, 0.25]])
    with torch.no_grad():
        model.user_emb.copy_(users)
        model.item_emb.copy_(items)
        model.tag_emb.copy_(tags)
    pu, pi, raw_u, raw_i = model.positions()
    own = torch.cat([users, items, tags])
    final = (own + expected_adj() @ own) / 2
    assert torch.allclose(pu, final[:2], atol=1e-6)
    assert torch.allclose(pi, final[2:4], atol=1e-6)
    assert raw_u is model.user_emb
    assert raw_i.shape == (2, 2 * dim)
    assert torch.allclose(raw_i[:, :dim], items)
    assert torch.allclose(raw_i[0, dim:], tags[0])
    assert torch.all(raw_i[1, dim:] == 0)


def test_symmetric_mm_gradient():
    model = built(dim=3)
    torch.manual_seed(1)
    x = torch.randn(5, 3, requires_grad=True)
    w = torch.randn(5, 3)
    (_SymmetricMM.apply(x, model.adj) * w).sum().backward()
    assert torch.allclose(x.grad, model.adj.to_dense().T @ w, atol=1e-6)


def test_tiny_fit_moves_tags_and_scores():
    model = LightGCNTags(item_tags=ITEM_TAGS, dim=4, layers=2, max_epochs=2, eval_every=1, batch_size=4)
    model.device = torch.device("cpu")
    initial = {}
    build = model.build

    def recording_build(*args):
        build(*args)
        initial["tag_emb"] = model.tag_emb.detach().clone()

    model.build = recording_build
    model.fit(TRAIN, N_USERS, N_ITEMS, validate=None)
    assert model.tag_emb.shape == (1, 4)
    assert not torch.allclose(model.tag_emb, initial["tag_emb"])
    assert model.score(np.array([0])).shape == (1, N_ITEMS)
