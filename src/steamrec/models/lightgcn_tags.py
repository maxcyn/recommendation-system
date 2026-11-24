"""LightGCN where tags are extra dots in the graph, connected to the games that carry them.

In LightGCN the graph has two kinds of dots, people and games, with a line between a person
and each game they liked. Here every tag is a third kind of dot with its own position, and
there is also a line between a game and each of its tags. The graph step is unchanged: `layers`
times over, every dot's position is replaced by the weighted average of its neighbours', so
a game now hears from its tags as well as from the people who liked it, and two games that
share tags pull towards each other even if nobody liked both.

A line's strength is 1/sqrt(lines at one end x lines at the other end), counting both kinds
of line. Only people and games are scored; the tags' final positions are not used. Scoring and
training are the same as in LightGCN, so the tag dots are the only difference between the two.
"""
import numpy as np
import torch

from steamrec.models.lightgcn import LightGCN, _SymmetricMM
from steamrec.models.mf import MatrixFactorisation
from steamrec.tags import load_item_tags, tag_average_matrix


class LightGCNTags(LightGCN):
    def __init__(self, item_tags=None, **kwargs):
        """item_tags: (item_idx, tag_idx, n_tags) for every game-tag line; read from the split if None."""
        super().__init__(**kwargs)
        assert self.norm == "sym"
        self.item_tags = item_tags

    def build(self, train, n_users, n_items):
        MatrixFactorisation.build(self, train, n_users, n_items)
        tag_items, tag_idx, n_tags = self.item_tags if self.item_tags is not None else load_item_tags()
        tag_items, tag_idx = np.asarray(tag_items), np.asarray(tag_idx)
        self.n_tags = n_tags
        self.tag_emb = torch.nn.Parameter(torch.randn(n_tags, self.dim, device=self.device) * 0.1)
        u = train.user_idx.to_numpy()
        i = train.item_idx.to_numpy() + n_users
        g = tag_items + n_users
        t = tag_idx + n_users + n_items
        rows = np.concatenate([u, i, g, t])
        cols = np.concatenate([i, u, t, g])
        n = n_users + n_items + n_tags
        deg = np.bincount(rows, minlength=n).astype(np.float64)
        inv_sqrt = np.zeros_like(deg)
        inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
        weights = inv_sqrt[rows] * inv_sqrt[cols]
        self.adj = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([rows, cols])), torch.as_tensor(weights, dtype=torch.float32), (n, n)
        ).coalesce().to_sparse_csr().to(self.device)
        self.tag_avg = tag_average_matrix(tag_items, tag_idx, n_items, n_tags, self.device)

    def parameters(self):
        return [self.user_emb, self.item_emb, self.tag_emb]

    def state_dict(self):
        return {"user_emb": self.user_emb, "item_emb": self.item_emb, "tag_emb": self.tag_emb}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.user_emb.copy_(state["user_emb"])
            self.item_emb.copy_(state["item_emb"])
            self.tag_emb.copy_(state["tag_emb"])

    def positions(self):
        x = torch.cat([self.user_emb, self.item_emb, self.tag_emb])
        total = x
        for _ in range(self.layers):
            x = _SymmetricMM.apply(x, self.adj)
            total = total + x
        final = total / (self.layers + 1)
        # the penalty covers a game's own row and the average of its tags' rows
        raw_items = torch.cat([self.item_emb, torch.sparse.mm(self.tag_avg, self.tag_emb)], dim=1)
        return final[: self.n_users], final[self.n_users : self.n_users + self.n_items], self.user_emb, raw_items
