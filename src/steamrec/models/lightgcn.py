"""LightGCN: matrix factorisation plus a graph step.

Each person and game starts with its own position, as in matrix factorisation. Then, `layers`
times over, every person's position is replaced by the average of the games they liked, and
every game's position by the average of the people who liked it (averages are weighted so that
very popular games and very active people do not dominate). The final position is the mean
of the starting position and the position after each round. After 1 round a person's position
reflects their games; after 2, other people who liked those games; after 3, those people's games.

Only training lines are used to build the graph. Scoring and training are the same as in
matrix factorisation, so the graph step is the only difference between the two models.
"""
import numpy as np
import torch

from steamrec.models.mf import MatrixFactorisation


class _SymmetricMM(torch.autograd.Function):
    """adj @ x for a symmetric adj. Same result as torch.sparse.mm, about 3x faster to train,
    because the backward step reuses adj where the built-in one rebuilds its transpose."""

    @staticmethod
    def forward(ctx, x, adj):
        ctx.adj = adj
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx, grad):
        return torch.sparse.mm(ctx.adj, grad), None


class LightGCN(MatrixFactorisation):
    def __init__(self, layers=2, norm="sym", **kwargs):
        """norm="sym": a line's strength is 1/sqrt(person's likes x game's likes) (the standard LightGCN).
        norm="mean": each person and game takes the plain average of what its neighbours send."""
        super().__init__(**kwargs)
        assert norm in ("sym", "mean")
        self.layers, self.norm = layers, norm

    def build(self, train, n_users, n_items):
        super().build(train, n_users, n_items)
        u = train.user_idx.to_numpy()
        i = train.item_idx.to_numpy() + n_users
        rows = np.concatenate([u, i])
        cols = np.concatenate([i, u])
        deg = np.bincount(rows, minlength=n_users + n_items).astype(np.float64)
        inv_sqrt = np.zeros_like(deg)
        inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
        weights = inv_sqrt[rows] * inv_sqrt[cols]
        n = n_users + n_items
        self.adj = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([rows, cols])), torch.as_tensor(weights, dtype=torch.float32), (n, n)
        ).coalesce().to_sparse_csr().to(self.device)
        # plain average = D^-1 A = D^-1/2 (sym-normalised A) D^1/2, so the symmetric matrix is reused
        self.deg_sqrt = torch.as_tensor(np.sqrt(deg), dtype=torch.float32, device=self.device)[:, None]
        self.deg_inv_sqrt = torch.as_tensor(inv_sqrt, dtype=torch.float32, device=self.device)[:, None]

    def positions(self):
        x = torch.cat([self.user_emb, self.item_emb])
        total = x
        for _ in range(self.layers):
            if self.norm == "mean":
                x = self.deg_inv_sqrt * _SymmetricMM.apply(self.deg_sqrt * x, self.adj)
            else:
                x = _SymmetricMM.apply(x, self.adj)
            total = total + x
        final = total / (self.layers + 1)
        return final[: self.n_users], final[self.n_users :], self.user_emb, self.item_emb

    def settings(self) -> dict:
        return {**super().settings(), "layers": self.layers, "norm": self.norm}
