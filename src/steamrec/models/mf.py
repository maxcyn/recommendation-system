"""Matrix factorisation trained with the BPR loss.

Every person and every game gets a list of `dim` numbers (a position on a "taste map").
A person's score for a game is the dot product of the two positions. Training repeatedly
takes a game a person liked and a random game they did not, and nudges the positions so
the liked game scores higher than the other one.

This is LightGCN without the graph step, with the same loss, so the gap between the two
models shows what the graph adds.
"""
import numpy as np
import pandas as pd
import torch


class BPRTrainer:
    """Shared training loop for models that produce person and game positions."""

    def __init__(self, dim=64, lr=1e-3, reg=3e-3, batch_size=4096, max_epochs=1000, eval_every=5, patience=5, seed=0):
        self.dim, self.lr, self.reg, self.batch_size = dim, lr, reg, batch_size
        self.max_epochs, self.eval_every, self.patience, self.seed = max_epochs, eval_every, patience, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []

    # subclasses define build() and positions()
    def build(self, train: pd.DataFrame, n_users: int, n_items: int):
        raise NotImplementedError

    def positions(self):
        """Return (person positions, game positions) and the raw parameters to regularise."""
        raise NotImplementedError

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, validate=None):
        """`validate(person_positions, game_positions) -> score` is called every `eval_every` epochs."""
        torch.manual_seed(self.seed)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.n_users, self.n_items = n_users, n_items
        self.build(train, n_users, n_items)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        users = torch.as_tensor(train.user_idx.to_numpy(), device=self.device)
        items = torch.as_tensor(train.item_idx.to_numpy(), device=self.device)
        n = len(users)
        liked_keys = torch.sort(users * n_items + items).values
        best, best_state, bad = -1.0, None, 0

        for epoch in range(1, self.max_epochs + 1):
            perm = torch.randperm(n, device=self.device, generator=gen)
            neg = torch.randint(0, n_items, (n,), device=self.device, generator=gen)
            # redraw any "random game" the person actually liked in training
            for _ in range(20):
                keys = users * n_items + neg
                clash = liked_keys[torch.searchsorted(liked_keys, keys).clamp(max=n - 1)] == keys
                if not clash.any():
                    break
                neg[clash] = torch.randint(0, n_items, (int(clash.sum()),), device=self.device, generator=gen)
            total = torch.zeros((), device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                u, i, j = users[idx], items[idx], neg[idx]
                pu, pi, raw_u, raw_i = self.positions()
                pos = (pu[u] * pi[i]).sum(1)
                negs = (pu[u] * pi[j]).sum(1)
                loss = torch.nn.functional.softplus(negs - pos).mean()
                l2 = (raw_u[u].pow(2).sum() + raw_i[i].pow(2).sum() + raw_i[j].pow(2).sum()) / (2 * len(idx))
                opt.zero_grad()
                (loss + self.reg * l2).backward()
                opt.step()
                total += loss.detach() * len(idx)

            if validate is not None and epoch % self.eval_every == 0:
                with torch.no_grad():
                    pu, pi, _, _ = self.positions()
                    score = validate(pu, pi)
                self.history.append({"epoch": epoch, "train_loss": total.item() / n, "val_recall@20": score})
                print(self.history[-1], flush=True)
                if score > best:
                    best, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.best_val = best
        self._freeze()
        return self

    def _freeze(self):
        with torch.no_grad():
            pu, pi, _, _ = self.positions()
            self._pu, self._pi = pu.cpu().numpy(), pi.cpu().numpy()

    def score(self, users: np.ndarray) -> np.ndarray:
        return self._pu[users] @ self._pi.T

    def settings(self) -> dict:
        return {k: getattr(self, k) for k in ("dim", "lr", "reg", "batch_size", "max_epochs", "eval_every", "patience", "seed")}


class MatrixFactorisation(BPRTrainer):
    def build(self, train, n_users, n_items):
        self.user_emb = torch.nn.Parameter(torch.randn(n_users, self.dim, device=self.device) * 0.1)
        self.item_emb = torch.nn.Parameter(torch.randn(n_items, self.dim, device=self.device) * 0.1)

    def parameters(self):
        return [self.user_emb, self.item_emb]

    def state_dict(self):
        return {"user_emb": self.user_emb, "item_emb": self.item_emb}

    def load_state_dict(self, state):
        with torch.no_grad():
            self.user_emb.copy_(state["user_emb"])
            self.item_emb.copy_(state["item_emb"])

    def positions(self):
        return self.user_emb, self.item_emb, self.user_emb, self.item_emb
