"""Matrix factorisation where a game's position also depends on its tags.

Every person and every game gets its own position, as in matrix factorisation, and every tag
gets a position too. A game's final position is its own row plus the average of its tags' rows
(a game without tags just keeps its own row). Two games that share tags therefore start out
close together, even if few people have played one of them.

Scoring and training are the same as in matrix factorisation, so the tag part is the only
difference between the two models.
"""
import torch

from steamrec.models.mf import MatrixFactorisation
from steamrec.tags import load_item_tags, tag_average_matrix


class MatrixFactorisationTags(MatrixFactorisation):
    def __init__(self, item_tags=None, **kwargs):
        """item_tags: optional (item_idx, tag_idx, n_tags); read from the metadata file when not given."""
        super().__init__(**kwargs)
        self.item_tags = item_tags

    def build(self, train, n_users, n_items):
        super().build(train, n_users, n_items)
        item_idx, tag_idx, n_tags = self.item_tags if self.item_tags is not None else load_item_tags()
        self.tag_emb = torch.nn.Parameter(torch.randn(n_tags, self.dim, device=self.device) * 0.1)
        self.tag_avg = tag_average_matrix(item_idx, tag_idx, n_items, n_tags, self.device)

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
        tag_part = torch.sparse.mm(self.tag_avg, self.tag_emb)
        # the penalty covers both the game's own row and its tag part
        return self.user_emb, self.item_emb + tag_part, self.user_emb, torch.cat([self.item_emb, tag_part], dim=1)
