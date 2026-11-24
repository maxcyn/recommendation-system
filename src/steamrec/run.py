"""Train one model and score it on the hidden games.

Usage: python -m steamrec.run popularity|mf|lightgcn [--on test|val] [--tag NAME] [--seed 0 --dim 64 --lr 0.001 --reg 0.0001 --layers 3]
Writes results/<name>_<on>.json and results/<name>_<on>_per_user.csv.
Models that need tuning stop training when validation Recall@20 stops improving.
"""
import argparse
import json
import time

import pandas as pd

from steamrec.data import ROOT
from steamrec.evaluate import FastRecall, evaluate_model, load_split
from steamrec.models.popularity import Popularity

ACTIVITY_BINS = [2, 5, 10, 20, 50, 100000]


def make_model(name: str, overrides: dict):
    if name == "popularity":
        return Popularity()
    from steamrec.models.lightgcn import LightGCN
    from steamrec.models.mf import MatrixFactorisation

    if name in ("mf", "mf_tags"):
        overrides.pop("layers", None)
        overrides.pop("norm", None)
    if name == "mf_tags":
        from steamrec.models.mf_tags import MatrixFactorisationTags

        return MatrixFactorisationTags(**overrides)
    if name == "lightgcn_tags":
        from steamrec.models.lightgcn_tags import LightGCNTags

        return LightGCNTags(**overrides)
    return {"mf": MatrixFactorisation, "lightgcn": LightGCN}[name](**overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["popularity", "mf", "lightgcn", "mf_tags", "lightgcn_tags"])
    ap.add_argument("--on", choices=["val", "test", "both"], default="test")
    ap.add_argument("--tag", default=None)
    for name, typ in (("dim", int), ("lr", float), ("reg", float), ("batch_size", int), ("max_epochs", int), ("seed", int), ("layers", int), ("norm", str)):
        ap.add_argument(f"--{name}", type=typ, default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("model", "on", "tag") and v is not None}

    parts, n_users, n_items = load_split()
    # the model learns from training lines only, whichever set it is scored on
    start = time.time()
    model = make_model(args.model, overrides)
    validate = FastRecall("val", k=20, device=model.device) if args.model != "popularity" else None
    model.fit(parts["train"], n_users, n_items, validate=validate)
    name = args.tag or args.model
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for on in (["val", "test"] if args.on == "both" else [args.on]):
        summary, per_user = evaluate_model(model.score, on=on)
        summary.update(model=args.model, settings=model.settings(), best_val_recall_at_20=model.best_val,
                       training_seconds=round(time.time() - start, 1), history=model.history)
        summary["by_known_liked_games"] = _table(per_user, pd.cut(per_user.n_known, ACTIVITY_BINS, right=False))
        summary["by_hidden_games"] = _table(per_user, per_user.n_hidden.clip(upper=6))
        (out / f"{name}_{on}.json").write_text(json.dumps(summary, indent=2))
        per_user.to_csv(out / f"{name}_{on}_per_user.csv", index=False)
        summary.pop("history")
        print(json.dumps(summary, indent=2))


def _table(per_user, groups):
    return {
        str(k): {
            "people": int(len(g)),
            "recall@10": float(g["recall@10"].mean()),
            "recall_capped@10": float(g["recall_capped@10"].mean()),
            "ndcg@10": float(g["ndcg@10"].mean()),
            "mean_hits@10": float(g["hits@10"].mean()),
        }
        for k, g in per_user.groupby(groups, observed=True)
    }


if __name__ == "__main__":
    main()
