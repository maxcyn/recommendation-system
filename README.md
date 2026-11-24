# Steam game recommendation with graph neural networks

A learning project that compares three recommenders on the same data and split — a popularity baseline, matrix factorisation and a graph neural network (LightGCN) — to answer: *does the graph add anything, and where does it work and where does it break?*

Trained and evaluated on the UCSD "Steam Video Game and Bundle Data" (Australian users, ~2017-18): 51,633 users, 9,111 games and 1.56M liked games inferred from playtime.

Full findings, lessons and limits: **[REPORT.md](REPORT.md)**.

## Architecture

```
data/raw/ucsd/*.json.gz            (owned games + playtime, reviews, game metadata)
        |
        v
  preprocess.py      flatten to CSV, no filtering     -> data/processed/
        |
        v
  split.py           "liked" label (per-game median playtime, >=30 min),
                     keep users with 5+ liked games,
                     hide ~10% / ~20% of each user's likes -> val / test
                                                          -> data/processed/split/
        |
        +----------------+----------------------+------------------------+
        v                v                      v                        v
  models/popularity   models/mf.py        models/lightgcn.py      models/mf_tags.py
                      64-dim embeddings   same weights + 2 rounds models/lightgcn_tags.py
                      BPR loss            of message passing     (tag side information)
                        |                 (norm="sym" | "mean")
                        +-----------------+-------+
                                          |
                                          v
                                    run.py   train one model, early stop
                                             on validation Recall@20
                                          |
                                          v
                            evaluate.py + metrics.py
                            rank all 9,111 games, drop known games,
                            Recall@K / NDCG@K, activity bands
                                          |
              +---------------------------+------------------------+
              v                           v                        v
         compare.py                  diagnose.py              decompose.py
     gaps with 95% intervals    what each model recommends   which part of LightGCN
     (resample users + seeds)   by game popularity           causes the popularity lean
```

Both learned models share the same embeddings, dot-product score and BPR pairwise loss, so the only difference between matrix factorisation and LightGCN is the graph propagation step. LightGCN's variant `norm="mean"` replaces the standard symmetric edge weight `1/sqrt(deg_u * deg_i)` with a plain per-receiver average.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124  # GPU build; skip for CPU-only
.venv/Scripts/python -m pip install -e ".[dev]"
```

(On macOS/Linux, use `.venv/bin/python` instead of `.venv/Scripts/python`.)

## Usage

```bash
# build the dataset (requires the UCSD files in data/raw/ucsd/, not committed)
.venv/Scripts/python -m steamrec.preprocess
.venv/Scripts/python -m steamrec.split

# train and evaluate one model (mf ~6 min, lightgcn ~20-40 min on an RTX 4060)
.venv/Scripts/python -m steamrec.run popularity
.venv/Scripts/python -m steamrec.run mf --on both --seed 0 --tag mf_final_seed0
.venv/Scripts/python -m steamrec.run lightgcn --on both --seed 0 --tag lightgcn_final_seed0

# compare models with 95% intervals (expects seeds 0-2 of each)
.venv/Scripts/python -m steamrec.compare

# diagnostics: popularity mix of recommendations, LightGCN decomposition
.venv/Scripts/python -m steamrec.diagnose train mf        # then `train lightgcn`, then `analyse`
.venv/Scripts/python -m steamrec.decompose

# variants
.venv/Scripts/python -m steamrec.diagnose train lightgcn --norm mean --reg 0.0003  # plain-average LightGCN
.venv/Scripts/python -m steamrec.run mf_tags --tag mf_tags          # matrix factorisation + tag embeddings
.venv/Scripts/python -m steamrec.run lightgcn_tags --tag lgcn_tags  # tags as extra graph nodes

# tests
.venv/Scripts/python -m pytest
```

Run trainings one at a time: parallel runs exhaust memory on a 16 GB machine.

## Results at a glance

| | Popularity | Matrix factorisation | LightGCN |
|---|---|---|---|
| Recall@10 | 0.147 | 0.223 | 0.226 |
| NDCG@10 | 0.126 | 0.198 | 0.205 |
| Recall@20 | 0.209 | 0.318 | 0.319 |
| Distinct games recommended (top 10) | 33 | 2,296 | 1,324 |

Headline: matrix factorisation beats popularity by +52% Recall@10 (established); LightGCN's +1.2% over matrix factorisation is **not** established (95% interval includes zero), though it orders the list slightly better (NDCG@10 +3.8%, established). Standard LightGCN leans towards blockbusters; the plain-average variant removes that lean at the same recall in one run. Details in [REPORT.md](REPORT.md).

## Repository layout

```
src/steamrec/
    preprocess.py    raw UCSD JSON -> processed CSV tables
    split.py         label + per-user train/val/test masking
    data.py          loading processed tables
    tags.py          game-tag preparation
    run.py           train + evaluate one model (CLI)
    evaluate.py      all-game ranking, known-game filtering, band summaries
    metrics.py       Recall@K, NDCG@K
    compare.py       multi-seed comparison with bootstrap intervals
    diagnose.py      popularity-bias diagnostics
    decompose.py     LightGCN component decomposition (own / round 1 / round 2)
    models/
        popularity.py     most-liked training games
        mf.py             matrix factorisation + shared BPR trainer
        lightgcn.py       LightGCN (norm="sym" standard, norm="mean" plain average)
        mf_tags.py        matrix factorisation with tag embeddings
        lightgcn_tags.py  LightGCN with tags as graph nodes
tests/               unit tests (13)
data/                raw and processed data (gitignored)
results/             run outputs (gitignored)
REPORT.md            full findings write-up
```

`AGENTS.md` (gitignored) is the working decision log kept during the project: every choice, alternative and assumption with its defence, for agent-assisted continuation.
