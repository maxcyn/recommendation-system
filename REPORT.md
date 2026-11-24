# Report: does a graph-based recommender beat simpler methods for Steam games?

## Conclusion

On this data, a graph model (LightGCN) recommends about as well as matrix factorisation, and both are far better than recommending popular games. The graph did not add accuracy. What it changed was which games get recommended, and that came down to one design detail: how strongly a connection counts. The standard formula pushes LightGCN towards blockbusters; replacing it with a plain average removes that push and, in a single run, gave the widest range of recommendations at the same accuracy.

## Question and data

- **Question:** does a graph-based deep learning recommender suggest games a Steam user will like better than simpler methods, and where does it work and where does it break?
- **Data:** UCSD "Steam Video Game and Bundle Data", Australian users, collected about 2017-18. Each record is a user's complete library: every owned game with total minutes played. A separate file holds the same users' reviews (thumbs up/down). Library records carry no dates, so no test by time is possible.
- **"Liked":** a person liked a game if they played it at least as much as the typical player of that game (the median playtime among people who played it) and for at least 30 minutes. Checked against reviews: 91.7% of games the rule calls liked got a thumbs up, but the rule excludes only 39.1% of thumbs-downs.
- **Who:** 51,633 people with 5 or more liked games, 9,111 games, 1,556,394 liked pairs. No game was dropped for being rare.

## Method

- **Test:** for each person, about 20% of their liked games were hidden at random for the test and 10% for tuning (validation); models learned from the rest. A person with exactly 5 liked games has 3 training / 1 validation / 1 test. One split (seed 42).
- **Score:** each model ranks all 9,111 games for each person, their known games (training plus validation) are removed, and the top K is compared with the hidden games. Recall is the share of hidden games in the top 10 or 20; NDCG also rewards placing them higher. Both are averaged over people.
- **Models:** popularity (the globally most-liked training games), matrix factorisation, and LightGCN. The last two are deliberately identical — 64-dimensional embeddings, dot-product scores, the same pairwise ranking loss (BPR) and settings — except that LightGCN builds each position as the average of its own embedding and two rounds of graph propagation over the training interactions. The graph's mixing rounds are therefore the only difference being tested.
- **Judging gaps:** three training runs per model with different random starts; a 95% interval from 2,000 resamples that redraw both people and runs. A gap counts as established only if the interval excludes zero.

## Results

| | Popularity | Matrix factorisation | LightGCN |
|---|---|---|---|
| Recall@10 | 0.147 | 0.223 | 0.226 |
| NDCG@10 | 0.126 | 0.198 | 0.205 |
| Recall@20 | 0.209 | 0.318 | 0.319 |
| Different games recommended | 33 | 2,296 | 1,324 |

| Comparison | Gap in Recall@10 | 95% interval | Established? |
|---|---|---|---|
| Matrix factorisation over popularity | +0.077 (+52%) | +0.073 to +0.080 | yes |
| LightGCN over matrix factorisation | +0.003 (+1.2%) | -0.002 to +0.007 | no |

The same model's Recall@10 moves by about 0.003 between random starts — as much as the gap between the two trained models.

## What works

- **Personalisation.** Both trained models beat popularity by about half, in every activity band (established).
- **Ordering.** LightGCN places hits slightly higher: NDCG@10 +0.008, interval +0.003 to +0.011 (established, small).
- **Robustness to tuning.** Across five regularisation settings, LightGCN's validation score moved by 0.011 where matrix factorisation's moved by 0.039 (one run per setting).
- **A plain average in the graph.** Replacing the standard `1/sqrt(degree_u * degree_i)` edge weight with a per-receiver average recommended 2,723 different games and caught 0.149 lesser-known games per person, against 2,296 and 0.129 for matrix factorisation, at the same recall (one run). This matches published work on popularity bias in graph recommenders (Zhao et al., SIGIR 2022, arXiv 2204.12326), so it is a known remedy, reproduced here.

## What breaks

- **No accuracy gain from the graph.** A well-tuned matrix factorisation matches it on recall (established).
- **Popularity lean.** Standard LightGCN gave 71% of its top-10 slots to the 50 most-liked games and caught half as many lesser-known games as matrix factorisation. The cause: a game's propagated position is partly the sum of its fans' embeddings, and the standard edge weight only half cancels the number of fans, so popular games end up with larger vectors (median length 7.8 for top-50 games against 1.1 for games ranked below 1,000, versus 2.1–2.9 flat for matrix factorisation). One-run diagnostics.
- **The lightest users.** People with exactly 5 liked games did worse with LightGCN (0.278 against 0.299; interval excludes zero, though five bands were tested). Only partly explained.
- **Rare games.** No model ever put a game with fewer than 5 training likes in a top 10, even when given tags (one run).
- **Extra information did not rescue it.** Adding game tags to both models (tag embeddings for matrix factorisation, tag nodes in the graph for LightGCN) changed nothing meaningful (one run each).
- **Cost.** LightGCN trains in 20 to 75 minutes against about 6 for matrix factorisation, and its best regularisation setting shifted tenfold when the edge formula changed.

## Predictions and how they turned out

| Prediction | Whose | Outcome |
|---|---|---|
| Popularity catches about 1 of a person's 4 hidden games | User | Close: 0.61 |
| Matrix factorisation beats popularity "by a bit" | User | Understated: +52% |
| The graph helps most for people with more games | User | Partly: a small established gain for 50+ games, but it came from blockbusters |
| LightGCN over-penalises popular games | User | True for the lightest users only; the opposite elsewhere |
| A plain average finds more lesser-known games; effect on recall unclear | User | Borne out in one run; recall unchanged |
| LightGCN only ties because the graph lacks tags and similar information | User | Not supported by the tags test |
| The first matrix factorisation tuning was "near what the model can do" | Assistant | Wrong: retuning raised Recall@10 from 0.194 to 0.223 |
| A plain average would fix the lightest users' loss | Assistant | Not at top 10; partly at top 20 |
| Tags would help rarely-liked games | Assistant | No effect |

## Lessons

- **A baseline is only as fair as its tuning.** With light regularisation LightGCN led matrix factorisation by 0.031 on the validation score; with both models searched over the same range the lead was 0.001.
- **One run is not a result.** Run-to-run variation was as large as the gap being judged.
- **The same overall score can hide different behaviour.** Three models within 0.003 of each other in recall recommended very different mixes of games.
- **A small design detail can matter more than the model family.** The edge-weight formula changed what LightGCN recommends more than the graph itself changed accuracy.

## Limits

- Australian users from about 2017-18; nothing here is claimed about Steam users today.
- "Liked" is inferred from playtime and validated only on people who write reviews — a self-selected minority.
- One random hiding of games; three runs per model for the main comparison.
- The diagnostics, the plain-average result and the tags test are single runs without intervals.
- Hidden games are a random sample of what a person liked, so the test is easier than predicting future behaviour, and it cannot tell whether a recommended blockbuster was already known to the person.

## If the project continues

1. Three runs and intervals for the plain-average LightGCN, to confirm its edge on lesser-known games.
2. A score for lesser-known hidden games only, reported next to recall.
3. Repeating the random hiding with other seeds.
4. A graph model built for people and games with few likes, since that is where LightGCN was weakest.
