# Idea list

Status: `todo` / `done: <result>` / `scrap: <why>`. Baselines on the 3 grouped layouts:
round-3 blend 0.8514 (singles 0.827), + 30% CatBoost 0.8528 (singles 0.829). Ceiling estimate ~0.90 (label noise).

## Round 6: neural nets and ensembling with boosting
1. done 0.819 (singles 0.784), weaker than PLR: sklearn MLP on the new feature set (quantile-scaled, uri_count etc.), epochs fixed by an inner split, 5-seed bag.
2. done 0.831 at 20 epochs (40: 0.826, 80: 0.823; drop 0.5 + wd 1e-2: 0.829), tuning more: PyTorch MLP with periodic (PLR) numeric embeddings, dropout, AdamW, seed ensemble.
3. scrap 0.820/0.820 at 0.1/0.2, hurts: label smoothing in the PyTorch MLP (labels look ~8.6% noisy).
4. done: still nothing after tuning; greedy selection over 12 models scores 0.8527 on the layouts it didn't pick weights on, vs 0.8524 current best. NN at 0.831 adds nothing (blend 0.8514 to 0.8514 at 20% weight; rank corr 0.92 with trees). Retry once NN improves: rank-average the best NN with CatBoost + round-3 blend; search a few weights.
5. scrap 0.8514, worse than rank averaging: stacked logistic meta-learner on OOF predictions (outer grouped CV over the meta level).
6. scrap: conflicting repeats already appear as identical-feature rows with both labels, which is a soft label for trees: soft labels for repeated tracks (mean label per uri) with sample weights in the trees.
7. done: lr 0.01 x 2000 best at 0.8466 (was 0.8454); depth 4/8, l2 20, rsm 0.5, unweighted all <=0.846; PairLogit 0.842: CatBoost tuning (depth, lr, iterations) and a pairwise ranking loss (AUC-aligned).
8. scrap 0.816: NN fed with boosted-tree leaf embeddings (GBDT leaves one-hot -> embedding + PLR numerics).
9. done 0.841 = HGB alone, no gain: boost-then-NN residual: NN fits the residual logit of HGB (init score), so it only learns what trees miss.
10. done: 10 epochs 0.823, 15 epochs 0.828, lr 3e-4 0.825; 20 epochs (0.831) stays best: NN tuning: fewer epochs (10-15), lower lr, smaller embedding, 10-seed bag.

## Round 10: features from the brainstorm (single-feature AUC checks)
11. scrap 0.8479 vs 0.8503 (ET+RF 1-seed):  vocal presence `(1-instrumentalness)-speechiness` (single-feature AUC 0.690, best found) + `instrumentalness==0` flag.
12. scrap 0.8494:  era interactions: energy, valence, speechiness, loudness-adjusted energy x era indicator (pre-2015, 2015-19, 2020-22, 2023+). Direction flips in 2015-19.
13. scrap 0.8498:  danceability percentile within release year (0.545 -> 0.581).
14. done 0.8513 (+0.001, noise level):  calm-acoustic flag (acousticness>0.6 & energy<0.4), months before snapshot (2024-11-22) for 2023+.
15. scrap: all features together 0.8472 (trees), 0.8286 (NN, was 0.831), 0.8468 (CatBoost), 0.8488 (full blend, was 0.8514):  re-test NN + CatBoost + trees ensemble on the new features (interactions should help the NN most).

## Status after round 10
List empty. Best stays the round-3 blend + 30% CatBoost (0.8528 CV, `../submission_blend_cat.csv`).
About 45 experiments since round 3 all land within +/-0.002 of it. The label-noise ceiling is ~0.90.
Single-feature AUC gains didn't carry over: trees already find these combinations, and extra columns dilute their feature sampling.

## Train-only rebuild (no test-set data, 2026-11-01)
`train_only.py`: uri_count, date_count, near_dup_count and the year medians use training rows as the only reference.
Honest CV: blend 0.8278, CatBoost 0.8214, blend + 30% CatBoost **0.8289**. Writes `../submission_train_only.csv`.
The earlier 0.85x scores counted repeats over train+test (and inside CV, over the validation fold), so they don't carry over.
Drop conflicting tracks from training only (`--drop-conflicts`, 66 rows, validation untouched): blend 0.8272, CatBoost 0.8201,
blend + 30% CatBoost 0.8282 vs 0.8289. Slightly lower on all 3 layouts, so scrapped. `../submission_train_only_noconflict.csv`.

## Rounds 11-14: test features allowed again (2026-11-01, user reversed the ban)
`build()` counts over train+test again (reproduces round 1 exactly: RF 0.8463). round11/12 restored.
- Album mates (cross-fitted same-date High rate from train labels), full blend + 30% CatBoost: 0.8541 (singles 0.8300). `../submission_mates.csv`
- Re-release override toward a near-identical train neighbour's label: scrap, always lower (0.8514-0.8526).
- Transductive smoothing toward same-date mates' predicted scores in the unlabelled pool (validation + test): scrap, lower at every alpha.
- **Similarity-weighted mates** (same-date mates weighted by exp(-audio distance), tau 1) + mate count, full blend + 30% CatBoost:
  **0.8551 (singles 0.8314)**, new best. `../submission_mates_sim.csv` via `make_submission_mates.py --sim`.
- Round 15, best model with conflicting tracks dropped from training folds: 0.8549 (singles 0.8308) vs 0.8551 (0.8314) kept.
  A tie within noise (2 of 3 layouts lower), so keep the conflicts.
- Round 16, conflicts dropped + one row per repeated track (training folds only): 0.8545 (singles 0.8315), vs 0.8551 kept and 0.8549 dropped.
  Lower on all 3 layouts than kept, though within noise. `../submission_mates_sim_noconflict_dedup.csv`.
- Round 17, logistic regression (with sim mates) added to the best ensemble: alone 0.829-0.830; best + 5-30% LR gives 0.8550 down to 0.8517,
  monotonically worse with weight on every layout (rank corr 0.92-0.94 with the ensemble). Scrapped.
- Round 18, soft voting (probability averaging) of the same members: 0.8537-0.8549, vs 0.8551 for rank averaging. Adding it to the rank
  ensemble at 20-50%: 0.8550/0.8549. Scrapped. Members' mean probabilities differ (HGB 0.35 vs forests/CatBoost 0.38-0.40 from class weights),
  which probability averaging is sensitive to.
