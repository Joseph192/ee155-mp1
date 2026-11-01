# MP1 Popularity Report

Every experiment run in this codebase to raise ROC AUC on the EE155 MP1 Spotify High/Low popularity task, and what the data revealed. It covers your original sweeps, a parallel agent session, and rounds 1–18 in `experiments/`. Report date: 1 Nov 2026.

| Headline | Value |
|---|---:|
| Best CV AUC, using test-set features (similarity-weighted album mates) | **0.8551** |
| Best CV AUC, training data only | **0.8289** |
| Best public Kaggle score reported (old model: 0.83415) | **0.84644** |
| Target | 0.90 (not reached) |

## Setup

- **Data:** 3,864 training rows and 863 test rows. Each row is a Spotify track with 10 audio features (danceability, energy, loudness, …), key, mode, time signature, an album release date and a track ID (`uri`). The label is High (35%) or Low.
- **Scoring:** cross-validated AUC. Unless noted, each score is the mean over 3 layouts (split seeds 155, 2026 and 42) of track-grouped, stratified 3-fold CV, so copies of the same track never sit on both sides of a fold. Some numbers from the original work use one layout or four.
- **Single-track score:** test rows rarely repeat (2.5% of test rows versus 11% of train rows). Later rounds therefore also report AUC on single-occurrence validation tracks only, the closest match to the test set.

## The AUC ladder

| Milestone | CV AUC | Uses test-set features |
|---|---:|:---:|
| Original histogram boosted trees (1 layout) | 0.806 | no |
| Original random forest, reproduced on 3 layouts (Kaggle 0.83415) | 0.8256 | no |
| Blend + CatBoost, training data only (`train_only.py`) | 0.8289 | no |
| Era experts, parallel session (4 layouts) | 0.8293 | no |
| + repeat count (`uri_count`) | 0.8462 | yes |
| + extra-trees / random forest rank blend | 0.8503 | yes |
| + 3 model seeds and boosted trees (Kaggle 0.84644) | 0.8514 | yes |
| + 30% CatBoost | 0.8528 | yes |
| + album mates | 0.8541 | yes |
| **+ similarity weighting (best)** | **0.8551** | yes |

The full range from the original model to the best is about 0.05 AUC.

## What the data revealed

### A repeated track means a popular track
203 tracks appear more than once in train, as exact copies: every feature is identical and only the label can differ. The copies sit about 1,150 rows apart, so this isn't a merge artifact. The dataset looks like one row per track per playlist, and a popular song sits in more playlists.

| Times the track appears (train + test) | Training rows | High share |
|---|---:|---:|
| Once | 3,439 | 29.7% |
| Twice | 370 | 76.8% |
| 3 times | 54 | 96.3% |

A `uri_count` feature lifted CV from 0.826 to 0.846, the largest single gain in the project.

### The split keeps all copies of a track together, which limits that signal
No test track ID appears in train. Counted against training rows alone, every test row gets a count of 1 and the feature does nothing. It only reaches the 11 tracks repeated within test when test rows are counted too.

On single-occurrence validation tracks the repeat feature barely moves AUC (0.8218 → 0.8224, from the parallel session's `occurrence_audit.json`). The strict train-only rebuild (`train_only.py`) scores 0.8289, close to the original forest.

> Most of the gap between 0.829 and 0.855 comes from counting test rows. Treat the two numbers as answers to different questions: training data only, or every feature available.

### Album mates carry popularity, especially when they sound alike
A song released on the same date as a High song is High 47% of the time, against 25% when the same-date song is Low.

- A cross-fitted "same-date High rate" feature added **+0.0013**.
- Weighting each same-date song by audio similarity, `exp(−distance)`, picks out likely same-artist tracks and added another **+0.001**. That gives the best model, at 0.8551 (0.8314 on single tracks).
- Training rows get the feature from an inner grouped split, so no row ever sees its own label.

### What sounds popular depends on the era
Every song released before 1990 is High. After that, energy and valence point to High for 2023+ releases but slightly the other way for 2015–19, where speechiness matters instead (the rap era). The model is weakest exactly there.

| Release era | High share | Out-of-fold AUC |
|---|---:|---:|
| Before 2000 | 69% | 0.885 |
| 2000–09 | 52% | 0.810 |
| 2010–14 | 46% | 0.824 |
| **2015–19** | 40% | **0.734** |
| 2020–22 | 25% | 0.823 |
| 2023 and later | 26% | 0.853 |

*(Round-3 blend, split seed 155.)* The parallel session turned this into separate pre/post-2023 forests blended 50/50 with a shared one. It scored **0.8293** on four layouts without any test-set features, which makes it the strongest training-only idea found.

### Diversity runs out fast on 3.8k tabular rows
- Rank-averaging extra-trees with a random forest added +0.004.
- Seed averaging, boosted trees and CatBoost added about +0.001 each.
- After that, nothing helped:
  - a PyTorch MLP with periodic numeric embeddings (0.831, vs 0.784 for the original MLP)
  - NN–boosting hybrids
  - stacking and greedy ensemble selection
  - logistic regression and soft voting

  Their predictions were 0.92–0.94 rank-correlated with the trees, so they added a weaker copy of the same ordering.
- Rank averaging beat probability averaging. Balanced class weights push the forests' and CatBoost's probabilities up (mean 0.38–0.40), but not the boosted model's (0.35).

### Conflicting labels exist, but the ceiling they imply is uncertain
32 of the 203 repeated tracks carry both labels with identical features (66 rows). Removing them from training made no difference (0.8549 vs 0.8551), and neither did also keeping one row per track (0.8545).

> **Correction:** earlier messages said a perfect model would top out near 0.90. That figure assumes the 16% disagreement on repeated tracks applies to every song. The parallel session showed that the 32 conflicts alone cap AUC only near 0.9998 (`label_conflict_audit.json`). So 0.90 is an extrapolation, not a proven bound. The plateau around 0.855 across about 100 experiments is the stronger evidence that these features are close to exhausted.

### Ideas that consistently failed
- **Strong single-feature AUCs didn't survive inside models.** Vocal presence (0.690 alone), era interaction copies and in-year percentiles were all neutral or negative. The trees already find these combinations.
- **Unsupervised structure didn't help:** k-means, Gaussian mixtures, PCA, kNN density, anomaly scores and kNN label features never beat the baseline.
- **Transductive tricks beyond counting hurt:** pseudo-labelling, label spreading, copying a near-identical train song's label, and smoothing scores across same-date test songs.
- **Training tweaks didn't help:** label smoothing, pairwise ranking loss, and removing or reweighting duplicates.

## The best model

- **Features (26):**
  - The original per-row features: the 10 audio columns, release year, day, fractional year and weekday, time signature and mode.
  - Counts over train + test: `uri_count`, `near_dup_count` and `date_count`.
  - Six audio-minus-year-median columns.
  - The album-mate pair `mate_sim` and `mate_n`.
- **Models:** 3 extra-trees and 3 random forests (500 trees, at least 3 samples per leaf, entropy, balanced class weights) and one histogram boosted-tree model, rank-averaged. That score is mixed 70/30 by rank with CatBoost (learning rate 0.01, 2,000 rounds, depth 6).
- **Built by:** `experiments/make_submission_mates.py --sim`, which writes `submission_mates_sim.csv`.

## Every experiment

### Original work and the parallel session
| Experiment | CV AUC | Result |
|---|---:|---|
| Histogram boosted trees, one-hot features | 0.806 | baseline |
| scikit-learn MLP (epoch picked on validation, so optimistic) | 0.784 | weaker |
| RBF SVM · extra-trees · classical gradient boosting | 0.772 · 0.805 · 0.798 | no gain |
| Fractional year, release day, weekday; drop key and month | 0.816 | kept |
| Reweight or deduplicate repeated tracks | 0.809 | no gain |
| Random forest with date features (Kaggle 0.83415) | 0.824 | kept |
| Year as both number and category, leaf 1, half features (4 layouts) | 0.8278 | small gain |
| Cohort target rates; similarity / kNN; feature ablations | ≤ 0.824 | no gain |
| Release-era experts, 2023 cutoff, 50/50 blend (4 layouts) | 0.8293 | best training-only |

### `experiments/` rounds 1–18
| Round | Idea | CV AUC | Single tracks | Result |
|---|---|---:|---:|---|
| 1 | Random forest reproduced on 3 layouts | 0.8256 | 0.8218 | reference |
| 1 | + `uri_count` (train + test) | 0.8462 | 0.8224 | kept |
| 1 | + near-duplicate count · + release-date count | 0.8462 · 0.8278 | – | kept, small |
| 1 | + audio relative to year median | 0.8245 | – | neutral |
| 2 | Extra-trees + random forest rank blend | 0.8503 | 0.8258 | kept |
| 2 | Spline logistic regression · kNN | 0.8206 · 0.7985 | – | scrapped |
| 3 | 3 model seeds + boosted trees (Kaggle 0.84644) | 0.8514 | 0.8273 | kept |
| 3 | Test pseudo-labelling | 0.8493 | 0.8244 | scrapped |
| 4 | k-means / PCA / kNN density / anomaly score | 0.848–0.849 | 0.824 | scrapped |
| 4 | Gaussian mixtures · label spreading | 0.8446 · 0.7598 | – | scrapped |
| 5 | kNN neighbour label rate (cross-fitted) | 0.8480 | 0.8231 | scrapped |
| 5 | Blend + 30% CatBoost | 0.8528 | 0.8288 | kept |
| 6 | PyTorch MLP with periodic embeddings, 20 epochs | 0.8310 | 0.8029 | no blend gain |
| 6 | Label smoothing | 0.8205 | – | scrapped |
| 7 | CatBoost tuned (learning rate 0.01, 2,000 rounds) · pairwise loss | 0.8466 · 0.8419 | 0.8218 | tuned kept |
| 8 | NN on boosted-tree leaves · NN correcting boosted logits | 0.8164 · 0.8410 | – | scrapped |
| 9 | Stacking · greedy ensemble selection (unseen layouts) | 0.8514 · 0.8527 | – | no gain |
| 10 | Vocal presence, era interactions, in-year percentile | 0.847–0.851 | – | scrapped |
| – | All counts from training folds only (`train_only.py`) | 0.8289 | – | honest score |
| – | Same, conflicting tracks dropped | 0.8282 | – | no gain |
| 11–12 | Album mates (same-date High rate) | 0.8541 | 0.8300 | kept |
| 11 | Copy label from a near-identical train song | 0.851–0.853 | – | scrapped |
| 13 | Smooth scores across same-date test songs | 0.831–0.853 | – | scrapped |
| 13–14 | **Similarity-weighted album mates** | **0.8551** | **0.8314** | **best** |
| 15–16 | Drop conflicting tracks · also keep one row per track | 0.8549 · 0.8545 | 0.8308 · 0.8315 | tie |
| 17 | + logistic regression at 5–30% | 0.8550–0.8517 | – | worse |
| 18 | Soft voting instead of rank averaging | 0.8537–0.8549 | – | no gain |

The per-idea detail and reasoning are in `experiments/IDEAS.md`.

## Submissions in the main folder

| File | Model | Test features | CV AUC | Kaggle |
|---|---|:---:|---:|---:|
| `submission.csv` | Original random forest (rewritten by the parallel session at 23:27) | no | 0.824 | 0.83415 |
| `submission_blend.csv` | Round-3 tree blend | yes | 0.8514 | 0.84644 |
| `submission_blend_cat.csv` | + 30% CatBoost | yes | 0.8528 | – |
| `submission_train_only.csv` | Same blend, training data only | no | 0.8289 | – |
| `submission_train_only_noconflict.csv` | Same, conflicts dropped | no | 0.8282 | – |
| `submission_mates.csv` | + album mates | yes | 0.8541 | – |
| **`submission_mates_sim.csv`** | **+ similarity-weighted album mates** | yes | **0.8551** | – |
| `submission_mates_sim_noconflict.csv` | Same, conflicts dropped | yes | 0.8549 | – |
| `submission_mates_sim_noconflict_dedup.csv` | Same, plus one row per track | yes | 0.8545 | – |

## How far to trust these numbers

- **The CV scores are somewhat optimistic.** The same three fold layouts were reused across roughly 100 experiments. The Kaggle leaderboard is the only score nobody tuned against, and its public part is small enough to move about ±0.01 by chance.
- **The leaderboard can't rank most submissions.** They differ by 0.001–0.002 in CV and are over 0.98 rank-correlated, so a single upload can't tell them apart.
- **Two promising combinations are untried:** era experts with album mates, and CatBoost with native categorical year and date. Both can be done with training data only.
- **Reaching 0.90 most likely needs information the dataset lacks**, such as artist popularity, follower counts or playlist names. Check the course rules before adding outside data.

## Reproduce

```sh
cd experiments
.venv/bin/python make_submission_mates.py --sim   # best submission (needs CatBoost in .venv)
.venv/bin/python train_only.py                    # training-data-only version
python3 round1.py                                 # any round: python3 roundN.py (CatBoost/torch rounds need .venv)
```

**Sources:** `EXPERIMENTS.md`, `experiments/IDEAS.md`, `experiments/RESULTS.md`, the parallel session's `*_results.jsonl` and audit files, and the saved out-of-fold predictions in `experiments/oof/`.
