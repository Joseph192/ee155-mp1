# MP1 Popularity Report

Every experiment run in this codebase to raise ROC AUC on the EE155 MP1 Spotify High/Low popularity task: what was tried, why it was tried, and why each idea was kept or dropped. It covers your original sweeps, a parallel agent session, and rounds 1–18 in `experiments/`. Report date: 1 Nov 2026.

| Headline | Value |
|---|---:|
| Best CV AUC, using test-set features (similarity-weighted album mates) | **0.8551** |
| Best CV AUC, training data only | **0.8289** |
| Best public Kaggle score reported (old model: 0.83415) | **0.84644** |
| Target | 0.90 (not reached) |

## Setup

- **Data:** 3,864 training rows and 863 test rows. Each row is a Spotify track with 10 audio features (danceability, energy, loudness, …), key, mode, time signature, an album release date and a track ID (`uri`). The label is High (35%) or Low.
- **Scoring:** cross-validated AUC. Unless noted, each score is the mean over 3 layouts (split seeds 155, 2026 and 42) of track-grouped, stratified 3-fold CV. Some numbers from the original work use one layout or four.
- **Single-track score:** later rounds also report AUC on single-occurrence validation tracks only (see decision 3).

## Decisions and reasoning

These choices shaped every experiment, so they come before the results.

**1. Folds are grouped by track ID.** The training set contains 221 exact duplicate rows (203 tracks appearing 2–3 times). With ordinary folds, a copy in the training fold would "predict" its twin in the validation fold and inflate AUC. Grouping by `uri` keeps all copies on one side, which is how the test set was built: no test track appears in train. Your original scripts already did this. My harness asserts it on every fold, because getting it wrong silently would invalidate every number after it.

**2. Each score averages 3 fold layouts, not 1.** With about 1,300 validation rows per fold, AUC moves by roughly ±0.003 between split seeds. Many ideas in this project change AUC by less than that. Averaging three layouts makes a 0.002 difference more trustworthy, and an idea counts as a real gain only when it wins on most layouts, not just on average. Three layouts instead of four or more kept each experiment to about a minute, which mattered over about 100 experiments.

**3. Single-track AUC is reported alongside the overall score.** After finding that repeated tracks are mostly High, I noticed the test set has far fewer repeats (2.5% of rows vs 11% in train). A feature that only helps on repeated tracks would look great in CV and do little on Kaggle. Scoring single-occurrence tracks separately measures what the test set mostly contains, and it caught exactly that: `uri_count` lifted overall CV by 0.02 but single-track AUC by only 0.0006.

**4. The data was explored before modelling.** Your models had already plateaued around 0.824 through tuning, so more tuning was unlikely to reach 0.9. The higher-leverage move was to find information the models weren't using. I checked for leaks first (row order, test `ID`, track-ID characters, overlap between train and test). That's cheap and would have been decisive, but none showed anything usable. Duplicates were next, and they turned up the repeat-count signal.

**5. Anything built from labels is cross-fitted.** Features such as "High rate of nearby songs" or "High rate of same-date songs" use training labels. Computed naively, a row can see its own label. Even leaving the row out still leaks through pairs: if A and B are neighbours, A's feature contains B's label and B's contains A's, so trees learn to decode it. Round 4 showed this: the leave-own-track-out kNN feature scored 0.840 and hurt. Every label-based feature since is cross-fitted, meaning training rows get values from an inner grouped split that never saw them.

**6. Models are combined by rank, not probability.** AUC depends only on ordering, so ranks keep exactly what matters. The members' probabilities also sit on different scales: balanced class weights push the forests and CatBoost to a mean of 0.38–0.40, while the boosted model averages 0.35. A probability average would quietly let the most spread-out model dominate. Round 18 confirmed it: soft voting scored 0.8537–0.8549 against 0.8551 for rank averaging.

**7. Weights stay simple, and blend weights come from flat regions.** CatBoost's 30% weight came from a grid of 20%, 30% and 40%. All three were within 0.0001, so I took the middle of a flat region instead of chasing the peak. Stacking and greedy ensemble selection fit more parameters to the same folds. Greedy selection scored 0.8527 on layouts it didn't choose weights on, against 0.8524 for the simple blend, so it wasn't adopted: the extra flexibility would mostly fit noise.

**8. A loop stops when gains drop to noise level.** Each loop ran until new ideas produced at most +0.001 to +0.002, which is inside the run-to-run spread from decision 2. Continuing past that point mostly fits the reused folds. After about 100 experiments on the same three layouts, the best CV score is already somewhat optimistic.

**9. The project was organised for safety and reproducibility.**
- All new code lives in `experiments/` and imports your helpers (`make_features`, `build_forest_preprocessor`) unchanged, as you asked.
- CatBoost and PyTorch went into an isolated `experiments/.venv` rather than your system Python, so nothing outside the folder changed.
- New submissions got new file names instead of overwriting old ones, so the file with a known Kaggle score (`submission_blend.csv`, 0.84644) was never lost.

**10. Leakage policy followed your instructions as they changed.** You first allowed test-set features, then banned them, then allowed them again.
- When banned, I rebuilt every count to use training rows only (`train_only.py`), including inside CV, so validation tracks are compared only against their training fold. That produced the honest 0.8289.
- When re-allowed, I restored the train+test counting and verified it reproduced round 1 exactly (0.8463) before building on it.
- `train_only.py` was kept so either version can be regenerated.

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

## What the data revealed

### A repeated track means a popular track
**Why I looked:** duplicates show up in any data audit, and identical rows with a label attached are suspicious. They're either noise to clean up or a signal about how the dataset was built.

203 tracks appear more than once in train, as exact copies: every feature is identical and only the label can differ. The copies sit about 1,150 rows apart, so this isn't a merge artifact. The most plausible explanation is one row per track per playlist: a popular song sits in more playlists.

| Times the track appears (train + test) | Training rows | High share |
|---|---:|---:|
| Once | 3,439 | 29.7% |
| Twice | 370 | 76.8% |
| 3 times | 54 | 96.3% |

**Why I kept `uri_count`:** it lifted CV from 0.826 to 0.846, the largest single gain in the project. Even though decision 3 shows most of that comes from training repeats, it still flags the 11 test tracks that repeat within test.

### The split keeps all copies of a track together, which limits that signal
**Why this matters:** it explains how much of the CV gain will actually show up on Kaggle.

No test track ID appears in train. Counted against training rows alone, every test row gets a count of 1 and the feature does nothing; it reaches only the 11 within-test repeats when test rows are counted too. On single-occurrence validation tracks it barely moves AUC (0.8218 → 0.8224, from the parallel session's `occurrence_audit.json`). The strict train-only rebuild scores 0.8289.

> Most of the gap between 0.829 and 0.855 comes from counting test rows. That's why the report keeps two "best" numbers: they answer different questions.

### Album mates carry popularity, especially when they sound alike
**Why I looked:** tracks released on the same date are often from the same album, and the same artist. Popularity is largely an artist-level property, which the audio features don't capture.

A song released on the same date as a High song is High 47% of the time, against 25% when the same-date song is Low.

- **Plain album mates (+0.0013).** A cross-fitted "same-date High rate", pulled toward the overall High rate when a date has few songs. The pull is needed because a date with one mate would otherwise output a hard 0 or 1.
- **Similarity weighting (+0.001 more).** Not every same-date song is from the same album; a big release Friday mixes many artists. Weighting each mate by audio similarity, `exp(−distance)`, favours likely same-artist tracks, because an artist's songs tend to share production style. The distance scale was compared at 0.5, 1 and 2; 1 was best.
- Result: **0.8551** (0.8314 on single tracks), the current best. The gain held on all three layouts and on single tracks, which is why I trust it more than other gains of similar size.

### What sounds popular depends on the era
**Why I looked:** the per-era error analysis was done to find where the model fails, since improving the weakest segment is usually more productive than polishing the average.

Every song released before 1990 is High. After that, energy and valence point to High for 2023+ releases but slightly the other way for 2015–19, where speechiness matters instead (the rap era).

| Release era | High share | Out-of-fold AUC |
|---|---:|---:|
| Before 2000 | 69% | 0.885 |
| 2000–09 | 52% | 0.810 |
| 2010–14 | 46% | 0.824 |
| **2015–19** | 40% | **0.734** |
| 2020–22 | 25% | 0.823 |
| 2023 and later | 26% | 0.853 |

*(Round-3 blend, split seed 155.)*

- **My version (dropped):** I added per-era copies of the flipping features, expecting it to help the neural net most. It didn't help either the trees or the net. Trees already split on year first, so the explicit copies only diluted their feature sampling.
- **The parallel session's version:** separate pre/post-2023 forests blended 50/50 with a shared one. It scored **0.8293** with training data only, the strongest training-only idea found. It works where my version didn't because each expert gets a whole model dedicated to its era, rather than extra columns competing for splits.

### Diversity runs out fast on 3.8k tabular rows
**Why I kept trying model families:** an ensemble gains most from members that make *different* mistakes, so each new family was a bet that it captured something the trees missed.

- **Extra-trees with a random forest added +0.004.** Extra-trees pick random split thresholds, so their errors differ from the random forest's.
- **Seed averaging, boosted trees and CatBoost added about +0.001 each.** Boosting builds trees sequentially on residuals, a different bias from bagging.
- **The PyTorch MLP (0.831) didn't help the blend.** I used periodic numeric embeddings because plain MLPs are known to underperform trees on tabular data, and these embeddings are one of the documented fixes. Epochs were fixed instead of picked on validation, so its score is honest, unlike your original MLP's 0.784. It still added nothing: its rankings were 0.92 correlated with the trees', so it was a weaker copy of the same ordering.
- **Hybrids, stacking, logistic regression and soft voting were all tried for the same reason** (a different inductive bias or a smarter combiner), and all failed for the same reason: no new information to add.

### Conflicting labels exist, but the ceiling they imply is uncertain
**Why I tested removal:** identical inputs with opposite labels look like pure noise, and cleaning noise often helps.

32 of the 203 repeated tracks carry both labels (66 rows). Removing them from training made no difference (0.8549 vs 0.8551), and neither did also keeping one row per track (0.8545). Removal doesn't help because keeping both copies already trains the model toward about 50% for those songs, which is the honest answer. Removing them just discards 32 mostly-popular songs. They were always kept in validation, because test presumably contains the same kind of noise, and removing them there would make CV optimistic.

> **Correction:** earlier messages said a perfect model would top out near 0.90. That figure assumes the 16% disagreement on repeated tracks applies to every song. The parallel session showed that the 32 conflicts alone cap AUC only near 0.9998 (`label_conflict_audit.json`). 0.90 is an extrapolation, not a proven bound. The stronger evidence that these features are close to exhausted is the plateau around 0.855 across about 100 experiments.

### Ideas that consistently failed, and why
- **Brainstorm features with strong single-feature AUCs** (vocal presence at 0.690 alone, era copies, in-year percentiles). A feature that's strong on its own is often a combination of columns the trees already split on, so it adds no new information and dilutes their feature sampling.
- **Unsupervised structure** (k-means, Gaussian mixtures, PCA, density, anomaly scores). Tried as genre proxies; the raw audio columns already carry that information, and trees find the clusters themselves.
- **Transductive tricks beyond counting** (pseudo-labelling, label spreading, copying a near-identical train song's label, smoothing across same-date test songs). Each one pushes predictions toward labels the model already roughly knew, so it adds confirmation bias rather than information.
- **Label smoothing and pairwise ranking loss.** Tried because of the label noise and because AUC is a pairwise metric. Trees and the net already rank well under ordinary log loss, so changing the objective only removed signal.

## The best model

- **Features (26):**
  - The original per-row features: the 10 audio columns, release year, day, fractional year and weekday, time signature and mode.
  - Counts over train + test: `uri_count`, `near_dup_count` and `date_count`.
  - Six audio-minus-year-median columns. They were neutral on their own but left in because removing them didn't help either.
  - The album-mate pair `mate_sim` and `mate_n`. `mate_n` tells the model how much to trust `mate_sim`.
- **Models:** 3 extra-trees and 3 random forests (500 trees, at least 3 samples per leaf, entropy, balanced class weights) and one histogram boosted-tree model, rank-averaged, then mixed 70/30 by rank with CatBoost (learning rate 0.01, 2,000 rounds, depth 6). CatBoost used a lower learning rate with more rounds because that was its best tuned setting (round 7); slower learning generally generalises better on small data.
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
| Round | Idea | Why it was tried | CV AUC | Single tracks | Outcome and reason |
|---|---|---|---:|---:|---|
| 1 | Random forest on 3 layouts | Reference point on the new harness | 0.8256 | 0.8218 | reference |
| 1 | + `uri_count` | Repeats are 77–96% High | 0.8462 | 0.8224 | kept; biggest gain, mostly from training repeats |
| 1 | + near-duplicate count | Same song re-released under a new ID | 0.8462 | – | kept; rare, so small |
| 1 | + release-date count | Rough album size or big release day | 0.8278 | – | kept; small gain |
| 1 | + audio relative to year median | The loudness war shifts what "loud" means | 0.8245 | – | neutral; trees already condition on year |
| 2 | Extra-trees + random forest rank blend | Different split rules give different errors | 0.8503 | 0.8258 | kept |
| 2 | Spline logistic regression · kNN | Different inductive biases for the blend | 0.8206 · 0.7985 | – | scrapped; too weak, hurt the blend |
| 3 | 3 model seeds + boosted trees | Smooth seed noise; add a boosting bias | 0.8514 | 0.8273 | kept (Kaggle 0.84644) |
| 3 | Test pseudo-labelling | More training data from test | 0.8493 | 0.8244 | scrapped; reinforces existing errors |
| 4 | k-means / PCA / density / anomaly score | Genre and "typical sound" proxies | 0.848–0.849 | 0.824 | scrapped; raw columns already carry it |
| 4 | Gaussian mixtures · label spreading | Soft genre clusters; semi-supervised graph | 0.8446 · 0.7598 | – | scrapped |
| 4 | kNN neighbour label rate (leave-own-out) | Songs that sound alike share popularity | 0.8395 | 0.8132 | scrapped; pair leakage (decision 5) |
| 5 | Same, cross-fitted | Fix that leakage | 0.8480 | 0.8231 | scrapped; honest but no gain |
| 5 | Blend + 30% CatBoost | Strongest available boosting library | 0.8528 | 0.8288 | kept |
| 6 | PyTorch MLP, periodic embeddings, 20 epochs | NN diversity, as you suggested | 0.8310 | 0.8029 | no blend gain; 0.92 correlated with trees |
| 6 | Label smoothing | Labels look noisy | 0.8205 | – | scrapped; removed signal |
| 7 | CatBoost tuned (learning rate 0.01, 2,000 rounds) | Default settings untested | 0.8466 | 0.8218 | kept |
| 7 | Pairwise ranking loss | AUC is a pairwise metric | 0.8419 | – | scrapped |
| 8 | NN on boosted-tree leaves | Give the NN the trees' splits | 0.8164 | – | scrapped; overfits leaf one-hots |
| 8 | NN correcting boosted logits | NN learns only what trees miss | 0.8410 | – | scrapped; equal to boosting alone |
| 9 | Stacking · greedy ensemble selection | Learn better weights | 0.8514 · 0.8527 | – | not adopted; fits noise (decision 7) |
| 10 | Vocal presence, era copies, in-year percentile | Brainstorm with strong single-feature AUCs | 0.847–0.851 | – | scrapped; trees already find them |
| – | All counts from training folds only | Test-set features banned at the time | 0.8289 | – | kept as the honest version |
| – | Same, conflicting tracks dropped | Remove identical-input noise | 0.8282 | – | no gain |
| 11–12 | Album mates (same-date High rate) | Same date ≈ same album ≈ same artist | 0.8541 | 0.8300 | kept; gain held on single tracks |
| 11 | Copy label from a near-identical train song | Re-releases agree 77–97% | 0.851–0.853 | – | scrapped; model already scores them right |
| 13 | Smooth scores across same-date test songs | Let test songs inform each other | 0.831–0.853 | – | scrapped; blurs good predictions |
| 13–14 | **Similarity-weighted album mates** | Same-artist tracks sound alike | **0.8551** | **0.8314** | **best**; won on every layout |
| 15–16 | Drop conflicts · also keep one row per track | Clean noise in the best model | 0.8549 · 0.8545 | 0.8308 · 0.8315 | tie; both copies already teach 50/50 |
| 17 | + logistic regression at 5–30% | A linear view for the blend | 0.8550–0.8517 | – | worse; weaker and 0.92 correlated |
| 18 | Soft voting instead of ranks | Common ensembling practice | 0.8537–0.8549 | – | no gain; members' probability scales differ |

More detail per idea is in `experiments/IDEAS.md`.

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

**Why I recommend `submission_mates_sim.csv`:** it has the best CV score, and it's the best on single tracks too, so the gain isn't just from repeated tracks. The no-conflict variants tie with it, and keeping all real training rows is the simpler choice when results tie.

## How far to trust these numbers

- **The CV scores are somewhat optimistic.** The same three fold layouts were reused across roughly 100 experiments. The Kaggle leaderboard is the only score nobody tuned against, and its public part is small enough to move about ±0.01 by chance.
- **The leaderboard can't rank most submissions.** They differ by 0.001–0.002 in CV and are over 0.98 rank-correlated, so a single upload can't tell them apart.
- **Two combinations are worth trying next:** era experts with album mates, and CatBoost with native categorical year and date. Both target the weakest segment (2015–19) and the artist-level signal, the two places evidence says information is still missing.
- **Reaching 0.90 most likely needs data the dataset lacks**, such as artist popularity, follower counts or playlist names, because the experiments point to artist-level popularity as the missing driver. Check the course rules before adding outside data.

## Reproduce

```sh
cd experiments
.venv/bin/python make_submission_mates.py --sim   # best submission (needs CatBoost in .venv)
.venv/bin/python train_only.py                    # training-data-only version
python3 round1.py                                 # any round: python3 roundN.py (CatBoost/torch rounds need .venv)
```

**Sources:** `EXPERIMENTS.md`, `experiments/IDEAS.md`, `experiments/RESULTS.md`, the parallel session's `*_results.jsonl` and audit files, and the saved out-of-fold predictions in `experiments/oof/`.
