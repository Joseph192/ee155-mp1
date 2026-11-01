# Round results (3 track-grouped 3-fold layouts, seeds 155/2026/42)

"Singles" = AUC on validation rows whose track appears once, the closest match to test (1.3% repeated tracks vs 6% in train).

| Idea | CV AUC | Singles | Verdict |
| --- | ---: | ---: | --- |
| Old RF features (reproduced) | 0.826 | – | baseline |
| + `uri_count` (repeats in train+test) | 0.846 | – | keep: repeated tracks are 77–96% High |
| + near-duplicate audio count | 0.846 | – | negligible |
| + audio relative to release-year median | 0.825 | – | scrap |
| + tracks sharing the release date | 0.828 | – | small, kept in `all` |
| ET / RF / HGB tuning on `all` | 0.847 | 0.822 | ET leaf 3, 30% features best |
| Spline logistic, kNN | 0.821 / 0.799 | – | scrap, hurt blends |
| ET+RF rank blend | 0.850 | 0.826 | keep |
| 3 model seeds + HGB in blend | **0.851** | **0.827** | final (`make_submission.py`) |
| Test pseudo-labelling | 0.849 | 0.824 | scrap |
| Round 4 (single-seed ET+RF baseline here: 0.850 / 0.826) | | | |
| + k-means distance/size/id | 0.848 | 0.824 | scrap |
| + GMM memberships | 0.845 | 0.820 | scrap |
| + PCA (5 comps) | 0.849 | 0.824 | scrap |
| + kNN density, isolation-forest score | 0.848 | 0.824 | scrap |
| + kNN neighbour label rate (leave-own-track-out) | 0.840 | 0.813 | scrap, train-side rates optimistic |
| + same, nested cross-fitted (round 5) | 0.848 | 0.823 | scrap |
| Label spreading on kNN graph incl. test | 0.760 | 0.762 | scrap |
| CatBoost alone (round 5, `.venv`) | 0.845 | 0.821 | weak alone |
| Final blend + 30% CatBoost | 0.853 | 0.829 | +0.0014, within noise |

Kaggle public score of the final blend (`../submission_blend.csv`): **0.84644** (old RF: 0.83415).

0.9 not reached. 32 of 203 repeated tracks carry both labels, so the label depends on context missing
from the features (likely the source playlist). That caps achievable AUC well below 0.9 with these columns.

Ceiling estimate: 15.8% of repeated tracks disagree, which implies ~8.6% symmetric label noise. A perfect ranker
would then score about 0.90. Per-era OOF AUC is weakest for 2015-19 (0.73) and best before 2000 (0.89).
