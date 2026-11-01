# Popularity AUC experiments

All scores below use three stratified, track-grouped folds and only the supplied training data. These are model-selection scores, not an untouched test estimate.

| Experiment | Mean AUC | Finding |
| --- | ---: | --- |
| Original histogram boosted trees, one-hot features | 0.806 | Stronger than the neural network's 0.784 |
| Add fractional release year, release day, date precision | 0.809 | Small gain |
| Native categorical tree inputs and hyperparameter sweep | 0.809 | No material gain over one-hot encoding |
| Extra trees | 0.805 | No gain |
| Classical gradient boosting | 0.798 | No gain |
| RBF SVM | 0.772 | Worse |
| Reweight or deduplicate repeated tracks | 0.809 | No gain |
| Folded tempo and combined key/mode | 0.808 and 0.802 | Worse |
| Add release weekday | 0.812 | Small gain, also improved on two other fold layouts |
| Remove key and categorical release month | 0.816 | Best current feature set; fractional year still uses month when known |
| Random forest with date-enriched features | 0.824 | Best current method; entropy splits, balanced weights, 800 trees, 3 minimum samples per leaf, square-root feature sampling |

The boosted-tree model is in `improved_tree.py`. Its mean AUC by split seed is 0.8161 (155), 0.8198 (2026), 0.8114 (42), and 0.8182 (777). Its parameters are 31 maximum leaves, 5 minimum samples per leaf, learning rate 0.03, and 300 boosting rounds. On a related feature set, training AUC was about 1.00 versus validation AUC about 0.81, indicating substantial overfitting.

The random-forest sweep is in `random_forest_sweep.py`. Its selected setting scored 0.8239 (155), 0.8311 (2026), 0.8240 (42), and 0.8273 (777). Restoring `key` or categorical release month lowered its AUC in a separate feature check.

The user reported a public Kaggle score of **0.83415** for the generated random-forest submission. This is a public-leaderboard result, not a private score.

A 75% random-forest / 25% boosted-tree probability blend scored 0.8271 averaged across four grouped three-fold layouts, versus 0.8266 for the random forest alone. The difference is too small to justify a new submission on its own.

The original 0.92 target and the user's revised 0.90 target have not been reached. Further tuning on these same folds makes model selection increasingly optimistic. None of the repeated fold layouts is an untouched final test.

## Feature-engineering round: target AUC above 0.90

All transforms are fitted on the training portion of each fold. URI is used only to keep repeated tracks together, never as a predictor. No external training data or test labels are used. The existing submission and its generating script remain unchanged.

### Release-year encoding

`feature_sweep.py` tests three interpretations of categorical year:

- `year_onehot`: replace numeric year with one-hot year; retain fractional year.
- `year_both`: add one-hot year alongside both numeric year and fractional year.
- `year_only_categorical`: remove both numeric year representations, so year information is exclusively categorical.

Initial screening uses 400 trees and split seed 155. Every value is a mean across three track-grouped folds.

| Year representation | Square-root feature sampling | Half of features per split |
| --- | ---: | ---: |
| Original numeric-year baseline | 0.82256 | 0.82316 |
| One-hot year, retaining fractional year | 0.81650 | 0.82318 |
| Numeric year plus one-hot year | 0.82231 | 0.82424 |
| Exclusively categorical year | 0.78556 | 0.77825 |

Adding many one-hot columns changes the effective feature sampling, so both sampling settings were compared. The evidence favors retaining the ordered year information, with a small benefit from also supplying year categories when half the columns are sampled.

### Other feature hypotheses

| Experiment family | Best screening AUC | Decision |
| --- | ---: | --- |
| Cyclic month and release quarter | 0.82106 | No improvement |
| Date precision / first-of-month indicators | 0.82261 | Negligible |
| Instrumental / spoken / live threshold flags | 0.82393 | Initial gain did not hold on additional splits |
| Audio interactions and estimated beat count | 0.82009 | No improvement |
| Musical key/mode, circular key, folded tempo | 0.81907 | No improvement |
| Combine initial engineered feature families | 0.81822 | No improvement |
| Release-cohort counts and smoothed target rates | 0.82404 | Small screening gain only |
| Audio compared with its five-year release era | 0.82081 | No improvement |
| Audio cluster distances, 8 or 32 clusters | 0.81336 | No improvement |
| Nearest-neighbor / forest probability blends | 0.82120 | No improvement |
| Duration rounding and measurement precision | 0.82044 | No improvement |
| Histogram boosting with native categorical year | 0.82146 | Below forest |

These are screening results with different feature families and explicitly recorded settings, not a single controlled ranking of algorithms. JSONL files contain exact parameters and fold scores. Most forest screens use 400 trees; feature-removal screens use 300. Histogram boosting uses 500 rounds, learning rate 0.03, minimum leaf size 10, maximum leaves 7/15/31, and L2 1/10.

`cohort_sweep.py` constructs popularity rates for release year, year-month, and exact date. It uses inner track-grouped cross-fitting for training features and outer-training-only statistics for validation features, with smoothing 2/10/50. Counts/rates/both are tested. It never computes a validation song's feature from that song's label. Counts use the same inner splits, so training counts are smaller than validation counts; results apply to this implementation.

`similarity_sweep.py` fits scaling and clusters within each training fold. It tests 8/32 clusters, era-relative audio, and distance-weighted nearest neighbors with 5/15/40/100 neighbors, year weights 0/1/3/10, and probability blend weights 0.25/0.5/1.

Individual removal of 14 inputs did not produce a compelling gain. Removing loudness or instrumentalness hurt most. A subsequent minimum-leaf-size sweep (1/2/3/5/10/20, with and without mode) favored smaller leaves for the numeric-plus-categorical-year forest.

### Best completed candidate

800-tree random forest, entropy, balanced class weights, unlimited depth, minimum leaf size **1**, `max_features=0.5`, model seed 155. It uses `year_both`; no original inputs are removed.

| Grouped split seed | Original forest | New candidate |
| --- | ---: | ---: |
| 155 | 0.82390 | 0.82586 |
| 2026 | 0.83107 | 0.83222 |
| 42 | 0.82402 | 0.82366 |
| 777 | 0.82730 | 0.82951 |
| Mean | **0.82657** | **0.82781** |

The increase is only **0.00124 AUC** and is not an independent test result or evidence that 0.90 is attainable. The analogous new forest with minimum leaf size 3 averages 0.82749. The only available public leaderboard result remains the original forest's user-reported **0.83415**.

Reproduce the candidate evaluation:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 feature_sweep.py --variants year_both --seeds 155 2026 42 777 --trees 800 --max-features 0.5 --leaf 1
```

Results append to `feature_results.jsonl`. Other records: `cohort_results.jsonl`, `similarity_results.jsonl`, `ablation_results.jsonl`, `regularization_results.jsonl`, and `boosted_year_results.jsonl`. Feature removal and leaf-size experiments can be reproduced with `--drop COLUMN` and `--leaf N`.

### Next experiment: CatBoost (installation blocked)

The user approved installing CatBoost in the local `.venv`. Environment creation succeeded, but installation could not connect to `pypi.org` (DNS/network failure). No CatBoost model has been trained or scored.

`catboost_sweep.py` is prepared for baseline/year/date/full-key categorical feature sets, depths 4/6/8, and L2 3/10. It uses native categories as recommended in the [CatBoost documentation](https://catboost.ai/docs/en/features/categorical-features). A separate inner grouped split chooses the number of boosting rounds, up to 3000 with patience 150; the model is then refit on the full outer-training fold before evaluation. Outer validation is not used for early stopping.

From a terminal with network access:

```sh
.venv/bin/python -m pip install catboost
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python catboost_sweep.py
```

Verification: completed sklearn experiments ran successfully; feature self-checks cover missing/partial dates and unseen years; target-encoding checks cover unseen categories and exclusion of a row's own label; grouped-fold separation and finite feature arrays are asserted. CatBoost preprocessing and Python compilation pass, but CatBoost training is unverified pending installation.

## Follow-up: occurrence-count audit and release-era experts

The separately created `experiments/` files were inspected without modifying them. Their occurrence-count features are computed on combined train/test inputs (without labels), so they use held-out feature information. Stored predictions show up to 0.84731 mean grouped-fold AUC. However, URI singletons constitute 89.00% of training rows versus 97.45% of test rows.

An audit of those saved predictions, using only validation tracks occurring once in the training dataset, gives:

| Saved model | All validation rows | Singleton validation rows |
| --- | ---: | ---: |
| `r1_base_rf` | 0.82555 | 0.82184 |
| `r1_all_rf` | 0.84627 | 0.82238 |
| `r1_all_et` | 0.84691 | 0.82183 |
| `r2_et_leaf3_mf0.3` | 0.84731 | 0.82182 |

These use the existing three split seeds, each with three grouped folds. Audit results are in `occurrence_audit.json`. This does not establish a ceiling on achievable AUC; it shows that the large aggregate gain is not reproduced on the subset most prevalent in the competition test set. These transductive scores are not a replacement for the fold-fitted baseline comparison above.

`era_expert_sweep.py` tests separate pre-/post-release-year forests at cutoffs 2015, 2020, and 2023. Each specialist fits preprocessing only on its own training rows. Predictions are blended with the common forest at specialist weights 0.25/0.5/1.0. The common model uses numeric plus one-hot year, 400 trees, minimum leaf size 1, half-feature sampling, entropy, balanced weights, and seed 155. Validation remains the same three grouped folds; assertions check track separation and finite probability outputs.

The 2015 cutoff initially won on split seed 155, but the **2023 cutoff with 50% specialist / 50% shared-model predictions** performed best averaged across all four layouts:

| Split seed | Common forest (400 trees) | 2023 expert blend |
| --- | ---: | ---: |
| 155 | 0.82568 | 0.82647 |
| 2026 | 0.83132 | 0.83316 |
| 42 | 0.82202 | 0.82557 |
| 777 | 0.83020 | 0.83220 |
| Mean | **0.82731** | **0.82935** |

This is the best completed fold-fitted candidate so far, improving the matched baseline on all four layouts. The cutoff/weight selection uses these same layouts, so 0.82935 remains a model-selection result, not an unbiased test estimate. It is still well below 0.90. No new competition submission was uploaded. The 2015/50% blend averages 0.82813 and does not improve on seed 777.

Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 era_expert_sweep.py --seeds 155 2026 42 777`. Results append to `era_expert_results.jsonl`. Python compilation and all executed assertions passed. CatBoost availability was rechecked and it remains uninstalled.

### Final audit before dependency unblock

Later results in the separately created `experiments/` directory were checked directly from their stored out-of-fold arrays. The three-seed ET/RF/HGB blend scores 0.85140 overall and 0.82730 on singleton validation tracks. The ET/RF blend scores 0.85075 / 0.82640; pseudo-labeling scores 0.84932 / 0.82438. They retain the transductive occurrence-count caveat above and none reaches 0.90.

The statement in `experiments/RESULTS.md` that 32 conflicting repeated-track labels cap achievable AUC well below 0.90 is **not supported** by that count. As a diagnostic only, assigning every track its observed training-label mean yields training AUC 0.99985 even while forcing identical scores for each repeated track. This deliberately uses the labels and is **not** a predictive model or validation result; it simply demonstrates that these observed contradictions alone do not imply the claimed ceiling. `label_conflict_audit.json` records the calculation. No claim is made that such an oracle score can generalize.

The 0.90 objective remains unmet. CatBoost is still unavailable after checks in three consecutive goal turns; the approved installation failed due to network/DNS access. The next prepared model-family experiment requires installing that dependency externally. The goal is blocked pending that external-state change, not complete and not proven impossible.
