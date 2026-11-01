"""Compare a shallow tree and boosted trees on track-grouped AUC."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import build_preprocessor


SEED = 155


def main():
    frame = pd.read_csv("data/train.csv")
    labels = frame["Popularity_Type"].eq("High").to_numpy(dtype=int)
    groups = frame["uri"].fillna(pd.Series("missing_track_" + frame.index.astype(str), index=frame.index)).to_numpy()
    folds = []
    for train_idx, valid_idx in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(frame, labels, groups):
        preprocessor = build_preprocessor()
        folds.append((
            preprocessor.fit_transform(frame.iloc[train_idx]), labels[train_idx],
            preprocessor.transform(frame.iloc[valid_idx]), labels[valid_idx],
        ))

    def score(model):
        aucs = []
        for train_x, train_y, valid_x, valid_y in folds:
            model.fit(train_x, train_y)
            aucs.append(roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1]))
        return np.mean(aucs), np.std(aucs)

    baseline = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=SEED)
    mean_auc, std_auc = score(baseline)
    print(f"shallow_tree AUC={mean_auc:.4f} ± {std_auc:.4f}", flush=True)

    results = []
    for leaves, learning_rate, rounds in product((7, 15, 31), (0.03, 0.1), (100, 300)):
        model = HistGradientBoostingClassifier(
            max_leaf_nodes=leaves,
            learning_rate=learning_rate,
            max_iter=rounds,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=SEED,
        )
        mean_auc, std_auc = score(model)
        results.append((mean_auc, leaves, learning_rate, rounds))
        print(f"boosted leaves={leaves} lr={learning_rate} rounds={rounds} AUC={mean_auc:.4f} ± {std_auc:.4f}", flush=True)

    print(f"Best boosted setting: {max(results)}")


if __name__ == "__main__":
    main()
