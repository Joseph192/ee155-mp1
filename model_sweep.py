"""Compare small MLP architectures, then learning rates, using grouped AUC."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from data_preprocessing import build_preprocessor


MAX_EPOCHS = 200
PATIENCE = 20
SEED = 155


def score_setting(frame, labels, folds, layers, learning_rate):
    scores = []
    epochs = []
    for train_idx, valid_idx in folds:
        preprocessor = build_preprocessor()
        train_x = preprocessor.fit_transform(frame.iloc[train_idx])
        valid_x = preprocessor.transform(frame.iloc[valid_idx])
        train_y, valid_y = labels[train_idx], labels[valid_idx]
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=64,
            learning_rate_init=learning_rate,
            random_state=SEED,
        )
        best_auc, best_epoch = -np.inf, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            model.partial_fit(train_x, train_y, classes=np.array([0, 1]))
            auc = roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1])
            if auc > best_auc + 1e-4:
                best_auc, best_epoch = auc, epoch
            if epoch - best_epoch >= PATIENCE:
                break
        scores.append(best_auc)
        epochs.append(best_epoch)
    return np.mean(scores), np.std(scores), epochs


def main():
    frame = pd.read_csv("data/train.csv")
    labels = frame["Popularity_Type"].eq("High").to_numpy(dtype=int)
    groups = frame["uri"].fillna(pd.Series("missing_track_" + frame.index.astype(str), index=frame.index)).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(frame, labels, groups))

    architectures = [(width,) for width in (8, 16, 32, 48, 64, 96, 128)]
    architectures += list(product((16, 32, 48, 64, 96), repeat=2))
    architectures += [
        (16, 16, 16), (32, 32, 32), (64, 64, 64),
        (64, 32, 16), (96, 64, 32), (32, 64, 32),
    ]
    results = []
    for layers in architectures:
        mean_auc, std_auc, epochs = score_setting(frame, labels, folds, layers, 1e-3)
        results.append((mean_auc, layers))
        print(f"layers={layers} lr=0.001 AUC={mean_auc:.4f} ± {std_auc:.4f} best_epochs={epochs}", flush=True)

    best_layers = max(results)[1]
    print(f"Best architecture: {best_layers}", flush=True)
    for learning_rate in (3e-4, 1e-3, 3e-3, 1e-2):
        mean_auc, std_auc, epochs = score_setting(frame, labels, folds, best_layers, learning_rate)
        print(f"layers={best_layers} lr={learning_rate:g} AUC={mean_auc:.4f} ± {std_auc:.4f} best_epochs={epochs}", flush=True)


if __name__ == "__main__":
    main()
