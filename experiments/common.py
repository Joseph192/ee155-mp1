"""Shared track-grouped CV harness for the new experiments. Run scripts from the repo root."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from improved_tree import make_features  # noqa: E402  (old code, imported unchanged)
from random_forest_sweep import build_forest_preprocessor  # noqa: E402

SEEDS = (155, 2026, 42)
OUT = Path(__file__).resolve().parent / "oof"


def load():
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    labels = train["Popularity_Type"].eq("High").to_numpy(dtype=int)
    groups = train["uri"].fillna(pd.Series("missing_" + train.index.astype(str), index=train.index)).to_numpy()
    return train, test, labels, groups


def folds(labels, groups, seed):
    return list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(labels, labels, groups))


def oof(fit_predict, features, labels, groups, seeds=SEEDS):
    """fit_predict(train_x, train_y, valid_x, train_idx, valid_idx) -> probabilities. Returns {seed: oof array}."""
    result = {}
    for seed in seeds:
        pred = np.full(len(labels), np.nan)
        for train_idx, valid_idx in folds(labels, groups, seed):
            assert set(groups[train_idx]).isdisjoint(groups[valid_idx])
            pred[valid_idx] = fit_predict(features.iloc[train_idx], labels[train_idx], features.iloc[valid_idx], train_idx, valid_idx)
        result[seed] = pred
    return result


def fold_auc(pred, labels, groups, seed):
    """Mean of per-fold AUCs, matching how EXPERIMENTS.md scores were computed."""
    return np.mean([roc_auc_score(labels[v], pred[v]) for _, v in folds(labels, groups, seed)])


def report(name, preds, labels, groups):
    aucs = [fold_auc(p, labels, groups, s) for s, p in preds.items()]
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"{name}.npz", **{str(s): p for s, p in preds.items()})
    print(f"{name:40s} mean={np.mean(aucs):.4f} " + " ".join(f"{s}={a:.4f}" for s, a in zip(preds, aucs)), flush=True)
    return np.mean(aucs)
