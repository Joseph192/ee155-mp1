"""Round 4: unsupervised and neighbour-based features on top of the 'all' set.

Unsupervised pieces (clusters, PCA, density, anomaly) are fit on train+test features, never labels.
Neighbour-label features use only the training fold, and a training row never sees its own track's label.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from common import load, oof, report
from features import AUDIO, build
from round2 import single_auc
from round3 import blend_models

SPACE = AUDIO + ["release_fractional_year"]


def embed(both):
    """Standardised audio + date matrix; log duration so it doesn't dominate distances."""
    x = both[SPACE].copy()
    x["duration_ms"] = np.log(x.duration_ms)
    x = SimpleImputer(strategy="median").fit_transform(x)
    return StandardScaler().fit_transform(x)


def unsupervised(both, z, variant):
    out = pd.DataFrame(index=both.index)
    if variant in ("kmeans", "unsup_all"):
        for k in (8, 25):
            km = KMeans(k, n_init=10, random_state=155).fit(z)
            dist = km.transform(z)
            out[f"km{k}_dist"] = dist.min(axis=1)
            out[f"km{k}_size"] = pd.Series(km.labels_).map(pd.Series(km.labels_).value_counts()).to_numpy()
            out[f"km{k}_id"] = km.labels_
    if variant in ("gmm", "unsup_all"):
        gmm = GaussianMixture(10, covariance_type="full", random_state=155).fit(z)
        for i, col in enumerate(gmm.predict_proba(z).T):
            out[f"gmm{i}"] = col
        out["gmm_loglik"] = gmm.score_samples(z)
    if variant in ("pca", "unsup_all"):
        for i, col in enumerate(PCA(5, random_state=155).fit_transform(z).T):
            out[f"pc{i}"] = col
    if variant in ("density", "unsup_all"):
        dist, _ = NearestNeighbors(n_neighbors=21).fit(z).kneighbors(z)
        out["knn_dist5"] = dist[:, 1:6].mean(axis=1)
        out["knn_dist20"] = dist[:, 1:21].mean(axis=1)
        out["iso_score"] = IsolationForest(n_estimators=300, random_state=155).fit(z).score_samples(z)
    return out


def neighbour_labels(z_train, y_train, g_train, z_query, g_query, ks=(10, 30, 100)):
    """Mean label of the k nearest training rows, skipping rows of the query's own track."""
    extra = 4  # a track repeats at most 3 times in train
    dist, idx = NearestNeighbors(n_neighbors=min(max(ks) + extra, len(z_train))).fit(z_train).kneighbors(z_query)
    same = g_train[idx] == g_query[:, None]
    cols = {}
    for k in ks:
        keep = (~same).cumsum(axis=1) <= k
        mask = keep & ~same
        cols[f"nn{k}_rate"] = (y_train[idx] * mask).sum(axis=1) / mask.sum(axis=1)
        w = mask / (dist + 1e-3)
        cols[f"nn{k}_wrate"] = (y_train[idx] * w).sum(axis=1) / w.sum(axis=1)
    return pd.DataFrame(cols)


def self_check():
    z = np.array([[0.0], [0.0], [1.0], [5.0]])
    y = np.array([1, 1, 0, 0])
    g = np.array(["a", "a", "b", "c"])
    out = neighbour_labels(z, y, g, z[:1], g[:1], ks=(1,))
    assert out.nn1_rate.iloc[0] == 0  # nearest non-self-track neighbour is 'b' (label 0), not its duplicate


def main():
    self_check()
    train, test, y, groups = load()
    base, test_base = build(train, test, "all")
    both_raw = pd.concat([base, test_base], ignore_index=True)
    z_all = embed(both_raw)
    z = z_all[:len(train)]
    singles = (base.uri_count == 1).to_numpy()
    variants = sys.argv[1:] or ["kmeans", "gmm", "pca", "density", "unsup_all", "nnlabel", "nnlabel_unsup", "spread"]

    for variant in variants:
        features = base.copy()
        if variant in ("kmeans", "gmm", "pca", "density", "unsup_all", "nnlabel_unsup"):
            u = unsupervised(both_raw, z_all, "unsup_all" if variant == "nnlabel_unsup" else variant).iloc[:len(train)]
            features = pd.concat([features, u.reset_index(drop=True)], axis=1)

        if variant in ("nnlabel", "nnlabel_unsup"):
            def fit_predict(tx, ty, vx, ti, vi):
                tr_nn = neighbour_labels(z[ti], ty, groups[ti], z[ti], groups[ti])
                va_nn = neighbour_labels(z[ti], ty, groups[ti], z[vi], groups[vi])
                return blend_models(pd.concat([tx.reset_index(drop=True), tr_nn], axis=1), ty,
                                    pd.concat([vx.reset_index(drop=True), va_nn], axis=1), model_seeds=(155,))
        elif variant == "spread":
            def fit_predict(tx, ty, vx, ti, vi):
                # Transductive: labels from the training fold spread over a kNN graph of train fold + validation + test.
                pool = np.vstack([z[ti], z[vi], z_all[len(train):]])
                labels = np.concatenate([ty, np.full(len(pool) - len(ty), -1)])
                model = LabelSpreading(kernel="knn", n_neighbors=15, alpha=0.5, max_iter=200).fit(pool, labels)
                return model.label_distributions_[len(ti):len(ti) + len(vi), 1]
        else:
            def fit_predict(tx, ty, vx, *_):
                return blend_models(tx, ty, vx, model_seeds=(155,))

        preds = oof(fit_predict, features, y, groups)
        report(f"r4_{variant}", preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
