"""Round 11: transductive signals found by inspecting test features.

A. Album mates: smoothed High rate of other tracks with the same full release date (cross-fitted for training rows,
   so a row never sees labels from a model that saw it; plain leave-one-out leaks via pairs).
B. Re-release override: if a row's nearest other-track labelled neighbour is within distance 0.3 in audio space,
   move its score toward that neighbour's label (train label agreement 0.77-0.97 at that range).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from common import OUT, SEEDS, fold_auc, folds, load, oof, report
from features import build
from round2 import single_auc
from round3 import blend_models
from round4 import embed


def mate_rate(dates_ref, y_ref, g_ref, dates_q, g_q, smoothing=5):
    """Smoothed High rate of reference rows sharing the query's full date, excluding the query's own track."""
    prior = y_ref.mean()
    ref = pd.DataFrame({"d": dates_ref, "y": y_ref, "g": g_ref})
    full = ref[ref.d.str.len() == 10]
    by_date = full.groupby("d").y.agg(["sum", "count"])
    by_date_track = full.groupby(["d", "g"]).y.agg(["sum", "count"])
    q = pd.DataFrame({"d": dates_q, "g": g_q})
    s = q.d.map(by_date["sum"]).fillna(0).to_numpy()
    n = q.d.map(by_date["count"]).fillna(0).to_numpy()
    own = q.set_index(["d", "g"]).index.map(lambda k: tuple(by_date_track.loc[k]) if k in by_date_track.index else (0, 0))
    s -= np.array([o[0] for o in own])
    n -= np.array([o[1] for o in own])
    return (s + smoothing * prior) / (n + smoothing), n


def with_mates(dates, groups):
    def fit_predict(tx, ty, vx, ti, vi):
        tr_rate, tr_n = np.empty(len(ti)), np.empty(len(ti))
        for fit, held in StratifiedGroupKFold(3, shuffle=True, random_state=911).split(ti, ty, groups[ti]):
            tr_rate[held], tr_n[held] = mate_rate(dates[ti][fit], ty[fit], groups[ti][fit], dates[ti][held], groups[ti][held])
        va_rate, va_n = mate_rate(dates[ti], ty, groups[ti], dates[vi], groups[vi])
        tx = tx.assign(mate_rate=tr_rate, mate_n=tr_n)
        vx = vx.assign(mate_rate=va_rate, mate_n=va_n)
        return blend_models(tx, ty, vx, model_seeds=(155,))
    return fit_predict


def override(scores, z_ref, y_ref, g_ref, z_q, g_q, radius=0.3, weight=0.8):
    """Pull scores (in rank units, 0-1) toward the label of a near-identical labelled other-track neighbour."""
    dist, idx = NearestNeighbors(n_neighbors=min(5, len(z_ref))).fit(z_ref).kneighbors(z_q)
    out = scores.copy()
    hits = 0
    for i in range(len(z_q)):
        for d, j in zip(dist[i], idx[i]):
            if g_ref[j] != g_q[i]:
                if d < radius:
                    out[i] = (1 - weight) * scores[i] + weight * y_ref[j]
                    hits += 1
                break
    return out, hits


def self_check():
    rate, n = mate_rate(np.array(["2020-01-01"] * 3), np.array([1, 0, 1]), np.array(["a", "b", "c"]),
                        np.array(["2020-01-01"]), np.array(["a"]), smoothing=0)
    assert n[0] == 2 and rate[0] == 0.5  # own track 'a' excluded
    s, hits = override(np.array([0.5]), np.array([[0.0], [5.0]]), np.array([1, 0]), np.array(["x", "y"]), np.array([[0.1]]), np.array(["q"]))
    assert hits == 1 and np.isclose(s[0], 0.9)


def main():
    self_check()
    train, test, y, groups = load()
    features, test_features = build(train, test, "all")
    singles = (features.uri_count == 1).to_numpy()
    dates = train.track_album_release_date.fillna("?").to_numpy()

    preds = oof(with_mates(dates, groups), features, y, groups)
    report("r11_mates_etrf", preds, y, groups)
    print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}  (baseline ET+RF 1-seed 0.8503 / 0.8258)", flush=True)

    # B on top of the current best OOF (round-3 blend + 30% CatBoost).
    z = embed(pd.concat([features, test_features], ignore_index=True))[:len(train)]
    r3, cat = np.load(OUT / "r3_blend_3seeds_hgb.npz"), np.load(OUT / "r5_catboost.npz")
    best = {s: (0.7 * rankdata(r3[str(s)]) + 0.3 * rankdata(cat[str(s)])) / len(y) for s in SEEDS}
    for radius in (0.05, 0.3, 0.6):
        for weight in (0.5, 0.8, 1.0):
            adj, n_hits = {}, 0
            for s in SEEDS:
                adj[s] = best[s].copy()
                for ti, vi in folds(y, groups, s):
                    adj[s][vi], h = override(best[s][vi], z[ti], y[ti], groups[ti], z[vi], groups[vi], radius, weight)
                    n_hits += h
            aucs = [fold_auc(adj[s], y, groups, s) for s in SEEDS]
            print(f"override r={radius} w={weight}: mean={np.mean(aucs):.4f} singles={single_auc(adj, y, groups, singles):.4f} hits/layout={n_hits // 3}  (best 0.8528 / 0.8288)", flush=True)


if __name__ == "__main__":
    main()
