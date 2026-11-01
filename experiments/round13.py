"""Round 13: test features shaping predictions.

16. Similarity-weighted album mates: same-date training mates weighted by audio similarity (same artist sounds alike).
17. Transductive smoothing: each unlabelled row's score moves toward the predicted scores of its same-date mates in the
    unlabelled pool. In CV the pool is validation fold + test, both scored by the fold's model.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

from common import SEEDS, fold_auc, folds, load, report
from features import build
from round2 import single_auc
from round3 import blend_models
from round4 import embed
from round11 import mate_rate


def mate_rate_sim(dates_ref, y_ref, g_ref, z_ref, dates_q, g_q, z_q, tau=1.0, smoothing=2):
    """Like mate_rate, but each same-date mate counts exp(-distance / tau)."""
    prior = y_ref.mean()
    out = np.full(len(dates_q), prior)
    by_date = pd.Series(np.arange(len(dates_ref))).groupby(dates_ref).apply(np.array)
    for i, d in enumerate(dates_q):
        if len(d) != 10 or d not in by_date.index:
            continue
        j = by_date[d]
        j = j[g_ref[j] != g_q[i]]
        w = np.exp(-np.linalg.norm(z_ref[j] - z_q[i], axis=1) / tau)
        out[i] = (w @ y_ref[j] + smoothing * prior) / (w.sum() + smoothing)
    return out


def pool_smooth(scores, dates, groups, alpha):
    """Blend each score with the mean score of other-track same-date rows in the same pool."""
    s = pd.Series(scores)
    frame = pd.DataFrame({"s": scores, "d": dates, "g": groups})
    frame = frame[frame.d.str.len() == 10]
    tot = frame.groupby("d").s.agg(["sum", "count"])
    own = frame.groupby(["d", "g"]).s.agg(["sum", "count"])
    out = s.copy()
    for i, row in frame.iterrows():
        o = own.loc[(row.d, row.g)]
        n = tot.loc[row.d, "count"] - o["count"]
        if n > 0:
            out[i] = (1 - alpha) * row.s + alpha * (tot.loc[row.d, "sum"] - o["sum"]) / n
    return out.to_numpy()


def self_check():
    r = mate_rate_sim(np.array(["2020-01-01"] * 2), np.array([1, 0]), np.array(["a", "b"]), np.array([[0.0], [10.0]]),
                      np.array(["2020-01-01"]), np.array(["q"]), np.array([[0.0]]), tau=1.0, smoothing=0)
    assert r[0] > 0.99  # the near mate (label 1) dominates the far one
    sm = pool_smooth(np.array([0.0, 1.0, 0.5]), np.array(["2020-01-01", "2020-01-01", "1999"]), np.array(["a", "b", "c"]), 0.5)
    assert np.allclose(sm, [0.5, 0.5, 0.5])


def main():
    self_check()
    train, test, y, groups = load()
    tx, vx = build(train, test, "all")
    feats = pd.concat([tx, vx], ignore_index=True)
    n = len(train)
    dates = pd.concat([train.track_album_release_date, test.track_album_release_date], ignore_index=True).fillna("?").to_numpy()
    g_all = np.concatenate([groups, test.uri.fillna("test_missing").to_numpy()])
    z = embed(feats)
    test_idx = np.arange(n, len(feats))
    singles = (tx.uri_count == 1).to_numpy()

    def mates(ti, qi, sim):
        """Cross-fitted mate features for training rows ti, full-reference ones for query rows qi."""
        rate = mate_rate_sim if sim else (lambda dr, yr, gr, zr, dq, gq, zq: mate_rate(dr, yr, gr, dq, gq)[0])
        tr = np.empty(len(ti))
        for fit, held in StratifiedGroupKFold(3, shuffle=True, random_state=911).split(ti, y[ti], g_all[ti]):
            a, b = ti[fit], ti[held]
            tr[held] = rate(dates[a], y[a], g_all[a], z[a], dates[b], g_all[b], z[b])
        return tr, rate(dates[ti], y[ti], g_all[ti], z[ti], dates[qi], g_all[qi], z[qi])

    for sim in (False, True):
        pooled = {}
        for s in SEEDS:
            pooled[s] = np.full(n, np.nan), {}
            for ti, vi in folds(y, groups, s):
                qi = np.concatenate([vi, test_idx])
                t_rate, q_rate = mates(ti, qi, sim)
                p = blend_models(feats.iloc[ti].assign(mate=t_rate), y[ti], feats.iloc[qi].assign(mate=q_rate), model_seeds=(155,))
                pooled[s][1][tuple(vi[:3])] = (vi, qi, p)
        tag = "sim" if sim else "plain"
        base = {s: np.full(n, np.nan) for s in SEEDS}
        for s in SEEDS:
            for vi, qi, p in pooled[s][1].values():
                base[s][vi] = p[:len(vi)]
        report(f"r13_mates_{tag}_etrf", base, y, groups)
        print(f"{'':40s} singles_only={single_auc(base, y, groups, singles):.4f}", flush=True)
        for alpha in (0.1, 0.2, 0.3, 0.5):
            sm = {s: np.full(n, np.nan) for s in SEEDS}
            for s in SEEDS:
                for vi, qi, p in pooled[s][1].values():
                    sm[s][vi] = pool_smooth(rankdata(p) / len(p), dates[qi], g_all[qi], alpha)[:len(vi)]
            aucs = [fold_auc(sm[s], y, groups, s) for s in SEEDS]
            print(f"  + pool smoothing alpha={alpha}: mean={np.mean(aucs):.4f} singles={single_auc(sm, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
