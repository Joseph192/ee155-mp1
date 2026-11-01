"""Round 12: album-mate features in the full models. Run with .venv/bin/python (CatBoost)."""

import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

from common import SEEDS, fold_auc, load, oof, report
from features import build
from round2 import single_auc
from round3 import blend_models
from round7 import CONFIGS as CAT
from round11 import mate_rate


def add_mates(fit_predict, dates, groups, smoothing=5):
    """Wrap any fit_predict so it also gets cross-fitted album-mate rate and count."""
    def wrapped(tx, ty, vx, ti, vi):
        tr_rate, tr_n = np.empty(len(ti)), np.empty(len(ti))
        for fit, held in StratifiedGroupKFold(3, shuffle=True, random_state=911).split(ti, ty, groups[ti]):
            tr_rate[held], tr_n[held] = mate_rate(dates[ti][fit], ty[fit], groups[ti][fit], dates[ti][held], groups[ti][held], smoothing)
        va_rate, va_n = mate_rate(dates[ti], ty, groups[ti], dates[vi], groups[vi], smoothing)
        return fit_predict(tx.assign(mate_rate=tr_rate, mate_n=tr_n), ty, vx.assign(mate_rate=va_rate, mate_n=va_n), ti, vi)
    return wrapped


def main():
    train, test, y, groups = load()
    features, _ = build(train, test, "all")
    singles = (features.uri_count == 1).to_numpy()
    dates = train.track_album_release_date.fillna("?").to_numpy()
    etrf = lambda tx, ty, vx, *_: blend_models(tx, ty, vx, model_seeds=(155,))
    runs = {
        "r12_mates_s2_etrf": add_mates(etrf, dates, groups, 2),
        "r12_mates_s20_etrf": add_mates(etrf, dates, groups, 20),
        "r12_mates_blend3_hgb": add_mates(lambda tx, ty, vx, *_: blend_models(tx, ty, vx, with_hgb=True), dates, groups),
        "r12_mates_cat": add_mates(CAT["r7_cat_d6_lr01_2000"], dates, groups),
    }
    preds = {}
    for name in sys.argv[1:] or runs:
        preds[name] = oof(runs[name], features, y, groups)
        report(name, preds[name], y, groups)
        print(f"{'':40s} singles_only={single_auc(preds[name], y, groups, singles):.4f}", flush=True)
    if {"r12_mates_blend3_hgb", "r12_mates_cat"} <= preds.keys():
        for w in (0.2, 0.3, 0.4):
            b = {s: (1 - w) * rankdata(preds["r12_mates_blend3_hgb"][s]) + w * rankdata(preds["r12_mates_cat"][s]) for s in SEEDS}
            print(f"mates blend + {w} cat: mean={np.mean([fold_auc(b[s], y, groups, s) for s in SEEDS]):.4f} "
                  f"singles={single_auc(b, y, groups, singles):.4f}  (previous best 0.8528 / 0.8288)", flush=True)


if __name__ == "__main__":
    main()
