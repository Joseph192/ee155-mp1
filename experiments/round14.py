"""Round 14: similarity-weighted album mates (+ plain mate count) in the full blend and CatBoost. Run with .venv/bin/python."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

from common import SEEDS, fold_auc, load, oof, report
from features import build
from round2 import single_auc
from round3 import blend_models
from round4 import embed
from round7 import CONFIGS as CAT
from round11 import mate_rate
from round13 import mate_rate_sim


def add_sim_mates(fit_predict, dates, groups, z, tau=1.0):
    """Cross-fitted similarity-weighted mate rate and plain mate count for training rows; full reference for query rows."""
    def feats(a, b):
        return mate_rate_sim(dates[a], y_all[a], groups[a], z[a], dates[b], groups[b], z[b], tau), mate_rate(dates[a], y_all[a], groups[a], dates[b], groups[b])[1]

    def wrapped(tx, ty, vx, ti, vi):
        y_all[ti] = ty
        tr_rate, tr_n = np.empty(len(ti)), np.empty(len(ti))
        for fit, held in StratifiedGroupKFold(3, shuffle=True, random_state=911).split(ti, ty, groups[ti]):
            tr_rate[held], tr_n[held] = feats(ti[fit], ti[held])
        va_rate, va_n = feats(ti, vi)
        return fit_predict(tx.assign(mate_sim=tr_rate, mate_n=tr_n), ty, vx.assign(mate_sim=va_rate, mate_n=va_n), ti, vi)
    y_all = np.zeros(len(dates))  # labels filled per fold from training rows only
    return wrapped


def main():
    train, test, y, groups = load()
    tx, vx = build(train, test, "all")
    singles = (tx.uri_count == 1).to_numpy()
    z = embed(pd.concat([tx, vx], ignore_index=True))[:len(train)]
    dates = train.track_album_release_date.fillna("?").to_numpy()
    etrf = lambda a, b, c, *_: blend_models(a, b, c, model_seeds=(155,))
    preds = {}

    def run(name, fit_predict):
        preds[name] = oof(fit_predict, tx, y, groups)
        report(name, preds[name], y, groups)
        print(f"{'':40s} singles_only={single_auc(preds[name], y, groups, singles):.4f}", flush=True)

    for tau in (0.5, 2.0):
        run(f"r14_sim_tau{tau}_etrf", add_sim_mates(etrf, dates, groups, z, tau))
    run("r14_sim_blend3_hgb", add_sim_mates(lambda a, b, c, *_: blend_models(a, b, c, with_hgb=True), dates, groups, z))
    run("r14_sim_cat", add_sim_mates(CAT["r7_cat_d6_lr01_2000"], dates, groups, z))
    for w in (0.2, 0.3, 0.4):
        b = {s: (1 - w) * rankdata(preds["r14_sim_blend3_hgb"][s]) + w * rankdata(preds["r14_sim_cat"][s]) for s in SEEDS}
        print(f"sim blend + {w} cat: mean={np.mean([fold_auc(b[s], y, groups, s) for s in SEEDS]):.4f} "
              f"singles={single_auc(b, y, groups, singles):.4f}  (round-12 best 0.8541 / 0.8300)", flush=True)


if __name__ == "__main__":
    main()
