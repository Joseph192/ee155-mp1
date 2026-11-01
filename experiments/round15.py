"""Round 15: best model (round 14: similarity-weighted mates + blend + 30% CatBoost) with conflicting tracks dropped
from each training fold only. Compared against the saved round-14 OOF (same code, conflicts kept). Run with .venv/bin/python.
--dedup (round 16) also keeps one row per remaining repeated track; compared with kept and conflicts-dropped.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import OUT, SEEDS, fold_auc, load, oof, report
from features import build
from round2 import single_auc
from round3 import blend_models
from round4 import embed
from round7 import CONFIGS as CAT
from round14 import add_sim_mates
from train_only import conflicting


def drop_conflicts(fit_predict, groups, dedup=False):
    """Remove tracks labelled both ways in this training fold (fold labels only) before mates and model see them.
    dedup: also keep only the first row of each remaining repeated track (uri_count still says it repeated)."""
    def wrapped(tx, ty, vx, ti, vi):
        keep = ~conflicting(groups[ti], ty)
        if dedup:
            keep &= ~pd.Series(groups[ti]).duplicated().to_numpy()
        return fit_predict(tx[keep], ty[keep], vx, ti[keep], vi)
    return wrapped


def main():
    train, test, y, groups = load()
    tx, vx = build(train, test, "all")
    singles = (tx.uri_count == 1).to_numpy()
    z = embed(pd.concat([tx, vx], ignore_index=True))[:len(train)]
    dates = train.track_album_release_date.fillna("?").to_numpy()
    models = {"blend3_hgb": lambda a, b, c, *_: blend_models(a, b, c, with_hgb=True), "cat": CAT["r7_cat_d6_lr01_2000"]}
    dedup = "--dedup" in sys.argv
    kept = {m: dict(np.load(OUT / f"r14_sim_{m}.npz")) for m in models}
    variants = [("conflicts kept", {m: {s: kept[m][str(s)] for s in SEEDS} for m in models})]
    if dedup:
        prev = {m: dict(np.load(OUT / f"r15_noconflict_sim_{m}.npz")) for m in models}
        variants.append(("conflicts dropped", {m: {s: prev[m][str(s)] for s in SEEDS} for m in models}))
    dropped = {}
    for m, fit_predict in models.items():
        dropped[m] = oof(drop_conflicts(add_sim_mates(fit_predict, dates, groups, z), groups, dedup), tx, y, groups)
        report(f"r1{6 if dedup else 5}_noconflict{'_dedup' if dedup else ''}_sim_{m}", dropped[m], y, groups)
    variants.append(("conflicts dropped + one row per track" if dedup else "conflicts dropped", dropped))

    def show(name, preds):
        aucs = [fold_auc(preds[s], y, groups, s) for s in SEEDS]
        print(f"{name:34s} mean={np.mean(aucs):.4f} " + " ".join(f"{s}={a:.4f}" for s, a in zip(SEEDS, aucs))
              + f"  singles={single_auc(preds, y, groups, singles):.4f}")

    for label, src in variants:
        print(f"--- {label}")
        show("  blend (ET+RF+HGB, sim mates)", src["blend3_hgb"])
        show("  CatBoost (sim mates)", src["cat"])
        show("  blend + 30% CatBoost", {s: 0.7 * rankdata(src["blend3_hgb"][s]) + 0.3 * rankdata(src["cat"][s]) for s in SEEDS})


if __name__ == "__main__":
    main()
