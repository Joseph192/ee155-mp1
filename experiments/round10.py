"""Round 10 (ideas 11-15): brainstorm features, through trees, CatBoost and the PLR net. Run with .venv/bin/python.

Baselines on 'all': ET+RF 1-seed 0.8503, PLR e20 0.8310, CatBoost lr0.01 0.8466, full blend 0.8514.
"""

import sys

from common import load, oof, report
from features import build
from nets import plr_mlp
from round2 import single_auc
from round3 import blend_models
from round7 import CONFIGS as CAT


def main():
    train, test, y, groups = load()
    variants = sys.argv[1:] or ["eng_vocal", "eng_era", "eng_dance", "eng_misc", "eng"]
    runs = [(v, "etrf", lambda tx, ty, vx, *_: blend_models(tx, ty, vx, model_seeds=(155,))) for v in variants]
    runs += [("eng", "plr_e20", plr_mlp(epochs=20)), ("eng", "cat_lr01", CAT["r7_cat_d6_lr01_2000"]),
             ("eng", "blend3_hgb", lambda tx, ty, vx, *_: blend_models(tx, ty, vx, with_hgb=True))]
    for variant, model, fit_predict in runs:
        features, _ = build(train, test, variant)
        singles = (features.uri_count == 1).to_numpy()
        preds = oof(fit_predict, features, y, groups)
        report(f"r10_{variant}_{model}", preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
