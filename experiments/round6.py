"""Round 6a: neural nets on the 'all' features (ideas 1-3). Run with .venv/bin/python."""

import sys

from common import load, oof, report
from features import build
from nets import plr_mlp, sk_mlp
from round2 import single_auc

CONFIGS = {
    "r6_skmlp": sk_mlp,
    "r6_plr_e20": plr_mlp(epochs=20),
    "r6_plr_e40": plr_mlp(epochs=40),
    "r6_plr_e80": plr_mlp(epochs=80),
    "r6_plr_e40_ls0.1": plr_mlp(epochs=40, smoothing=0.1),
    "r6_plr_e40_ls0.2": plr_mlp(epochs=40, smoothing=0.2),
    "r6_plr_e40_drop0.5_wd1e-2": plr_mlp(epochs=40, dropout=0.5, weight_decay=1e-2),
}


def main():
    train, test, y, groups = load()
    features, _ = build(train, test, "all")
    singles = (features.uri_count == 1).to_numpy()
    for name in sys.argv[1:] or CONFIGS:
        preds = oof(CONFIGS[name], features, y, groups)
        report(name, preds, y, groups)
        print(f"{'':40s} singles_only={single_auc(preds, y, groups, singles):.4f}", flush=True)


if __name__ == "__main__":
    main()
