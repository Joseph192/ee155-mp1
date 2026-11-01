"""Round 8 (ideas 8-10): NN-boosting hybrids and PLR tuning. Run with .venv/bin/python.

8: GBDT leaf one-hots as an extra NN input next to PLR numerics.
9: NN learns a correction on top of HGB's logit (offset), trained on inner cross-fitted HGB logits.
"""

import sys

import numpy as np
import torch
from scipy.special import logit
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from torch import nn

from common import load, oof, report
from features import build
from nets import PLR, plr_mlp, scaler
from round2 import single_auc


class HybridNet(nn.Module):
    def __init__(self, n_features, n_extra=0, hidden=256, dropout=0.3, dim=8):
        super().__init__()
        self.plr = PLR(n_features, dim=dim)
        self.extra = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_extra, 64), nn.ReLU()) if n_extra else None
        width = n_features * dim + (64 if n_extra else 0)
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x, extra=None):
        h = self.plr(x) if self.extra is None else torch.cat([self.plr(x), self.extra(extra)], dim=1)
        return self.head(h).squeeze(-1)


def train_net(tx, ty, vx, t_extra=None, v_extra=None, t_offset=None, v_offset=None, epochs=20, seeds=range(5), lr=1e-3):
    t = lambda a: None if a is None else torch.tensor(np.asarray(a), dtype=torch.float32)
    tx, vx, ty, t_extra, v_extra = t(tx), t(vx), t(ty), t(t_extra), t(v_extra)
    t_offset = torch.zeros(len(tx)) if t_offset is None else t(t_offset)
    v_offset = torch.zeros(len(vx)) if v_offset is None else t(v_offset)
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        model = HybridNet(tx.shape[1], 0 if t_extra is None else t_extra.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(tx) / 128)))
        for _ in range(epochs):
            model.train()
            for b in torch.randperm(len(tx)).split(128):
                out = model(tx[b], None if t_extra is None else t_extra[b]) + t_offset[b]
                loss = nn.functional.binary_cross_entropy_with_logits(out, ty[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(vx, v_extra) + v_offset).numpy())
    return np.mean(preds, axis=0)


def leaf_net(train_x, train_y, valid_x, *_):
    prep = scaler(train_x.columns)
    tx, vx = prep.fit_transform(train_x), prep.transform(valid_x)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=155).fit(tx, train_y)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    t_leaf = enc.fit_transform(gb.apply(tx)[:, :, 0])
    v_leaf = enc.transform(gb.apply(vx)[:, :, 0])
    return train_net(tx, train_y, vx, t_leaf, v_leaf)


def hgb():
    return HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.03, max_iter=300,
                                          l2_regularization=1.0, early_stopping=False, random_state=155)


def residual_net(scale):
    def fit_predict(train_x, train_y, valid_x, *_):
        # Train rows get out-of-fold HGB logits; in-sample logits would leave the NN nothing honest to correct.
        t_off = np.empty(len(train_x))
        for fit, held in StratifiedKFold(5, shuffle=True, random_state=7).split(train_x, train_y):
            t_off[held] = logit(np.clip(hgb().fit(train_x.iloc[fit], train_y[fit]).predict_proba(train_x.iloc[held])[:, 1], 1e-4, 1 - 1e-4))
        v_off = logit(np.clip(hgb().fit(train_x, train_y).predict_proba(valid_x)[:, 1], 1e-4, 1 - 1e-4))
        prep = scaler(train_x.columns)
        return train_net(prep.fit_transform(train_x), train_y, prep.transform(valid_x), t_offset=scale * t_off, v_offset=scale * v_off, epochs=10)
    return fit_predict


CONFIGS = {
    "r8_plr_e10_10seeds": plr_mlp(epochs=10, seeds=range(10)),
    "r8_plr_e15_10seeds": plr_mlp(epochs=15, seeds=range(10)),
    "r8_plr_e30_lr3e-4": plr_mlp(epochs=30, lr=3e-4),
    "r8_leaf_net": leaf_net,
    "r8_residual_net_1.0": residual_net(1.0),
    "r8_residual_net_0.5": residual_net(0.5),
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
