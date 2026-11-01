"""Neural nets for tabular data: a bagged sklearn MLP and a PyTorch MLP with periodic (PLR) numeric embeddings.

PLR = Periodic -> Linear -> ReLU per feature (Gorishniy et al. 2022, "On Embeddings for Numerical Features").
Run with experiments/.venv/bin/python (needs torch).
"""

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from common import build_forest_preprocessor

torch.set_num_threads(4)


def scaler(columns):
    return make_pipeline(build_forest_preprocessor(columns), QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=155))


def sk_mlp(train_x, train_y, valid_x, *_, seeds=(0, 1, 2, 3, 4)):
    prep = scaler(train_x.columns)
    tx, vx = prep.fit_transform(train_x), prep.transform(valid_x)
    preds = []
    for s in seeds:
        # Early stopping on an internal split of the training fold only; the outer validation fold is never seen.
        model = MLPClassifier((64, 32), alpha=1e-2, batch_size=64, learning_rate_init=1e-3, max_iter=300,
                              early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=s)
        preds.append(model.fit(tx, train_y).predict_proba(vx)[:, 1])
    return np.mean(preds, axis=0)


class PLR(nn.Module):
    def __init__(self, n_features, frequencies=16, dim=8, sigma=0.5):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_features, frequencies) * sigma)
        self.w = nn.Parameter(torch.randn(n_features, 2 * frequencies, dim) / np.sqrt(2 * frequencies))
        self.b = nn.Parameter(torch.zeros(n_features, dim))

    def forward(self, x):
        v = 2 * np.pi * self.c[None] * x[..., None]
        v = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        return torch.relu(torch.einsum("bnk,nkd->bnd", v, self.w) + self.b).flatten(1)


class PLRNet(nn.Module):
    def __init__(self, n_features, hidden=256, dropout=0.3, dim=8):
        super().__init__()
        self.net = nn.Sequential(
            PLR(n_features, dim=dim),
            nn.Linear(n_features * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def plr_mlp(epochs=40, smoothing=0.0, seeds=(0, 1, 2, 3, 4), lr=1e-3, weight_decay=1e-4, dropout=0.3):
    def fit_predict(train_x, train_y, valid_x, *_):
        prep = scaler(train_x.columns)
        tx = torch.tensor(prep.fit_transform(train_x), dtype=torch.float32)
        vx = torch.tensor(prep.transform(valid_x), dtype=torch.float32)
        ty = torch.tensor(train_y, dtype=torch.float32) * (1 - smoothing) + smoothing / 2
        preds = []
        for s in seeds:
            torch.manual_seed(s)
            model = PLRNet(tx.shape[1], dropout=dropout)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            steps = epochs * int(np.ceil(len(tx) / 128))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
            for _ in range(epochs):
                model.train()
                for batch in torch.randperm(len(tx)).split(128):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(tx[batch]), ty[batch])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            model.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(model(vx)).numpy())
        return np.mean(preds, axis=0)
    return fit_predict


if __name__ == "__main__":
    # Self-check: PLR net must learn a non-monotone 1-D rule a linear model cannot.
    rng = np.random.default_rng(0)
    import pandas as pd
    x = pd.DataFrame({"a": rng.uniform(-3, 3, 2000), "time_signature": 4.0, "mode": 1.0, "release_weekday": 4.0})
    y = (np.abs(x.a) < 1).astype(int).to_numpy()
    p = plr_mlp(epochs=30, seeds=(0,))(x[:1500], y[:1500], x[1500:])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[1500:], p)
    assert auc > 0.95, auc
    print(f"self-check ok, AUC={auc:.3f}")
