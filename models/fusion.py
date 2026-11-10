"""Fusion MLP (Person B): one small encoder per feature block, concatenated, then an MLP head.

Not run directly. Use it through the other CLIs:
  python -m models.baselines --models fusion --blocks drug,disease,text --seed 0
  python -m models.error_analysis --model fusion --blocks drug,disease,text
  python -m models.decision_run --model fusion --blocks drug,disease,text
Needs PyTorch (not in the root requirements.txt; CPU is enough, a GPU is used if present).

The official valid split picks the stopping epoch, so valid scores for this model are optimistic.
Test is never seen during training. Settings below were fixed before the first test run, not tuned.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

HIDDEN, DROP_IN, DROP, LR, WEIGHT_DECAY, EPOCHS, BATCH, PATIENCE = 128, 0.2, 0.5, 1e-3, 1e-2, 60, 128, 8


class _Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.enc = nn.ModuleList(
            nn.Sequential(nn.Dropout(DROP_IN), nn.Linear(k, HIDDEN), nn.ReLU(), nn.Dropout(DROP)) for k in dims)
        self.head = nn.Sequential(nn.Linear(HIDDEN * len(dims), HIDDEN), nn.ReLU(), nn.Dropout(DROP), nn.Linear(HIDDEN, 1))

    def forward(self, xs):
        return self.head(torch.cat([e(x) for e, x in zip(self.enc, xs)], 1)).squeeze(1)


class FusionMLP:
    """sklearn-shaped (fit / predict_proba). dims = width of each block inside X, in column order."""

    def __init__(self, dims, seed=0):
        assert all(k > 0 for k in dims), f"empty feature block: {dims}"
        self.dims, self.seed = list(dims), seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _split(self, X):
        X = np.asarray(X, dtype=np.float32)
        assert X.shape[1] == sum(self.dims), f"X has {X.shape[1]} columns, blocks add up to {sum(self.dims)}"
        cuts = np.cumsum(self.dims)[:-1]
        return [torch.tensor((b - m) / s, device=self.device) for b, (m, s) in zip(np.split(X, cuts, axis=1), self.scale)]

    def fit(self, X, y, X_val, y_val):
        assert len(X_val) > 0, "the fusion model needs a valid split for early stopping"
        torch.manual_seed(self.seed)
        # 0/1 blocks (fingerprints, ICD multi-hot) stay as they are; dense blocks (text) are standardised on train
        self.scale = [(0.0, 1.0) if np.isin(b, (0, 1)).all() else (b.mean(0), b.std(0) + 1e-6)
                      for b in np.split(np.asarray(X, dtype=np.float32), np.cumsum(self.dims)[:-1], axis=1)]
        xs, xv = self._split(X), self._split(X_val)
        yt = torch.tensor(np.asarray(y, dtype=np.float32), device=self.device)
        yv = np.asarray(y_val)
        yv_t = torch.tensor(yv.astype(np.float32), device=self.device)
        both = 0 < yv.sum() < len(yv)
        self.net = _Net(self.dims).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss = nn.functional.binary_cross_entropy_with_logits
        best, state, bad = -np.inf, None, 0
        for _ in range(EPOCHS):
            self.net.train()
            perm = torch.randperm(len(yt), device=self.device)
            for i in range(0, len(yt), BATCH):
                k = perm[i:i + BATCH]
                opt.zero_grad()
                loss(self.net([x[k] for x in xs]), yt[k]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                out = self.net(xv)
            # ROC-AUC on valid; it is undefined with one class, so fall back to the loss
            score = roc_auc_score(yv, out.cpu().numpy()) if both else -float(loss(out, yv_t))
            if score > best:
                best, state, bad = score, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net(self._split(X))).cpu().numpy()
        return np.column_stack([1 - p, p])
