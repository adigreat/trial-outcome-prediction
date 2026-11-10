"""Floor baselines (Person B): logistic regression and XGBoost, one model per phase.

Run from the repo root:  python -m models.baselines --blocks drug
The fusion MLP uses the same runner:  python -m models.baselines --models fusion --blocks drug,disease,text
Every run is appended to results/results.csv through eval.harness.evaluate (valid and test rows).
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from eval.harness import evaluate, load_contract

MODELS = {
    "logreg": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
    "xgboost": lambda seed: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                          subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=4),
}


def fit(name, seed, X, y, tr, va, dims):
    """Fit one model on rows tr. Only the fusion MLP uses va (early stopping) and dims (block widths)."""
    if name == "fusion":
        from models.fusion import FusionMLP  # imported here so the baselines run without PyTorch
        return FusionMLP(dims, seed).fit(X[tr], y[tr], X[va], y[va])
    return MODELS[name](seed).fit(X[tr], y[tr])


def load_blocks(d, blocks):
    """(X, dims): the chosen blocks side by side, and the width of each."""
    parts = [d[f"{b}_features"].reshape(len(d["labels"]), -1) for b in blocks]
    return np.hstack(parts), [p.shape[1] for p in parts]


def run(folder="data/processed", blocks=("drug",), seed=0, path=None, notes="", models=tuple(MODELS)):
    d = load_contract(folder)
    X, dims = load_blocks(d, blocks)
    y, phase = d["labels"], d["phase"]
    kw = {"path": path} if path else {}
    for name in models:
        for p in (1, 2, 3):
            tr = d["split_train"][phase[d["split_train"]] == p]
            if len(set(y[tr].tolist())) < 2:  # nothing to fit on
                continue
            va = d["split_valid"][phase[d["split_valid"]] == p]
            model = fit(name, seed, X, y, tr, va, dims)
            for split in ("valid", "test"):
                idx = d[f"split_{split}"][phase[d[f"split_{split}"]] == p]
                if len(idx) == 0:
                    continue
                cfg = {"model": name, "blocks": list(blocks), "split": split}
                evaluate(y[idx], model.predict_proba(X[idx])[:, 1], phase[idx], cfg, notes=notes, seed=seed, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="drug", help="comma list of drug,disease,text")
    ap.add_argument("--models", default="logreg,xgboost", help="comma list of logreg,xgboost,fusion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    run(blocks=tuple(a.blocks.split(",")), seed=a.seed, notes=a.notes, models=tuple(a.models.split(",")))
