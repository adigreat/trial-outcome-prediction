"""Evaluation harness (Person B). Contract: HANDOFF.md section 5. Results format: section 11."""
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CONTRACT = ["drug_features", "disease_features", "text_features", "trial_ids",
            "labels", "phase", "split_train", "split_valid", "split_test"]
COLUMNS = ["date", "config", "phase", "pr_auc", "roc_auc", "n", "pos_rate", "notes"]
RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def load_contract(folder="data/processed"):
    """Load the section 5 arrays and fail loudly on misalignment (the silent-bug zone)."""
    d = {k: np.load(Path(folder) / f"{k}.npy", allow_pickle=False) for k in CONTRACT}
    n = len(d["trial_ids"])
    for k in ("drug_features", "disease_features", "text_features", "labels", "phase"):
        assert len(d[k]) == n, f"{k} has {len(d[k])} rows, trial_ids has {n}"
    # Phase 1/2 and 2/3 trials appear in two phase files, so the key is (trial_id, phase), not trial_id
    assert len(set(zip(d["trial_ids"].tolist(), d["phase"].tolist()))) == n, "duplicate (trial_id, phase) rows"
    assert set(np.unique(d["labels"])) <= {0, 1}, "labels must be 0/1"
    assert set(np.unique(d["phase"])) <= {1, 2, 3}, "phase must be 1/2/3"
    tr, va, te = d["split_train"], d["split_valid"], d["split_test"]
    for s in (tr, va, te):
        assert s.min() >= 0 and s.max() < n, "split index out of range"
    assert not (set(tr.tolist()) & set(va.tolist()) or set(tr.tolist()) & set(te.tolist())
                or set(va.tolist()) & set(te.tolist())), "train/valid/test overlap"
    return d


def evaluate(y_true, y_score, phase, config, notes="", seed=None, path=RESULTS):
    """Score per phase, append one row per phase to the results CSV, return the rows.

    config: dict or str describing the run. Seed is recorded inside it.
    """
    y_true, y_score, phase = map(np.asarray, (y_true, y_score, phase))
    assert len(y_true) == len(y_score) == len(phase), "length mismatch"
    cfg = dict(config) if isinstance(config, dict) else {"name": str(config)}
    if seed is not None:
        cfg["seed"] = seed
    rows = []
    for p in sorted(set(phase.tolist())):
        m = phase == p
        y = y_true[m]
        # ROC-AUC and PR-AUC are undefined with one class present
        both = 0 < y.sum() < len(y)
        rows.append({
            "date": date.today().isoformat(), "config": json.dumps(cfg, sort_keys=True),
            "phase": int(p),
            "pr_auc": round(float(average_precision_score(y, y_score[m])), 4) if both else "",
            "roc_auc": round(float(roc_auc_score(y, y_score[m])), 4) if both else "",
            "n": int(m.sum()), "pos_rate": round(float(y.mean()), 4), "notes": notes,
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:  # append-only, never overwrite
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return rows
