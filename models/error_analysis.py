"""Error analysis by disease area (Person B): where does the model do well or badly on the test split?

Run from the repo root:  python -m models.error_analysis --model xgboost --blocks disease
Appends to results/error_analysis.csv. Disease area = ICD-10 chapter letter, read from the
`icd_chapter_*` columns of the disease block. A trial with codes in two chapters counts in both.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.harness import load_contract
from models.baselines import MODELS

OUT = Path(__file__).resolve().parent.parent / "results" / "error_analysis.csv"
COLUMNS = ["date", "config", "phase", "area", "n", "pos_rate", "mean_pred", "pr_auc", "roc_auc",
           "accuracy_at_0.5", "wrong_fail", "wrong_success"]
# shortcut: letter-level chapters, so D mixes benign neoplasms with blood disorders; split D at D50 if that matters
AREAS = {"A": "infectious", "B": "infectious", "C": "cancer", "D": "neoplasm/blood", "E": "endocrine/metabolic",
         "F": "mental", "G": "nervous system", "H": "eye/ear", "I": "circulatory", "J": "respiratory",
         "K": "digestive", "L": "skin", "M": "musculoskeletal", "N": "genitourinary", "O": "pregnancy",
         "P": "perinatal", "Q": "congenital", "R": "symptoms", "S": "injury", "T": "injury/poisoning",
         "Y": "external causes", "Z": "health-status factors"}


def by_group(y_true, p_success, groups, min_n=30):
    """One row per group. groups: {name: boolean mask}. AUCs are blank below min_n or with one class."""
    y, p = np.asarray(y_true), np.asarray(p_success, dtype=float)
    rows = []
    for name, m in groups.items():
        m = np.asarray(m, dtype=bool)
        assert len(m) == len(y), f"mask for {name} has the wrong length"
        n = int(m.sum())
        if n == 0:
            continue
        yy, pp = y[m], p[m]
        ok = n >= min_n and 0 < yy.sum() < n
        pred = pp >= 0.5
        rows.append({
            "area": name, "n": n, "pos_rate": round(float(yy.mean()), 4), "mean_pred": round(float(pp.mean()), 4),
            "pr_auc": round(float(average_precision_score(yy, pp)), 4) if ok else "",
            "roc_auc": round(float(roc_auc_score(yy, pp)), 4) if ok else "",
            "accuracy_at_0.5": round(float((pred == yy).mean()), 4),
            "wrong_fail": int((~pred & (yy == 1)).sum()),     # predicted failure, trial succeeded
            "wrong_success": int((pred & (yy == 0)).sum()),   # predicted success, trial failed
        })
    return sorted(rows, key=lambda r: -r["n"])


def chapter_groups(disease_features, names):
    """{area label: mask} from the icd_chapter_* columns. Letters that share a label are merged."""
    groups = {}
    for j, name in enumerate(names):
        if name.startswith("icd_chapter_") and name != "icd_chapter_OOV":
            label = AREAS.get(name[len("icd_chapter_"):], name)
            groups[label] = groups.get(label, False) | (disease_features[:, j] > 0)
    return groups


def run(folder="data/processed", model="xgboost", blocks=("disease",), seed=0, path=OUT):
    folder = Path(folder)
    d = load_contract(folder)
    names = json.loads((folder / "feature_names" / "disease_features.json").read_text(encoding="utf-8"))
    groups = chapter_groups(d["disease_features"], names)
    X = np.hstack([d[f"{b}_features"].reshape(len(d["labels"]), -1) for b in blocks])
    y, phase = d["labels"], d["phase"]
    cfg = json.dumps({"model": model, "blocks": list(blocks), "seed": seed, "split": "test"}, sort_keys=True)
    out = []
    for p in (1, 2, 3):
        tr = d["split_train"][phase[d["split_train"]] == p]
        te = d["split_test"][phase[d["split_test"]] == p]
        if len(te) == 0 or len(set(y[tr].tolist())) < 2:
            continue
        score = MODELS[model](seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        g = {"ALL": np.ones(len(te), dtype=bool), **{k: m[te] for k, m in groups.items()}}
        out += [{"date": date.today().isoformat(), "config": cfg, "phase": p, **r} for r in by_group(y[te], score, g)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:  # append-only, like results.csv
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=list(MODELS))
    ap.add_argument("--blocks", default="disease", help="comma list of drug,disease,text")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for r in run(model=a.model, blocks=tuple(a.blocks.split(",")), seed=a.seed):
        print(r["phase"], f"{r['area']:24}", r["n"], r["pos_rate"], r["mean_pred"], r["pr_auc"], r["roc_auc"],
              r["accuracy_at_0.5"], r["wrong_fail"], r["wrong_success"])
