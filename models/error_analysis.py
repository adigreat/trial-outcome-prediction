"""Error analysis by disease area (owner: Ananya since 2026-11-10): where does the model do well or badly
on the test split?

Run from the repo root:  python -m models.error_analysis --model fusion --blocks drug,disease,text
Appends to results/error_analysis.csv. Disease area = ICD-10 chapter, read from the disease block:
- Chapter letter from the `icd_chapter_*` columns. A trial with codes in two areas counts in both.
- Letter D is split with the `icd_category_D*` columns: D00-D49 (in situ, benign and uncertain
  neoplasms) and D50-D89 (blood and immune). A trial whose only D codes are categories outside the
  train vocabulary goes to "D (unresolved)".
- Chapters P (perinatal) and Z (health-status factors) count only for trials that have no code in any
  other chapter. In TOP they are almost always side effects of matching a disease name to ICD codes:
  "diabetes" also pulls in P70.2 (neonatal diabetes) and Z83.3 (family history of diabetes), "HIV" pulls
  in Z21 and Z71.7, "cystic fibrosis" pulls in P09.4. Counting them would list the same diabetes or
  HIV trials a second time under a misleading name (HANDOFF section 10, answered 2026-11-11).
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.harness import load_contract
from models.baselines import MODELS, fit, load_blocks

OUT = Path(__file__).resolve().parent.parent / "results" / "error_analysis.csv"
COLUMNS = ["date", "config", "phase", "area", "n", "pos_rate", "mean_pred", "pr_auc", "roc_auc",
           "accuracy_at_0.5", "wrong_fail", "wrong_success"]
AREAS = {"A": "infectious", "B": "infectious", "C": "cancer", "E": "endocrine/metabolic",
         "F": "mental", "G": "nervous system", "H": "eye/ear", "I": "circulatory", "J": "respiratory",
         "K": "digestive", "L": "skin", "M": "musculoskeletal", "N": "genitourinary", "O": "pregnancy",
         "P": "perinatal (only code)", "Q": "congenital", "R": "symptoms", "S": "injury", "T": "injury/poisoning",
         "Y": "external causes", "Z": "health-status factors (only code)"}
D_LOW, D_HIGH, D_UNRESOLVED = "neoplasm in situ/benign (D00-D49)", "blood/immune (D50-D89)", "D (unresolved)"
SECONDARY = ("P", "Z")   # counted only when the trial has no other chapter


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
    """{area label: mask}. Letters that share a label are merged. See the module docstring for D, P and Z."""
    X = np.asarray(disease_features)
    chap = {n[len("icd_chapter_"):]: X[:, j] > 0 for j, n in enumerate(names)
            if n.startswith("icd_chapter_") and n != "icd_chapter_OOV"}
    if not chap:
        return {}
    primary = np.zeros(len(X), dtype=bool)
    for letter, m in chap.items():
        if letter not in SECONDARY:
            primary |= m
    groups = {}

    def add(label, m):
        groups[label] = groups.get(label, np.zeros(len(X), dtype=bool)) | m

    for letter, m in chap.items():
        if letter == "D":
            low = np.zeros(len(X), dtype=bool)
            high = np.zeros(len(X), dtype=bool)
            for j, n in enumerate(names):
                if n.startswith("icd_category_D") and n[len("icd_category_D"):][:2].isdigit():
                    (low if int(n[len("icd_category_D"):][:2]) < 50 else high)[:] |= X[:, j] > 0
            add(D_LOW, m & low)
            add(D_HIGH, m & high)
            add(D_UNRESOLVED, m & ~low & ~high)
        elif letter in SECONDARY:
            add(AREAS[letter], m & ~primary)
        else:
            add(AREAS.get(letter, f"icd_chapter_{letter}"), m)
    return groups


def run(folder="data/processed", model="xgboost", blocks=("disease",), seed=0, path=OUT):
    folder = Path(folder)
    d = load_contract(folder)
    names = json.loads((folder / "feature_names" / "disease_features.json").read_text(encoding="utf-8"))
    groups = chapter_groups(d["disease_features"], names)
    X, dims = load_blocks(d, blocks)
    y, phase = d["labels"], d["phase"]
    cfg = json.dumps({"model": model, "blocks": list(blocks), "seed": seed, "split": "test"}, sort_keys=True)
    out = []
    for p in (1, 2, 3):
        tr = d["split_train"][phase[d["split_train"]] == p]
        te = d["split_test"][phase[d["split_test"]] == p]
        if len(te) == 0 or len(set(y[tr].tolist())) < 2:
            continue
        va = d["split_valid"][phase[d["split_valid"]] == p]
        score = fit(model, seed, X, y, tr, va, dims).predict_proba(X[te])[:, 1]
        g = {"ALL": np.ones(len(te), dtype=bool), **{k: m[te] for k, m in groups.items()}}
        out += [{"date": date.today().isoformat(), "config": cfg, "phase": p, **r} for r in by_group(y[te], score, g)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:  # append-only, like results.csv
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=[*MODELS, "fusion"])
    ap.add_argument("--blocks", default="disease", help="comma list of drug,disease,text")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for r in run(model=a.model, blocks=tuple(a.blocks.split(",")), seed=a.seed):
        print(r["phase"], f"{r['area']:24}", r["n"], r["pos_rate"], r["mean_pred"], r["pr_auc"], r["roc_auc"],
              r["accuracy_at_0.5"], r["wrong_fail"], r["wrong_success"])
