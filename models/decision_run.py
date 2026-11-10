"""Decision analysis on real predictions (Person B).

Run from the repo root:  python -m models.decision_run --model xgboost --blocks disease
For each phase: fit on train, pick the kill threshold that maximises net value on VALID, then report
what that threshold does on TEST. Appends to results/decision_analysis.csv. Net 0 = "kill nothing".

Costs (HANDOFF section 9, 2026-11-10). Money saved by stopping a trial that would have failed = the
average per-study cost by phase from the ERG report for HHS ASPE (Sertkaya et al., 2014, Medidata
contract data from 2004 on, US, not inflation-adjusted). The value lost by stopping a trial that
would have succeeded has no source we trust, so it is a stated assumption: a multiple of the trial
cost, run at several values. Read the result as "how costly must a wrong kill be before the model
stops being useful", not as a dollar forecast.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from eval.decision import best_threshold, threshold_table
from eval.harness import load_contract
from models.baselines import MODELS, fit, load_blocks

OUT = Path(__file__).resolve().parent.parent / "results" / "decision_analysis.csv"
TRIAL_COST_MUSD = {1: 3.80, 2: 13.35, 3: 19.89}
LOSS_MULTIPLES = (1, 3, 10)
COLUMNS = ["date", "config", "phase", "trial_cost_musd", "loss_multiple", "threshold_from_valid", "n_test",
           "n_killed", "true_kills", "wrong_kills", "failures_caught", "successes_lost", "net_musd"]


def run(folder="data/processed", model="xgboost", blocks=("disease",), seed=0, path=OUT):
    d = load_contract(folder)
    X, dims = load_blocks(d, blocks)
    y, phase = d["labels"], d["phase"]
    cfg = json.dumps({"model": model, "blocks": list(blocks), "seed": seed}, sort_keys=True)
    out = []
    for p in (1, 2, 3):
        tr, va, te = (d[f"split_{s}"][phase[d[f"split_{s}"]] == p] for s in ("train", "valid", "test"))
        if min(len(va), len(te)) == 0 or len(set(y[tr].tolist())) < 2:
            continue
        m = fit(model, seed, X, y, tr, va, dims)
        p_va, p_te = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
        cost = TRIAL_COST_MUSD[p]
        for mult in LOSS_MULTIPLES:
            # threshold chosen on valid only, so the test row is not tuned on test
            t = best_threshold(threshold_table(y[va], p_va, cost, cost * mult))["threshold"]
            r = threshold_table(y[te], p_te, cost, cost * mult, thresholds=[t])[0]
            out.append({"date": date.today().isoformat(), "config": cfg, "phase": p, "trial_cost_musd": cost,
                        "loss_multiple": mult, "threshold_from_valid": t, "n_test": len(te),
                        "n_killed": r["n_killed"], "true_kills": r["true_kills"], "wrong_kills": r["wrong_kills"],
                        "failures_caught": r["failures_caught"], "successes_lost": r["successes_lost"],
                        "net_musd": round(r["net"], 2)})
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
    ap.add_argument("--model", default="xgboost", choices=[*MODELS, "fusion"])
    ap.add_argument("--blocks", default="disease", help="comma list of drug,disease,text")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for r in run(model=a.model, blocks=tuple(a.blocks.split(",")), seed=a.seed):
        print(*(r[k] for k in COLUMNS[2:]))
