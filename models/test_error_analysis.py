"""Self-check on hand-made and fake data. Run from the repo root: python -m models.test_error_analysis"""
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from eval.test_harness import fake_contract
from models.error_analysis import by_group, chapter_groups, run

y = np.array([1, 1, 0, 0, 1, 0])
p = np.array([0.9, 0.4, 0.2, 0.7, 0.8, 0.1])
g = {"a": [1, 1, 1, 1, 0, 0], "b": [0, 0, 0, 0, 1, 1], "empty": [0] * 6, "one_class": [1, 1, 0, 0, 0, 0]}
rows = {r["area"]: r for r in by_group(y, p, g, min_n=2)}
assert "empty" not in rows
# group a: predictions 1,0,0,1 vs truth 1,1,0,0 -> one missed success, one missed failure
assert (rows["a"]["n"], rows["a"]["wrong_fail"], rows["a"]["wrong_success"], rows["a"]["accuracy_at_0.5"]) == (4, 1, 1, 0.5)
assert rows["a"]["roc_auc"] == 0.75 and rows["b"]["roc_auc"] == 1.0
assert rows["one_class"]["roc_auc"] == "" and rows["one_class"]["pos_rate"] == 1.0
assert by_group(y, p, {"a": g["a"]}, min_n=30)[0]["roc_auc"] == "", "small groups get no AUC"

# A and B share the label "infectious" and must be merged; OOV is not an area
feats = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
cg = chapter_groups(feats, ["icd_chapter_A", "icd_chapter_B", "icd_chapter_C", "icd_chapter_OOV"])
assert set(cg) == {"infectious", "cancer"} and cg["infectious"].tolist() == [True, True, False]

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    fake_contract(t)
    (t / "feature_names").mkdir()
    (t / "feature_names" / "disease_features.json").write_text(json.dumps(
        ["icd_chapter_C", "icd_chapter_I", "icd_category_C34", "icd_chapter_OOV"]))
    out = run(folder=t, model="logreg", blocks=("drug", "disease"), path=t / "e.csv")
    assert {r["area"] for r in out} == {"ALL", "cancer", "circulatory"} and {r["phase"] for r in out} == {1, 2, 3}
    run(folder=t, model="logreg", path=t / "e.csv")
    assert len(list(csv.DictReader(open(t / "e.csv")))) == 2 * len(out), "second run must append"
print("ok")

# decision_run: 3 phases x 3 loss multiples, thresholds come from valid, nothing killed means net 0
from models.decision_run import LOSS_MULTIPLES, run as decision_run
with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    fake_contract(t)
    rows = decision_run(folder=t, model="logreg", blocks=("drug",), path=t / "d.csv")
    assert len(rows) == 3 * len(LOSS_MULTIPLES)
    assert all(r["n_killed"] == r["true_kills"] + r["wrong_kills"] <= r["n_test"] for r in rows)
    assert all(r["net_musd"] == 0 for r in rows if r["n_killed"] == 0)
    assert all(round(r["true_kills"] * r["trial_cost_musd"] * 1.0
                     - r["wrong_kills"] * r["trial_cost_musd"] * r["loss_multiple"], 2) == r["net_musd"] for r in rows)
print("ok")
