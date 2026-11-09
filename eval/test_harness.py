"""Self-check on fake data. Run from the repo root: python -m eval.test_harness"""
import csv
import tempfile
from pathlib import Path

import numpy as np

from eval.harness import CONTRACT, evaluate, load_contract


def fake_contract(folder, n=300, seed=0):
    r = np.random.default_rng(seed)
    labels = r.integers(0, 2, n)
    a = {"drug_features": r.random((n, 8)), "disease_features": r.random((n, 4)),
         "text_features": r.random((n, 6)), "trial_ids": np.array([f"NCT{i:08d}" for i in range(n)]),
         "labels": labels, "phase": r.integers(1, 4, n),
         "split_train": np.arange(0, 200), "split_test": np.arange(200, n)}
    for k in CONTRACT:
        np.save(Path(folder) / f"{k}.npy", a[k])
    return a


def test_all():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        a = fake_contract(t)
        d = load_contract(t)
        assert len(d["labels"]) == 300

        # perfect scores -> 1.0 in every phase; random scores -> near the positive rate
        rows = evaluate(a["labels"], a["labels"].astype(float), a["phase"], {"m": "oracle"}, seed=0, path=t / "r.csv")
        assert [r["phase"] for r in rows] == [1, 2, 3] and all(r["pr_auc"] == 1.0 and r["roc_auc"] == 1.0 for r in rows)
        evaluate(a["labels"], np.random.default_rng(1).random(300), a["phase"], "random", path=t / "r.csv")
        out = list(csv.DictReader(open(t / "r.csv")))
        assert len(out) == 6, "second call must append, not overwrite"
        assert all(0.3 < float(r["roc_auc"]) < 0.7 for r in out[3:]), "random scores should sit near 0.5"

        # a phase with one class must not crash
        r1 = evaluate([1, 1, 1], [0.2, 0.5, 0.9], [1, 1, 1], "single", path=t / "r2.csv")
        assert r1[0]["pr_auc"] == ""

        # misaligned arrays must fail loudly
        np.save(t / "labels.npy", a["labels"][:-1])
        try:
            load_contract(t)
            raise SystemExit("misalignment was not caught")
        except AssertionError:
            pass
    print("ok")


if __name__ == "__main__":
    test_all()
