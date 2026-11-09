"""Self-check on fake data. Run from the repo root: python -m models.test_baselines"""
import csv
import tempfile
from pathlib import Path

from eval.test_harness import fake_contract
from models.baselines import run


def test_all():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        fake_contract(t)
        run(folder=t, blocks=("drug", "disease", "text"), seed=0, path=t / "r.csv")
        rows = list(csv.DictReader(open(t / "r.csv")))
        # 2 models x 3 phases x (valid + test)
        assert len(rows) == 12, len(rows)
        assert {r["phase"] for r in rows} == {"1", "2", "3"}
        assert all(0.0 <= float(r["roc_auc"]) <= 1.0 for r in rows if r["roc_auc"])
    print("ok")


if __name__ == "__main__":
    test_all()
