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
        # fusion MLP through the same runner: 1 model x 3 phases x (valid + test)
        run(folder=t, blocks=("drug", "disease", "text"), seed=0, path=t / "f.csv", models=("fusion",))
        assert len(list(csv.DictReader(open(t / "f.csv")))) == 6

    # the fusion MLP must learn a signal that needs two blocks (XOR of a 0/1 column and a dense column's sign)
    import numpy as np
    from models.fusion import FusionMLP
    r = np.random.default_rng(0)
    X = np.hstack([r.integers(0, 2, (600, 4)), r.normal(size=(600, 5)) * 50 + 7])
    y = (X[:, 0] > 0) ^ (X[:, 4] > 7)
    m = FusionMLP([4, 5], seed=0).fit(X[:400], y[:400], X[400:500], y[400:500])
    p = m.predict_proba(X[500:])
    assert p.shape == (100, 2) and ((p[:, 1] >= 0.5) == y[500:]).mean() > 0.9, ((p[:, 1] >= 0.5) == y[500:]).mean()
    print("ok")


if __name__ == "__main__":
    test_all()
