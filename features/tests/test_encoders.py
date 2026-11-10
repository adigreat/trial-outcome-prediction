import numpy as np
import pytest

from features import disease, fingerprints
from features.tests.conftest import ASPIRIN, CAFFEINE, ETHANOL

# ----------------------------------------------------------------------------- fingerprints


def test_fingerprint_matches_rdkit_directly():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    want = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(
        Chem.MolFromSmiles(ASPIRIN))
    assert np.array_equal(fingerprints.smiles_to_bits(ASPIRIN), want)
    assert fingerprints.smiles_to_bits("not_a_smiles") is None


def test_fingerprint_block(small_master):
    m, _ = small_master
    X, diag, meta = fingerprints.encode(m)
    assert X.shape == (len(m), 2048) and X.dtype == np.float32
    assert set(np.unique(X)) <= {0.0, 1.0}
    rows = {n: i for i, n in enumerate(m.nctid)}
    eth, asp = fingerprints.smiles_to_bits(ETHANOL), fingerprints.smiles_to_bits(ASPIRIN)
    assert np.array_equal(X[rows["NCT00000003"]], (eth | asp).astype(np.float32))   # union of two drugs
    assert X[rows["NCT00000004"]].sum() == 0                     # only an invalid SMILES -> zeros
    assert X[rows["NCT00000005"]].sum() == 0                     # no SMILES -> zeros
    i4, i5 = rows["NCT00000004"], rows["NCT00000005"]
    assert (diag["n_smiles"][i4], diag["n_valid_smiles"][i4]) == (1, 0)
    assert (diag["n_smiles"][i5], diag["n_valid_smiles"][i5]) == (0, 0)
    assert meta["invalid_unique_smiles"] == 1 and len(meta["feature_names"]) == 2048
    X2, _, _ = fingerprints.encode(m)
    assert np.array_equal(X, X2)                                 # deterministic


def test_fingerprints_do_not_import_torch():
    import subprocess
    import sys
    code = ("import sys, features.fingerprints, features.build, features.assemble; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ----------------------------------------------------------------------------- disease


def test_disease_vocab_fit_on_train_only(small_master):
    m, _ = small_master
    vocab, _ = disease.fit_vocab(m, min_count=1)
    train_codes = {c for codes in m[m.split == "train"].icd_codes for c in codes}
    assert set(vocab["code"]) == train_codes
    assert "Z99.9" not in vocab["code"] and "C50.1" not in vocab["code"]     # test-only codes
    assert "Z" not in vocab["chapter"]


def test_disease_block_hierarchy_and_oov(small_master):
    m, _ = small_master
    X, diag, meta = disease.encode(m, min_count=1)
    names = meta["feature_names"]
    col = {n: j for j, n in enumerate(names)}
    rows = {n: i for i, n in enumerate(m.nctid)}
    r = X[rows["NCT00000012"]]           # test trial, C34.9 unseen, but its parents are known
    assert r[col["icd_chapter_C"]] == 1 and r[col["icd_category_C34"]] == 1
    assert r[col["icd_code_OOV"]] == 1 and r[col["icd_category_OOV"]] == 0
    r = X[rows["NCT00000005"]]           # Z99.9: nothing known at any level
    assert r[col["icd_chapter_OOV"]] == r[col["icd_category_OOV"]] == r[col["icd_code_OOV"]] == 1
    assert r.sum() == 3
    r = X[rows["NCT00000020"]]           # two codes, both known
    assert r[col["icd_code_C34.1"]] == r[col["icd_code_I10"]] == 1 and r[col["icd_code_OOV"]] == 0
    assert diag["n_icd"][rows["NCT00000020"]] == 2
    assert X.shape[1] == len(names)


def test_disease_counts_unique_trials_not_rows(small_master):
    m, _ = small_master
    # I10 is carried by NCT00000010 (listed in phase I and II train) and NCT00000020 -> 2 trials, not 3
    vocab, _ = disease.fit_vocab(m, min_count=3)
    assert "I10" not in vocab["code"]
    vocab, _ = disease.fit_vocab(m, min_count=2)
    assert "I10" in vocab["code"]
