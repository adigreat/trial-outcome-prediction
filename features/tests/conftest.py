"""Shared fixtures: a tiny synthetic TOP-format dataset, written as the same 9 CSVs."""
import csv
import json

import pytest

from features import config as C

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
ETHANOL = "CCO"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

CRITERIA = """
        Inclusion Criteria:

          -  Age >= 18 years.

          -  Histological diagnosis of metastatic disease not
             amenable to curative therapy

        Exclusion Criteria:

          -  Pregnancy
"""


def row(nctid, label, smiles=(ETHANOL,), icd=("C34.1",), criteria=CRITERIA, phase="phase 1",
        status="completed", why_stop=""):
    return {
        "nctid": nctid, "status": status, "why_stop": why_stop, "label": str(label), "phase": phase,
        "diseases": json.dumps(["lung cancer"]).replace('"', "'"),
        "icdcodes": str([str(list(icd))]),
        "drugs": str(["drug"]), "smiless": str(list(smiles)), "criteria": criteria,
    }


def write_raw(folder, rows_by_file):
    """rows_by_file: {(phase_name, split): [row dicts]}; missing files are written empty."""
    folder.mkdir(parents=True, exist_ok=True)
    cols = ["nctid", "status", "why_stop", "label", "phase", "diseases", "icdcodes", "drugs", "smiless", "criteria"]
    for pn in C.PHASES:
        for s in C.SPLITS:
            with open(folder / f"phase_{pn}_{s}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, cols)
                w.writeheader()
                w.writerows(rows_by_file.get((pn, s), []))
    return folder


@pytest.fixture
def small_raw(tmp_path):
    rows = {
        ("I", "train"): [row("NCT00000003", 1, (ETHANOL, ASPIRIN), ("C34.1", "C34.2")),
                         row("NCT00000001", 0, (CAFFEINE,), ("C34.1",)),
                         row("NCT00000010", 1, (ETHANOL,), ("I10",), phase="phase 1/phase 2")],
        ("I", "valid"): [row("NCT00000004", 1, ("not_a_smiles",), ("C34.1",))],
        ("I", "test"): [row("NCT00000005", 0, (), ("Z99.9",), criteria="")],
        ("II", "train"): [row("NCT00000010", 1, (ETHANOL,), ("I10",), phase="phase 1/phase 2"),
                          row("NCT00000011", 1, (ASPIRIN,), ("C34.1",))],
        ("II", "test"): [row("NCT00000012", 0, (ASPIRIN,), ("C34.9",))],
        ("III", "train"): [row("NCT00000020", 1, (CAFFEINE,), ("C34.1", "I10"))],
        ("III", "test"): [row("NCT00000021", 1, (ETHANOL,), ("C50.1",))],
    }
    return write_raw(tmp_path / "raw", rows), rows


@pytest.fixture
def small_master(small_raw):
    from features.data import load_master
    raw, _ = small_raw
    return load_master(raw, strict_version=False, check_counts=False)
