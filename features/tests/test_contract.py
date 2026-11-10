"""End to end: blocks -> data/processed contract -> Person B's own loader accepts it."""
import json

import numpy as np
import pytest

from features import disease, fingerprints
from features.assemble import CONTRACT, assemble
from features.blocks import load_block, save_block


def build(m, rep, tmp_path, blocks=("drug", "disease")):
    bd, out = tmp_path / "blocks", tmp_path / "processed"
    if "drug" in blocks:
        X, d, meta = fingerprints.encode(m)
        save_block("drug", X, m, meta, d, block_dir=bd)
    if "disease" in blocks:
        X, d, meta = disease.encode(m, min_count=1)
        save_block("disease", X, m, meta, d, block_dir=bd)
    manifest, report = assemble(m, rep, out_dir=out, block_dir=bd)
    return out, bd, manifest, report


def test_fingerprints_alone_satisfy_person_b_loader(small_master, tmp_path):
    harness = pytest.importorskip("eval.harness")
    m, rep = small_master
    out, _, manifest, report = build(m, rep, tmp_path, blocks=("drug",))
    d = harness.load_contract(out)                         # B's loader, unmodified
    assert d["drug_features"].shape == (len(m), 2048)
    assert d["disease_features"].shape == d["text_features"].shape == (len(m), 0)
    assert manifest["blocks"]["drug_features"]["status"] == "ready"
    assert manifest["blocks"]["text_features"]["status"] == "pending"
    assert "FAIL" not in report


def test_contract_alignment(small_master, tmp_path):
    m, rep = small_master
    out, _, manifest, _ = build(m, rep, tmp_path)
    a = {k: np.load(out / f"{k}.npy", allow_pickle=False) for k in CONTRACT + ["split_valid"]}
    assert a["trial_ids"].tolist() == m.nctid.tolist()
    assert a["labels"].tolist() == m.label.tolist() and a["phase"].tolist() == m.phase.tolist()
    for s in ("train", "valid", "test"):
        assert m.split.to_numpy()[a[f"split_{s}"]].tolist() == [s] * len(a[f"split_{s}"])
    # every row's drug features are the fingerprint of that row's own SMILES
    for i, sm in enumerate(m.smiles):
        bits = [fingerprints.smiles_to_bits(s) for s in sm]
        bits = [b for b in bits if b is not None]
        want = np.bitwise_or.reduce(bits).astype(np.float32) if bits else np.zeros(2048, np.float32)
        assert np.array_equal(a["drug_features"][i], want), m.nctid.iloc[i]
    names = json.loads((out / "feature_names" / "disease_features.json").read_text())
    assert len(names) == a["disease_features"].shape[1]
    for k, f in manifest["files"].items():
        assert list(np.load(out / k).shape) == f["shape"]
    meta = (out / "trial_meta.csv").read_text().splitlines()
    assert "n_valid_smiles" in meta[0] and "n_icd" in meta[0] and "status" not in meta[0]
    assert len(meta) == len(m) + 1


def test_stale_block_becomes_pending_not_misaligned(small_master, tmp_path):
    m, rep = small_master
    _, bd, _, _ = build(m, rep, tmp_path)
    m2 = m.iloc[1:].reset_index(drop=True)               # master changed (e.g. a row was dropped)
    X, reason = load_block("drug", m2, block_dir=bd)
    assert X is None and "stale" in reason
    m3 = m.copy()
    m3.loc[0, "label"] = 1 - m3.loc[0, "label"]          # same ids, different labels
    assert load_block("drug", m3, block_dir=bd)[0] is None


def test_save_block_rejects_bad_shapes(small_master, tmp_path):
    m, _ = small_master
    with pytest.raises(ValueError):
        save_block("x", np.zeros((len(m) - 1, 3)), m, {}, block_dir=tmp_path)
    with pytest.raises(ValueError):
        save_block("x", np.full((len(m), 3), np.nan), m, {}, block_dir=tmp_path)
