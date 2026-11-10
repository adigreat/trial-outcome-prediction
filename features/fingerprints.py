"""Encoder 1 — drug: Morgan fingerprints (RDKit).

Purpose   Represent the trial's drug(s) by substructure presence, with no training involved,
          so it cannot leak test information and is ready before anything else.
Input     master-table column `smiles`: list of SMILES strings per trial (TOP `smiless`).
Output    float32 (n_rows, FP_NBITS) of 0/1. Bit j = 1 if ANY of the trial's parsable drugs
          has Morgan bit j set (radius FP_RADIUS). Union, because a trial tests the
          combination; it keeps the vector binary for LR / XGBoost.
Missing   No SMILES, or none that RDKit can parse -> all-zero row. Counted per row in the
          block metadata and in data/processed/trial_meta.csv (n_smiles, n_valid_smiles).

No torch import anywhere in this path: fingerprints must never wait on the text encoder.
"""
import logging

import numpy as np

from features import config as C

log = logging.getLogger(__name__)


def _rdkit():
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")   # invalid SMILES are counted and reported, not spammed to stderr
    return Chem, rdFingerprintGenerator, rdBase.rdkitVersion


def smiles_to_bits(smiles, radius=C.FP_RADIUS, nbits=C.FP_NBITS):
    """Morgan bit vector for one SMILES, or None if RDKit cannot parse it."""
    Chem, gen_mod, _ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return gen_mod.GetMorganGenerator(radius=radius, fpSize=nbits).GetFingerprintAsNumPy(mol).astype(np.uint8)


def encode(master, radius=C.FP_RADIUS, nbits=C.FP_NBITS):
    """Return (X float32 (n, nbits), per-row diagnostics dict, metadata dict)."""
    Chem, gen_mod, version = _rdkit()
    gen = gen_mod.GetMorganGenerator(radius=radius, fpSize=nbits)
    cache = {}

    def bits(s):
        if s not in cache:
            mol = Chem.MolFromSmiles(s)
            cache[s] = None if mol is None else gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        return cache[s]

    n = len(master)
    X = np.zeros((n, nbits), dtype=np.uint8)
    n_smiles = np.zeros(n, dtype=np.int64)
    n_valid = np.zeros(n, dtype=np.int64)
    for i, sm in enumerate(master["smiles"]):
        n_smiles[i] = len(sm)
        for s in sm:
            b = bits(s)
            if b is not None:
                X[i] |= b
                n_valid[i] += 1
    invalid = sorted(s for s, b in cache.items() if b is None)
    diag = {"n_smiles": n_smiles, "n_valid_smiles": n_valid}
    meta = {
        "encoder": "morgan_fingerprint", "rdkit_version": version, "radius": radius, "n_bits": nbits,
        "aggregation": C.FP_AGGREGATION, "dtype_note": "0/1 stored as float32",
        "unique_smiles": len(cache), "invalid_unique_smiles": len(invalid),
        "invalid_smiles_examples": invalid[:20],
        "rows_no_smiles": int((n_smiles == 0).sum()),
        "rows_no_valid_smiles": int((n_valid == 0).sum()),
        "rows_some_invalid": int(((n_valid < n_smiles) & (n_valid > 0)).sum()),
        "mean_bits_on": float(X.sum(1).mean()),
        "feature_names": [f"morgan_r{radius}_{j}" for j in range(nbits)],
    }
    log.info("fingerprints: %d rows, %d unique SMILES (%d invalid), %d rows all-zero",
             n, len(cache), len(invalid), meta["rows_no_valid_smiles"])
    return X.astype(np.float32), diag, meta
