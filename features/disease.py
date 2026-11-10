"""Encoder 2 — disease: hierarchical ICD-10 multi-hot.

Purpose   HANDOFF section 3 asks for a *learned* ICD embedding. Learning has to happen inside
          Person B's model (on train labels only), so this encoder ships the input that
          embedding is learned from: a multi-hot over the ICD-10 hierarchy. A linear layer
          on this vector is exactly an EmbeddingBag over the trial's codes, so B's fusion MLP
          learns the disease embedding as its first layer, and LR / XGBoost can use the
          multi-hot directly.
          Three levels so rare codes still share signal through their parents:
            chapter  'C'      (first letter)
            category 'C34'    (3-character block)
            code     'C34.1'  (full code)
Input     master-table column `icd_codes`: validated, de-duplicated ICD-10 codes per trial.
Output    float32 (n_rows, d_disease) of 0/1. Columns = [chapter vocab | category vocab |
          code vocab | 3 OOV flags]. Names in feature_names.
Leakage   The vocabulary is fitted on official TRAIN rows only (ICD_FIT_SPLIT). A code first
          seen in valid/test cannot get its own column; it lights up its parent levels and
          the OOV flag for its level instead. Counting is per unique trial, not per row, so
          trials listed in two phases are not double-counted.
Missing   No codes -> all-zero row (diagnostic column n_icd = 0).
"""
import logging
from collections import Counter

import numpy as np

from features import config as C

log = logging.getLogger(__name__)

LEVEL_FN = {
    "chapter": lambda c: c[0],
    "category": lambda c: c.split(".")[0],
    "code": lambda c: c,
}


def fit_vocab(master, levels=C.ICD_LEVELS, min_count=C.ICD_MIN_COUNT, fit_split=C.ICD_FIT_SPLIT):
    fit = master[master["split"] == fit_split].drop_duplicates("nctid")
    vocab = {}
    for lvl in levels:
        cnt = Counter(t for codes in fit["icd_codes"] for t in {LEVEL_FN[lvl](c) for c in codes})
        vocab[lvl] = sorted(t for t, k in cnt.items() if k >= min_count)
    return vocab, int(len(fit))


def transform(master, vocab, levels=C.ICD_LEVELS):
    index, names, off = {}, [], 0
    for lvl in levels:
        for j, t in enumerate(vocab[lvl]):
            index[(lvl, t)] = off + j
        names += [f"icd_{lvl}_{t}" for t in vocab[lvl]]
        off += len(vocab[lvl])
    oov_col = {lvl: off + k for k, lvl in enumerate(levels)}
    names += [f"icd_{lvl}_OOV" for lvl in levels]
    X = np.zeros((len(master), len(names)), dtype=np.float32)
    n_icd = np.zeros(len(master), dtype=np.int64)
    n_oov = np.zeros(len(master), dtype=np.int64)
    for i, codes in enumerate(master["icd_codes"]):
        n_icd[i] = len(codes)
        for lvl in levels:
            for t in {LEVEL_FN[lvl](c) for c in codes}:
                j = index.get((lvl, t))
                if j is None:
                    X[i, oov_col[lvl]] = 1.0
                    if lvl == "code":
                        n_oov[i] += 1
                else:
                    X[i, j] = 1.0
    return X, names, {"n_icd": n_icd, "n_icd_oov": n_oov}


def encode(master, levels=C.ICD_LEVELS, min_count=C.ICD_MIN_COUNT, fit_split=C.ICD_FIT_SPLIT):
    vocab, n_fit = fit_vocab(master, levels, min_count, fit_split)
    X, names, diag = transform(master, vocab, levels)
    by_split = {s: float((X[(master["split"] == s).to_numpy(), -1] > 0).mean()) for s in C.SPLITS}
    meta = {
        "encoder": "icd10_hierarchical_multihot", "levels": list(levels), "min_count": min_count,
        "fit_split": fit_split, "fit_unique_trials": n_fit,
        "vocab_sizes": {lvl: len(vocab[lvl]) for lvl in levels}, "vocab": vocab,
        "rows_no_icd": int((diag["n_icd"] == 0).sum()),
        "share_rows_with_oov_code_by_split": by_split,
        "feature_names": names,
    }
    log.info("disease: vocab %s, %d dims; rows with an unseen full code: %s",
             meta["vocab_sizes"], X.shape[1], {k: round(v, 3) for k, v in by_split.items()})
    return X, diag, meta
