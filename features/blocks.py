"""Storage for finished feature blocks (one per encoder), keyed to the master table.

A block is three files in data/interim/blocks/:
  {name}.npy          float32 (n_rows, d), rows in master-table order
  {name}_ids.npy      the master table's nctid column, saved alongside as an alignment check
  {name}.json         metadata: table_hash, params, feature names, per-row diagnostics summary

Writes are atomic (tmp file + rename), so an interrupted run never leaves a half block that
looks finished.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from features import config as C
from features.data import DataError, table_hash


def _atomic_save_npy(path, arr):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_block(name, X, master, meta, diag=None, block_dir=C.BLOCK_DIR):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != len(master):
        raise DataError(f"block {name}: shape {X.shape} does not match {len(master)} master rows")
    if not np.isfinite(X).all():
        raise DataError(f"block {name}: contains NaN/inf")
    block_dir = Path(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, name=name, shape=list(X.shape), table_hash=table_hash(master),
                built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    # ids and meta first, matrix last: the matrix is what marks the block as complete
    _atomic_save_npy(block_dir / f"{name}_ids.npy", master["nctid"].to_numpy().astype("U11"))
    _atomic_write_text(block_dir / f"{name}.json", json.dumps(meta, indent=2, sort_keys=True))
    if diag:
        cols = sorted(diag)
        lines = [",".join(cols)] + [",".join(str(int(diag[c][i])) for c in cols) for i in range(len(master))]
        _atomic_write_text(block_dir / f"{name}_diag.csv", "\n".join(lines) + "\n")
    _atomic_save_npy(block_dir / f"{name}.npy", X)
    return block_dir / f"{name}.npy"


def load_block(name, master, block_dir=C.BLOCK_DIR):
    """Return (X, meta) if a finished block matches this master table, else (None, reason)."""
    block_dir = Path(block_dir)
    p, pm, pi = block_dir / f"{name}.npy", block_dir / f"{name}.json", block_dir / f"{name}_ids.npy"
    if not (p.exists() and pm.exists() and pi.exists()):
        return None, "not built"
    meta = json.loads(pm.read_text(encoding="utf-8"))
    if meta.get("table_hash") != table_hash(master):
        return None, "stale (built against a different master table)"
    ids = np.load(pi, allow_pickle=False)
    if not np.array_equal(ids, master["nctid"].to_numpy().astype("U11")):
        return None, "stale (trial-id order differs)"
    X = np.load(p, allow_pickle=False)
    if X.shape[0] != len(master):
        return None, f"stale (has {X.shape[0]} rows)"
    return X, meta
