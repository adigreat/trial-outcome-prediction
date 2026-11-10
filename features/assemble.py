"""Write the HANDOFF section 5 contract to data/processed/ from whichever blocks are finished.

Blocks that are not built yet are written as (n, 0) float32 placeholders, so Person B's
`eval.harness.load_contract()` (which loads all 8 arrays) works from the moment fingerprints
exist. manifest.json says which blocks are real and which are pending.

Files written (row i of every array is the same trial):
  contract: drug_features, disease_features, text_features, trial_ids, labels, phase,
            split_train, split_test                                           (.npy)
  extras:   split_valid.npy, manifest.json, trial_meta.csv, feature_names/{block}.json,
            validation_report.md
"""
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import config as C
from features.blocks import _atomic_save_npy, _atomic_write_text, load_block

log = logging.getLogger(__name__)

BLOCKS = {"drug": "drug_features", "disease": "disease_features", "text": "text_features"}
CONTRACT = ["drug_features", "disease_features", "text_features", "trial_ids",
            "labels", "phase", "split_train", "split_test"]
# HANDOFF section 3 positive rates, used only to flag drift in the report
EXPECTED_POS = {(1, "train"): .57, (1, "valid"): .57, (1, "test"): .55, (2, "train"): .48, (2, "valid"): .47,
                (2, "test"): .56, (3, "train"): .65, (3, "valid"): .67, (3, "test"): .75}


def _git_commit():
    try:
        sha = subprocess.run(["git", "-C", str(C.REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(C.REPO_ROOT), "status", "--porcelain", "features"],
                               capture_output=True, text=True).stdout.strip() != ""
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(master, master_report, out_dir=C.PROCESSED_DIR, block_dir=C.BLOCK_DIR):
    out_dir = Path(out_dir)
    (out_dir / "feature_names").mkdir(parents=True, exist_ok=True)
    n = len(master)
    arrays, block_info = {}, {}
    trial_meta = master[["nctid", "phase", "phase_raw", "split", "label"]].copy()
    trial_meta.index.name = "row"

    for name, fname in BLOCKS.items():
        X, meta = load_block(name, master, block_dir)
        if X is None:
            arrays[fname] = np.zeros((n, 0), dtype=np.float32)
            block_info[fname] = {"status": "pending", "reason": meta, "dim": 0}
            log.info("%s: %s -> writing (n, 0) placeholder", fname, meta)
            continue
        arrays[fname] = X.astype(np.float32, copy=False)
        names = meta.pop("feature_names", None)
        if names is not None:
            if len(names) != X.shape[1]:
                raise ValueError(f"{fname}: {len(names)} feature names for {X.shape[1]} columns")
            _atomic_write_text(out_dir / "feature_names" / f"{fname}.json", json.dumps(names))
        meta.pop("vocab", None)   # big; the names file already carries it
        block_info[fname] = {"status": "ready", "dim": int(X.shape[1]), **meta}
        diag = Path(block_dir) / f"{name}_diag.csv"
        if diag.exists():
            d = pd.read_csv(diag)
            if len(d) != n:
                raise ValueError(f"{diag} has {len(d)} rows, expected {n}")
            for c in d.columns:
                trial_meta[c] = d[c].to_numpy()

    arrays["trial_ids"] = master["nctid"].to_numpy().astype("U11")
    arrays["labels"] = master["label"].to_numpy().astype(np.int64)
    arrays["phase"] = master["phase"].to_numpy().astype(np.int64)
    for s in C.SPLITS:
        arrays[f"split_{s}"] = np.flatnonzero((master["split"] == s).to_numpy()).astype(np.int64)

    checks = validate(arrays, master)
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise AssertionError("validation failed, nothing written:\n" + "\n".join(f"- {c['check']}: {c['detail']}" for c in failed))

    for k, a in arrays.items():
        _atomic_save_npy(out_dir / f"{k}.npy", a)
    trial_meta.to_csv(out_dir / "trial_meta.csv")

    manifest = {
        "contract_version": "HANDOFF section 5 (2026-11-09)",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
        "dataset": {"name": C.DATASET_NAME, "upstream": C.UPSTREAM_REPO, "upstream_commit": C.UPSTREAM_COMMIT,
                    "csv_sha256": master_report["csv_sha256"],
                    "split": "official TOP temporal split; train/valid = completed before 2014 "
                             "(valid = random 10%), test = started 2014 or later"},
        "n_rows": n, "row_key": "(trial_id, phase)",
        "row_order": "phase 1,2,3 -> split train,valid,test -> nctid",
        "master_table_hash": master_report["table_hash"],
        "blocks": block_info,
        "splits": {s: int(len(arrays[f"split_{s}"])) for s in C.SPLITS},
        "cleaning": {k: master_report.get(k) for k in
                     ("rows_read", "rows_kept", "dropped", "icd_codes_malformed", "empty_smiles",
                      "empty_icd", "empty_criteria", "warnings")},
        "files": {f"{k}.npy": {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": _sha(out_dir / f"{k}.npy")}
                  for k, a in arrays.items()},
        "seed": C.SEED,
    }
    _atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    report = render_report(arrays, master, master_report, block_info, checks)
    _atomic_write_text(out_dir / "validation_report.md", report)
    log.info("contract written to %s (%s)", out_dir,
             ", ".join(f"{k}={v['status']}:{v['dim']}" for k, v in block_info.items()))
    return manifest, report


def validate(a, master):
    """Every check B's harness does, plus the ones that catch silent alignment bugs."""
    n = len(master)
    out = []

    def check(name, ok, detail=""):
        out.append({"check": name, "ok": bool(ok), "detail": detail})

    for k in ("drug_features", "disease_features", "text_features", "labels", "phase"):
        check(f"{k} rows == trial_ids rows", len(a[k]) == n, f"{len(a[k])} vs {n}")
    check("trial_ids is fixed-width unicode (loads with allow_pickle=False)", a["trial_ids"].dtype.kind == "U")
    check("trial_ids match master order", np.array_equal(a["trial_ids"], master["nctid"].to_numpy().astype("U11")))
    pairs = set(zip(a["trial_ids"].tolist(), a["phase"].tolist()))
    check("(trial_id, phase) unique", len(pairs) == n, f"{n - len(pairs)} duplicates")
    check("labels in {0,1}", set(np.unique(a["labels"]).tolist()) <= {0, 1})
    check("phase in {1,2,3}", set(np.unique(a["phase"]).tolist()) <= {1, 2, 3})
    check("labels match master", np.array_equal(a["labels"], master["label"].to_numpy()))
    tr, va, te = (set(a[f"split_{s}"].tolist()) for s in C.SPLITS)
    check("splits pairwise disjoint", not (tr & va or tr & te or va & te))
    check("splits cover every row", len(tr | va | te) == n)
    ids = a["trial_ids"]
    check("no trial id in both test and train/valid (any phase)",
          not (set(ids[list(te)]) & set(ids[list(tr | va)])))
    for s, idx in zip(C.SPLITS, (tr, va, te)):
        idx = np.fromiter(idx, int)
        check(f"split_{s} matches master split column", (master["split"].to_numpy()[idx] == s).all() if len(idx) else True)
    for k in ("drug_features", "disease_features", "text_features"):
        X = a[k]
        check(f"{k} finite", np.isfinite(X).all())
        check(f"{k} float32", X.dtype == np.float32)
        if X.shape[1]:
            # same trial in two phase files must get the identical feature row
            m = master.reset_index(drop=True)
            dup = m[m.duplicated("nctid", keep=False)]
            same = True
            for _, g in dup.groupby("nctid"):
                rows = g.index.to_numpy()
                if not (X[rows] == X[rows[0]]).all():
                    same = False
                    break
            check(f"{k} identical across phase copies of a trial", same)
    return out


def render_report(a, master, rep, block_info, checks):
    L = ["# Feature pipeline validation report", "",
         f"Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} · code `{_git_commit()}` · "
         f"dataset {C.DATASET_NAME} @ `{C.UPSTREAM_COMMIT[:7]}` · master hash `{rep['table_hash'][:12]}`", "",
         "## Blocks", "", "| file | status | dim | notes |", "|---|---|---|---|"]
    for k, v in block_info.items():
        if v["status"] == "ready":
            note = {"drug_features": lambda v: f"Morgan r={v['radius']}, {v['n_bits']} bits, union; "
                                               f"{v['rows_no_valid_smiles']} rows all-zero (no parsable SMILES); "
                                               f"{v['invalid_unique_smiles']}/{v['unique_smiles']} unique SMILES unparsable",
                    "disease_features": lambda v: f"ICD multi-hot, vocab fit on {v['fit_split']} "
                                                  f"(min_count {v['min_count']}): {v['vocab_sizes']}; "
                                                  f"rows with an unseen full code: "
                                                  + ", ".join(f"{s} {x:.1%}" for s, x in v['share_rows_with_oov_code_by_split'].items()),
                    "text_features": lambda v: f"{v['model']} ({v['pooling']} pool), {v['unique_bullets']} unique bullets; "
                                               f"{v['rows_no_inclusion']} rows no inclusion, {v['rows_no_exclusion']} no exclusion"}[k](v)
        else:
            note = v["reason"]
        L.append(f"| `{k}.npy` | {v['status']} | {v['dim']} | {note} |")
    L += ["", "## Rows per phase and split", "", "| phase | split | n | positive rate | HANDOFF rate |", "|---|---|---|---|---|"]
    for (p, s), g in master.groupby(["phase", "split"], sort=False):
        L.append(f"| {p} | {s} | {len(g)} | {g['label'].mean():.3f} | {EXPECTED_POS.get((p, s), float('nan')):.2f} |")
    L += ["", "## Cleaning", "",
          f"- rows read {rep['rows_read']}, kept {rep['rows_kept']}, dropped {rep['dropped'] or 'none'}",
          "- `status` and `why_stop` are never read (outcome information, not design-time)",
          f"- malformed ICD codes removed: {rep['icd_codes_malformed']}",
          f"- rows with no SMILES {rep['empty_smiles']}, no ICD codes {rep['empty_icd']}, empty criteria {rep['empty_criteria']} "
          "(kept; their block is all-zero)"]
    L += [f"- ⚠ {w}" for w in rep.get("warnings", [])]
    L += ["", "## Checks", ""] + [f"- {'PASS' if c['ok'] else 'FAIL'} — {c['check']}" + (f" ({c['detail']})" if c['detail'] and not c['ok'] else "") for c in checks]
    return "\n".join(L) + "\n"
