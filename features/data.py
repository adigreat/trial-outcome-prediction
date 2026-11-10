"""Loading, cleaning, validation and trial-ID alignment for the TOP benchmark.

Output is the *master table*: one row per (nctid, phase), in a fixed order
(phase 1-3, then train/valid/test, then nctid). Every encoder writes its block in this
row order, and every block records the hash of the master table it was built against,
so assembly can refuse to pair features with the wrong rows.

Only design-time columns are kept. `status` and `why_stop` are never read.
"""
import ast
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from features import config as C

log = logging.getLogger(__name__)

NCT_RE = re.compile(r"^NCT\d{8}$")
# ICD-10(-CM): letter, digit, digit-or-letter (C7A, D3A, O9A are real), optional .suffix
ICD_RE = re.compile(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
MASTER_COLUMNS = ["nctid", "phase", "phase_raw", "split", "label", "smiles", "icd_codes", "diseases", "criteria"]


class DataError(ValueError):
    """Raised when the input data breaks an invariant we rely on. Never caught silently."""


# --------------------------------------------------------------------------- fetch / verify

def file_sha256(path):
    """sha256 of a text file with CRLF normalised to LF (stable across OSes / git autocrlf)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fetch_raw(raw_dir=C.RAW_DIR, commit=C.UPSTREAM_COMMIT):
    """Clone the upstream TOP repo at the pinned commit and copy the 9 phase CSVs into raw_dir."""
    raw_dir = Path(raw_dir)
    if all((raw_dir / f).exists() for f in C.CSV_SHA256):
        log.info("raw CSVs already present in %s", raw_dir)
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        log.info("cloning %s @ %s", C.UPSTREAM_REPO, commit[:7])
        subprocess.run(["git", "clone", "--quiet", C.UPSTREAM_REPO, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "--quiet", commit], check=True)
        for f in C.CSV_SHA256:
            shutil.copy2(Path(tmp) / "data" / f, raw_dir / f)
    return raw_dir


def verify_raw(raw_dir=C.RAW_DIR, strict=True):
    """Check every CSV exists and matches the pinned hash. Returns {file: sha256}."""
    hashes = {}
    for f, want in C.CSV_SHA256.items():
        p = Path(raw_dir) / f
        if not p.exists():
            raise DataError(f"missing {p}. Run `python -m features.build fetch` or copy the TOP CSVs there.")
        got = file_sha256(p)
        hashes[f] = got
        if got != want:
            msg = f"{f}: sha256 {got[:12]}… does not match pinned {want[:12]}… (different dataset version?)"
            if strict:
                raise DataError(msg + " Pass --allow-unpinned only if you mean to change version (HANDOFF section 9).")
            log.warning(msg)
    return hashes


# --------------------------------------------------------------------------- field parsers

def _literal_list(s):
    """Parse a python-literal list stored as text. Returns (list_of_str, ok)."""
    s = (s or "").strip()
    if s in ("", "[]", "nan", "None"):
        return [], True
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return [], False
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return [], False
    return [str(x) for x in v], True


def _dedupe(xs):
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_smiles(s):
    """'["CCO", "CCO"]' -> ['CCO'] (stripped, de-duplicated, order kept). Validity is checked by RDKit later."""
    xs, ok = _literal_list(s)
    return _dedupe(x.strip() for x in xs), ok


def parse_icd(s):
    """TOP stores ICD codes as a list of *stringified* lists: '["[\\'C61\\', \\'C62\\']", ...]'.

    Returns (valid_codes, n_invalid, ok). Codes are upper-cased, de-duplicated, and checked
    against the ICD-10 shape; malformed codes are dropped and counted.
    """
    outer, ok = _literal_list(s)
    codes = []
    for item in outer:
        inner, ok_inner = _literal_list(item)
        if not ok_inner:                         # a bare code rather than a stringified list
            inner = [item]
        codes.extend(c.strip().upper() for c in inner)
    codes = _dedupe(codes)
    valid = [c for c in codes if ICD_RE.match(c)]
    return valid, len(codes) - len(valid), ok


def clean_criteria(s):
    """Normalise whitespace inside lines but keep line structure (bullets are line-based)."""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t ]+", " ", ln).rstrip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- load + clean

def _read_one(path, phase, split):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path.name}: missing columns {missing}")
    df = df[list(C.REQUIRED_COLUMNS)].copy()   # drops status / why_stop here, before anything else
    df["phase_file"], df["split"], df["source"] = phase, split, path.name
    return df


def load_master(raw_dir=C.RAW_DIR, strict_version=True, check_counts=True):
    """Load, clean and validate all 9 CSVs. Returns (master DataFrame, report dict)."""
    raw_dir = Path(raw_dir)
    hashes = verify_raw(raw_dir, strict=strict_version)
    frames = [_read_one(raw_dir / f"phase_{pn}_{s}.csv", p, s) for pn, p in C.PHASES.items() for s in C.SPLITS]
    df = pd.concat(frames, ignore_index=True)
    rep = {"rows_read": int(len(df)), "csv_sha256": hashes, "dropped": {}, "warnings": []}

    def drop(mask, reason):
        n = int(mask.sum())
        if n:
            log.warning("dropping %d rows: %s (%s)", n, reason, df.loc[mask, "nctid"].head(5).tolist())
            rep["dropped"][reason] = rep["dropped"].get(reason, 0) + n
        return df[~mask].copy()

    # --- identifiers and labels
    df["nctid"] = df["nctid"].str.strip().str.upper()
    df = drop(~df["nctid"].str.match(NCT_RE), "malformed nctid")
    df["label"] = df["label"].str.strip()
    df = drop(~df["label"].isin(["0", "1"]), "label not 0/1")
    df["label"] = df["label"].astype(np.int64)
    # phase = which phase file the row came from ("phase 1/phase 2" trials sit in both I and II)
    df["phase_raw"] = df["phase"].str.strip()
    df["phase"] = df.pop("phase_file").astype(np.int64)

    # --- exact duplicate (nctid, phase, split): identical copies are dropped, conflicting copies are fatal
    key = ["nctid", "phase"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        content = ["label", "smiless", "icdcodes", "criteria", "split"]
        conflicting = df[dup].groupby(key)[content].nunique().gt(1).any(axis=1)
        if conflicting.any():
            raise DataError(f"conflicting duplicate rows for {conflicting[conflicting].index.tolist()[:5]}")
        df = drop(df.duplicated(key, keep="first"), "exact duplicate (nctid, phase)")

    # --- parse fields
    sm = df["smiless"].map(parse_smiles)
    df["smiles"] = sm.map(lambda t: t[0])
    df = drop(~sm.map(lambda t: t[1]), "unparseable smiless")
    icd = df["icdcodes"].map(parse_icd)
    df["icd_codes"] = icd.map(lambda t: t[0])
    rep["icd_codes_malformed"] = int(icd.map(lambda t: t[1]).sum())
    df = drop(~icd.map(lambda t: t[2]), "unparseable icdcodes")
    df["diseases"] = df["diseases"].map(lambda s: _dedupe(x.strip().lower() for x in _literal_list(s)[0]))
    df["criteria"] = df["criteria"].map(clean_criteria)

    # --- cross-phase consistency: the same trial must carry the same label and inputs everywhere
    for col in ("label", "smiless", "icdcodes", "criteria"):
        bad = df.groupby("nctid")[col].nunique()
        bad = bad[bad > 1]
        if len(bad):
            raise DataError(f"trial(s) {bad.index.tolist()[:5]} have different '{col}' in different phase files")

    # --- leakage checks between splits
    within = df.groupby(["nctid", "phase"])["split"].nunique()
    if (within > 1).any():
        raise DataError("a trial appears in two splits of the same phase")
    sets = {s: set(df.loc[df.split == s, "nctid"]) for s in C.SPLITS}
    tt = (sets["train"] | sets["valid"]) & sets["test"]
    if tt:
        raise DataError(f"{len(tt)} trials are in test for one phase and train/valid for another: {sorted(tt)[:5]}")
    tv = sets["train"] & sets["valid"]
    rep["train_valid_cross_phase"] = sorted(tv)
    if tv:
        rep["warnings"].append(
            f"{len(tv)} trials are in train for one phase and valid for another (official split; "
            "harmless for per-phase models, slightly optimistic validation for a pooled model)")

    # --- content warnings (kept, not dropped: missing inputs become zero vectors + flags)
    rep["empty_smiles"] = int((df["smiles"].map(len) == 0).sum())
    rep["empty_icd"] = int((df["icd_codes"].map(len) == 0).sum())
    rep["empty_criteria"] = int((df["criteria"] == "").sum())

    # --- fixed order and final shape
    df["split_order"] = df["split"].map({s: i for i, s in enumerate(C.SPLITS)})
    df = df.sort_values(["phase", "split_order", "nctid"], kind="mergesort").reset_index(drop=True)
    df = df[MASTER_COLUMNS]

    counts = df.groupby(["phase", "split"]).size().to_dict()
    rep["counts"] = {f"{p}/{s}": int(n) for (p, s), n in sorted(counts.items())}
    if check_counts:
        diff = {k: (counts.get(k, 0), v) for k, v in C.EXPECTED_COUNTS.items() if counts.get(k, 0) != v}
        if diff:
            raise DataError(f"row counts differ from HANDOFF section 3 (got, expected): {diff}")
    rep["rows_kept"] = int(len(df))
    rep["table_hash"] = table_hash(df)
    log.info("master table: %d rows, hash %s", len(df), rep["table_hash"][:12])
    return df, rep


def table_hash(df):
    """Hash of the alignment-relevant content (ids, phase, split, label). Blocks store this."""
    h = hashlib.sha256()
    for r in df[["nctid", "phase", "split", "label"]].itertuples(index=False):
        h.update(f"{r.nctid}|{r.phase}|{r.split}|{r.label}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- persistence

def save_master(df, rep, out_dir=C.INTERIM_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = df.copy()
    for c in ("smiles", "icd_codes", "diseases"):
        m[c] = m[c].map(json.dumps)
    m.to_csv(out_dir / "master.csv", index_label="row")
    (out_dir / "master_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    return out_dir / "master.csv"


def read_master(path=C.INTERIM_DIR / "master.csv"):
    m = pd.read_csv(path, dtype={"nctid": str, "split": str, "phase_raw": str, "criteria": str},
                    keep_default_na=False, index_col="row")
    for c in ("smiles", "icd_codes", "diseases"):
        m[c] = m[c].map(json.loads)
    m["phase"] = m["phase"].astype(np.int64)
    m["label"] = m["label"].astype(np.int64)
    if not (m.index.to_numpy() == np.arange(len(m))).all():
        raise DataError("master.csv row index is not 0..n-1")
    return m


def get_master(raw_dir=C.RAW_DIR, interim_dir=C.INTERIM_DIR, strict_version=True, rebuild=False):
    """Load the cached master table, or build it. Rebuilds if the raw CSVs changed."""
    path = Path(interim_dir) / "master.csv"
    rep_path = Path(interim_dir) / "master_report.json"
    if path.exists() and rep_path.exists() and not rebuild:
        rep = json.loads(rep_path.read_text())
        current = verify_raw(raw_dir, strict=strict_version)
        if current == rep.get("csv_sha256"):
            m = read_master(path)
            if table_hash(m) != rep["table_hash"]:
                raise DataError("master.csv was edited by hand (hash mismatch); delete data/interim and rebuild")
            return m, rep
        log.info("raw CSVs changed since master.csv was built; rebuilding")
    m, rep = load_master(raw_dir, strict_version=strict_version)
    save_master(m, rep, interim_dir)
    return m, rep
