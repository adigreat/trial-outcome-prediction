# features/ — data and feature pipeline (Person A)

Turns the TOP benchmark CSVs into the row-aligned arrays in HANDOFF section 5. Person B
loads them with `eval.harness.load_contract()`; nothing in `eval/` or `models/` needs to change.

## Quick start (from the repo root)

```bash
pip install -r features/requirements.txt
python -m features.build fetch          # TOP @ 8dc0497 -> data/raw/top/ (or copy the 9 CSVs there yourself)
python -m features.build fingerprints   # ~5 s, CPU. Writes data/processed/ with text/disease as placeholders
python -m features.build disease        # ~1 s, CPU
pip install -r features/requirements-text.txt
python -m features.build text           # PubMedBERT, GPU recommended. Resumable: just re-run after a crash
python -m pytest features/tests -q      # 39 tests
```

Each step re-writes `data/processed/` as soon as its block is done, so the contract is always
loadable. Finished blocks are skipped on re-runs (`--force` rebuilds). The raw CSVs are
hash-checked against the pinned version; `--allow-unpinned` exists but changing the dataset
is a HANDOFF section 9 decision.

## What Person B gets — `data/processed/`

Row `i` of every array is the same trial-in-a-phase. Key = `(trial_id, phase)`: 876 trials
are listed in two phase files (e.g. "phase 1/phase 2") and appear once per phase, with identical features and label.

| file | shape / dtype | meaning |
|---|---|---|
| `drug_features.npy` | (12477, 2048) float32 | Morgan fingerprint bits (0/1), radius 2, OR over the trial's drugs |
| `disease_features.npy` | (12477, 1983) float32 | ICD-10 multi-hot: 22 chapters + 499 categories + 1459 codes + 3 OOV flags |
| `text_features.npy` | (12477, 1536) float32 | [mean PubMedBERT of inclusion bullets ‖ mean of exclusion bullets]. **(12477, 0) until built** |
| `trial_ids.npy` | (12477,) `<U11` | NCT id. The alignment key |
| `labels.npy` | (12477,) int64 | 1 = success, 0 = failure |
| `phase.npy` | (12477,) int64 | 1, 2, 3 = the TOP phase file the row came from |
| `split_train.npy` | (8143,) int64 | row indices, official TOP train |
| `split_test.npy` | (3427,) int64 | row indices, official TOP test (started 2014 or later) |
| `split_valid.npy` | (907,) int64 | **extra, not in the section 5 contract.** Official TOP valid (random 10% of pre-2014). Use it for early stopping / tuning, or merge into train for final fits |
| `manifest.json` | | dataset version + CSV hashes, code commit, per-block status (`ready` / `pending`), params, sha256 of every array |
| `feature_names/*.json` | | one column name per feature, e.g. `icd_category_C34`, `morgan_r2_17`, `text_excl_5` |
| `trial_meta.csv` | | per row: nctid, phase, raw phase string, split, label, n_smiles, n_valid_smiles, n_icd, n_icd_oov, n_incl, n_excl |
| `validation_report.md` | | counts, positive rates vs HANDOFF, cleaning summary, every alignment check |

Row order: phase 1→3, then train → valid → test, then nctid.

```python
from eval.harness import load_contract
import json, numpy as np
d = load_contract("data/processed")
valid = np.load("data/processed/split_valid.npy")
blocks = json.load(open("data/processed/manifest.json"))["blocks"]
ready = [k for k, v in blocks.items() if v["status"] == "ready"]   # don't train on (n, 0) placeholders
X = np.hstack([d[k] for k in ready])
```

## The three encoders

| | drug | disease | protocol |
|---|---|---|---|
| module | `fingerprints.py` | `disease.py` | `text.py` |
| input | `smiless` list | `icdcodes` list | `criteria` text |
| method | RDKit Morgan r=2, 2048 bits, bitwise OR across drugs | multi-hot over chapter / 3-char category / full code | split inclusion / exclusion → bullets → frozen PubMedBERT, masked mean pool → mean per section |
| fitted on data? | no | vocabulary from **train rows only**, codes in ≥5 train trials | no (frozen) |
| missing input | zero row (`n_valid_smiles = 0`) | zero row; unseen codes light parents + OOV flag | zero half for a missing section |
| needs | CPU, seconds | CPU, seconds | GPU ~6 GB, resumable sentence cache |

Disease "learned embedding" (HANDOFF section 3): the learning belongs in B's model, since it has
to use train labels only. A linear layer on the multi-hot is an EmbeddingBag over the trial's
codes, so the fusion MLP learns it as its first layer. LR / XGBoost use the multi-hot directly.

## Data checks (see `validation_report.md` for the live numbers)

- `status` and `why_stop` are never read (they describe outcomes, so they are leakage).
- The 9 CSVs must match pinned sha256 hashes (CRLF-normalised) and the HANDOFF row counts.
- Malformed NCT ids / labels are dropped and counted. Duplicate `(nctid, phase)` rows are dropped if identical and fatal if they conflict.
- A trial in test for one phase and train/valid for another is fatal (none exist). Different labels or inputs for the same trial across phases is fatal (none exist).
- 106 trials are in train for one phase and valid for another. That is the official split, so they are kept. It matters only if valid is used to tune a model pooled across phases.
- Every encoder block stores the master-table hash and the trial-id order. Assembly refuses a block built against different rows, and writes a placeholder instead of misaligned features.

## Known limitations

- 1 of 1,748 unique SMILES does not parse in RDKit. No trial ends up all-zero because of it.
- TOP's drug→SMILES mapping comes from upstream and is imperfect. For example, NCT01187615's regorafenib arms map to a platinum complex. We do not repair it.
- About 1,270 trials have no detectable exclusion section, mostly old free-text protocols ("DISEASE CHARACTERISTICS: …"). Their exclusion half is zero and `n_excl = 0`.
- In test, 16.5% of rows carry a full ICD code never seen in train (train itself: 8%). This is expected with a temporal split, and the parent levels absorb most of it.
