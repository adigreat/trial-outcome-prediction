# HANDOFF.md

Shared context file for this project. Both of us keep it current. Claude reads this at the
start of a session so it knows what we're building without being re-briefed every time.

**Rule: if you make a decision, write it here. If you change something another person
depends on, write it here before you push.**

---

## 1. What we're building

A model that predicts whether a clinical trial will fail, using only information available
at design time — the drug, the disease, and the eligibility criteria.

Phase III trials cost hundreds of millions and roughly half fail. Predicting that before
the money is committed is the point.

**Type:** binary classification, evaluated separately per trial phase.

---

## 2. Why this project (the differentiator)

Most clinical-trial projects on GitHub predict whether a trial *administratively completes*
— pulled from a raw ClinicalTrials.gov export, XGBoost, SHAP, Streamlit. Completion is a much
easier target and worth much less. At least one of those repos says outright that it is not
an efficacy label.

We use the TOP benchmark, which has a real efficacy label, multimodal inputs, and a temporal
split. Harder, more credible, and a clean answer when someone asks how ours is different.

**This paragraph goes in the README. It is the whole pitch.**

---

## 3. Data

**Source:** `futianfan/clinical-trial-outcome-prediction` (TOP benchmark). Free,
non-commercial use, which covers us.

**Shape:** ~17.5k trials, split into Phase I / II / III. Roughly balanced
success vs failure. Each trial gives us three modalities:

| Modality | What it is | How we encode it |
|---|---|---|
| Drug | SMILES string | Morgan fingerprints (RDKit) |
| Disease | ICD-10 code | learned embedding |
| Protocol | eligibility criteria, free text | frozen PubMedBERT embeddings |

**Split:** use the official temporal split — train on trials completed before Jan 2014,
test on trials starting after. Do not invent our own split. The whole credibility of the
project rests on this.

### ⚠ Version trap — decide this first

Two versions of this dataset are in circulation. The original Fu et al. release, and a
2024 update from Chen et al. used by TrialBench and later work. Published baselines differ
between them.

- **Version we are using:** Original Fu et al. TOP release (`futianfan/clinical-trial-outcome-prediction`),
  using the shipped `data/phase_{I,II,III}_{train,valid,test}.csv`. Split per `benchmark/data_split.py`:
  train/valid = completed before 2014 (valid is a random 10% of that), test = started 2014 or later.
  Trials that started before 2014 and finished in/after 2014 are dropped by that script.
- **Not using:** TrialBench (Chen et al. 2024). Random 80/20 split and mixed label sources.
- **Decided by:** Aditya, 2026-11-09
- **Counted from the shipped CSVs (2026-11-09)**, train / valid / test, positive rate in brackets:
  - Phase I: 1044 (0.57) / 117 (0.57) / 627 (0.55) = 1788
  - Phase II: 4005 (0.48) / 446 (0.47) / 1654 (0.56) = 6105
  - Phase III: 3094 (0.65) / 344 (0.67) / 1146 (0.75) = 4584

  These are close to the 1,787 / 6,102 / 4,576 in the MediTab table. The other set I found
  (1,160 / 4,449 / 3,436) does not match this release.
- **Leakage warning:** the CSVs contain `status` and `why_stop` columns. They describe what
  happened to the trial, so they are NOT design-time information. Never use them as features.
  Features come only from `smiless`, `icdcodes`/`diseases` and `criteria`.
- **Test set drifts:** Phase III positive rate rises from 0.65 (train) to 0.75 (test). Report
  PR-AUC with that in mind, since a model's PR-AUC baseline rises with the positive rate.

Only compare our numbers against results computed on the same version. State the version
in the README.

---

## 4. Ownership

Split by vertical, not by day. The handoff contract in section 5 is what lets us work
without blocking each other.

### Person A — data and features
- Dataset version decision
- Loading, cleaning, trial-ID alignment (this is where silent bugs live)
- All three encoders
- **Ships fingerprints first** so B is unblocked while text embeddings are still cooking

### Person B — modelling and evaluation
- Evaluation harness (built before any model)
- Baselines: logistic regression, XGBoost
- Training loop and experiment runner
- Results table

### Shared
Ablations, error analysis, and the decision-threshold work are split one each — see
section 6. Neither person owns both.

| Role | Who |
|---|---|
| Person A (data and features) | Ananya (`ananyabiradar`) |
| Person B (modelling and evaluation) | Aditya (`adigreat`) |

---

## 5. The interface contract

**STATUS: AGREED 2026-11-09 (Aditya, confirmed with Ananya). Changing it needs a section 9 entry first.**

A produces, B consumes:

```
data/processed/
  drug_features.npy       # (n_trials, d_drug)
  disease_features.npy    # (n_trials, d_disease)
  text_features.npy       # (n_trials, d_text)
  trial_ids.npy           # (n_trials,)  — alignment key, same order as all of the above
  labels.npy              # (n_trials,)  — 1 = success, 0 = failure
  phase.npy               # (n_trials,)  — 1, 2, or 3
  split_train.npy         # indices into the arrays above
  split_test.npy          # indices
```

Every array is row-aligned by index. `trial_ids.npy` is the ground truth for alignment —
if anything looks wrong downstream, check this first.

If A changes a shape or a filename, it goes in section 9 and B gets told before the push.

**Clarifications found when checking the real data (these are part of the contract):**

- **One row per (trial, phase-file), not per trial.** 876 trials appear in two phase files
  (the Phase 1/2 and Phase 2/3 trials). So `trial_ids` may repeat across phases, but the
  pair `(trial_id, phase)` must be unique. `phase.npy` means "which of the TOP files this row
  came from", not the trial's own label.
- **Every row's data comes from the CSVs' `smiless`, `icdcodes`/`diseases` and `criteria`
  columns only.** Never `status` or `why_stop` (section 3, leakage warning).
- **Each phase is evaluated separately** on its own official test file, as HINT does.
- **`split_train` = the official `*_train.csv` rows, `split_test` = the official `*_test.csv`
  rows.** The official `*_valid.csv` rows are not in the contract yet; see section 10.
- **B's checker:** `eval.harness.load_contract("data/processed")` validates all of the above.
  A should run it on the arrays before telling B they are ready.

---

## 6. Workstreams

No dates. Pick up the next thing that isn't blocked.

### Floor first
Evaluation harness, then logistic regression and XGBoost on simple features.
These numbers are the floor. Nothing fancy ships until the floor exists.

**Metrics:** PR-AUC and ROC-AUC, reported per phase, on the official temporal test split.
PR-AUC is primary — it is what the published baselines report.

### Encoders
Morgan fingerprints → ICD embeddings → PubMedBERT text embeddings, in that order.
Use frozen text embeddings, not fine-tuned. 6GB VRAM, and fine-tuning buys little here.

**Avoid PyTorch Geometric.** Fingerprints over graph networks. The CUDA version-matching
problem is a known timesink and the accuracy difference does not justify it.

### Fusion model
Concatenate the three feature blocks, train a simple MLP head. Keep it boring.

### Ablations ← *the part that makes this ours*
Drop one modality at a time, per phase. The interesting output looks like:
"eligibility text carries most of the signal in Phase II but barely matters in Phase III."

That sentence is worth more than a leaderboard number.

### Error analysis
Break failures down by disease area. Oncology trials fail differently from cardiology.

### Decision analysis
At what predicted probability would a sponsor actually act? How many trials would we
correctly kill, and how many successful ones would we wrongly kill? Put a cost on both.

---

## 7. Definition of done

- [ ] Floor baselines reproduced and logged
- [ ] All three encoders producing aligned arrays
- [ ] Fusion model trained, numbers on the official test split
- [ ] Ablation table, per phase
- [ ] Error analysis by disease area
- [ ] Decision-threshold analysis
- [ ] README written, including the section 2 differentiator and the dataset version
- [ ] Repo clean, runs from a fresh clone

---

## 8. Expectations

**We will probably not beat HINT.** It is a purpose-built research model with a
hierarchical architecture. Published numbers are roughly PR-AUC 0.57 / 0.65 / 0.81 for
Phase I / II / III — check these against whichever dataset version we pick before quoting them.

Getting within a few points using simpler components, with ablations that explain *why*,
is a better story than a vague state-of-the-art claim nobody believes. Do not inflate
results. An honest "we got X, here's what carried the signal" survives interview
questions; a padded number does not.

Also: **log failed experiments in the results table, don't delete them.** A good chunk of
the error analysis comes from things that didn't work.

---

## 9. Decisions log

Append here. Newest at the bottom. One line each.

| Date | Decision | Who |
|---|---|---|
| 2026-11-09 | Dataset version: original Fu et al. TOP, official 2014 temporal split (not TrialBench) | Aditya |
| 2026-11-09 | Person A (data and features) = Ananya, Person B (modelling and eval) = Aditya | Aditya |
| 2026-11-09 | Section 5 contract agreed, with the (trial_id, phase) clarification | Aditya |
| 2026-11-09 | Eval harness merged (`eval/harness.py`, PR #2). Results CSV has two extra columns, `n` and `pos_rate` | Aditya |
| 2026-11-09 | `.npy` files are not shared through git. Each person regenerates them locally with a script in `features/` from the public TOP CSVs | Aditya |
| 2026-11-09 | Repo made public. Branch protection on `main` enabled; all changes go through a PR | Aditya |

---

## 10. Open questions

- [x] Which dataset version (section 3): Fu et al. original, decided 2026-11-09
- [ ] **Validation set.** The contract has no validation split but TOP ships `*_valid.csv`.
  Proposal: add `split_valid.npy` (the official valid rows), keep `split_train` train-only.
  Ananya to confirm before building the arrays; Aditya then adds it to the harness checks.
- [ ] Do we report Phase I at all, or focus on II and III where the signal is stronger?
- [ ] Cost assumptions for the decision analysis — where do we source trial cost figures?

---

## 11. Conventions

- Separate directories per owner: `features/` (A), `eval/` and `models/` (B). Keeps git
  merges boring.
- Results go in one CSV, appended, never overwritten. Every row: date, config, phase,
  PR-AUC, ROC-AUC, notes.
- Random seeds fixed and recorded. An unreproducible result is not a result.
- No clinical claims anywhere in the repo. The README says: *research prototype, not
  validated for clinical use.* Overclaiming on medical results reads badly to anyone
  who knows the field.

---

## 12. Using this with Claude

We each have our own Claude — separate chats, separate memory. Nothing syncs between us
automatically. **This file is the sync.** What you write here, my Claude sees when I pull,
and the reverse.

Start a session by pointing Claude at this file. Then say what you're working on. Before
you finish for the day, update sections 9 and 10 so the other person's Claude picks up
where you left off.

---

## 13. Status and next up (update this every session)

**Last updated:** 2026-11-09, by Aditya.

**Done:** repo public with `main` protected; dataset and roles decided; section 5 contract agreed;
evaluation harness merged with a self-check (`python -m eval.test_harness`); raw TOP CSVs
counted; leakage columns identified.

**Getting the raw data** (gitignored, ~40 MB, run from the repo root):

```
mkdir -p data/raw && cd data/raw
for p in I II III; do for s in train valid test; do
  curl -fLO https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/phase_${p}_${s}.csv
done; done
```

**Next up, Person A (Ananya), in `features/`:**
1. Resolve the validation-set question in section 10.
2. Script that reads the nine CSVs and writes `trial_ids`, `labels`, `phase`, `split_train`,
   `split_test` to `data/processed/`. Row order must be identical across all arrays.
3. Morgan fingerprints into `drug_features.npy` **first**, so B is unblocked. Some trials list
   several drugs in `smiless`: decide how to combine them and write it in section 9.
4. Run `eval.harness.load_contract("data/processed")`; if it passes, tell B.
5. Then ICD embeddings, then PubMedBERT text embeddings (2 rows have empty `criteria`).

**Next up, Person B (Aditya), in `eval/` and `models/`:**
1. Wait for fingerprints; meanwhile build the baseline scripts against fake arrays.
2. Logistic regression and XGBoost per phase, logged through `eval.harness.evaluate`.

**Branches and PRs:** one branch per task (`ananya/...`, `aditya/...`), merge through a PR that
the other person approves. Never push to `main`.
