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
  split_train.npy         # indices into the arrays above (official *_train.csv rows)
  split_valid.npy         # indices (official *_valid.csv rows) — added 2026-11-09
  split_test.npy          # indices (official *_test.csv rows)
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
- **`split_train` / `split_valid` / `split_test` = the official `*_train.csv` / `*_valid.csv` /
  `*_test.csv` rows.** The three index sets must not overlap. Tune on valid, report on test.
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
hierarchical architecture. **Do not quote 0.57 / 0.65 / 0.81 — we could not source them
(2026-11-10).** What we did find, from the HINT arXiv paper (2102.04252 v3, Table 3, PR-AUC
Phase I / II / III): 0.764 / 0.602 / 0.618, on a different test set (312 / 789 / 1,141 trials,
registration-date cutoffs in 2014, 15% of train held out for validation). That is not our
release (627 / 1,654 / 1,146), so it is not a like-for-like comparison. See section 13.

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
| 2026-11-09 | Contract gains `split_valid.npy` (official valid rows). `load_contract` now requires it and checks the three splits do not overlap. Ananya must produce it | Aditya (Ananya to confirm) |
| 2026-11-09 | Report all three phases, Phase I included. Phase I is the smallest (627 test trials), so show `n` beside every number and say it is noisy. Dropping it would look like cherry-picking and HINT reports it | Aditya |
| 2026-11-09 | Baselines in `models/baselines.py` (logreg, xgboost). One model per phase, fit on train, logged on valid and test. Project venv `.venv/` from `requirements.txt` | Aditya |

---

## 10. Open questions

- [x] Which dataset version (section 3): Fu et al. original, decided 2026-11-09
- [x] Validation set: `split_valid.npy` added to the contract and the harness (2026-11-09).
- [x] Phase I: report all three phases, flag Phase I as noisy (section 9, 2026-11-09).
- [ ] Cost assumptions for the decision analysis. Candidate sources found 2026-11-10 (from
  search summaries, open the primary source before quoting): Sertkaya et al., Clin Trials 2016
  (per-study cost by phase and therapeutic area: Phase 1 about $1.4M to $6.6M, Phase 2 about
  $7.0M to $19.6M, Phase 3 about $11.5M to $52.9M); CBO per-drug figures (about $28M / $65M /
  $282M for Phase 1 / 2 / 3, per drug completing all phases, so not per trial). Decide which
  one the decision analysis uses and state it.
- [ ] Which HINT numbers are the right comparison for our release (section 13)?
- [ ] Ananya's Claude cannot work on the repo yet (section 13). Pick a route.

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

**Last updated:** 2026-11-10, by Aditya (with Claude). Research notes only: no code changed.
Still no activity from Ananya (checked `gh pr list`, branches, reviews on 2026-11-10).

**Done:** repo public with `main` protected (1 approving review required); dataset and roles
decided; section 5 contract agreed, now including `split_valid.npy`; evaluation harness merged
with a self-check (`python -m eval.test_harness`); baselines written and tested on fake data;
raw TOP CSVs counted; leakage columns identified; Phase I decision made (report all three).

**Where we stopped (2026-11-09, end of Aditya's session):**
- PR #3 (section 5 agreed, status section) and PR #4 (baselines, `split_valid`, `requirements.txt`)
  are OPEN with no reviews. Ananya is the only other collaborator, so she has to approve them.
  Merge #3 first. #4 is stacked on #3's branch (`aditya/section5-agreed`), so its diff shrinks
  once #3 is in. Until both merge, `main` does not have the contract status, `split_valid`,
  or the baselines.
- Aditya's local checkout was on branch `aditya/baselines`. After #3 merges: `git checkout main`,
  `git pull`, and if #4 shows conflicts, rebase `aditya/baselines` onto `main`.
- Claude must not merge an unreviewed PR or use `gh pr merge --admin` unless Aditya says so in
  that session. It was blocked once already. Aditya can still choose to do it.
- No activity from Ananya yet as of this update: no branches, reviews, or comments. We have not
  verified that she has seen the invite or the PRs, nor that she agreed to `split_valid`.
  Check `gh pr list` and `git branch -r` first.
- `results/results.csv` does not exist yet. No real numbers have been produced.

**Not verified, do not quote yet:**
- HINT's published PR-AUC for OUR release. 0.57 / 0.65 / 0.81 has no source. The arXiv v3
  paper (Table 3) gives 0.764 / 0.602 / 0.618, but its test sets (312 / 789 / 1,141) differ from
  ours. The abstract says 0.772 / 0.607 / 0.623, so the abstract and Table 3 disagree. The PubMed
  record for the published HINT paper (35465223) reports F1 0.665 / 0.620 / 0.847 on 627 / 1,653 /
  1,140 trials, which matches our CSV sizes, so that is probably the right version to compare
  against. We could not retrieve its PR-AUC, so find it in the full text first.
- The metric definition HINT uses: the paper only says "PR-AUC" with bootstrap error bars, no
  formula. Not yet checked in the HINT code. sklearn's `average_precision_score` is the
  step-wise version and may differ slightly.
- XGBoost settings (200 trees, depth 4) are untuned defaults, so the baselines are a first floor.
- Trial cost figures for the decision analysis: candidate sources are in section 10, none
  verified against the primary paper.

**Ananya's Claude cannot work yet (2026-11-10):** Ananya sees "The Claude GitHub App on adigreat
isn't connected. Only adigreat can connect it." The app IS installed on `adigreat` (all repos)
and connected on Aditya's Claude account. This only affects cloud/web Claude sessions. It does
not block her working locally: clone the repo, use her own `gh` login, run Claude Code in the
terminal. If she needs cloud sessions, the options are a fork under her own account (install the
app there, PR back to `adigreat`) or moving the repo to a GitHub org both of us belong to.

**Getting the raw data** (gitignored, ~40 MB, run from the repo root):

```
mkdir -p data/raw && cd data/raw
for p in I II III; do for s in train valid test; do
  curl -fLO https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/phase_${p}_${s}.csv
done; done
```

**Next up, Person A (Ananya), in `features/`:**
1. Validation set is decided: also write `split_valid.npy` (section 5).
2. Script that reads the nine CSVs and writes `trial_ids`, `labels`, `phase`, `split_train`,
   `split_valid`, `split_test` to `data/processed/`. Row order must be identical across all arrays.
3. Morgan fingerprints into `drug_features.npy` **first**, so B is unblocked. Some trials list
   several drugs in `smiless`: decide how to combine them and write it in section 9.
4. Run `eval.harness.load_contract("data/processed")`; if it passes, tell B.
5. Then ICD embeddings, then PubMedBERT text embeddings (2 rows have empty `criteria`).

**Next up, Person B (Aditya), in `eval/` and `models/`:**
1. DONE: `models/baselines.py` (logreg + xgboost per phase), tested on fake arrays with
   `python -m models.test_baselines`. Needs the real arrays to produce real numbers.
2. When Ananya ships fingerprints: `python -m models.baselines --blocks drug`, then review the
   first real rows in `results/results.csv`.
3. Setup for a fresh clone: `python -m venv .venv`, then `.venv/Scripts/pip install -r requirements.txt`
   (Windows; `.venv/bin/pip` elsewhere).

**Branches and PRs:** one branch per task (`ananya/...`, `aditya/...`), merge through a PR that
the other person approves. Never push to `main`.
