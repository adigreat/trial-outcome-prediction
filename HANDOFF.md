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

### Shared (redistributed 2026-11-10, see section 9)
The original rule was "ablations, error analysis and decision analysis are split one each". Now that
the features are finished, the remaining work is split like this:

| Work | Owner | Files |
|---|---|---|
| Fusion model (MLP) | Aditya | `models/` |
| Decision analysis | Aditya | `eval/decision.py`, `models/decision_run.py` |
| Error analysis by disease area | Ananya | `models/error_analysis.py`, `models/test_error_analysis.py` |
| Ablation table and multi-seed runs | Ananya | runs `models.baselines` (and the fusion CLI once it exists); rows go to `results/` |
| HINT comparison numbers | Ananya | sections 8 and 13 of this file |
| Fresh-clone test and `requirements.txt` | Ananya | repo root |
| Final README | Ananya drafts, Aditya adds the fusion numbers | `README.md` |

Ananya may edit the files listed against her name even though they sit under `models/`. Everything
else in `eval/` and `models/` stays Aditya's.

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
| 2026-11-10 | Dataset pinned to TOP upstream commit `8dc0497`; the 9 CSVs are hash-checked on load. Row key is `(trial_id, phase)`; row order is phase → train/valid/test → nctid | Ananya |
| 2026-11-10 | Contract additions (nothing renamed or reshaped): `split_valid.npy` (official TOP valid, 907 rows; `split_train` = official train only, 8143), `manifest.json`, `feature_names/`, `trial_meta.csv`, `validation_report.md` | Ananya |
| 2026-11-10 | Blocks not built yet are written as `(n, 0)` float32 placeholders so `load_contract()` works with fingerprints alone; `manifest.json` `blocks.*.status` says `ready` or `pending` | Ananya |
| 2026-11-10 | Drug = Morgan r=2, 2048 bits, OR over the trial's drugs, float32 0/1 | Ananya |
| 2026-11-10 | Disease = ICD-10 multi-hot (chapter + 3-char category + full code, ≥5 train trials, + 3 OOV flags = 1983 dims), vocab fit on train only. The "learned embedding" is learned by B's model as a linear layer on this | Ananya |
| 2026-11-10 | Text = frozen `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT), masked mean pool per criterion bullet, then [inclusion mean ‖ exclusion mean] = 1536 dims | Ananya |
| 2026-11-10 | PRs #3, #4, #5 merged with `gh pr merge --admin` on Aditya's explicit instruction, because Ananya had not reviewed them | Aditya |
| 2026-11-10 | Results CSV gains `pr_auc_hint`: average precision on predictions cut to 0/1 at 0.5, which is how HINT's code computes its "PR-AUC". `pr_auc` stays the standard threshold-free metric. Use `pr_auc_hint` only to compare with HINT's paper, and label it | Aditya |
| 2026-11-10 | `eval/decision.py`: threshold table (true kills, wrong kills, net value). Costs are inputs; the source is still open (section 10) | Aditya |
| 2026-11-10 | Model: first real floor run (drug, disease, drug+disease; table in section 13). Docs rule added to `CLAUDE.md`: HANDOFF.md, EASY_LANGUAGE.md and README.md are updated together whenever the state changes | Aditya |
| 2026-11-10 | Answer to Ananya: keep the 106 trials that are in train for one phase and valid for another. We train one model per phase, so there is no leak | Aditya |
| 2026-11-10 | Answer to Ananya: the multi-hot disease block is fine as is. No dense version for now (it is already the strongest block) | Aditya |
| 2026-11-10 | The PubMedBERT text step is run on Aditya's machine (RTX 3050 6 GB), because Ananya's workspace has no GPU. Same code, `python -m features.build text` | Aditya |
| 2026-11-10 | Decision-analysis costs: money saved per correctly stopped trial = average per-study cost from the ERG report for HHS ASPE (Sertkaya et al., 2014): $3.80M / $13.35M / $19.89M for Phase 1 / 2 / 3 (US, Medidata contract data from 2004 on, not inflation-adjusted). Value lost per wrongly stopped success has no trusted source, so it is an assumption: 1x, 3x and 10x the trial cost, all reported. Kill threshold is chosen on valid and reported on test (`models/decision_run.py`) | Aditya |
| 2026-11-10 | Error analysis by disease area = ICD-10 chapter letter from the disease block (`models/error_analysis.py`). Letter D mixes benign neoplasms and blood disorders | Aditya |
| 2026-11-10 | Text block built on Aditya's machine (PubMedBERT, 1536 dims). All three blocks are ready, so the XGBoost ablation over all seven block combinations is logged in `results/results.csv` | Aditya |
| 2026-11-10 | PR #8 approved by Ananya and merged normally (no admin merge) | Aditya |
| 2026-11-10 | Work redistributed (table in section 4). Ananya takes: error analysis (she now owns `models/error_analysis.py` and its test), the ablation table and multi-seed runs, the HINT number hunt, the fresh-clone test with `requirements.txt`, and the README draft. Aditya keeps: fusion model, decision analysis. This replaces "neither person owns both" in section 4 | Aditya |
| 2026-11-10 | Ananya's laptop has no CUDA GPU (Snapdragon X, Windows on ARM, 16 GB RAM), so she does not run PubMedBERT. Aditya sends her `text_features.npy` directly (not through git). Her zips were compared with Aditya's local build: drug, disease, labels, phase, splits and trial IDs are identical, so the row order matches | Aditya |
| 2026-11-10 | The redistribution PR (docs only) is admin-merged on Aditya's explicit instruction, so that Ananya sees her new list straight away | Aditya |

---

## 10. Open questions

- [x] Which dataset version (section 3): Fu et al. original, decided 2026-11-09
- [x] Validation set: `split_valid.npy` added to the contract and the harness (2026-11-09).
- [x] Phase I: report all three phases, flag Phase I as noisy (section 9, 2026-11-09).
- [x] Cost assumptions for the decision analysis: decided 2026-11-10, see section 9 (ASPE/ERG 2014
  per-study costs; the value of a lost success is a stated 1x / 3x / 10x assumption).
- [ ] Which HINT numbers are the right comparison for our release (section 13)?
- [x] Ananya's Claude: she works from her own environment and opened PR #6, so this is not blocking.
- [x] (Aditya) 106 trials in train for one phase and valid for another: keep them (section 9, 2026-11-10).
- [x] (Aditya) Disease block as a multi-hot: fine as is (section 9, 2026-11-10).
- [ ] (Ananya) 115 Phase III test trials carry an ICD chapter P (perinatal) code and 194 a chapter Z code, which looks high. Is that real in TOP's `icdcodes`, or a parsing artifact? Seen in `results/error_analysis.csv`
- [x] Fingerprints shipped 2026-11-10: `python -m features.build fingerprints`, see `features/README.md`. Disease block also ready; text pending a GPU run on Ananya's machine

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

**Last updated:** 2026-11-10, by Aditya.
PRs #3 to #8 are merged into `main`. #3 to #7 were admin-merged on Aditya's instruction, without a
review from the other person. #8 was approved by Ananya and merged normally. **The work was
redistributed on 2026-11-10: see the table in section 4 and the two "Next up" lists below.** For #6 (Ananya's features pipeline), Aditya first merged `main`
into `ananya/features-pipeline` and pushed it, keeping both sides' rows in `HANDOFF.md`; her code was
not changed. Ananya: please look over what landed (`features/`, section 9) and say if anything is wrong.

**Real numbers so far (2026-11-10, Aditya's machine, seed 0, untuned defaults).** Arrays built locally
with `python -m features.build fetch / fingerprints / disease / text`, then
`python -m models.baselines --blocks ...`. Test split, XGBoost, PR-AUC (the standard, threshold-free
one) with ROC-AUC in brackets. The "always predict success" floor for PR-AUC is the positive rate.
Logistic regression rows, `pr_auc_hint` and the valid split are in `results/results.csv` (84 rows).

| blocks (xgboost) | Phase I (n=627) | Phase II (n=1654) | Phase III (n=1146) |
|---|---|---|---|
| positive rate (PR-AUC floor) | 0.553 | 0.555 | 0.750 |
| drug | 0.609 (0.556) | 0.607 (0.557) | 0.829 (0.628) |
| disease | 0.652 (0.593) | 0.682 (0.631) | 0.858 (0.706) |
| text | 0.628 (0.586) | 0.667 (0.617) | 0.851 (0.678) |
| drug+disease | 0.626 (0.588) | 0.660 (0.603) | 0.852 (0.665) |
| drug+text | 0.630 (0.594) | 0.685 (0.635) | 0.849 (0.670) |
| disease+text | 0.645 (0.594) | 0.683 (0.629) | 0.836 (0.666) |
| drug+disease+text | 0.655 (0.616) | 0.668 (0.626) | 0.844 (0.666) |

What this says, and does not say:
- Disease and text each carry real signal and are close to each other. Drug fingerprints are the
  weakest block in every phase.
- Stacking blocks does not help a plain XGBoost: no combination clearly beats disease alone, and in
  Phase III every combination is below it. Differences of a point or two are noise (one seed, untuned).
- Phase III PR-AUC looks high only because 75% of its test trials succeed. Its ROC-AUC (about 0.67 to
  0.71) is the fairer read.
- Nothing here is comparable to HINT's published numbers (see "Not verified").

**Error analysis by disease area** (`python -m models.error_analysis`, rows in
`results/error_analysis.csv`): cancer is the hard area. It has the lowest success rate (0.44 / 0.46 /
0.60 against 0.55 / 0.56 / 0.75 overall) and within cancer the models are near chance in Phase I
(ROC-AUC 0.51 disease-only, 0.54 with all blocks). In Phase III the disease-only model under-predicts
cancer successes (mean prediction 0.41, actual 0.60). Circulatory was near chance with disease only
(0.53 / 0.51 in Phase II / III) and better with all three blocks (0.58 / 0.69), on about 100 trials.

**Decision analysis** (`python -m models.decision_run`, rows in `results/decision_analysis.csv`, costs in
section 9): with the kill threshold picked on valid, these models only come out ahead on test when
a wrong kill costs no more than the trial itself, and not in Phase III. At 3x and 10x they lose money
or stop almost nothing. Honest reading: the current models are not good enough to stop trials on.

**Done:** repo public with `main` protected (1 approving review required); dataset and roles
decided; section 5 contract agreed, including `split_valid.npy`; evaluation harness with a self-check
(`python -m eval.test_harness`); `pr_auc_hint` column; decision-analysis code (`eval/decision.py`);
baselines (logreg, xgboost) run on the real arrays for all seven block combinations; features
pipeline from Ananya, with all three blocks built on Aditya's machine (text = PubMedBERT, 1536 dims);
error analysis by disease area; decision analysis on real predictions; leakage columns identified;
Phase I decision made (report all three).

**Where we stopped (2026-11-10, Aditya's session):**
- `main` is up to date and there are no open PRs except whatever is listed by `gh pr list`.
- Still to do: fusion MLP (the main model), tuning and more seeds, then the final README. The
  XGBoost ablation, error analysis and decision analysis exist but should be redone on the fusion model.
- The text block took about 25 minutes on an RTX 3050 6 GB (197,232 criterion bullets). 3 rows have an
  all-zero text vector (no usable criteria).
- Claude must not merge an unreviewed PR or use `gh pr merge --admin` unless Aditya says so in
  that session. Aditya said so for PRs #3 to #7.
- Ananya reviewed and approved PR #8, and said she will take the Windows UTF-8 fix and the ICD chapter
  P / Z question in a separate PR. She has not commented on PRs #3 to #7.
- Ananya's laptop: Snapdragon X (8 cores, Windows on ARM), 16 GB RAM, Adreno GPU, so no CUDA. Everything
  except the PubMedBERT step runs on CPU there (XGBoost, the error analysis, and the small fusion MLP
  once it exists). Not checked: whether `rdkit`, `xgboost` and `torch` install on Windows-on-ARM
  Python. If one does not, use the x64 Python build, which runs under emulation. Her builds so far ran
  in a Linux workspace, so her laptop is a real second environment for the fresh-clone test.
- Ananya's two zips (fingerprints; fingerprints + disease) match Aditya's local build exactly. Neither
  has the text block; it exists only on Aditya's machine until he sends her `text_features.npy`.
- Windows bug found while building (2026-11-10): `python -m features.build fingerprints` crashes at the
  end with `UnicodeEncodeError ... '⚠'` because the validation report is written without
  `encoding="utf-8"`. Workaround that needs no code change: set `PYTHONUTF8=1` before running. The
  proper fix is one line in `features/` (Ananya's folder), so it is hers to make.

**Not verified, do not quote yet:**
- HINT's published PR-AUC for OUR release. 0.57 / 0.65 / 0.81 has no source. The arXiv v3
  paper (Table 3) gives 0.764 / 0.602 / 0.618, but its test sets (312 / 789 / 1,141) differ from
  ours. The abstract says 0.772 / 0.607 / 0.623, so the abstract and Table 3 disagree. The PubMed
  record for the published HINT paper (35465223) reports F1 0.665 / 0.620 / 0.847 on 627 / 1,653 /
  1,140 trials, which matches our CSV sizes, so that is probably the right version to compare
  against. We could not retrieve its PR-AUC, so find it in the full text first.
- HINT's metric definition (checked in `HINT/model.py`, `Interaction.evaluation`, through a fetch
  summary, so confirm against the file before quoting): ROC-AUC uses raw probabilities, but PR-AUC is
  `average_precision_score(label, predict)` called AFTER the predictions are cut to 0/1 at a fixed
  0.5. That is not a threshold-free PR-AUC. Ours (`pr_auc`) scores raw probabilities, the standard
  definition. `pr_auc_hint` in the results CSV mimics HINT. HINT's bootstrap defaults to 20 resamples.
- XGBoost settings (200 trees, depth 4) are untuned defaults, so the baselines are a first floor.
- Trial cost figures for the decision analysis: candidate sources are in section 10, none
  verified against the primary paper.

**Ananya's Claude cannot work yet (2026-11-10):** Ananya sees "The Claude GitHub App on adigreat
isn't connected. Only adigreat can connect it." The app IS installed on `adigreat` (all repos)
and connected on Aditya's Claude account. This only affects cloud/web Claude sessions. It does
not block her working locally: clone the repo, use her own `gh` login, run Claude Code in the
terminal. If she needs cloud sessions, the options are a fork under her own account (install the
app there, PR back to `adigreat`) or moving the repo to a GitHub org both of us belong to.

**Getting the data and the arrays** (run from the repo root; `data/` is gitignored; on Windows set
`PYTHONUTF8=1` first, see "Where we stopped"):

```
pip install -r requirements.txt -r features/requirements.txt
python -m features.build fetch          # clones TOP at the pinned commit into data/raw/top/
python -m features.build fingerprints
python -m features.build disease
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the build for your CUDA
pip install -r features/requirements-text.txt
python -m features.build text           # PubMedBERT, about 25 min on a 6 GB GPU, resumable
```

**Next up, Person A (Ananya).** Updated 2026-11-10 after the redistribution (section 4). None of this
needs a GPU.
1. Fix the Windows `UnicodeEncodeError` in `features/` (write the validation report with
   `encoding="utf-8"`).
2. Answer the ICD chapter P / Z question in section 10.
3. Error analysis is yours now (`models/error_analysis.py`, `models/test_error_analysis.py`):
   a. split ICD letter D into benign neoplasms (D00 to D49) and blood disorders (D50 to D89);
   b. rerun `python -m models.error_analysis` and update the paragraph in this section;
   c. rerun it on the fusion model when Aditya has merged it, and write the result up for the README.
4. Ablation table with several seeds: `python -m models.baselines --blocks <combo> --seed <n>` for the
   seven block combinations and seeds 1 to 4 (seed 0 is logged). Report mean and spread per phase. Needs
   `text_features.npy` from Aditya (put it in `data/processed/`, then check that
   `load_contract("data/processed")` passes and `manifest.json` still describes the blocks). Do the same
   with the fusion CLI once it exists.
5. Find HINT's PR-AUC for our release: the full text of the published HINT paper (PubMed 35465223),
   test sets of 627 / 1,653 / 1,140 trials. Record the numbers with the table or page they come from in
   section 8, and close the open question in section 10. Also confirm the metric definition in
   `HINT/model.py` (see "Not verified").
6. Fresh-clone test on your laptop: clone, create the venv, install, run the build commands above and
   the four self-checks. Fix what is missing in the root `requirements.txt` (`torch` is not listed) and
   write down anything Windows-on-ARM specific.
7. Draft the final README: the section 2 pitch, the dataset version, how to run, the results table,
   the error analysis, limits. Leave marked gaps for the fusion numbers.

**Next up, Person B (Aditya), in `eval/` and `models/`:**
1. Send `text_features.npy` to Ananya directly (Drive or similar, never git). She is blocked on it for
   item 4 of her list.
2. Fusion MLP: concatenate the three blocks, small MLP head, tune on valid. Give it the same CLI shape
   as `models.baselines` (`--blocks`, `--seed`) and log through `eval.harness.evaluate`, so Ananya can
   run the ablation and the error analysis on it. PyTorch is installed in the venv (cu124).
3. Rerun the decision analysis on the fusion model.
4. Add the fusion numbers to Ananya's README draft.
5. Review Ananya's PRs.

**Branches and PRs:** one branch per task (`ananya/...`, `aditya/...`), merge through a PR that
the other person approves. Never push to `main`.
