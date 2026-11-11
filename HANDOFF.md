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

  These are close to the 1,787 / 6,102 / 4,576 in the MediTab table. The other set
  (1,160 / 4,449 / 3,436) is HINT's train + valid in the published paper (1,044 + 116, 4,004 + 445,
  3,092 + 344), so it IS this release, within 1 to 6 trials per split (corrected 2026-11-11, Ananya).
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
hierarchical architecture.

**HINT's numbers for our release (found 2026-11-11, Ananya).** Source: the published paper, Fu et al.,
*Patterns* 3:100445 (2022), Table 2, "Empirical results of various approaches for phase-level-outcome
predictions on test sets", mean ± SD of 30 runs with different seeds. Test sets 627 / 1,653 / 1,140
(ours 627 / 1,654 / 1,146), train 1,044 / 4,004 / 3,092, valid 116 / 445 / 344. Read from the open
copy at Temple University's repository (scholarshare.temple.edu), the PDF of the published article.

| HINT, Phase | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|
| I | 0.567 ± 0.010 | 0.665 ± 0.010 | 0.576 ± 0.008 |
| II | 0.629 ± 0.009 | 0.620 ± 0.008 | 0.645 ± 0.006 |
| III | 0.811 ± 0.007 | 0.847 ± 0.009 | 0.723 ± 0.006 |

How to compare:
- **ROC-AUC is the like-for-like number.** Both sides compute it on probabilities.
- **PR-AUC is ambiguous.** The paper defines PR-AUC over thresholds, but HINT's released code at our
  pinned commit (`HINT/model.py`, `evaluation`, lines 96 to 100) cuts predictions to 0/1 at 0.5 and then
  calls `average_precision_score`, inside a 20-resample bootstrap (`bootstrap_test`, lines 137 to 165).
  Compare HINT's PR-AUC with our `pr_auc_hint`, label it, and never with our `pr_auc`.
- 0.57 / 0.65 / 0.81 (the old line in this section) matches 0.567 and 0.811 for Phases I and III; its
  0.65 for Phase II does not (it is close to HINT's Phase II ROC-AUC, 0.645). Use the table above. The arXiv v3 numbers (0.764 / 0.602 / 0.618, Table 3) are on a different test set
  (312 / 789 / 1,141). Do not use them.

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
| 2026-11-10 | Fusion model = `models/fusion.py`: one encoder per block (dropout 0.2, linear to 128, ReLU, dropout 0.5), concatenated, then a 128-unit head. AdamW, lr 1e-3, weight decay 1e-2, batch 128, at most 60 epochs. One model per phase. The official valid split picks the stopping epoch (ROC-AUC, patience 8), so valid scores for this model are optimistic; test is never used in training. Settings were fixed before the first test run and are not tuned. 0/1 blocks are fed as they are, the text block is standardised on train | Aditya |
| 2026-11-10 | The fusion model runs through the existing commands (`models.baselines --models fusion`, `models.error_analysis --model fusion`, `models.decision_run --model fusion`). For that, Aditya changed three lines in `models/error_analysis.py` (Ananya's file) to call the shared `fit` helper in `models/baselines.py` | Aditya |
| 2026-11-11 | PR #10 (fusion MLP) approved by Ananya and merged normally. Ananya re-ran fusion drug+disease seed 0 on CPU: test PR-AUC 0.618 / 0.670 / 0.870 against Aditya's GPU 0.634 / 0.679 / 0.871, so fusion rows depend on the device; quote the multi-seed mean | Ananya |
| 2026-11-11 | All text I/O in `features/` is UTF-8 (fixes the Windows `UnicodeEncodeError`). `PYTHONUTF8=1` is no longer needed. A test runs the pipeline with Python's `EncodingWarning` turned into an error | Ananya |
| 2026-11-11 | ICD chapters P and Z in TOP are real (verbatim in `icdcodes`, not a parsing slip) but are side effects of TOP's disease-name to ICD matching: "diabetes" also gets P70.2 (neonatal diabetes) and Z83.3 (family history), "HIV" gets Z21 / Z71.7 / Z11.4, "cystic fibrosis" gets P09.4. Only 2 of 115 (P) and 21 of 194 (Z) Phase III test trials have nothing else, and the shares are the same in train and test. The disease block keeps them (they are inputs, and the features stay as TOP gives them); the error analysis counts P and Z as an area only when the trial has no other chapter | Ananya |
| 2026-11-11 | Error analysis splits letter D at D50 using the `icd_category_D*` columns: "neoplasm in situ/benign (D00-D49)", "blood/immune (D50-D89)", and "D (unresolved)" for D codes outside the train vocabulary (15 test rows). Labels for P and Z are now "... (only code)", so old and new rows in `results/error_analysis.csv` cannot be confused | Ananya |
| 2026-11-11 | HINT comparison: published Patterns 2022 Table 2 (section 8). ROC-AUC is the like-for-like metric; HINT's PR-AUC is compared only with `pr_auc_hint` | Ananya |
| 2026-11-11 | Root `requirements.txt` gains `torch` (CPU is enough): `models.test_baselines` imports the fusion model, so a fresh clone could not pass the four checks without it | Ananya |
| 2026-11-11 | Aditya ran the two items Ananya was blocked on for lack of `text_features.npy`: logreg and XGBoost seeds 1 to 4 for text, drug+text, disease+text and drug+disease+text (192 rows), and the fusion error analysis with the D split and P/Z rule. Every model and block combination now has 5 seeds | Aditya |
| 2026-11-11 | Spreads in the README are sample standard deviations (n - 1) everywhere, to match the HINT table. The fusion table in section 13 was written with the population formula, so a few of its spreads are 0.001 smaller | Aditya |
| 2026-11-11 | HINT's Table 2 numbers and its metric code were checked a second time by Aditya: the paper's full text from Europe PMC (PMC9024011) and `HINT/model.py` at commit `8dc0497`. Both agree with section 8 | Aditya |
| 2026-11-11 | Answer to Ananya: the P/Z side-effect codes stay in the disease features, as TOP gives them. Removing them would mean a new disease block and rerunning every result, for codes the models can learn to ignore. The README limits section already says we document this and do not fix it | Aditya |
| 2026-11-11 | Ananya's PR #11 approved by Aditya (reviewed at commit `b27769d`) | Aditya |

---

## 10. Open questions

- [x] Which dataset version (section 3): Fu et al. original, decided 2026-11-09
- [x] Validation set: `split_valid.npy` added to the contract and the harness (2026-11-09).
- [x] Phase I: report all three phases, flag Phase I as noisy (section 9, 2026-11-09).
- [x] Cost assumptions for the decision analysis: decided 2026-11-10, see section 9 (ASPE/ERG 2014
  per-study costs; the value of a lost success is a stated 1x / 3x / 10x assumption).
- [x] Which HINT numbers are the right comparison for our release: Patterns 2022 Table 2 (section 8, 2026-11-11).
- [x] Ananya's Claude: she works from her own environment and opened PR #6, so this is not blocking.
- [x] (Aditya) 106 trials in train for one phase and valid for another: keep them (section 9, 2026-11-10).
- [x] (Aditya) Disease block as a multi-hot: fine as is (section 9, 2026-11-10).
- [x] (Ananya) ICD chapter P / Z counts: real in TOP, side effects of its disease-name matching, not our parsing (section 9, 2026-11-11).
- [x] (Aditya) P/Z side-effect codes in the disease *features*: keep them, as TOP gives them (section 9, 2026-11-11). No rebuild, no new runs.
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

**Last updated:** 2026-11-11, by Aditya, after Ananya's session the same day. Both
sessions are summarised under "Where we stopped (2026-11-11)".
PRs #3 to #10 are merged into `main` (#9 and #10 on 2026-11-10/11; #10 approved by Ananya). #3 to #7 were admin-merged on Aditya's instruction, without a
review from the other person. #8 was approved by Ananya and merged normally. **The work was
redistributed on 2026-11-10: see the table in section 4 and the two "Next up" lists below.**
The fusion model is built, run and merged (PR #10). For #6 (Ananya's features pipeline), Aditya first merged `main`
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

**Fusion MLP (2026-11-10, `models/fusion.py`, settings in section 9).** Test split, PR-AUC, mean and
standard deviation over 5 training seeds (0 to 4), mean ROC-AUC in brackets. The spread is over seeds
only; it says nothing about how much the test sample itself would vary. All 210 rows (valid and test)
are in `results/results.csv` with `"model": "fusion"`.

| blocks (fusion MLP) | Phase I (n=627) | Phase II (n=1654) | Phase III (n=1146) |
|---|---|---|---|
| positive rate (PR-AUC floor) | 0.553 | 0.555 | 0.750 |
| best XGBoost row above (disease, 1 seed) | 0.652 (0.593) | 0.682 (0.631) | 0.858 (0.706) |
| drug | 0.628 ± 0.007 (0.565) | 0.612 ± 0.003 (0.563) | 0.835 ± 0.004 (0.630) |
| disease | 0.635 ± 0.010 (0.571) | 0.683 ± 0.002 (0.633) | 0.861 ± 0.001 (0.704) |
| text | 0.682 ± 0.007 (0.636) | 0.682 ± 0.003 (0.640) | 0.862 ± 0.003 (0.691) |
| drug+disease | 0.633 ± 0.003 (0.580) | 0.677 ± 0.006 (0.618) | 0.867 ± 0.004 (0.700) |
| drug+text | 0.682 ± 0.010 (0.636) | 0.688 ± 0.007 (0.640) | 0.872 ± 0.005 (0.696) |
| disease+text | 0.674 ± 0.009 (0.631) | 0.696 ± 0.005 (0.653) | 0.874 ± 0.003 (0.719) |
| drug+disease+text | 0.675 ± 0.012 (0.627) | 0.693 ± 0.005 (0.644) | 0.882 ± 0.002 (0.726) |

What the ablation says:
- With the fusion model, combining blocks helps, which it did not with XGBoost. All three blocks beat
  the best XGBoost row in every phase (by about 0.02, 0.01 and 0.02 PR-AUC).
- Phase I: the eligibility text carries the signal. Text alone (0.682) is as good as all three (0.675);
  adding disease or drug changes nothing beyond seed noise.
- Phase II: disease and text together (0.696) do the work. Adding the drug block does not help (0.693).
- Phase III: every block adds a little and all three is best (0.882; ROC-AUC 0.726).
- The drug fingerprint is the weakest block alone in every phase.
- The gains over XGBoost are small, and XGBoost has one seed. Ananya's multi-seed XGBoost runs are
  needed before saying how big the gap is.
- `pr_auc_hint` for all three blocks on test is 0.597 / 0.611 / 0.804 (mean of 5 seeds). Still not
  compared with HINT (see "Not verified").

**Fusion model, error analysis (seed 0, all three blocks):** cancer is still the hard area. ROC-AUC
within cancer is 0.55 / 0.59 / 0.57 against 0.63 / 0.65 / 0.73 overall, and in Phase III the model
still under-predicts cancer successes (mean prediction 0.45, actual 0.60). Circulatory is 0.69 / 0.61 /
0.73 on 30 / 105 / 80 trials.

**Fusion model, decision analysis (seed 0, all three blocks):** the reading does not change. At a 1x
loss the model comes out ahead in Phase I (+$137M, 154 trials stopped, 59 of them wrongly) and
Phase II (+$1,055M, 839 stopped, 380 wrongly), and loses in Phase III (-$438M). At 3x it loses in
every phase. At 10x it stops almost nothing. Not good enough to stop trials on.

**XGBoost with 5 seeds (2026-11-11, Ananya).** Seeds 1 to 4 added for drug, disease and drug+disease
(logreg and XGBoost, 144 rows, run in Ananya's Linux workspace; seed 0 is Aditya's). Test PR-AUC, mean
± SD (ROC-AUC): drug 0.610 ± 0.008 / 0.607 ± 0.003 / 0.829 ± 0.001; disease 0.650 ± 0.004 (0.591) /
0.680 ± 0.003 (0.630) / 0.858 ± 0.001 (0.704); drug+disease 0.628 ± 0.002 / 0.654 ± 0.006 / 0.853 ±
0.002. XGBoost barely moves with the seed, so the one-seed rows above were fair. Fusion all three beats
XGBoost disease by 0.025 / 0.013 / 0.024 PR-AUC, more than either model's seed spread. Re-running seed 0
here gave PR-AUC within 0.004 of Aditya's rows (different xgboost build).
Added by Aditya the same day, on his machine (seeds 1 to 4, 192 rows): text 0.634 ± 0.011 (0.596) /
0.672 ± 0.009 (0.625) / 0.846 ± 0.007 (0.672); drug+text 0.635 ± 0.009 / 0.671 ± 0.011 / 0.846 ± 0.004;
disease+text 0.641 ± 0.015 / 0.671 ± 0.008 / 0.839 ± 0.004; drug+disease+text 0.643 ± 0.018 (0.604) /
0.668 ± 0.009 (0.624) / 0.843 ± 0.005 (0.669). So with 5 seeds everywhere, XGBoost's best row is still
disease alone, and fusion all three beats XGBoost all three by 0.032 / 0.025 / 0.039 PR-AUC.

**Comparison with HINT (2026-11-11, Ananya; numbers in section 8).** ROC-AUC, fusion all three (5
seeds) against HINT (30 seeds): 0.627 vs 0.576 / 0.644 vs 0.645 / 0.726 vs 0.723. Our `pr_auc_hint`
against HINT's PR-AUC: 0.597 vs 0.567 / 0.611 vs 0.629 / 0.804 vs 0.811. So: level with HINT in
Phases II and III, above it in Phase I (the smallest, noisiest test set). Not a claim to beat it.

**Error analysis with the D split and the P/Z rule (2026-11-11, Ananya; XGBoost, disease block, seed
0; the fusion rerun waits for `text_features.npy`).** D00-D49 (in situ / benign neoplasms) behaves like
cancer: 156 / 387 / 118 test trials, 81-90% of them also carry a cancer code, success rate 0.41 in
Phase II and 0.61 in Phase III, and the model under-predicts them in Phase III (mean prediction 0.33).
Blood / immune (D50-D89) is small: 19 / 58 / 40 trials. With the P/Z rule, "perinatal (only code)" has
3 / 4 / 2 trials and "health-status factors (only code)" 37 / 26 / 21, so neither is a real area.
Fusion rerun with the same rules (Aditya, 2026-11-11, all three blocks, seed 0; the ALL and cancer
rows are identical to the 2026-11-10 run): D00-D49 has ROC-AUC 0.56 / 0.60 / 0.47, and in Phase III the
model predicts 0.42 where 61% succeed. Blood / immune: 0.63 in Phase II (58 trials) and 0.58 in
Phase III (40 trials); Phase I is too small to score.

The paragraphs below describe the XGBoost runs from 2026-11-10 (before the D split).

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
error analysis by disease area; decision analysis on real predictions; fusion MLP with a 5-seed
ablation, error analysis and decision analysis; leakage columns identified;
Phase I decision made (report all three).

**Where we stopped (2026-11-11, Ananya's session):**
- PR #10 reviewed (all checks re-run, fusion reproduced on CPU, see section 9), approved and merged.
- Windows UTF-8 bug fixed in `features/` (section 9). The `PYTHONUTF8=1` workaround is no longer needed
  once this PR is merged.
- ICD P / Z question answered (section 9 and 10). New open question for Aditya in section 10.
- Error analysis: D split and P/Z rule done and tested; rerun on XGBoost disease. Fusion rerun waits for
  `text_features.npy`.
- XGBoost seeds 1 to 4 done for the three combinations without text.
- HINT numbers found and the metric definition confirmed in the code (section 8).
- README drafted, with `TODO(Aditya)` markers where the fusion write-up is his.
- Fresh-clone test done on a clean Linux clone (see "Fresh-clone test" below; not an empty venv). Still to do on Ananya's
  Windows-on-ARM laptop.
- `text_features.npy` has not arrived yet.

**Where we stopped (2026-11-11, Aditya's session, after Ananya's):**
- Reviewed Ananya's PR (UTF-8 fix, error analysis, XGBoost seeds, HINT numbers, README draft): all four
  self-checks print `ok` and `pytest features/tests` gives 39 passed on Windows 11 without `PYTHONUTF8`.
  HINT's numbers and metric code checked independently (section 9).
- Ran what Ananya was blocked on (section 9): XGBoost and logreg seeds 1 to 4 for the four text
  combinations, and the fusion error analysis with the new rules. `results/results.csv` now has 630
  rows with no duplicates.
- README: both `TODO(Aditya)` notes filled, results table completed, error-analysis and
  decision-analysis sections updated, spreads made consistent.
- The section 10 question about P/Z codes in the disease features is answered: keep them (section 9).
  Section 10 has no open questions left.
- Ananya no longer needs `text_features.npy` for any run. She needs it only if she wants the text
  block in her laptop fresh-clone test without a multi-hour CPU build.

**Fresh-clone test (2026-11-11, Ananya, Linux workspace, Python 3.13):**
- Clean clone of this branch with no `data/`: `features.build fetch`, `fingerprints` and `disease` took 11
  seconds in total. `load_contract` passes, all four self-checks print `ok`, `pytest features/tests` gives
  39 passed, and the drug and disease arrays are byte-identical to the earlier build.
- Found: root `requirements.txt` had no `torch`, but `models.test_baselines` imports the fusion model, so
  a fresh clone could not pass the checks. Added (section 9). An import scan of every `.py` file finds
  exactly numpy, pandas, scikit-learn, xgboost, torch (root), rdkit, pytest (`features/requirements.txt`)
  and transformers (`features/requirements-text.txt`), so nothing else is missing.
- Not completed: a from-scratch `pip install` into an empty venv. The workspace's network was too slow
  (on Linux, `xgboost` pulls in the ~300 MB `nvidia-nccl` wheel, then `torch` is larger still), so the
  checks above ran on the workspace's own Python with the same packages. The empty-venv install is part
  of the Windows-on-ARM laptop test.
- For the laptop (not checked yet): if `pip install torch`, `rdkit` or `xgboost` finds no wheel for
  Windows-on-ARM Python, use the x64 Python build, which runs under emulation (see the note further down).
  Write down which one was needed.

**Where we stopped (2026-11-10, Aditya's session):**
- `main` is up to date and there are no open PRs except whatever is listed by `gh pr list`.
- Still to do: multi-seed XGBoost runs, the ICD letter D split, HINT's numbers, the fresh-clone test and
  the final README (all on Ananya's list), then putting the fusion numbers into the README.
- The fusion settings are untuned. Tuning on valid is possible but valid already picks the stopping
  epoch, so a tuned model would need its valid scores read with even more care. Not planned.
- The fusion run was interrupted once (the machine ran low on memory) and finished on a second go.
  Two rows of the cut-off run were removed from the uncommitted `results/results.csv` before rerunning
  that seed, so there are no duplicate or missing rows: 84 earlier rows + 35 runs x 6.
- A fusion run takes about 25 seconds per block combination and seed on the RTX 3050, most of it
  loading. Training on a GPU is not bit-for-bit repeatable, so expect differences in the third decimal.
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
- Windows bug found while building (2026-11-10): `python -m features.build fingerprints` crashed at the
  end with `UnicodeEncodeError ... '⚠'`. FIXED 2026-11-11 by Ananya (section 9).

**Not verified, do not quote yet:**
- RESOLVED 2026-11-11: HINT's numbers for our release and its metric definition are in section 8
  (published Patterns 2022 Table 2; `HINT/model.py` read directly at the pinned commit). Still unknown:
  which version of the code produced the paper's table, so the PR-AUC definition behind it is inferred
  from the released code, not stated by the paper.
- XGBoost settings (200 trees, depth 4) are untuned defaults, so the baselines are a first floor.
- Trial cost figures for the decision analysis: candidate sources are in section 10, none
  verified against the primary paper.

**Ananya's Claude cannot work yet (2026-11-10):** Ananya sees "The Claude GitHub App on adigreat
isn't connected. Only adigreat can connect it." The app IS installed on `adigreat` (all repos)
and connected on Aditya's Claude account. This only affects cloud/web Claude sessions. It does
not block her working locally: clone the repo, use her own `gh` login, run Claude Code in the
terminal. If she needs cloud sessions, the options are a fork under her own account (install the
app there, PR back to `adigreat`) or moving the repo to a GitHub org both of us belong to.

**Getting the data and the arrays** (run from the repo root; `data/` is gitignored; `PYTHONUTF8=1` is no
longer needed once the 2026-11-11 UTF-8 fix is merged):

```
pip install -r requirements.txt -r features/requirements.txt
python -m features.build fetch          # clones TOP at the pinned commit into data/raw/top/
python -m features.build fingerprints
python -m features.build disease
pip install torch --index-url https://download.pytorch.org/whl/cu124   # pick the build for your CUDA
pip install -r features/requirements-text.txt
python -m features.build text           # PubMedBERT, about 25 min on a 6 GB GPU, resumable
```

**Next up, Person A (Ananya).** Updated 2026-11-11. None of this needs a GPU.
1. DONE 2026-11-11: Windows UTF-8 fix; ICD P / Z answer; error-analysis D split and P/Z rule; HINT
   numbers; README draft; fresh-clone test in Linux; XGBoost seeds 1 to 4 for the combinations without text.
2. DONE by Aditya 2026-11-11 (he has the text array): XGBoost seeds 1 to 4 for the text combinations
   and the fusion error analysis with the D split and P/Z rule. Check his wording of your
   error-analysis section in the README.
3. Fresh-clone test on the Windows-on-ARM laptop: same steps as the Linux run above, plus an empty-venv `pip install`. Write down whether
   `rdkit`, `xgboost` and `torch` install on ARM Python, or whether the x64 build under emulation is needed.
   The text step can be skipped there (the build writes a placeholder), or use `text_features.npy`
   from Aditya if it has arrived.
4. Finish the README: Aditya's parts are filled (2026-11-11). Remove the DRAFT comment when you are happy with it.

**Next up, Person B (Aditya), in `eval/` and `models/`:**
1. DONE 2026-11-11: section 10 question answered (the P/Z codes stay in the disease features).
2. Optional: send `text_features.npy` to Ananya directly (Drive or similar, never git) for her laptop
   test. No run is blocked on it any more.
3. DONE: fusion MLP with ablation, error and decision analysis; README parts; review of Ananya's PR.
4. Review whatever Ananya changes in the README, then the project is at its definition of done except
   for the laptop fresh-clone test.

**Branches and PRs:** one branch per task (`ananya/...`, `aditya/...`), merge through a PR that
the other person approves. Never push to `main`.
