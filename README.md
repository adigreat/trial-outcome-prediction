# Clinical Trial Outcome Prediction

> **Research prototype, not validated for clinical use.** Nothing here should inform a decision
> about a real trial.

Predicts whether a clinical trial will succeed or fail, using only what is known when the trial is
designed: the drug, the disease, and the eligibility criteria.

<!-- DRAFT (Ananya, 2026-11-11). Aditya's parts were filled in on 2026-11-11. -->

## How this differs from most trial-prediction repos

Most clinical-trial projects on GitHub predict whether a trial *administratively completes*,
using a raw ClinicalTrials.gov export, XGBoost, SHAP and Streamlit. Completion is a much
easier target and worth much less; at least one of those repos says outright that it is not
an efficacy label. We use the TOP benchmark, which has a real efficacy label, multimodal
inputs and a temporal split. It is harder and more credible.

## Dataset

**TOP benchmark, original Fu et al. release** (not the 2024 TrialBench update), pinned to upstream
commit [`8dc0497`](https://github.com/futianfan/clinical-trial-outcome-prediction). The build checks
every CSV against a stored hash. TOP's licence allows non-commercial use only.

- **Official temporal split:** train and valid hold trials completed before 1 January 2014 (valid is a
  random 10% of them), test holds trials that started in 2014 or later. We never make our own split.
- **Sizes**, train / valid / test (share of successes in test): Phase I 1,044 / 117 / 627 (0.55),
  Phase II 4,005 / 446 / 1,654 (0.56), Phase III 3,094 / 344 / 1,146 (0.75).
- **One model per phase**, scored on that phase's own test file.
- **Leakage guard:** the CSVs' `status` and `why_stop` columns describe what happened to the trial, so
  the loader never reads them.

## What the model sees

| Block | Input | Encoding | Size |
|---|---|---|---|
| Drug | SMILES of each drug | Morgan fingerprint (RDKit, radius 2), OR across the trial's drugs | 2,048 |
| Disease | ICD-10 codes | multi-hot over chapter, 3-character category and full code, vocabulary fitted on train only | 1,983 |
| Eligibility text | inclusion and exclusion criteria | frozen PubMedBERT, mean over criterion bullets, inclusion and exclusion kept separate | 1,536 |

Models: logistic regression and XGBoost (untuned floor), and a **fusion MLP** (one small encoder per
block, concatenated, MLP head; settings fixed before the first test run, early stopping on valid).
Details: [`features/README.md`](features/README.md), [`models/fusion.py`](models/fusion.py).

## Results

Test split. PR-AUC is the standard threshold-free one (`average_precision_score` on probabilities).
The "always predict success" row is the PR-AUC floor, and it is high in Phase III because 75% of those
test trials succeed, so read the gap above it, or the ROC-AUC. Mean ± standard deviation over training
seeds; the spread says nothing about how much the test sample itself would vary.

**PR-AUC (ROC-AUC in brackets)**

| Model, blocks | seeds | Phase I (n=627) | Phase II (n=1,654) | Phase III (n=1,146) |
|---|---|---|---|---|
| Always predict success | | 0.553 | 0.555 | 0.750 |
| XGBoost, drug | 5 | 0.610 ± 0.008 (0.553) | 0.607 ± 0.003 (0.558) | 0.829 ± 0.001 (0.624) |
| XGBoost, disease | 5 | 0.649 ± 0.004 (0.591) | 0.680 ± 0.003 (0.630) | 0.858 ± 0.001 (0.704) |
| XGBoost, text | 5 | 0.634 ± 0.011 (0.596) | 0.672 ± 0.009 (0.625) | 0.846 ± 0.007 (0.672) |
| XGBoost, drug + disease | 5 | 0.628 ± 0.002 (0.586) | 0.654 ± 0.006 (0.603) | 0.853 ± 0.002 (0.671) |
| XGBoost, disease + text | 5 | 0.641 ± 0.015 (0.597) | 0.671 ± 0.008 (0.626) | 0.839 ± 0.004 (0.667) |
| XGBoost, drug + disease + text | 5 | 0.643 ± 0.018 (0.604) | 0.668 ± 0.009 (0.624) | 0.843 ± 0.005 (0.669) |
| Fusion MLP, text | 5 | 0.682 ± 0.008 (0.636) | 0.682 ± 0.004 (0.640) | 0.862 ± 0.003 (0.691) |
| Fusion MLP, disease + text | 5 | 0.674 ± 0.010 (0.631) | **0.696** ± 0.005 (0.653) | 0.874 ± 0.004 (0.719) |
| Fusion MLP, drug + disease + text | 5 | 0.675 ± 0.013 (0.627) | 0.693 ± 0.006 (0.644) | **0.882** ± 0.002 (0.726) |

All seven block combinations for every model, the valid rows and `pr_auc_hint` are in
[`results/results.csv`](results/results.csv); failed and weak runs stay in it.

**What carries the signal (ablation)**

- Phase I: the eligibility text. Text alone is as good as all three blocks together.
- Phase II: disease and text together; adding the drug block does not help.
- Phase III: every block adds a little, and all three is best.
- The drug fingerprint is the weakest block alone in every phase.
- With XGBoost, stacking blocks does not beat disease alone. With the fusion MLP it does, by about
  0.01 to 0.03 PR-AUC.

Why the fusion MLP combines blocks and XGBoost does not: we did not test this, so it is a guess. The
fusion MLP gives each block its own encoder with heavy dropout before the blocks meet, so a wide,
noisy block cannot crowd out a narrow, useful one. XGBoost sees one flat table of 5,567 mostly sparse
columns for 1,000 to 4,000 training trials per phase, and each extra block mainly adds ways to overfit.

## Comparison with HINT

HINT (Fu et al., *Patterns* 3:100445, 2022, Table 2) is the published model on this benchmark. Its
test sets have 627 / 1,653 / 1,140 trials against our 627 / 1,654 / 1,146, so this is the same release
to within six trials per phase. HINT reports the mean of 30 seeds; ours are 5.

| Phase | HINT ROC-AUC | Ours, fusion all three, ROC-AUC | HINT PR-AUC | Ours `pr_auc_hint` | Ours `pr_auc` |
|---|---|---|---|---|---|
| I | 0.576 ± 0.008 | 0.627 ± 0.013 | 0.567 ± 0.010 | 0.597 ± 0.009 | 0.675 ± 0.013 |
| II | 0.645 ± 0.006 | 0.644 ± 0.009 | 0.629 ± 0.009 | 0.611 ± 0.005 | 0.693 ± 0.006 |
| III | 0.723 ± 0.006 | 0.726 ± 0.004 | 0.811 ± 0.007 | 0.804 ± 0.005 | 0.882 ± 0.002 |

- **ROC-AUC is the clean comparison**, because both sides compute it on probabilities. On it, the
  fusion MLP matches HINT in Phases II and III and is above it in Phase I, where the test set is
  smallest and noisiest.
- **PR-AUC is not cleanly comparable.** HINT's released code (`HINT/model.py`, `evaluation`, at the
  pinned commit) turns predictions into 0/1 at 0.5 before calling `average_precision_score`, while the
  paper describes the threshold-free curve. `pr_auc_hint` copies the code's version, and HINT's numbers
  sit close to it. Our standard `pr_auc` is much higher, but that is a different metric, not a win.
- We do not claim to beat HINT. A model built from simple parts lands in the same range.

## Where the model struggles (error analysis)

Disease area = ICD-10 chapter, with letter D split at D50. Chapters P and Z only count when a trial has
no other code, because in TOP they come from the disease-name matching (a diabetes trial also gets
"neonatal diabetes" and "family history of diabetes"). Code: `models/error_analysis.py`. Rows:
`results/error_analysis.csv`.

- **Cancer is the hard area.** It has the lowest success rate (0.44 / 0.46 / 0.60 against 0.55 / 0.56 /
  0.75 overall). The fusion model's ROC-AUC within cancer is 0.55 / 0.59 / 0.57, against 0.63 / 0.65 /
  0.73 overall.
- **The models are too pessimistic about cancer in Phase III**: the fusion model predicts 0.45 success
  on average (XGBoost on the disease block 0.41), and 60% succeed.
- **In situ and benign neoplasms (D00-D49) behave like cancer**, because 81-90% of those trials (by phase)
  are cancer trials that also carry an in-situ code. In Phase III the fusion model is no better than
  chance inside this group (ROC-AUC 0.47 on 118 trials; it predicts 0.42 where 61% succeed).
- **Blood and immune disorders (D50-D89)** are a small group (19 / 58 / 40 test trials), too small to
  say much about.

The fusion figures are from one run (seed 0, all three blocks).

## Would a sponsor act on it? (decision analysis)

Not yet. We price a correctly stopped trial at the average per-study cost (ASPE/ERG 2014: $3.8M /
$13.4M / $19.9M for Phase 1 / 2 / 3) and a wrongly stopped success at 1x, 3x or 10x that. The value of
a lost success has no source we trust, so those multiples are assumptions. The kill threshold is
picked on valid and scored on test (fusion model, all three blocks, seed 0).

- **At 1x** the model comes out ahead in Phase I (154 of 627 trials stopped, 59 of them wrongly) and
  Phase II (839 of 1,654 stopped, 380 wrongly), and behind in Phase III.
- **At 3x** it is behind in every phase.
- **At 10x** it stops almost nothing: 11 trials in Phase I (3 of them wrongly, so still behind), 2 in
  Phase II and none in Phase III.

So the model is only useful if wrongly stopping a successful trial costs no more than running it,
which is not a realistic assumption for a drug that would have worked. Code: `models/decision_run.py`.
Rows: `results/decision_analysis.csv`.

## Running it

From the repo root. Tested with Python 3.13 on Linux, Python 3.12 on Windows 11 (x64, with a CUDA GPU) and Python 3.14 x64 on Windows 11 on ARM (Snapdragon X, under emulation, CPU only); Windows notes are in HANDOFF section 13. git is not needed: `fetch` downloads the data over HTTPS. `data/` is gitignored; nothing in it is committed.

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r features/requirements.txt

python -m features.build fetch          # downloads the 9 TOP CSVs at the pinned commit (no git needed)
python -m features.build fingerprints   # seconds, CPU
python -m features.build disease        # seconds, CPU
pip install -r features/requirements-text.txt
python -m features.build text           # PubMedBERT, about 25 min on a 6 GB GPU; resumable

python -m models.baselines --blocks drug,disease,text --seed 0                     # logreg + XGBoost
python -m models.baselines --models fusion --blocks drug,disease,text --seed 0     # fusion MLP
python -m models.error_analysis --model fusion --blocks drug,disease,text
python -m models.decision_run --model fusion --blocks drug,disease,text
```

Checks: `python -m eval.test_harness`, `python -m eval.test_decision`, `python -m models.test_baselines`,
`python -m models.test_error_analysis`, `python -m pytest features/tests -q`.

On Windows, if a first run stops with `ImportError: DLL load failed ... An Application Control policy
has blocked this file`, Smart App Control is checking a newly installed library; run the command again.

The fusion MLP trains on CPU in under a minute per phase. It is not bit-for-bit repeatable across
machines (CPU and GPU differ in the third decimal), so quote the multi-seed mean.

## Limits

- **One benchmark, one split.** The test set is a single time period. Phase I has 627 test trials, so
  differences of a point or two there are noise.
- **Untuned.** XGBoost and the fusion MLP use fixed settings. Valid picks the fusion model's stopping
  epoch, so its valid scores are optimistic. Test is never used for any choice.
- **Upstream data quality.** TOP's drug-to-SMILES and disease-to-ICD mappings are automatic and
  sometimes wrong (for example, side-effect P and Z codes). We document these issues and do not fix them.
- **The label is TOP's.** "Success" follows the benchmark's own definition.
- **No clinical use.** See the note at the top.

## More

[HANDOFF.md](HANDOFF.md) holds decisions, ownership and status (the source of truth).
[EASY_LANGUAGE.md](EASY_LANGUAGE.md) tells the same story in plain words.
