# The project in easy language

Written for Ananya and Aditya. Last updated: 2026-11-10.

`HANDOFF.md` is still the source of truth. This file is the same story in plain words.
If the two disagree, `HANDOFF.md` wins, and please fix this file.

> Research prototype, not validated for clinical use.

---

## 1. What are we building?

A model that guesses **whether a clinical trial will fail**, using only what is known
*before* the trial starts:

- the **drug** (its chemical structure),
- the **disease** (its ICD-10 code),
- the **eligibility rules** (the free text saying who can join).

Why it matters: a late-stage trial costs a huge amount of money and about half of them fail.
Knowing that early would be useful.

## 2. Why is this different from other projects on GitHub?

Most projects predict whether a trial *finished*. That is easy and not very useful.
We predict whether a trial *succeeded or failed*, using the TOP benchmark, which has a real
success/failure label and a fair split by time. It is harder, and it is more believable.

## 3. The data, in four lines

- Source: the original Fu et al. TOP release (not TrialBench).
- Nine files: three phases (I, II, III) times train / valid / test.
- Split by time: train on trials from before 2014, test on trials from 2014 on.
- Each phase is scored on its own.

**Never use the `status` or `why_stop` columns as features.** They tell you what happened to
the trial, so using them is cheating. We only use `smiless`, `icdcodes`/`diseases` and
`criteria`.

## 4. Who does what

| Person | Job | Folder |
|---|---|---|
| Ananya (A) | Turn the raw files into numbers (features) | `features/` |
| Aditya (B) | Score models fairly and train baselines | `eval/`, `models/` |

Stay in your own folder. It stops us from fighting over the same files.

## 5. The handoff: how A's work reaches B

Ananya produces a set of `.npy` files in `data/processed/`. Aditya reads them.
Every file lines up by row, so row 5 in one file is the same trial as row 5 in every other.

| File | What it holds |
|---|---|
| `drug_features.npy` | drug fingerprints |
| `disease_features.npy` | disease encoding |
| `text_features.npy` | eligibility text encoding |
| `trial_ids.npy` | which trial each row is (the key that keeps rows lined up) |
| `labels.npy` | 1 = success, 0 = failure |
| `phase.npy` | 1, 2 or 3 (which phase file the row came from) |
| `split_train.npy`, `split_valid.npy`, `split_test.npy` | which rows are train, valid, test |

The `.npy` files are **not** stored in git. Each of us makes them on our own machine, or
Ananya sends them to Aditya as a zip.

Check before sending: `eval.harness.load_contract("data/processed")` must pass.

## 6. Where we are right now

**Done**
- Dataset and roles decided, handoff format agreed.
- Aditya's scoring code (`eval/harness.py`) is merged and passes its self-check.
- Aditya's first models (logistic regression and XGBoost) are written and tested on fake data.
- Raw data downloaded to Aditya's machine.
- Claude GitHub App installed on `adigreat`.

**Waiting**
- PR #3 and PR #4 are open and need Ananya's approval (`main` needs 1 review).
- Ananya's feature files do not exist yet, so we have no real results.

**Not done yet**
- Everything that needs real numbers: baselines, fusion model, ablations, error analysis,
  decision analysis, final README.

## 7. Next steps, why, and how

### Ananya
1. **Review and approve PR #3, then PR #4.**
   *Why:* `main` is locked until someone else approves. Until then, `main` does not have the
   agreed handoff format, `split_valid`, or the baselines. Her checkout is also behind, which
   is why her Claude thought `split_valid` had no slot.
   *How:* open the PR on GitHub, look at the changes, click Approve. Merge #3 first.
2. **Pull the latest code.**
   *How:* `git checkout main` then `git pull`, after #3 and #4 are merged.
3. **Build the fingerprints first.**
   *Why:* it unblocks Aditya while the slower text step is still running.
   *How:* a script in `features/` reads the nine CSVs, writes the id, label, phase and split
   files, then drug fingerprints (Morgan, radius 2, 2048 bits, all of a trial's drugs combined).
   Write the combining choice in `HANDOFF.md` section 9.
4. **Check, then tell Aditya.**
   *How:* run `eval.harness.load_contract("data/processed")`. If it passes, send the arrays.
5. **Then the disease encoding, then the text encoding.**
   *Why:* same order as the plan. Her Claude chose a multi-hot encoding of the ICD codes, using
   only codes seen in train. Log that choice in section 9.
   *How:* frozen PubMedBERT for the text. Her Claude's workspace has no GPU, so run the real
   PubMedBERT step on a machine that has one.

### Aditya
1. **Ask Ananya to approve PR #3 and #4.** Claude cannot do this for her.
2. **Wait for the fingerprints, then run the first real baseline.**
   *How:* `python -m models.baselines --blocks drug`, then read the new rows in
   `results/results.csv`.
3. **Open the HANDOFF research PR** (branch `aditya/handoff-research`) once the others are in.
4. **Settle two open questions** in `HANDOFF.md` section 10:
   - which HINT numbers are the right comparison for our dataset version,
   - which source we use for trial cost figures.
5. **Later:** fusion model, ablations, decision analysis.

## 8. About the HINT comparison (be careful here)

HINT is the published model we are compared against. We probably will not beat it, and that
is fine. A small gap with honest ablations is a better story than a big claim.

- The old numbers in `HANDOFF.md` (0.57 / 0.65 / 0.81) could not be traced to a source.
- The HINT arXiv paper gives PR-AUC of about 0.76 / 0.60 / 0.62 for Phase I / II / III, but on
  a **different test set** than ours, so we cannot compare directly.
- Do not quote any HINT number until we have matched it to our dataset version.

## 9. Rules we both follow

- Never push to `main`. Work on a branch, open a PR, the other person approves.
- Never commit data, `.npy` files or secrets. The repo is public.
- Never use `status` or `why_stop` as features.
- Fix random seeds and write them down.
- Keep failed experiments in the results table. Do not delete them.
- No clinical claims. This is a research prototype.
- Before you stop for the day, update `HANDOFF.md` sections 9, 10 and 13. Each of our Claudes
  only knows what is written there.

## 10. Words you might not know

| Word | Meaning |
|---|---|
| PR | Pull request: a proposed change that someone else reviews before it goes into `main` |
| Branch | Your own copy of the code to work on safely |
| Fingerprint (Morgan) | A fixed-length list of 0s and 1s describing a molecule's structure |
| ICD-10 | The standard code system for diseases |
| PubMedBERT | A language model trained on medical papers, used here to turn text into numbers |
| Frozen | We use the model as is and do not train it further |
| PR-AUC | Precision-recall area under the curve. Our main score. Higher is better |
| ROC-AUC | A second score. 0.5 means guessing, 1.0 means perfect |
| Ablation | Remove one input (drug, disease or text) and see how much the score drops |
| Leakage | Accidentally giving the model information it would not have in real life |
| Baseline | A simple model that sets the floor everything else must beat |
