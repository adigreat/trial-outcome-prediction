# The project in easy language

Written for Ananya and Aditya. Last updated: 2026-11-10 (all three feature blocks built, first ablation done).

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
- Aditya's scoring code is merged. It also logs a second PR-AUC that copies how HINT computes its own
  (`pr_auc_hint`), so we can compare fairly later.
- Ananya's feature pipeline is merged. All three feature blocks now exist: drug, disease and
  eligibility text. Aditya ran the text step on his laptop GPU (about 25 minutes).
- First real baselines for every combination of the three blocks.
- A first error analysis by disease area and a first decision analysis.
- Pull requests #3 to #7 are merged. Aditya merged them as admin because Ananya had not reviewed them
  yet. Ananya, please look over what landed and say if anything is wrong.

**The numbers (test set, PR-AUC, XGBoost, higher is better)**

| What the model sees | Phase I | Phase II | Phase III |
|---|---|---|---|
| Nothing (always guess "success") | 0.55 | 0.56 | 0.75 |
| Drug only | 0.61 | 0.61 | 0.83 |
| Disease only | 0.65 | 0.68 | 0.86 |
| Text only | 0.63 | 0.67 | 0.85 |
| Drug + disease | 0.63 | 0.66 | 0.85 |
| Drug + text | 0.63 | 0.69 | 0.85 |
| Disease + text | 0.65 | 0.68 | 0.84 |
| All three | 0.66 | 0.67 | 0.84 |

What it means in plain words:
- The **disease** and the **eligibility text** both help, by about the same amount. The **drug**
  fingerprint helps least.
- Putting the blocks together does **not** beat the disease alone with this simple model. That is the
  job of the fusion model, which is next.
- Phase III looks great only because 75% of its test trials succeed anyway. Look at the gap above
  the "always guess success" row, not the raw number.
- These are untuned, one-seed numbers. A point or two is noise.
- They cannot be compared with HINT's paper yet.

**Where the model struggles (error analysis)**
- **Cancer trials are the hard ones.** They fail more often than other areas, and inside cancer the
  model is close to guessing in Phase I.
- In Phase III the model is too pessimistic about cancer: it expects about 41% to succeed and 60% do.

**Would a sponsor act on it? (decision analysis)**
- Not yet. If stopping a good trial by mistake costs more than about one trial's budget, the model
  loses money or stops almost nothing. We use published average trial costs ($3.8M / $13.4M / $19.9M
  for Phase 1 / 2 / 3) and try several guesses for what a lost success is worth.

**Not done yet**
- The fusion model (the main model), tuning, more seeds, and the final README.

## 7. Next steps, why, and how

### Ananya
1. **Look at what was merged and what Aditya decided.**
   *Why:* PRs #3 to #7 went in without your review, and your two questions were answered (keep the 106
   trials; the disease multi-hot is fine as is).
   *How:* read sections 9, 10 and 13 of `HANDOFF.md` and say if anything is wrong.
2. **Fix the Windows crash in `features/`.**
   *Why:* on Windows the build stops at the end with a `UnicodeEncodeError`. Aditya got around it with
   `PYTHONUTF8=1`.
   *How:* open the validation report file with `encoding="utf-8"`.
3. **Check one odd count.** Many trials carry an ICD chapter P (perinatal) or Z code. Is that really in
   the data, or a parsing slip? (`HANDOFF.md` section 10.)
4. **Then pick a shared task:** writing up the error analysis or the decision analysis.

### Aditya
1. **Fusion model:** join the three blocks and train a small neural network (MLP).
   *Why:* simple stacking did not beat the disease block alone, so we need a model that can combine them.
   *How:* PyTorch is already installed in the project venv.
2. **Redo the ablation, error analysis and decision analysis on the fusion model**, with several seeds.
   *Why:* one seed is not enough to trust a difference of a point or two.
3. **Find HINT's real PR-AUC for our dataset version** before comparing anything.
4. **README:** add the results and the dataset version once the model is done.

## 8. About the HINT comparison (be careful here)

HINT is the published model we are compared against. We probably will not beat it, and that
is fine. A small gap with honest ablations is a better story than a big claim.

- The old numbers in `HANDOFF.md` (0.57 / 0.65 / 0.81) could not be traced to a source.
- HINT's own code computes "PR-AUC" after turning predictions into 0/1 at 0.5, which is not the
  usual way. We log both: `pr_auc` (usual) and `pr_auc_hint` (copies HINT).
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
- Keep `HANDOFF.md`, this file and `README.md` in step. When something changes (a PR merges, a
  result comes in, a decision is made, the next step changes), update all of them in the same
  change. A stale doc counts as a bug. (This is a rule in `CLAUDE.md`.)

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
