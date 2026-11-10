# The project in easy language

Written for Ananya and Aditya. Last updated: 2026-11-10 (after the first real baseline run).

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
- Aditya's scoring code is merged. It now also logs a second PR-AUC that copies how HINT computes
  its own (`pr_auc_hint`), so we can compare fairly later.
- Ananya's feature pipeline is merged. Drug fingerprints and the disease encoding are ready.
- Aditya built the arrays on his own machine and ran the first real baselines.
- Decision-analysis code is written and tested (costs still to be chosen).
- All the pull requests so far (#3 to #7) are merged. Aditya merged them as admin because Ananya had
  not reviewed them yet. Ananya, please look over what landed and say if anything is wrong.

**First real numbers (test set, PR-AUC, higher is better)**

| What the model sees | Phase I | Phase II | Phase III |
|---|---|---|---|
| Nothing (always guess "success") | 0.55 | 0.56 | 0.75 |
| Drug only (best of 2 models) | 0.61 | 0.61 | 0.83 |
| Disease only (best of 2 models) | 0.65 | 0.68 | 0.86 |
| Drug + disease (best of 2 models) | 0.63 | 0.66 | 0.85 |

What it means in plain words:
- The **disease** carries more signal than the **drug** in every phase.
- Adding the drug to the disease made it a little **worse**, not better. Probably the drug
  fingerprint is too big and noisy for this few trials.
- Phase III looks great only because 75% of its test trials succeed anyway. Look at the gap above
  the "always guess success" row, not the raw number.
- These are first, untuned, one-seed numbers. A point or two is noise.
- We do not have a text number yet, so we cannot say how much the eligibility text helps.
- These numbers cannot be compared with HINT's paper yet (different test set and a different way of
  computing PR-AUC).

**Not done yet**
- The eligibility-text encoding (PubMedBERT). It needs a computer with a GPU, which is Ananya's job.
- The fusion model, the three-way ablation, error analysis by disease area, the decision analysis
  on real predictions, and the final README.

## 7. Next steps, why, and how

### Ananya
1. **Look at what was merged and answer two questions** in `HANDOFF.md` section 10.
   *Why:* we merged your pipeline (PR #6) after fixing a small text clash in HANDOFF, and Aditya's
   Claude needs two answers: keep or drop the 106 trials that sit in train for one phase and valid
   for another, and whether the disease multi-hot is fine or a dense version is also wanted.
   *How:* read section 9 and section 10 of `HANDOFF.md`, then write your answers there.
2. **Run the PubMedBERT text step on a computer with a GPU.**
   *Why:* it is the last missing input, and we cannot ablate or fuse the text without it.
   *How:* `pip install -r features/requirements-text.txt`, then `python -m features.build text`
   (it can resume if it crashes). Check `load_contract`, then tell Aditya the text block is ready.
3. **Fix the Windows crash in `features/`.** On Windows the build stops at the very end with a
   `UnicodeEncodeError` because the report file is written without UTF-8. Aditya got around it with
   `PYTHONUTF8=1`. The real fix is to open that file with `encoding="utf-8"`.

### Aditya
1. **Fusion model**, once text exists: join the blocks and train a small MLP.
   *Why:* it is the main model of the project. *How:* needs PyTorch (not installed yet).
2. **Ablation table:** drug / disease / text, each alone and in combination, for every phase.
   *Why:* this is the part that makes the project ours. *How:* `python -m models.baselines --blocks ...`.
3. **Error analysis by disease area**, then the **decision analysis** on real predictions with
   `eval.decision.threshold_table`. We still need to choose the cost source for it.
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
