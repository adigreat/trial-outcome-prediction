# The project in easy language

Written for Ananya and Aditya. Last updated: 2026-11-11 (HINT numbers found, Windows fix, all runs have 5 seeds, README nearly done).

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
| Ananya (A) | Features (done). Now also: error analysis, the ablation runs with several seeds, finding HINT's numbers, the fresh-clone test, the README draft | `features/`, plus `models/error_analysis.py` and its test |
| Aditya (B) | Scoring code, baselines, the fusion model, the decision analysis | `eval/`, `models/` |

The work was re-split on 2026-11-10, because the features are finished and everything left was on
Aditya's side.

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
- **The fusion model (the main model) is built and run**, with 5 seeds for every combination of blocks.
- Pull requests #3 to #8 are merged. Ananya reviewed and approved #8. #3 to #7 were merged by Aditya
  as admin before she could review them.
- Ananya's two zips were checked against Aditya's own build: the arrays are identical.

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

**The fusion model (test set, PR-AUC, average of 5 runs)**

| What the model sees | Phase I | Phase II | Phase III |
|---|---|---|---|
| Nothing (always guess "success") | 0.55 | 0.56 | 0.75 |
| Best simple model above (disease only) | 0.65 | 0.68 | 0.86 |
| Drug only | 0.63 | 0.61 | 0.84 |
| Disease only | 0.64 | 0.68 | 0.86 |
| Text only | 0.68 | 0.68 | 0.86 |
| Drug + disease | 0.63 | 0.68 | 0.87 |
| Drug + text | 0.68 | 0.69 | 0.87 |
| Disease + text | 0.67 | 0.70 | 0.87 |
| All three | 0.68 | 0.69 | 0.88 |

What the fusion model tells us:
- **Combining the blocks now helps.** All three together beat the best simple model in every phase,
  by one or two points. That is a small gain, not a big one.
- **Phase I: the eligibility text does the work.** Text alone is as good as everything together.
- **Phase II: disease plus text is enough.** The drug block adds nothing.
- **Phase III: every block adds a little,** and all three together is best.
- The five runs agree with each other to about 0.01, so these patterns are not luck of the seed.
- Cancer is still the hard area, and the model is still not good enough to stop trials on.

**Where the model struggles (error analysis)**
- **Cancer trials are the hard ones.** They fail more often than other areas, and inside cancer the
  model is close to guessing in Phase I.
- In Phase III the model is too pessimistic about cancer: it expects about 41% to succeed and 60% do.

**Would a sponsor act on it? (decision analysis)**
- Not yet. If stopping a good trial by mistake costs more than about one trial's budget, the model
  loses money or stops almost nothing. We use published average trial costs ($3.8M / $13.4M / $19.9M
  for Phase 1 / 2 / 3) and try several guesses for what a lost success is worth.

**New on 2026-11-11 (Ananya)**
- Aditya's fusion pull request (#10) was checked, approved and merged. Ananya re-ran part of it on a
  normal processor and got nearly the same numbers (within about 0.02). Fusion results change slightly
  from machine to machine, so we quote the average of 5 runs.
- **The Windows crash is fixed.** The build now writes every text file as UTF-8. The `PYTHONUTF8=1`
  trick is no longer needed.
- **The odd P and Z disease codes are real, but they are noise.** TOP turns disease names into codes
  automatically, and that drags in extra codes: a diabetes trial also gets "diabetes in a newborn" (P)
  and "family history of diabetes" (Z). The error analysis now ignores P and Z unless a trial has
  nothing else.
- **Error analysis:** the letter D is now split into "tumours that are not cancer yet" (D00 to D49) and
  "blood disorders" (D50 to D89). The first group behaves like cancer, because most of those trials
  are cancer trials.
- **The simple model (XGBoost) now has 5 runs** for drug, disease, and drug + disease. It hardly changes
  between runs, so the fusion model's lead (about 0.01 to 0.03) is real, not luck.
- **We found HINT's real numbers for our dataset** (the published paper, Table 2). On the fairest
  score (ROC-AUC), our fusion model is level with HINT in Phases II and III and a bit above it in
  Phase I. See section 8.
- **Fresh-clone test** (Linux): a clean copy builds and passes every check. One thing was missing:
  `torch` in `requirements.txt`. Now added.
- **README drafted.** Gaps for Aditya are marked `TODO(Aditya)`.

**Also on 2026-11-11 (Aditya)**
- Ananya's pull request was checked on Aditya's Windows machine: every check passes, and the Windows
  crash is gone.
- HINT's numbers were checked a second time against the published paper. They are right.
- Aditya ran the two things Ananya could not run without the text file: XGBoost with 5 runs for the
  combinations that use text, and the error analysis on the fusion model with the new D split.
  **Every model now has 5 runs.** XGBoost with all three blocks gets 0.64 / 0.67 / 0.84, so the fusion
  model (0.68 / 0.69 / 0.88) is ahead by about 0.03 to 0.04.
- The README is filled in: Aditya's two gaps are written, and the tables are complete.

**Not done yet**
- The fresh-clone test on Ananya's Windows laptop.
- A last read of the README by Ananya.

## 7. Next steps, why, and how

### Ananya
Nothing here needs a GPU. Your laptop (Snapdragon X, 16 GB RAM) is enough.
1. **Done 2026-11-11:** Windows fix, P/Z answer, D split, HINT numbers, README draft, fresh-clone test on
   Linux, more XGBoost runs without text.
2. **Done by Aditya:** the XGBoost runs with text and the fusion error analysis. Please check how he
   worded your error-analysis section in the README.
3. **Fresh-clone test on your laptop.**
   *Why:* Windows on ARM is a different kind of machine, and nobody has tried a clean install there.
   *How:* clone, make a venv, `pip install -r requirements.txt -r features/requirements.txt`, build,
   run the checks. If a package will not install, try the x64 Python build. Write down what happened.
4. **Finish the README.** Aditya's parts are filled. Remove the DRAFT note when you are happy.

### Aditya
1. **Answered 2026-11-11:** the noisy P/Z codes stay in the disease *features*. Taking them out would
   mean rebuilding the disease block and redoing every run.
2. **Optional: send `text_features.npy` to Ananya** (Drive or similar, never git). No run needs it any
   more; it only saves her a long build on her laptop.
3. **Done:** the fusion model and its analyses, the README gaps, and the review of Ananya's pull request.

## 8. About the HINT comparison (be careful here)

HINT is the published model we are compared against. We did not expect to beat it, and we do not
claim to.

- **The right numbers** come from the published HINT paper (*Patterns*, 2022, Table 2). Its test set is
  the same as ours to within six trials per phase.
- HINT's ROC-AUC is 0.576 / 0.645 / 0.723 for Phase I / II / III. Our fusion model gets 0.627 / 0.644 /
  0.726. So: **level in Phases II and III, a bit above in Phase I**, where the test set is smallest and
  the noise is biggest.
- **Be careful with PR-AUC.** HINT's code turns predictions into 0/1 before computing it, which is not
  the usual way. Compare HINT's PR-AUC (0.567 / 0.629 / 0.811) only with our `pr_auc_hint` (0.597 /
  0.611 / 0.804), never with our usual `pr_auc`.
- The old numbers in `HANDOFF.md` (0.57 / 0.65 / 0.81) were partly right and partly wrong. The arXiv
  numbers (about 0.76 / 0.60 / 0.62) are on a different test set. Use neither.

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
