# Clinical Trial Outcome Prediction

> **Research prototype, not validated for clinical use.**

Predicts whether a clinical trial will fail, using only information available at design
time: the drug, the disease, and the eligibility criteria.

## How this differs from most trial-prediction repos

Most clinical-trial projects on GitHub predict whether a trial *administratively completes*,
using a raw ClinicalTrials.gov export, XGBoost, SHAP and Streamlit. Completion is a much
easier target and worth much less; at least one of those repos says outright that it is not
an efficacy label. We use the TOP benchmark, which has a real efficacy label, multimodal
inputs and a temporal split. It is harder and more credible.

## Dataset

TOP benchmark (Fu et al.), original release, using its official temporal split: train on
trials completed before 2014, test on trials that started in 2014 or later. Baselines are
only compared against results computed on this same version.

## Status

Work in progress. All three feature blocks (drug, disease, eligibility text) are built. A floor
(logistic regression and XGBoost, untuned, one seed) and the fusion MLP (untuned, five seeds) have
been run for every block combination on the official temporal test split, with an error analysis by
disease area and a decision analysis. With all three blocks the fusion MLP reaches a test PR-AUC of
0.675 / 0.693 / 0.882 for Phase I / II / III, where always predicting success gives 0.553 / 0.555 /
0.750. Multi-seed baselines and the final write-up are still to do; who does what is in
[HANDOFF.md](HANDOFF.md) section 4. The numbers are in [HANDOFF.md](HANDOFF.md) section 13 and are not
comparable to published results yet. See [HANDOFF.md](HANDOFF.md) for plan, ownership and decisions,
and [EASY_LANGUAGE.md](EASY_LANGUAGE.md) for the same story in plain words.
