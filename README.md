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

Work in progress. Drug and disease features are built and a first floor (logistic regression and
XGBoost, untuned) has been run on the official temporal test split. The eligibility-text features,
the fusion model, the ablations and the error analysis are still to do. Early numbers are in
[HANDOFF.md](HANDOFF.md) section 13 and are not comparable to published results yet. See
[HANDOFF.md](HANDOFF.md) for plan, ownership and decisions, and
[EASY_LANGUAGE.md](EASY_LANGUAGE.md) for the same story in plain words.
