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

Work in progress. All three feature blocks (drug, disease, eligibility text) are built, and a first
floor (logistic regression and XGBoost, untuned, one seed) has been run for every block combination
on the official temporal test split, with a first error analysis by disease area and a first
decision analysis. The fusion model, multi-seed runs and the final write-up are still to do; who does what is in
[HANDOFF.md](HANDOFF.md) section 4. Early numbers are
in [HANDOFF.md](HANDOFF.md) section 13 and are not comparable to published results yet. See
[HANDOFF.md](HANDOFF.md) for plan, ownership and decisions, and
[EASY_LANGUAGE.md](EASY_LANGUAGE.md) for the same story in plain words.
