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

TOP benchmark (Fu et al.), original release. Version: _TBD, see HANDOFF.md section 9._

## Status

Work in progress. See [HANDOFF.md](HANDOFF.md) for plan, ownership and decisions.
