# Predicting in-hospital death after percutaneous coronary intervention

Analysis code for a study that developed and temporally validated models to predict death during the hospitalisation of a
first percutaneous coronary intervention (PCI), within 30 days, from electronic health record data recorded before the
procedure in a Brazilian integrated health system. Six models are compared on equal footing: logistic regression, gradient
boosting, XGBoost, CatBoost and two tabular foundation models, TabPFN and Kumo Tabular. Reporting follows TRIPOD+AI.

**This repository contains no data and no data-extraction queries.** Patient data are not available outside the health
system. The scripts expect, in a `dados/` folder that is not versioned, one row per person with a pseudonymised identifier:
`icp_dataset.csv` (predictors, year of PCI and outcome), `icp_extra.csv` (bedside variables) and `icp_dataset_vespera.csv`
(laboratory results restricted to the day before PCI, for one sensitivity analysis).

## Contents

| File | What it does |
|---|---|
| `notebooks/gera_notebook_gpu.py` | Builds the analysis notebook for one predictor set (`--conjunto expandido \| cathpci \| mayo \| sem_idade \| vespera \| casos_completos`) |
| `notebooks/gera_notebook_unico.py` | Joins all predictor sets into a single notebook, `notebooks/ICP_notebook_unico.ipynb` (no outputs) |
| `notebooks/executa_notebook.py` | Runs the single notebook end to end and saves the executed copy |
| `22_bootstrap_final.py` | Paired bootstrap (2,000 replicates) for all metrics and differences between models and predictor sets |
| `23_figuras_paper.py` | Manuscript figures |
| `24_shap_catboost.py` | Exact TreeSHAP for CatBoost, after checking that the refitted model reproduces the original predictions |
| `25_subgrupos_ic.py` | Bootstrap confidence intervals by sex and age group |

## Design in brief

- Development set: PCIs in 2023–2024; temporal test set: PCIs in 2025; one row per person and no person in both sets.
- Every predictor recorded before the start of the procedure.
- Hyperparameters of the four classical models tuned with Optuna (TPE, 50 trials) in stratified five-fold cross-validation
  within the development set; TabPFN and Kumo Tabular used without tuning.
- Logistic recalibration on out-of-fold development predictions; decision threshold equal to the development-set prevalence.
- Shapley values for the model with the highest test-set AUROC.
- A single seed (`20260929`) controls all random processes; boosting runs on CPU with a fixed number of threads.

## Software

Python 3.14, scikit-learn 1.9.1, XGBoost 3.4.1, CatBoost 1.2.10, Optuna 5.0.0, tabpfn 9.0.0, structured-data-models
0.1.0.dev0 (Kumo Tabular) and PyTorch 2.11.0. TabPFN downloads its weights after license acceptance; set the `TABPFN_TOKEN`
environment variable before running and never store the token in the code.
