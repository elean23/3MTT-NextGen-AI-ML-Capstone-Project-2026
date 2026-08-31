# 3MTT Capstone Mobile Money Fraud Detector (MVP)
## Eleojo A Nwokeoji | FE/25/9769331301 | AI/ML

An intelligent system for detecting suspicious mobile money transactions, built as part of the 3MTT program. The project trains and evaluates fraud-detection models on two datasets: **PaySim** (a widely used public synthetic mobile-money dataset) and **NSMMF-250K**, a custom-built synthetic dataset simulating Nigerian mobile-money behaviour.

> **Status:** Experimental fraud-risk and decision-support prototype. Not validated on real transaction data and not intended for production fraud-blocking use. See [Limitations](#limitations).

---

## Problem

Mobile money agents and financial service providers in Nigeria regularly encounter fraudulent transaction patterns that result in financial losses. This project builds a machine learning system to flag anomalous or potentially fraudulent transactions and assign each one a fraud risk score.

## What this project delivers

- Fraud / Not Fraud classification for each transaction
- A continuous anomaly/risk score per transaction, not just a binary label
- Full model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) reported on an untouched, chronologically held-out test set
- A hybrid architecture combining a supervised classifier with an unsupervised anomaly detector


## Data
The notebook uses two synthetic mobile-money datasets:

### 1. **PaySim**
A public synthetic dataset of ~6.3M mobile-money transactions with an extreme class imbalance (~0.13% fraud). Fraud is concentrated almost entirely in `TRANSFER` and `CASH_OUT` transaction types.
> [PaySim Kaggle dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)  
> Kaggle handle: `ealaxi/paysim1`

### 2. NSMMF_fraud_Dataset
A custom-generated synthetic dataset built specifically for the Nigerian context: 250,000 transactions, 15,000 wallets, 1,000 agents, 2,500 merchants, across 36 states + FCT over 90 simulated days, with a fixed 0.5% fraud rate spread across seven fraud scenarios (social engineering, account takeover, SIM/device compromise, mule activity, rapid cash-out, velocity fraud, dormant-wallet takeover). Transactions were generated chronologically per customer rather than sampled independently, with behavioural features calculated from prior history only. This was intended to improve temporal and behavioural consistency, although subsequent ablation analysis identified strong synthetic relationships in some features, particularly `is_new_recipient`.

> [NSMMF_fraud_Dataset Kaggle dataset](https://www.kaggle.com/datasets/elenwai/nsmmf-synthetic-fraud-dataset)
> Kaggle handle: `elenwai/nsmmf-synthetic-fraud-dataset`

The two datasets are **modelled separately** throughout this project; their rows are never concatenated, since they differ in schema, scale, and fraud mechanics.


## Method

1. **Pre-modelling stage:** load, validate, clean, and feature-engineer both datasets independently; audit for potential leakage; check categorical cardinality and numeric correlation; and define chronological train / validation / test splits with asserted non-overlap. For PaySim, post-transaction balance fields are excluded from the primary feature set because cancelled fraudulent transactions can create simulator-specific balance behaviour that may provide unrealistically strong predictive information.
2. **Baseline supervised model:** Logistic Regression with `class_weight="balanced"`, using only leakage-safe features.
3. **Non-linear supervised model:** Random Forest, trained on a fraud-preserving subsample of the training partition to keep runtime practical, with `class_weight="balanced_subsample"`.
4. **Anomaly detector:** Isolation Forest, fitted only on legitimate training transactions, producing an anomaly percentile score.
5. **Model selection:** the strongest supervised model for each dataset is selected using validation PR-AUC, rather than default-threshold Precision, Recall, or F1. PR-AUC evaluates ranking performance across thresholds and is particularly informative under severe class imbalance.
6. **Threshold tuning:** the classification threshold is optimised on the validation set only, targeting maximum F1. The resulting threshold is treated as an experimental operating point rather than a production-optimal decision threshold.
7. **Hybrid fraud-risk score:** a weighted blend of the supervised model's fraud-risk score and the Isolation Forest anomaly percentile, with the blend weight tuned on validation data. The hybrid is adopted only where it improves validation PR-AUC over the supervised model alone.
8. **Final evaluation:** the chronological test set is scored only after model choice, threshold selection, and hybrid configuration have been frozen using validation data.

Two additional sensitivity and robustness analyses were conducted::
- **NSMMF feature ablation** — tests how strongly *Random Forest* performance depends on potentially dominant synthetic features. Removing `is_new_recipient` alone reduces validation PR-AUC by approximately 38%, while removing both `is_new_recipient` and `location_changed` reduces PR-AUC by approximately 46%. This indicates that a substantial share of Random Forest performance on NSMMF depends on synthetic feature relationships that may not generalise to real transaction data.
- **PaySim balance-feature sensitivity** — adds the excluded post-transaction balance fields back into the model to quantify their effect. Doing so increases PR-AUC to approximately 1.0000 and Precision/Recall to approximately 0.999–1.000. This near-perfect performance is consistent with simulator-specific balance mechanics and leakage risk, supporting the decision to exclude these fields from the primary benchmark.

## Results

Final results on the untouched chronological test sets:

| Dataset | Final system | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC (Bootstrap 95% CI) |
|---|---|---|---|---|---|---|---|
| NSMMF | Logistic Regression (supervised) | 0.9842 | 0.4845 | 0.2831 | 0.3574 | 0.9663 | 0.2938 (0.2195–0.3761) |
| PaySim | Random Forest + Isolation Forest (hybrid) | 0.9835 | 0.7376 | 0.3226 | 0.4489 | 0.9179 | 0.4257 (0.3948–0.4508) |

The 95% bootstrap confidence intervals reflect resampling uncertainty for the frozen test predictions. They do not capture uncertainty arising from model retraining, different random seeds, changes in synthetic-data generation, or deployment on a different population.

For NSMMF, the supervised model alone was selected over the hybrid because the anomaly detector did not improve validation PR-AUC. For PaySim, the hybrid improved validation PR-AUC from 0.3528 to 0.3864 and was therefore adopted as the final system.

**Recall by NSMMF fraud scenario (test set):**

| Scenario | Recall |
|---|---|
| Account takeover | 0.79 |
| SIM/device compromise | 0.58 |
| Velocity fraud | 0.20 |
| Dormant-wallet takeover | 0.19 |
| Mule activity | 0.17 |
| Rapid cash-out | 0.06 |
| Social engineering | 0.05 |

Overall Recall hides a wide spread across fraud types. The system is comparatively strong at catching account takeover and SIM/device compromise, which leave clear device/location signals, and weak at catching social engineering and rapid cash-out, which resemble normal transaction behaviour more closely.

**What drives the predictions:** for NSMMF, `is_new_recipient` is the strongest positive predictor (consistent with the ablation result above), followed by `P2P_TRANSFER` transaction type and `is_new_device`; `CASH_IN` is the strongest negative predictor. For PaySim, permutation importance was calculated for the selected Random Forest supervised model, which forms the supervised component of the final hybrid system

Stratified bootstrap resampling was applied to the frozen test scores to give a sense of estimation uncertainty (this does not capture variance across different random seeds or retraining runs — see the notebook for interval widths).

## Key finding

**Recall is the main weakness of both systems.** At the operating thresholds selected, both models miss a majority of fraud cases (NSMMF misses ~72%, PaySim misses ~68%) in exchange for keeping false positives low. This is a direct consequence of optimising for F1 at severe class imbalance, and it is the single most important thing to address before this system could support real fraud operations — likely via a recall-weighted or cost-sensitive threshold policy, because the relative costs of missed fraud and false alerts should be explicitly incorporated into deployment decisions.

## Limitations

- Both datasets are synthetic. Results describe model behaviour on simulated data, not real Nigerian mobile-money transactions, and have not been externally validated.
- NSMMF's supervised performance is partly attributable to synthetic design choices (see the ablation result above), not purely to generalisable fraud behaviour.
- PaySim's balance fields exhibit simulator-specific artefacts and were deliberately excluded from the primary benchmark.
- Random Forest and Isolation Forest on PaySim were trained on capped subsamples (not the full 6.3M rows) for computational practicality within Colab; this changes the class distribution seen during training relative to the true population rate and may affect probability calibration.
- The selected thresholds optimise F1 on validation data; they are experimental choices, not cost-calibrated production thresholds.
- The project does not address production concerns such as formal probability recalibration, concept drift, fairness, privacy, governance, latency, or investigation cost.


## Repository contents

| File / Folder | Description |
|---|---|
| `3MTT_Mobile_Money_Fraud_Detector_Modelling_Complete.ipynb` | Complete end-to-end project notebook covering data loading, validation, cleaning, exploratory analysis, feature engineering, leakage checks, chronological train/validation/test splitting, supervised modelling, anomaly detection, hybrid scoring, threshold tuning, sensitivity and ablation experiments, interpretability, calibration diagnostics, error analysis, bootstrap uncertainty, and final test evaluation. |
| `README.md` | Main project overview, methodology, results, limitations, dataset access instructions, and reproduction guidance. |
| `NSMMF_250K/` | Folder containing the Nigerian synthetic mobile-money fraud dataset resources and supporting documentation. |
| `NSMMF_250K/README.md` | Dataset-specific documentation for NSMMF-250K, including generation methodology, assumptions, schema, usage notes, and limitations. |
| `NSMMF_250K/generate_nsmmf.py` | Script used to generate the NSMMF-250K synthetic dataset. |
| `NSMMF_250K/data_dictionary.csv` | Definitions and modelling guidance for NSMMF-250K variables. |
| `NSMMF_250K/ASSUMPTIONS.md` | Records the simulation assumptions and distinguishes externally calibrated values from synthetic design choices. |
| `requirements.txt` | Python package dependencies required to reproduce the project. |



## How to run
1. Open `3MTT_Mobile_Money_Fraud_Detector_Modelling_Complete.ipynb` in Google Colab or Jupyter and run all cells to produce the cleaned, validated, and leakage-audited datasets and splits, the full modelling pipeline, from baseline model through final test evaluation.
2. Requires: `pandas`, `numpy`, `scikit-learn`, `matplotlib`. No GPU required; the notebook is designed to run within standard Google Colab RAM limits.


## Tools used
Python, Pandas, Scikit-learn, Google Colab.
