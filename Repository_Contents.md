## Repository contents

| File / Folder | Description |
|---|---|
| `3MTT_Mobile_Money_Fraud_Detector_Modelling_Complete.ipynb` | Complete end-to-end project notebook covering data loading, validation, cleaning, exploratory analysis, feature engineering, leakage checks, chronological train/validation/test splitting, supervised modelling, anomaly detection, hybrid scoring, threshold tuning, sensitivity and ablation experiments, interpretability, calibration diagnostics, error analysis, bootstrap uncertainty, and final test evaluation. |
| `README.md` | Main project overview, methodology, results, limitations, dataset access instructions, and reproduction guidance. |
| `NSMMF-250K/` | Folder containing the Nigerian synthetic mobile-money fraud dataset resources and supporting documentation. |
| `NSMMF-250K/README.md` | Dataset-specific documentation for NSMMF-250K, including generation methodology, assumptions, schema, usage notes, and limitations. |
| `NSMMF-250K/generate_nsmmf.py` | Script used to generate the NSMMF-250K synthetic dataset. |
| `NSMMF-250K/data_dictionary.csv` | Definitions and modelling guidance for NSMMF-250K variables. |
| `NSMMF-250K/ASSUMPTIONS.md` | Records the simulation assumptions and distinguishes externally calibrated values from synthetic design choices. |

### Dataset access

The modelling notebook uses two synthetic mobile-money datasets:

1. **PaySim** — [Kaggle dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)  
   Kaggle handle: `ealaxi/paysim1`

2. **NSMMF-250K** — [Kaggle dataset](https://www.kaggle.com/datasets/elenwai/nsmmf-synthetic-fraud-dataset)  
   Kaggle handle: `elenwai/nsmmf-synthetic-fraud-dataset`

The two datasets are modelled separately and are not concatenated.
