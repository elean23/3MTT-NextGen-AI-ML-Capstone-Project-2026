# NSMMF-250K

**Nigerian Synthetic Mobile Money Fraud Dataset — Version 1**

Designed for the **Mobile Money Fraud Detector** MVP.

## Scope
- 250,000 synthetic transactions
- 15,000 wallets
- 1,000 agents
- 2,500 merchants
- 36 Nigerian states + FCT
- 90 simulated days
- 1,250 synthetic fraud-labelled transactions (0.5%)

## Important limitation
These records are synthetic. They are not real NIBSS, CBN, MMO, bank, agent, or customer records.
The Nigerian public statistics discussed in the project methodology are calibration/context sources;
many behavioural distributions in Version 1 are explicit simulation assumptions.

## Files
- `nsmmf_full.csv` — audit/EDA data including IDs, post-transaction fields, and fraud-scenario metadata.
- `nsmmf_ml.csv` — leakage-reduced feature set plus `is_fraud`.
- `nsmmf_unlabelled.csv` — same feature set with the label removed for anomaly detection.
- `data_dictionary.csv` — column definitions and model-use guidance.
- `simulation_parameters.json` — core design parameters and assumptions.
- `validation_report.json` — generated integrity checks.

## Fraud scenarios
1. Social engineering
2. Account takeover
3. SIM/device compromise
4. Mule activity
5. Rapid cash-out
6. Velocity fraud
7. Dormant-wallet takeover

## Recommended modelling split
Use a **chronological train/validation/test split**, not a purely random split, because this is a
transaction-stream problem. Keep `timestamp` for splitting and derive temporal predictors rather
than encoding the raw timestamp directly.
