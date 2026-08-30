# NSMMF-250K V1.1 — Assumption Register

## Dataset purpose
Synthetic Nigerian mobile-money transaction data for the **Mobile Money Fraud Detector** MVP.

## Public-context/calibration concepts used in the project methodology
- Mobile Money Operators are treated separately from ordinary mobile-app banking.
- Nigerian mobile-money transaction scale/value statistics are used only as aggregate calibration context.
- KYC tiers and daily limits are treated as regulatory constraints in the simulator.
- Fraud technique categories are informed by Nigerian fraud reporting, but the scenario probabilities below are synthetic.

## Explicit V1.1 simulation assumptions
- 250,000 transactions across 90 days.
- 15,000 wallets, 1,000 agents, 2,500 merchants.
- Fraud prevalence fixed at 0.5% for ML experimentation; this is **not** an estimate of Nigeria's real fraud rate.
- Customer profile mix: occasional 25%, everyday 40%, trader/business 15%, cash-heavy 15%, high-value 5%.
- Synthetic KYC population mix: Tier 1 55%, Tier 2 35%, Tier 3 10%.
- Geographic activity weights are synthetic.
- Transaction-type, hourly-activity, device, counterparty-reuse, and fraud-injection distributions are synthetic assumptions.
- Seven fraud scenarios are generated: social engineering, account takeover, SIM/device compromise, mule activity, rapid cash-out, velocity fraud, dormant-wallet takeover.

## Leakage controls
- Rolling behavioural features use only prior transactions.
- `fraud_type` and `fraud_event_id` are audit-only.
- `customer_profile` is simulation-only.
- Post-transaction balances and transaction status are excluded from `nsmmf_ml.csv`.
- Raw identifiers are excluded from `nsmmf_ml.csv`.
- Raw timestamp is retained for chronological splitting but should not be encoded directly as a predictor.

## V1.1 validation headline
- 250,000 rows
- 1,250 fraud rows (0.5%)
- Mean amount ≈ ₦9,003
- No duplicate transaction IDs
- No negative transaction amounts
- No negative sender balances after processing
- No missing values in ML dataset
- No successful KYC daily-limit violations
