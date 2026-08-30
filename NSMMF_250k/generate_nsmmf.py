"""NSMMF-250K V1.1 generator

Generates the Nigerian Synthetic Mobile Money Fraud Dataset used for the
Mobile Money Fraud Detector project.

Important:
- This creates synthetic data, not observed Nigerian customer transactions.
- Several behavioural distributions and fraud-injection probabilities are explicit
  project assumptions.
- The generated `nsmmf_ml.csv` deliberately excludes obvious identifiers,
  post-transaction balances, status, hidden customer archetypes, and fraud metadata.

Run:
    python generate_nsmmf.py
"""

import numpy as np
import pandas as pd
import json, os, zipfile
from pathlib import Path
from collections import defaultdict, deque

OUT = Path(__file__).resolve().parent / "generated"
if OUT.exists():
    for f in OUT.iterdir():
        if f.is_file():
            f.unlink()
else:
    OUT.mkdir(parents=True)

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# V1 PARAMETERS
# -----------------------------
N_TX = 250_000
N_CUSTOMERS = 15_000
N_AGENTS = 1_000
N_MERCHANTS = 2_500
SIM_DAYS = 90
START_DATE = pd.Timestamp("2025-01-01")
TARGET_FRAUD_RATE = 0.005
N_FRAUD = int(N_TX * TARGET_FRAUD_RATE)

states = [
    "Abia","Adamawa","Akwa Ibom","Anambra","Bauchi","Bayelsa","Benue","Borno",
    "Cross River","Delta","Ebonyi","Edo","Ekiti","Enugu","Gombe","Imo","Jigawa",
    "Kaduna","Kano","Katsina","Kebbi","Kogi","Kwara","Lagos","Nasarawa","Niger",
    "Ogun","Ondo","Osun","Oyo","Plateau","Rivers","Sokoto","Taraba","Yobe",
    "Zamfara","FCT"
]
geo_zone = {
    "Abia":"South East","Adamawa":"North East","Akwa Ibom":"South South","Anambra":"South East",
    "Bauchi":"North East","Bayelsa":"South South","Benue":"North Central","Borno":"North East",
    "Cross River":"South South","Delta":"South South","Ebonyi":"South East","Edo":"South South",
    "Ekiti":"South West","Enugu":"South East","Gombe":"North East","Imo":"South East",
    "Jigawa":"North West","Kaduna":"North West","Kano":"North West","Katsina":"North West",
    "Kebbi":"North West","Kogi":"North Central","Kwara":"North Central","Lagos":"South West",
    "Nasarawa":"North Central","Niger":"North Central","Ogun":"South West","Ondo":"South West",
    "Osun":"South West","Oyo":"South West","Plateau":"North Central","Rivers":"South South",
    "Sokoto":"North West","Taraba":"North East","Yobe":"North East","Zamfara":"North West",
    "FCT":"North Central"
}

profile_names = np.array(["occasional","everyday","trader_business","cash_heavy","high_value"])
profile_probs = np.array([0.25,0.40,0.15,0.15,0.05])
activity_weight = {
    "occasional":0.35, "everyday":1.0, "trader_business":3.2, "cash_heavy":1.5, "high_value":1.2
}
amount_mult = {
    "occasional":0.7, "everyday":1.0, "trader_business":1.3, "cash_heavy":1.1, "high_value":3.0
}
profile_tx_mix = {
    "occasional":       np.array([0.28,0.22,0.22,0.08,0.08,0.12]),
    "everyday":         np.array([0.27,0.21,0.19,0.11,0.10,0.12]),
    "trader_business":  np.array([0.35,0.28,0.06,0.11,0.08,0.12]),
    "cash_heavy":       np.array([0.18,0.12,0.10,0.25,0.25,0.10]),
    "high_value":       np.array([0.32,0.22,0.05,0.15,0.14,0.12]),
}
tx_types = np.array(["P2P_TRANSFER","MERCHANT_PAYMENT","AIRTIME_PURCHASE","CASH_OUT","CASH_IN","BILL_PAYMENT"])

kyc_levels = np.array([1,2,3])
kyc_probs = np.array([0.55,0.35,0.10])
daily_outflow_limit = {1:50_000.0, 2:200_000.0, 3:5_000_000.0}
balance_cap = {1:300_000.0, 2:500_000.0, 3:10_000_000.0}  # Tier-3 finite simulation ceiling only

fraud_types = np.array([
    "social_engineering",
    "account_takeover",
    "sim_device_compromise",
    "mule_activity",
    "rapid_cashout",
    "velocity_fraud",
    "dormant_wallet_takeover",
])
fraud_probs = np.array([0.25,0.20,0.15,0.15,0.10,0.075,0.075])

# -----------------------------
# CUSTOMER / ENTITY POPULATIONS
# -----------------------------
profiles = rng.choice(profile_names, size=N_CUSTOMERS, p=profile_probs)
kyc = rng.choice(kyc_levels, size=N_CUSTOMERS, p=kyc_probs)

state_weights = np.ones(len(states), dtype=float)
for s, m in {"Lagos":3.0,"FCT":2.0,"Kano":1.7,"Rivers":1.6,"Oyo":1.5,"Ogun":1.4,"Kaduna":1.4}.items():
    state_weights[states.index(s)] *= m
state_weights /= state_weights.sum()
cust_state_idx = rng.choice(len(states), size=N_CUSTOMERS, p=state_weights)

major_states = {"Lagos","FCT","Rivers","Oyo","Kano","Ogun","Kaduna"}
urban_prob = np.array([0.85 if states[i] in major_states else 0.55 for i in cust_state_idx])
r1 = rng.random(N_CUSTOMERS)
r2 = rng.random(N_CUSTOMERS)
cust_urban = np.where(r1 < urban_prob, "urban", np.where(r2 < 0.55, "semi_urban", "rural"))

device_type = np.where(rng.random(N_CUSTOMERS) < 0.70, "smartphone", "feature_phone")
account_age_days = rng.integers(30, 3650, size=N_CUSTOMERS)
device_age_days = np.maximum(1, rng.integers(1, 1500, size=N_CUSTOMERS))
preferred_hour = np.clip(rng.normal(15, 4.0, size=N_CUSTOMERS), 6, 23)
base_avg_amount = np.exp(rng.normal(np.log(7000), 0.65, size=N_CUSTOMERS))
base_avg_amount *= np.array([amount_mult[p] for p in profiles])

pre_sim_inactive_days = rng.integers(0, 20, size=N_CUSTOMERS)
dormant_candidates = rng.choice(N_CUSTOMERS, size=max(400, N_CUSTOMERS//25), replace=False)
pre_sim_inactive_days[dormant_candidates] = rng.integers(45, 121, size=len(dormant_candidates))

start_balance = np.exp(rng.normal(np.log(45_000), 1.0, size=N_CUSTOMERS))
start_balance *= np.array([amount_mult[p] for p in profiles])
start_balance = np.minimum(start_balance, np.array([balance_cap[int(k)] for k in kyc]))

agent_state_idx = rng.choice(len(states), size=N_AGENTS, p=state_weights)
merchant_state_idx = rng.choice(len(states), size=N_MERCHANTS, p=state_weights)

# -----------------------------
# BASE TRANSACTIONS
# -----------------------------
sender_activity = np.array([activity_weight[p] for p in profiles], dtype=float)
sender_activity /= sender_activity.sum()
sender_idx = rng.choice(N_CUSTOMERS, size=N_TX, p=sender_activity)

type_idx = np.empty(N_TX, dtype=np.int8)
for p in profile_names:
    mask = profiles[sender_idx] == p
    type_idx[mask] = rng.choice(len(tx_types), size=mask.sum(), p=profile_tx_mix[p])
transaction_type = tx_types[type_idx].copy()

day_offsets = rng.integers(0, SIM_DAYS, size=N_TX)
hour_bins = np.array([[0,6],[6,9],[9,13],[13,18],[18,22],[22,24]])
hour_bin_probs = np.array([0.03,0.09,0.26,0.31,0.25,0.06])
bin_idx = rng.choice(len(hour_bins), size=N_TX, p=hour_bin_probs)
hours = np.fromiter((rng.integers(hour_bins[b,0], hour_bins[b,1]) for b in bin_idx), count=N_TX, dtype=np.int16)
minutes = rng.integers(0,60,size=N_TX)
seconds = rng.integers(0,60,size=N_TX)
timestamps = (
    START_DATE.to_datetime64()
    + day_offsets.astype("timedelta64[D]")
    + hours.astype("timedelta64[h]")
    + minutes.astype("timedelta64[m]")
    + seconds.astype("timedelta64[s]")
).astype("datetime64[ns]")

receiver_kind = np.empty(N_TX, dtype=object)
receiver_num = np.empty(N_TX, dtype=np.int32)
agent_num = np.full(N_TX, -1, dtype=np.int32)

p2p_mask = transaction_type == "P2P_TRANSFER"
receiver_kind[p2p_mask] = "customer"
receiver_num[p2p_mask] = rng.integers(0, N_CUSTOMERS, size=p2p_mask.sum())
same = p2p_mask & (receiver_num == sender_idx)
receiver_num[same] = (receiver_num[same] + 1) % N_CUSTOMERS

merch_mask = np.isin(transaction_type, ["MERCHANT_PAYMENT","BILL_PAYMENT","AIRTIME_PURCHASE"])
receiver_kind[merch_mask] = "merchant"
receiver_num[merch_mask] = rng.integers(0, N_MERCHANTS, size=merch_mask.sum())

cash_mask = np.isin(transaction_type, ["CASH_IN","CASH_OUT"])
receiver_kind[cash_mask] = "agent"
agent_num[cash_mask] = rng.integers(0, N_AGENTS, size=cash_mask.sum())
receiver_num[cash_mask] = agent_num[cash_mask]

means = {
    "P2P_TRANSFER":12000, "MERCHANT_PAYMENT":4500, "AIRTIME_PURCHASE":1250,
    "CASH_OUT":17000, "CASH_IN":15000, "BILL_PAYMENT":9000
}
sigmas = {
    "P2P_TRANSFER":0.95, "MERCHANT_PAYMENT":0.75, "CASH_OUT":0.90,
    "CASH_IN":0.90, "BILL_PAYMENT":0.70
}
amount = np.empty(N_TX, dtype=float)
airtime_values = np.array([100,200,500,1000,2000,5000], dtype=float)
airtime_probs = np.array([0.04,0.08,0.30,0.32,0.18,0.08])
for t in tx_types:
    m = transaction_type == t
    if t == "AIRTIME_PURCHASE":
        amount[m] = rng.choice(airtime_values, size=m.sum(), p=airtime_probs)
    else:
        sigma = sigmas[t]
        mu = np.log(means[t]) - 0.5*sigma*sigma
        amount[m] = rng.lognormal(mean=mu, sigma=sigma, size=m.sum())
        amount[m] *= np.array([amount_mult[p] for p in profiles[sender_idx[m]]])

roundable = ~np.isin(transaction_type, ["AIRTIME_PURCHASE","BILL_PAYMENT"])
rv = rng.random(N_TX)
idx1000 = roundable & (rv < 0.25)
idx500 = roundable & (rv >= 0.25) & (rv < 0.45)
idx100 = roundable & (rv >= 0.45) & (rv < 0.65)
amount[idx1000] = np.maximum(100, np.round(amount[idx1000]/1000)*1000)
amount[idx500] = np.maximum(100, np.round(amount[idx500]/500)*500)
amount[idx100] = np.maximum(100, np.round(amount[idx100]/100)*100)
amount = np.maximum(amount, 50.0)

initiation_channel = np.empty(N_TX, dtype=object)
for t in tx_types:
    m = transaction_type == t
    smart = device_type[sender_idx[m]] == "smartphone"
    n = m.sum()
    if t in ["CASH_IN","CASH_OUT"]:
        probs_smart, probs_feat = [0.02,0.03,0.95], [0.00,0.04,0.96]
    elif t == "AIRTIME_PURCHASE":
        probs_smart, probs_feat = [0.60,0.35,0.05], [0.05,0.90,0.05]
    elif t == "P2P_TRANSFER":
        probs_smart, probs_feat = [0.70,0.25,0.05], [0.05,0.90,0.05]
    else:
        probs_smart, probs_feat = [0.65,0.30,0.05], [0.05,0.90,0.05]
    out = np.empty(n, dtype=object)
    if smart.any():
        out[smart] = rng.choice(["APP","USSD","AGENT"], size=smart.sum(), p=probs_smart)
    if (~smart).any():
        out[~smart] = rng.choice(["APP","USSD","AGENT"], size=(~smart).sum(), p=probs_feat)
    initiation_channel[m] = out

is_new_device = rng.random(N_TX) < 0.04
sim_changed_last_7d = rng.random(N_TX) < 0.015
transaction_state_idx = cust_state_idx[sender_idx].copy()
loc_change_base = rng.random(N_TX) < 0.035
transaction_state_idx[loc_change_base] = rng.choice(len(states), size=loc_change_base.sum(), p=state_weights)
status = rng.choice(["successful","failed","reversed"], size=N_TX, p=[0.975,0.018,0.007])

# -----------------------------
# FRAUD INJECTION
# -----------------------------
is_fraud = np.zeros(N_TX, dtype=np.int8)
fraud_type = np.full(N_TX, "none", dtype=object)
fraud_event_id = np.full(N_TX, "", dtype=object)
fraud_indices = rng.choice(N_TX, size=N_FRAUD, replace=False)
assigned_types = rng.choice(fraud_types, size=N_FRAUD, p=fraud_probs)

for j, (idx, ft) in enumerate(zip(fraud_indices, assigned_types)):
    is_fraud[idx] = 1
    fraud_type[idx] = ft
    fraud_event_id[idx] = f"FE{j+1:06d}"

mule_wallets = rng.choice(N_CUSTOMERS, size=40, replace=False)
velocity_wallets = rng.choice(N_CUSTOMERS, size=80, replace=False)

for ft in fraud_types:
    inds = fraud_indices[assigned_types == ft]
    if len(inds) == 0:
        continue
    if ft == "social_engineering":
        change_rec = rng.random(len(inds)) < 0.75
        ii = inds[change_rec]
        receiver_kind[ii] = "customer"
        receiver_num[ii] = rng.integers(0, N_CUSTOMERS, size=len(ii))
        transaction_type[ii] = "P2P_TRANSFER"
        boost = rng.random(len(inds)) < 0.55
        amount[inds[boost]] *= rng.uniform(1.8, 4.5, size=boost.sum())
        is_new_device[inds] = rng.random(len(inds)) < 0.10
    elif ft == "account_takeover":
        is_new_device[inds] = rng.random(len(inds)) < 0.75
        sim_changed_last_7d[inds] |= rng.random(len(inds)) < 0.20
        change_rec = rng.random(len(inds)) < 0.70
        ii = inds[change_rec]
        receiver_kind[ii] = "customer"
        receiver_num[ii] = rng.integers(0, N_CUSTOMERS, size=len(ii))
        transaction_type[ii] = "P2P_TRANSFER"
        boost = rng.random(len(inds)) < 0.60
        amount[inds[boost]] *= rng.uniform(3.0, 8.0, size=boost.sum())
        loc = rng.random(len(inds)) < 0.40
        transaction_state_idx[inds[loc]] = rng.choice(len(states), size=loc.sum(), p=state_weights)
    elif ft == "sim_device_compromise":
        is_new_device[inds] = rng.random(len(inds)) < 0.90
        sim_changed_last_7d[inds] = rng.random(len(inds)) < 0.85
        change_rec = rng.random(len(inds)) < 0.65
        ii = inds[change_rec]
        receiver_kind[ii] = "customer"
        receiver_num[ii] = rng.integers(0, N_CUSTOMERS, size=len(ii))
        transaction_type[ii] = "P2P_TRANSFER"
        boost = rng.random(len(inds)) < 0.55
        amount[inds[boost]] *= rng.uniform(2.0, 5.0, size=boost.sum())
    elif ft == "mule_activity":
        receiver_kind[inds] = "customer"
        receiver_num[inds] = rng.choice(mule_wallets, size=len(inds))
        transaction_type[inds] = "P2P_TRANSFER"
        amount[inds] *= rng.uniform(1.2, 3.5, size=len(inds))
    elif ft == "rapid_cashout":
        transaction_type[inds] = "CASH_OUT"
        receiver_kind[inds] = "agent"
        agent_num[inds] = rng.integers(0, N_AGENTS, size=len(inds))
        receiver_num[inds] = agent_num[inds]
        initiation_channel[inds] = "AGENT"
        amount[inds] *= rng.uniform(1.4, 4.0, size=len(inds))
    elif ft == "velocity_fraud":
        groups = np.array_split(inds, max(1, len(inds)//5))
        for g in groups:
            if len(g) == 0:
                continue
            wallet = int(rng.choice(velocity_wallets))
            base_t = timestamps[g[0]]
            sender_idx[g] = wallet
            timestamps[g] = base_t + rng.integers(0, 45, size=len(g)).astype("timedelta64[m]")
            receiver_kind[g] = "customer"
            receiver_num[g] = rng.integers(0, N_CUSTOMERS, size=len(g))
            transaction_type[g] = "P2P_TRANSFER"
            amount[g] = rng.choice([7500,8000,9000,11000,12000,15000,18000], size=len(g))
            initiation_channel[g] = np.where(device_type[wallet]=="smartphone","APP","USSD")
    elif ft == "dormant_wallet_takeover":
        dormant = rng.choice(dormant_candidates, size=len(inds), replace=True)
        sender_idx[inds] = dormant
        is_new_device[inds] = rng.random(len(inds)) < 0.75
        receiver_kind[inds] = "customer"
        receiver_num[inds] = rng.integers(0, N_CUSTOMERS, size=len(inds))
        transaction_type[inds] = "P2P_TRANSFER"
        amount[inds] *= rng.uniform(2.5, 6.0, size=len(inds))
        initiation_channel[inds] = np.where(device_type[dormant]=="smartphone","APP","USSD")

same = (receiver_kind == "customer") & (receiver_num == sender_idx)
receiver_num[same] = (receiver_num[same] + 1) % N_CUSTOMERS

# Recompute transaction state baseline for fraud rows whose sender changed.
sender_changed_ft = np.isin(fraud_type, ["velocity_fraud","dormant_wallet_takeover"])
transaction_state_idx[sender_changed_ft] = cust_state_idx[sender_idx[sender_changed_ft]]

# Sort chronologically.
order = np.argsort(timestamps)
timestamps = pd.DatetimeIndex(timestamps[order])
sender_idx = sender_idx[order]
transaction_type = transaction_type[order]
receiver_kind = receiver_kind[order]
receiver_num = receiver_num[order]
agent_num = agent_num[order]
amount = amount[order]
initiation_channel = initiation_channel[order]
is_new_device = is_new_device[order]
sim_changed_last_7d = sim_changed_last_7d[order]
transaction_state_idx = transaction_state_idx[order]
status = status[order]
is_fraud = is_fraud[order]
fraud_type = fraud_type[order]
fraud_event_id = fraud_event_id[order]

# -----------------------------
# CHRONOLOGICAL LEDGER + HISTORY FEATURES
# -----------------------------
balances = start_balance.astype(float).copy()
daily_outflow = defaultdict(float)

hist_1h = [deque() for _ in range(N_CUSTOMERS)]
hist_24h = [deque() for _ in range(N_CUSTOMERS)]
hist_30d = [deque() for _ in range(N_CUSTOMERS)]
seen_recipients = [set() for _ in range(N_CUSTOMERS)]
last_tx_time = np.array(
    [(START_DATE - pd.Timedelta(days=int(d))).to_datetime64() for d in pre_sim_inactive_days],
    dtype="datetime64[ns]"
)
last_incoming = np.full(N_CUSTOMERS, np.datetime64("NaT"), dtype="datetime64[ns]")
recv_24h = [deque() for _ in range(N_CUSTOMERS)]

agent_day_count = defaultdict(int)
agent_day_cashout = defaultdict(float)
agent_day_customers = defaultdict(set)

sender_balance_before = np.zeros(N_TX)
sender_balance_after = np.zeros(N_TX)
receiver_balance_before = np.full(N_TX, np.nan)
receiver_balance_after = np.full(N_TX, np.nan)
tx_1h = np.zeros(N_TX, dtype=np.int16)
tx_24h = np.zeros(N_TX, dtype=np.int16)
amt_24h = np.zeros(N_TX)
avg_30d = np.zeros(N_TX)
amount_dev = np.zeros(N_TX)
mins_since_last = np.zeros(N_TX)
uniq_rec_24h = np.zeros(N_TX, dtype=np.int16)
is_new_recipient_arr = np.zeros(N_TX, dtype=np.int8)
days_since_last_activity = np.zeros(N_TX)
cashout_24h = np.zeros(N_TX)
failed_24h = np.zeros(N_TX, dtype=np.int16)
is_unusual_hour_arr = np.zeros(N_TX, dtype=np.int8)
recv_incoming_24h = np.zeros(N_TX, dtype=np.int16)
recv_unique_senders_24h = np.zeros(N_TX, dtype=np.int16)
mins_since_recent_incoming = np.full(N_TX, 999999.0)
agent_tx_today = np.zeros(N_TX, dtype=np.int16)
agent_cashout_today = np.zeros(N_TX)
agent_unique_customers_today = np.zeros(N_TX, dtype=np.int16)

for i in range(N_TX):
    t = pd.Timestamp(timestamps[i])
    s = int(sender_idx[i])
    typ = transaction_type[i]
    amt = float(amount[i])
    rk = receiver_kind[i]
    rn = int(receiver_num[i])
    day_key = t.date()

    cutoff1 = t - pd.Timedelta(hours=1)
    cutoff24 = t - pd.Timedelta(hours=24)
    cutoff30 = t - pd.Timedelta(days=30)

    h1 = hist_1h[s]
    while h1 and h1[0][0] < cutoff1:
        h1.popleft()
    h24 = hist_24h[s]
    while h24 and h24[0][0] < cutoff24:
        h24.popleft()
    h30 = hist_30d[s]
    while h30 and h30[0][0] < cutoff30:
        h30.popleft()

    tx_1h[i] = len(h1)
    tx_24h[i] = len(h24)
    if h24:
        vals24 = list(h24)
        amt_24h[i] = sum(x[1] for x in vals24)
        cashout_24h[i] = sum(x[1] for x in vals24 if x[3])
        failed_24h[i] = sum(1 for x in vals24 if x[4])
        uniq_rec_24h[i] = len({x[2] for x in vals24})
    avg_30d[i] = (sum(x[1] for x in h30)/len(h30)) if h30 else base_avg_amount[s]
    amount_dev[i] = amt / max(avg_30d[i], 1.0)

    last_t = pd.Timestamp(last_tx_time[s])
    delta_min = max(0.0, (t - last_t).total_seconds()/60.0)
    mins_since_last[i] = delta_min
    days_since_last_activity[i] = delta_min/1440.0

    key = (rk, rn)
    is_new_recipient_arr[i] = int(key not in seen_recipients[s])

    circ_dist = abs(t.hour - preferred_hour[s])
    circ_dist = min(circ_dist, 24-circ_dist)
    is_unusual_hour_arr[i] = int(circ_dist > 6.0)

    if rk == "customer":
        rh = recv_24h[rn]
        while rh and rh[0][0] < cutoff24:
            rh.popleft()
        recv_incoming_24h[i] = len(rh)
        recv_unique_senders_24h[i] = len({x[1] for x in rh})

    li = last_incoming[s]
    if not np.isnat(li):
        mins_since_recent_incoming[i] = max(
            0.0, float((t.to_datetime64() - li) / np.timedelta64(1, "m"))
        )

    if rk == "agent":
        akey = (rn, day_key)
        agent_tx_today[i] = agent_day_count[akey]
        agent_cashout_today[i] = agent_day_cashout[akey]
        agent_unique_customers_today[i] = len(agent_day_customers[akey])

    sender_balance_before[i] = balances[s]
    if rk == "customer":
        receiver_balance_before[i] = balances[rn]

    final_status = status[i]
    outflow = typ in ["P2P_TRANSFER","MERCHANT_PAYMENT","AIRTIME_PURCHASE","CASH_OUT","BILL_PAYMENT"]
    inflow = typ == "CASH_IN"

    if final_status == "successful" and outflow:
        lim = daily_outflow_limit[int(kyc[s])]
        used = daily_outflow[(s, day_key)]
        if amt > balances[s] or used + amt > lim:
            final_status = "failed"

    if final_status == "successful":
        if outflow:
            balances[s] -= amt
            daily_outflow[(s, day_key)] += amt
        elif inflow:
            balances[s] = min(balances[s] + amt, balance_cap[int(kyc[s])])

        if typ == "P2P_TRANSFER" and rk == "customer":
            balances[rn] = min(balances[rn] + amt, balance_cap[int(kyc[rn])])
            last_incoming[rn] = t.to_datetime64()

    status[i] = final_status
    sender_balance_after[i] = balances[s]
    if rk == "customer":
        receiver_balance_after[i] = balances[rn]

    failed_flag = final_status == "failed"
    h1.append((t, amt))
    h24.append((t, amt, key, typ=="CASH_OUT", failed_flag))
    h30.append((t, amt))
    seen_recipients[s].add(key)
    last_tx_time[s] = t.to_datetime64()

    if rk == "customer":
        recv_24h[rn].append((t, s))
    if rk == "agent":
        akey = (rn, day_key)
        agent_day_count[akey] += 1
        if typ == "CASH_OUT":
            agent_day_cashout[akey] += amt
        agent_day_customers[akey].add(s)

# -----------------------------
# FINAL DATAFRAME
# -----------------------------
transaction_id = np.array([f"TX{i+1:07d}" for i in range(N_TX)], dtype=object)
sender_id = np.array([f"W{s+1:06d}" for s in sender_idx], dtype=object)
receiver_id = np.empty(N_TX, dtype=object)
for kind in ["customer","merchant","agent"]:
    m = receiver_kind == kind
    prefix = {"customer":"W","merchant":"M","agent":"A"}[kind]
    width = {"customer":6,"merchant":5,"agent":4}[kind]
    receiver_id[m] = [f"{prefix}{x+1:0{width}d}" for x in receiver_num[m]]
agent_id = np.where(receiver_kind=="agent", receiver_id, "")

sender_state = np.array([states[x] for x in cust_state_idx[sender_idx]], dtype=object)
receiver_state = np.empty(N_TX, dtype=object)
cust_recv = receiver_kind=="customer"
merch_recv = receiver_kind=="merchant"
agent_recv = receiver_kind=="agent"
receiver_state[cust_recv] = [states[x] for x in cust_state_idx[receiver_num[cust_recv]]]
receiver_state[merch_recv] = [states[x] for x in merchant_state_idx[receiver_num[merch_recv]]]
receiver_state[agent_recv] = [states[x] for x in agent_state_idx[receiver_num[agent_recv]]]
transaction_state = np.array([states[x] for x in transaction_state_idx], dtype=object)
location_changed = (transaction_state != sender_state).astype(np.int8)

hour = timestamps.hour.astype(np.int8)
day_of_week = timestamps.dayofweek.astype(np.int8)
is_weekend = (day_of_week >= 5).astype(np.int8)

sender_account_age_days = account_age_days[sender_idx].astype(np.int32)
receiver_account_age_days = np.zeros(N_TX, dtype=np.int32)
receiver_account_age_days[cust_recv] = account_age_days[receiver_num[cust_recv]]
kyc_out = kyc[sender_idx].astype(np.int8)
dev_type_out = device_type[sender_idx]
days_on_current_device = device_age_days[sender_idx].astype(np.int32)
urban_rural = cust_urban[sender_idx]
geo_zone_out = np.array([geo_zone[s] for s in sender_state], dtype=object)

df = pd.DataFrame({
    "transaction_id": transaction_id,
    "timestamp": timestamps.astype(str),
    "sender_id": sender_id,
    "receiver_id": receiver_id,
    "agent_id": agent_id,
    "customer_profile": profiles[sender_idx],
    "transaction_type": transaction_type,
    "amount": np.round(amount,2),
    "initiation_channel": initiation_channel,
    "status": status,
    "sender_balance_before": np.round(sender_balance_before,2),
    "sender_balance_after": np.round(sender_balance_after,2),
    "receiver_balance_before": np.round(receiver_balance_before,2),
    "receiver_balance_after": np.round(receiver_balance_after,2),
    "sender_account_age_days": sender_account_age_days,
    "receiver_account_age_days": receiver_account_age_days,
    "kyc_level": kyc_out,
    "sender_state": sender_state,
    "receiver_state": receiver_state,
    "transaction_state": transaction_state,
    "geo_zone": geo_zone_out,
    "urban_rural": urban_rural,
    "device_type": dev_type_out,
    "is_new_device": is_new_device.astype(np.int8),
    "days_on_current_device": days_on_current_device,
    "sim_changed_last_7d": sim_changed_last_7d.astype(np.int8),
    "location_changed": location_changed,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "transactions_last_1h": tx_1h,
    "transactions_last_24h": tx_24h,
    "amount_last_24h": np.round(amt_24h,2),
    "avg_amount_30d": np.round(avg_30d,2),
    "amount_deviation_ratio": np.round(amount_dev,4),
    "time_since_last_tx_min": np.round(mins_since_last,2),
    "unique_recipients_24h": uniq_rec_24h,
    "is_new_recipient": is_new_recipient_arr,
    "days_since_last_activity": np.round(days_since_last_activity,3),
    "cashout_amount_24h": np.round(cashout_24h,2),
    "prior_failed_attempts_24h": failed_24h,
    "is_unusual_hour": is_unusual_hour_arr,
    "receiver_incoming_tx_24h": recv_incoming_24h,
    "receiver_unique_senders_24h": recv_unique_senders_24h,
    "minutes_since_recent_incoming": np.round(mins_since_recent_incoming,2),
    "agent_transactions_today": agent_tx_today,
    "agent_cashout_today": np.round(agent_cashout_today,2),
    "agent_unique_customers_today": agent_unique_customers_today,
    "is_fraud": is_fraud,
    "fraud_type": fraud_type,
    "fraud_event_id": fraud_event_id,
})

# Model-safe export: timestamp retained for chronological splitting, but should not be directly encoded as a raw predictor.
ml_features = [
    "timestamp","transaction_type","amount","initiation_channel",
    "sender_balance_before","sender_account_age_days","receiver_account_age_days","kyc_level",
    "sender_state","receiver_state","transaction_state","geo_zone","urban_rural",
    "device_type","is_new_device","days_on_current_device","sim_changed_last_7d","location_changed",
    "hour","day_of_week","is_weekend",
    "transactions_last_1h","transactions_last_24h","amount_last_24h","avg_amount_30d",
    "amount_deviation_ratio","time_since_last_tx_min","unique_recipients_24h",
    "is_new_recipient","days_since_last_activity","cashout_amount_24h",
    "prior_failed_attempts_24h","is_unusual_hour","receiver_incoming_tx_24h",
    "receiver_unique_senders_24h","minutes_since_recent_incoming",
    "agent_transactions_today","agent_cashout_today","agent_unique_customers_today",
]
df_ml = df[ml_features + ["is_fraud"]].copy()
df_unlabelled = df[ml_features].copy()

# -----------------------------
# VALIDATION
# -----------------------------
outflow_mask = df["transaction_type"].isin(
    ["P2P_TRANSFER","MERCHANT_PAYMENT","AIRTIME_PURCHASE","CASH_OUT","BILL_PAYMENT"]
) & (df["status"]=="successful")
tmp = df.loc[outflow_mask, ["sender_id","timestamp","amount","kyc_level"]].copy()
tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date
daily = tmp.groupby(["sender_id","date","kyc_level"], as_index=False)["amount"].sum()
daily["limit"] = daily["kyc_level"].map(daily_outflow_limit)

validation = {
    "rows": int(len(df)),
    "fraud_rows": int(df["is_fraud"].sum()),
    "fraud_rate": round(float(df["is_fraud"].mean()), 6),
    "mean_amount_ngn": round(float(df["amount"].mean()), 2),
    "median_amount_ngn": round(float(df["amount"].median()), 2),
    "duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
    "negative_amounts": int((df["amount"] < 0).sum()),
    "negative_sender_balances_after": int((df["sender_balance_after"] < -1e-9).sum()),
    "missing_values_in_ml_dataset": int(df_ml.isna().sum().sum()),
    "successful_kyc_daily_limit_violations": int((daily["amount"] > daily["limit"] + 1e-6).sum()),
    "fraud_scenario_counts": df.loc[df["is_fraud"]==1, "fraud_type"].value_counts().to_dict(),
    "legitimate_anomaly_rates": {
        "new_device_rate": round(float(df.loc[df.is_fraud==0,"is_new_device"].mean()),4),
        "new_recipient_rate": round(float(df.loc[df.is_fraud==0,"is_new_recipient"].mean()),4),
        "unusual_hour_rate": round(float(df.loc[df.is_fraud==0,"is_unusual_hour"].mean()),4),
        "amount_gt_3x_personal_avg_rate": round(float((df.loc[df.is_fraud==0,"amount_deviation_ratio"]>3).mean()),4),
        "high_velocity_5plus_1h_rate": round(float((df.loc[df.is_fraud==0,"transactions_last_1h"]>=5).mean()),4),
    },
    "status_distribution": df["status"].value_counts(normalize=True).round(4).to_dict(),
    "transaction_type_distribution": df["transaction_type"].value_counts(normalize=True).round(4).to_dict(),
}

# -----------------------------
# DATA DICTIONARY
# -----------------------------
descriptions = {
    "transaction_id":"Unique synthetic transaction identifier.",
    "timestamp":"Transaction date/time; retain for chronological splitting, not as a raw predictor.",
    "sender_id":"Synthetic sender wallet identifier.",
    "receiver_id":"Synthetic receiver wallet, merchant, or agent identifier.",
    "agent_id":"Agent identifier when an agent is involved.",
    "customer_profile":"Hidden simulation archetype; excluded from ML inputs.",
    "transaction_type":"P2P transfer, merchant payment, airtime, cash-out, cash-in, or bill payment.",
    "amount":"Transaction amount in Nigerian naira.",
    "initiation_channel":"APP, USSD, or AGENT interface within the mobile-money ecosystem.",
    "status":"Successful, failed, or reversed; excluded from main pre-authorisation feature set.",
    "sender_balance_before":"Sender wallet balance immediately before processing.",
    "sender_balance_after":"Sender wallet balance after processing; excluded to reduce leakage risk.",
    "receiver_balance_before":"Receiver wallet balance before transfer, when applicable.",
    "receiver_balance_after":"Receiver wallet balance after transfer; excluded from main ML dataset.",
    "sender_account_age_days":"Age of sender wallet/account in days.",
    "receiver_account_age_days":"Age of recipient wallet/account in days when applicable; zero for non-wallet receivers.",
    "kyc_level":"Synthetic KYC tier 1, 2, or 3.",
    "sender_state":"Synthetic Nigerian state associated with sender.",
    "receiver_state":"Synthetic state associated with receiver/entity.",
    "transaction_state":"Synthetic state where transaction is initiated.",
    "geo_zone":"Sender geopolitical zone.",
    "urban_rural":"Synthetic urban/semi-urban/rural classification.",
    "device_type":"Smartphone or feature phone.",
    "is_new_device":"Whether transaction is made from a recently introduced device.",
    "days_on_current_device":"Synthetic age of current device relationship.",
    "sim_changed_last_7d":"Whether a simulated SIM change occurred in the prior seven days.",
    "location_changed":"Whether transaction state differs from sender state.",
    "hour":"Hour of day derived from timestamp.",
    "day_of_week":"Monday=0 through Sunday=6.",
    "is_weekend":"1 for Saturday/Sunday.",
    "transactions_last_1h":"Sender transaction count in the prior one hour.",
    "transactions_last_24h":"Sender transaction count in the prior 24 hours.",
    "amount_last_24h":"Sender transaction value attempted in prior 24 hours.",
    "avg_amount_30d":"Sender rolling prior-30-day average amount; baseline profile used before enough history exists.",
    "amount_deviation_ratio":"Current amount divided by sender historical average.",
    "time_since_last_tx_min":"Minutes since sender previous transaction.",
    "unique_recipients_24h":"Distinct sender recipients in prior 24 hours.",
    "is_new_recipient":"1 if sender has not previously interacted with the receiver in the simulation.",
    "days_since_last_activity":"Days since prior sender activity, including pre-simulation inactivity baseline.",
    "cashout_amount_24h":"Sender cash-out value in prior 24 hours.",
    "prior_failed_attempts_24h":"Sender failed attempts in prior 24 hours.",
    "is_unusual_hour":"1 when transaction hour differs strongly from simulated sender preference.",
    "receiver_incoming_tx_24h":"Incoming transfer count to wallet receiver in prior 24 hours.",
    "receiver_unique_senders_24h":"Distinct senders to wallet receiver in prior 24 hours.",
    "minutes_since_recent_incoming":"Minutes since sender most recently received a P2P transfer; large sentinel when none.",
    "agent_transactions_today":"Agent transactions already observed that day.",
    "agent_cashout_today":"Agent cash-out value already observed that day.",
    "agent_unique_customers_today":"Unique customers already served by agent that day.",
    "is_fraud":"Binary target: 1 synthetic fraud, 0 legitimate.",
    "fraud_type":"Synthetic fraud scenario; never use as predictor.",
    "fraud_event_id":"Audit identifier for synthetic fraud event; never use as predictor."
}
model_usage = {}
for c in df.columns:
    if c in ml_features:
        model_usage[c] = "Split/derive" if c == "timestamp" else "Yes"
    elif c == "is_fraud":
        model_usage[c] = "Target"
    else:
        model_usage[c] = "No"

dd = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "description": [descriptions.get(c,"") for c in df.columns],
    "model_use": [model_usage[c] for c in df.columns],
})

# -----------------------------
# WRITE FILES
# -----------------------------
full_path = OUT/"nsmmf_full.csv"
ml_path = OUT/"nsmmf_ml.csv"
unlabelled_path = OUT/"nsmmf_unlabelled.csv"
dd_path = OUT/"data_dictionary.csv"
params_path = OUT/"simulation_parameters.json"
validation_path = OUT/"validation_report.json"
readme_path = OUT/"README.md"

df.to_csv(full_path, index=False)
df_ml.to_csv(ml_path, index=False)
df_unlabelled.to_csv(unlabelled_path, index=False)
dd.to_csv(dd_path, index=False)

params = {
    "dataset_name":"NSMMF-250K — Nigerian Synthetic Mobile Money Fraud Dataset",
    "version":"1.0",
    "random_seed":SEED,
    "transactions":N_TX,
    "customers":N_CUSTOMERS,
    "agents":N_AGENTS,
    "merchants":N_MERCHANTS,
    "simulation_days":SIM_DAYS,
    "start_date":str(START_DATE.date()),
    "target_fraud_rate":TARGET_FRAUD_RATE,
    "fraud_mix_assumption":dict(zip(fraud_types.tolist(), fraud_probs.tolist())),
    "customer_profile_mix_assumption":dict(zip(profile_names.tolist(), profile_probs.tolist())),
    "kyc_population_mix_assumption":{"1":0.55,"2":0.35,"3":0.10},
    "kyc_daily_outflow_limits_ngn":{"1":50000,"2":200000,"3":5000000},
    "transaction_types":tx_types.tolist(),
    "design_notes":[
        "Synthetic data only; not real NIBSS or customer transaction data.",
        "Fraud prevalence, population mix, geography weights, transaction mix, timing and fraud injection are simulation assumptions unless separately sourced in the project methodology.",
        "nsmmf_ml.csv excludes post-transaction values, IDs, hidden customer archetypes, status and fraud-scenario metadata.",
        "History-derived behavioural features are calculated using prior transactions only."
    ]
}
params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

readme_path.write_text(
f"""# NSMMF-250K

**Nigerian Synthetic Mobile Money Fraud Dataset — Version 1**

Designed for the **Mobile Money Fraud Detector** MVP.

## Scope
- {N_TX:,} synthetic transactions
- {N_CUSTOMERS:,} wallets
- {N_AGENTS:,} agents
- {N_MERCHANTS:,} merchants
- 36 Nigerian states + FCT
- {SIM_DAYS} simulated days
- {N_FRAUD:,} synthetic fraud-labelled transactions ({TARGET_FRAUD_RATE:.1%})

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
""", encoding="utf-8")

# Save a runnable notebook-friendly loader/helper script.
helper_path = OUT/"load_nsmmf.py"
helper_path.write_text(
"""from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent

def load_ml():
    df = pd.read_csv(ROOT / "nsmmf_ml.csv", parse_dates=["timestamp"])
    return df

def load_full():
    df = pd.read_csv(ROOT / "nsmmf_full.csv", parse_dates=["timestamp"])
    return df

def load_parameters():
    return json.loads((ROOT / "simulation_parameters.json").read_text())

if __name__ == "__main__":
    df = load_ml()
    print(df.shape)
    print(df["is_fraud"].value_counts())
""", encoding="utf-8")

zip_path = OUT.parent / "NSMMF_250K_generated_bundle.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        z.write(p, arcname=p.name)

print("Generated files:")
for p in [full_path, ml_path, unlabelled_path, dd_path, params_path, validation_path, readme_path, helper_path, zip_path]:
    print(f"- {p.name}: {p.stat().st_size/1024/1024:.2f} MB")

print("\nValidation summary:")
print(json.dumps(validation, indent=2))


# ===== REFINEMENT PASS =====

# Refinement pass for NSMMF-250K V1.1 using the generated transaction stream.
# This corrects the realism issues surfaced by the first validation run.

import numpy as np, pandas as pd, json, zipfile
from pathlib import Path
from collections import defaultdict, deque

# 1) Recalibrate amounts toward the ~NGN 9k overall target.
non_airtime = transaction_type != "AIRTIME_PURCHASE"
amount[non_airtime] *= 0.748

# Preserve sensible human-entered rounding after scaling.
roundable = ~np.isin(transaction_type, ["AIRTIME_PURCHASE","BILL_PAYMENT"])
rv2 = rng.random(N_TX)
m = roundable & (rv2 < 0.25)
amount[m] = np.maximum(100, np.round(amount[m]/1000)*1000)
m = roundable & (rv2 >= 0.25) & (rv2 < 0.45)
amount[m] = np.maximum(100, np.round(amount[m]/500)*500)
m = roundable & (rv2 >= 0.45) & (rv2 < 0.65)
amount[m] = np.maximum(100, np.round(amount[m]/100)*100)
amount = np.maximum(amount, 50.0)

# 2) Build recurring social/payment networks per wallet.
# Small recurring pools create realistic repeat recipients while retaining exploration/new-recipient behaviour.
cust_pools = []
merchant_pools = []
agent_pools = []
all_customers = np.arange(N_CUSTOMERS)
all_merchants = np.arange(N_MERCHANTS)
all_agents = np.arange(N_AGENTS)

for s in range(N_CUSTOMERS):
    cp = rng.choice(all_customers[all_customers != s], size=int(rng.integers(2,5)), replace=False)
    mp = rng.choice(all_merchants, size=int(rng.integers(2,5)), replace=False)
    ap = rng.choice(all_agents, size=int(rng.integers(1,4)), replace=False)
    cust_pools.append(cp)
    merchant_pools.append(mp)
    agent_pools.append(ap)

legit = is_fraud == 0
for i in np.where(legit)[0]:
    s = int(sender_idx[i])
    typ = transaction_type[i]
    # Roughly 88-94% of routine transactions reuse a familiar counterparty.
    if typ == "P2P_TRANSFER":
        if rng.random() < 0.90:
            receiver_kind[i] = "customer"
            receiver_num[i] = int(rng.choice(cust_pools[s]))
        else:
            receiver_kind[i] = "customer"
            r = int(rng.integers(0, N_CUSTOMERS))
            receiver_num[i] = (r + 1) % N_CUSTOMERS if r == s else r
    elif typ in ["MERCHANT_PAYMENT","BILL_PAYMENT","AIRTIME_PURCHASE"]:
        receiver_kind[i] = "merchant"
        receiver_num[i] = int(rng.choice(merchant_pools[s])) if rng.random() < 0.92 else int(rng.integers(0,N_MERCHANTS))
    elif typ in ["CASH_IN","CASH_OUT"]:
        receiver_kind[i] = "agent"
        a = int(rng.choice(agent_pools[s])) if rng.random() < 0.94 else int(rng.integers(0,N_AGENTS))
        receiver_num[i] = a
        agent_num[i] = a

# 3) Explicit legitimate velocity bursts among traders/business users.
# These prevent "high velocity = fraud" from becoming a deterministic rule.
eligible_traders = np.where(np.isin(profiles, ["trader_business","cash_heavy"]))[0]
legit_indices = np.where(is_fraud == 0)[0]
burst_groups = 420
for _ in range(burst_groups):
    gsize = int(rng.integers(6,10))
    g = rng.choice(legit_indices, size=gsize, replace=False)
    wallet = int(rng.choice(eligible_traders))
    base_t = timestamps[g[0]].to_datetime64()
    sender_idx[g] = wallet
    timestamps_arr = timestamps.to_numpy(dtype="datetime64[ns]")
    timestamps_arr[g] = base_t + rng.integers(0,55,size=gsize).astype("timedelta64[m]")
    timestamps = pd.DatetimeIndex(timestamps_arr)
    # Mostly normal P2P/merchant activity in the burst.
    for idx in g:
        if rng.random() < 0.55:
            transaction_type[idx] = "P2P_TRANSFER"
            receiver_kind[idx] = "customer"
            receiver_num[idx] = int(rng.choice(cust_pools[wallet]))
        else:
            transaction_type[idx] = "MERCHANT_PAYMENT"
            receiver_kind[idx] = "merchant"
            receiver_num[idx] = int(rng.choice(merchant_pools[wallet]))
        initiation_channel[idx] = "APP" if device_type[wallet] == "smartphone" else "USSD"

# 4) Reset baseline transaction processing status to a lower operational failure rate.
status = rng.choice(["successful","failed","reversed"], size=N_TX, p=[0.990,0.005,0.005])

# 5) Give wallets more realistic opening liquidity so ordinary spending doesn't create artificial failures.
start_balance = np.empty(N_CUSTOMERS, dtype=float)
for s in range(N_CUSTOMERS):
    if kyc[s] == 1:
        start_balance[s] = rng.uniform(130_000, 260_000)
    elif kyc[s] == 2:
        start_balance[s] = rng.uniform(220_000, 450_000)
    else:
        start_balance[s] = rng.lognormal(mean=np.log(900_000), sigma=0.55)
        start_balance[s] = min(start_balance[s], 4_000_000)

# Re-sort after legitimate burst insertion.
order2 = np.argsort(timestamps.to_numpy())
timestamps = pd.DatetimeIndex(timestamps.to_numpy()[order2])
for name in [
    "sender_idx","transaction_type","receiver_kind","receiver_num","agent_num","amount",
    "initiation_channel","is_new_device","sim_changed_last_7d","transaction_state_idx",
    "status","is_fraud","fraud_type","fraud_event_id"
]:
    globals()[name] = globals()[name][order2]

# Ensure transaction state follows sender for ordinary rows whose sender was modified in bursts.
transaction_state_idx[is_fraud==0] = np.where(
    rng.random((is_fraud==0).sum()) < 0.965,
    cust_state_idx[sender_idx[is_fraud==0]],
    transaction_state_idx[is_fraud==0]
)

# -----------------------------
# Re-run chronological ledger and all history features
# -----------------------------
balances = start_balance.astype(float).copy()
daily_outflow = defaultdict(float)
hist_1h = [deque() for _ in range(N_CUSTOMERS)]
hist_24h = [deque() for _ in range(N_CUSTOMERS)]
hist_30d = [deque() for _ in range(N_CUSTOMERS)]
seen_recipients = [set() for _ in range(N_CUSTOMERS)]
last_tx_time = np.array(
    [(START_DATE - pd.Timedelta(days=int(d))).to_datetime64() for d in pre_sim_inactive_days],
    dtype="datetime64[ns]"
)
last_incoming = np.full(N_CUSTOMERS, np.datetime64("NaT"), dtype="datetime64[ns]")
recv_24h = [deque() for _ in range(N_CUSTOMERS)]
agent_day_count = defaultdict(int)
agent_day_cashout = defaultdict(float)
agent_day_customers = defaultdict(set)

sender_balance_before = np.zeros(N_TX)
sender_balance_after = np.zeros(N_TX)
receiver_balance_before = np.full(N_TX, np.nan)
receiver_balance_after = np.full(N_TX, np.nan)
tx_1h = np.zeros(N_TX, dtype=np.int16)
tx_24h = np.zeros(N_TX, dtype=np.int16)
amt_24h = np.zeros(N_TX)
avg_30d = np.zeros(N_TX)
amount_dev = np.zeros(N_TX)
mins_since_last = np.zeros(N_TX)
uniq_rec_24h = np.zeros(N_TX, dtype=np.int16)
is_new_recipient_arr = np.zeros(N_TX, dtype=np.int8)
days_since_last_activity = np.zeros(N_TX)
cashout_24h = np.zeros(N_TX)
failed_24h = np.zeros(N_TX, dtype=np.int16)
is_unusual_hour_arr = np.zeros(N_TX, dtype=np.int8)
recv_incoming_24h = np.zeros(N_TX, dtype=np.int16)
recv_unique_senders_24h = np.zeros(N_TX, dtype=np.int16)
mins_since_recent_incoming = np.full(N_TX, 999999.0)
agent_tx_today = np.zeros(N_TX, dtype=np.int16)
agent_cashout_today = np.zeros(N_TX)
agent_unique_customers_today = np.zeros(N_TX, dtype=np.int16)

for i in range(N_TX):
    t = pd.Timestamp(timestamps[i])
    s = int(sender_idx[i])
    typ = transaction_type[i]
    amt = float(amount[i])
    rk = receiver_kind[i]
    rn = int(receiver_num[i])
    day_key = t.date()
    cutoff1 = t - pd.Timedelta(hours=1)
    cutoff24 = t - pd.Timedelta(hours=24)
    cutoff30 = t - pd.Timedelta(days=30)

    h1 = hist_1h[s]
    while h1 and h1[0][0] < cutoff1:
        h1.popleft()
    h24 = hist_24h[s]
    while h24 and h24[0][0] < cutoff24:
        h24.popleft()
    h30 = hist_30d[s]
    while h30 and h30[0][0] < cutoff30:
        h30.popleft()

    tx_1h[i] = len(h1)
    tx_24h[i] = len(h24)
    if h24:
        vals24 = list(h24)
        amt_24h[i] = sum(x[1] for x in vals24)
        cashout_24h[i] = sum(x[1] for x in vals24 if x[3])
        failed_24h[i] = sum(1 for x in vals24 if x[4])
        uniq_rec_24h[i] = len({x[2] for x in vals24})
    avg_30d[i] = (sum(x[1] for x in h30)/len(h30)) if h30 else base_avg_amount[s]
    amount_dev[i] = amt / max(avg_30d[i], 1.0)

    delta_min = max(0.0, (t - pd.Timestamp(last_tx_time[s])).total_seconds()/60.0)
    mins_since_last[i] = delta_min
    days_since_last_activity[i] = delta_min / 1440.0
    key = (rk, rn)
    is_new_recipient_arr[i] = int(key not in seen_recipients[s])

    # Looser customer-specific unusual-hour definition, to avoid labelling a third of legitimate activity anomalous.
    circ_dist = abs(t.hour - preferred_hour[s])
    circ_dist = min(circ_dist, 24-circ_dist)
    is_unusual_hour_arr[i] = int(circ_dist > 8.5)

    if rk == "customer":
        rh = recv_24h[rn]
        while rh and rh[0][0] < cutoff24:
            rh.popleft()
        recv_incoming_24h[i] = len(rh)
        recv_unique_senders_24h[i] = len({x[1] for x in rh})

    li = last_incoming[s]
    if not np.isnat(li):
        mins_since_recent_incoming[i] = max(0.0, float((t.to_datetime64()-li)/np.timedelta64(1,"m")))

    if rk == "agent":
        akey = (rn, day_key)
        agent_tx_today[i] = agent_day_count[akey]
        agent_cashout_today[i] = agent_day_cashout[akey]
        agent_unique_customers_today[i] = len(agent_day_customers[akey])

    sender_balance_before[i] = balances[s]
    if rk == "customer":
        receiver_balance_before[i] = balances[rn]

    final_status = status[i]
    outflow = typ in ["P2P_TRANSFER","MERCHANT_PAYMENT","AIRTIME_PURCHASE","CASH_OUT","BILL_PAYMENT"]
    inflow = typ == "CASH_IN"

    if final_status == "successful" and outflow:
        lim = daily_outflow_limit[int(kyc[s])]
        if amt > balances[s] or daily_outflow[(s,day_key)] + amt > lim:
            final_status = "failed"

    if final_status == "successful":
        if outflow:
            balances[s] -= amt
            daily_outflow[(s,day_key)] += amt
        elif inflow:
            balances[s] = min(balances[s] + amt, balance_cap[int(kyc[s])])
        if typ == "P2P_TRANSFER" and rk == "customer":
            balances[rn] = min(balances[rn] + amt, balance_cap[int(kyc[rn])])
            last_incoming[rn] = t.to_datetime64()

    status[i] = final_status
    sender_balance_after[i] = balances[s]
    if rk == "customer":
        receiver_balance_after[i] = balances[rn]

    failed_flag = final_status == "failed"
    h1.append((t,amt))
    h24.append((t,amt,key,typ=="CASH_OUT",failed_flag))
    h30.append((t,amt))
    seen_recipients[s].add(key)
    last_tx_time[s] = t.to_datetime64()

    if rk == "customer":
        recv_24h[rn].append((t,s))
    if rk == "agent":
        akey = (rn,day_key)
        agent_day_count[akey] += 1
        if typ == "CASH_OUT":
            agent_day_cashout[akey] += amt
        agent_day_customers[akey].add(s)

# Rebuild dataframe from refined stream.
transaction_id = np.array([f"TX{i+1:07d}" for i in range(N_TX)], dtype=object)
sender_id = np.array([f"W{s+1:06d}" for s in sender_idx], dtype=object)
receiver_id = np.empty(N_TX, dtype=object)
for kind in ["customer","merchant","agent"]:
    mk = receiver_kind == kind
    prefix = {"customer":"W","merchant":"M","agent":"A"}[kind]
    width = {"customer":6,"merchant":5,"agent":4}[kind]
    receiver_id[mk] = [f"{prefix}{x+1:0{width}d}" for x in receiver_num[mk]]
agent_id = np.where(receiver_kind=="agent", receiver_id, "")

sender_state = np.array([states[x] for x in cust_state_idx[sender_idx]], dtype=object)
receiver_state = np.empty(N_TX, dtype=object)
cust_recv = receiver_kind=="customer"
merch_recv = receiver_kind=="merchant"
agent_recv = receiver_kind=="agent"
receiver_state[cust_recv] = [states[x] for x in cust_state_idx[receiver_num[cust_recv]]]
receiver_state[merch_recv] = [states[x] for x in merchant_state_idx[receiver_num[merch_recv]]]
receiver_state[agent_recv] = [states[x] for x in agent_state_idx[receiver_num[agent_recv]]]
transaction_state = np.array([states[x] for x in transaction_state_idx], dtype=object)
location_changed = (transaction_state != sender_state).astype(np.int8)

hour = timestamps.hour.astype(np.int8)
day_of_week = timestamps.dayofweek.astype(np.int8)
is_weekend = (day_of_week >= 5).astype(np.int8)
sender_account_age_days = account_age_days[sender_idx].astype(np.int32)
receiver_account_age_days = np.zeros(N_TX,dtype=np.int32)
receiver_account_age_days[cust_recv] = account_age_days[receiver_num[cust_recv]]
kyc_out = kyc[sender_idx].astype(np.int8)
dev_type_out = device_type[sender_idx]
days_on_current_device = device_age_days[sender_idx].astype(np.int32)
urban_rural = cust_urban[sender_idx]
geo_zone_out = np.array([geo_zone[s] for s in sender_state], dtype=object)

df = pd.DataFrame({
    "transaction_id": transaction_id,
    "timestamp": timestamps.astype(str),
    "sender_id": sender_id,
    "receiver_id": receiver_id,
    "agent_id": agent_id,
    "customer_profile": profiles[sender_idx],
    "transaction_type": transaction_type,
    "amount": np.round(amount,2),
    "initiation_channel": initiation_channel,
    "status": status,
    "sender_balance_before": np.round(sender_balance_before,2),
    "sender_balance_after": np.round(sender_balance_after,2),
    "receiver_balance_before": np.round(receiver_balance_before,2),
    "receiver_balance_after": np.round(receiver_balance_after,2),
    "sender_account_age_days": sender_account_age_days,
    "receiver_account_age_days": receiver_account_age_days,
    "kyc_level": kyc_out,
    "sender_state": sender_state,
    "receiver_state": receiver_state,
    "transaction_state": transaction_state,
    "geo_zone": geo_zone_out,
    "urban_rural": urban_rural,
    "device_type": dev_type_out,
    "is_new_device": is_new_device.astype(np.int8),
    "days_on_current_device": days_on_current_device,
    "sim_changed_last_7d": sim_changed_last_7d.astype(np.int8),
    "location_changed": location_changed,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "transactions_last_1h": tx_1h,
    "transactions_last_24h": tx_24h,
    "amount_last_24h": np.round(amt_24h,2),
    "avg_amount_30d": np.round(avg_30d,2),
    "amount_deviation_ratio": np.round(amount_dev,4),
    "time_since_last_tx_min": np.round(mins_since_last,2),
    "unique_recipients_24h": uniq_rec_24h,
    "is_new_recipient": is_new_recipient_arr,
    "days_since_last_activity": np.round(days_since_last_activity,3),
    "cashout_amount_24h": np.round(cashout_24h,2),
    "prior_failed_attempts_24h": failed_24h,
    "is_unusual_hour": is_unusual_hour_arr,
    "receiver_incoming_tx_24h": recv_incoming_24h,
    "receiver_unique_senders_24h": recv_unique_senders_24h,
    "minutes_since_recent_incoming": np.round(mins_since_recent_incoming,2),
    "agent_transactions_today": agent_tx_today,
    "agent_cashout_today": np.round(agent_cashout_today,2),
    "agent_unique_customers_today": agent_unique_customers_today,
    "is_fraud": is_fraud,
    "fraud_type": fraud_type,
    "fraud_event_id": fraud_event_id,
})

df_ml = df[ml_features + ["is_fraud"]].copy()
df_unlabelled = df[ml_features].copy()

outflow_mask = df["transaction_type"].isin(
    ["P2P_TRANSFER","MERCHANT_PAYMENT","AIRTIME_PURCHASE","CASH_OUT","BILL_PAYMENT"]
) & (df["status"]=="successful")
tmp = df.loc[outflow_mask, ["sender_id","timestamp","amount","kyc_level"]].copy()
tmp["date"] = pd.to_datetime(tmp["timestamp"]).dt.date
daily = tmp.groupby(["sender_id","date","kyc_level"],as_index=False)["amount"].sum()
daily["limit"] = daily["kyc_level"].map(daily_outflow_limit)

validation = {
    "version":"1.1",
    "rows":int(len(df)),
    "fraud_rows":int(df.is_fraud.sum()),
    "fraud_rate":round(float(df.is_fraud.mean()),6),
    "mean_amount_ngn":round(float(df.amount.mean()),2),
    "median_amount_ngn":round(float(df.amount.median()),2),
    "duplicate_transaction_ids":int(df.transaction_id.duplicated().sum()),
    "negative_amounts":int((df.amount<0).sum()),
    "negative_sender_balances_after":int((df.sender_balance_after < -1e-9).sum()),
    "missing_values_in_ml_dataset":int(df_ml.isna().sum().sum()),
    "successful_kyc_daily_limit_violations":int((daily.amount > daily.limit + 1e-6).sum()),
    "fraud_scenario_counts":df.loc[df.is_fraud==1,"fraud_type"].value_counts().to_dict(),
    "legitimate_anomaly_rates":{
        "new_device_rate":round(float(df.loc[df.is_fraud==0,"is_new_device"].mean()),4),
        "new_recipient_rate":round(float(df.loc[df.is_fraud==0,"is_new_recipient"].mean()),4),
        "unusual_hour_rate":round(float(df.loc[df.is_fraud==0,"is_unusual_hour"].mean()),4),
        "amount_gt_3x_personal_avg_rate":round(float((df.loc[df.is_fraud==0,"amount_deviation_ratio"]>3).mean()),4),
        "high_velocity_5plus_1h_rate":round(float((df.loc[df.is_fraud==0,"transactions_last_1h"]>=5).mean()),4),
    },
    "status_distribution":df.status.value_counts(normalize=True).round(4).to_dict(),
    "transaction_type_distribution":df.transaction_type.value_counts(normalize=True).round(4).to_dict(),
}

# overwrite exports
full_path = OUT/"nsmmf_full.csv"
ml_path = OUT/"nsmmf_ml.csv"
unlabelled_path = OUT/"nsmmf_unlabelled.csv"
validation_path = OUT/"validation_report.json"
params_path = OUT/"simulation_parameters.json"
df.to_csv(full_path,index=False)
df_ml.to_csv(ml_path,index=False)
df_unlabelled.to_csv(unlabelled_path,index=False)
validation_path.write_text(json.dumps(validation,indent=2),encoding="utf-8")

params = json.loads(params_path.read_text())
params["version"]="1.1"
params["refinements"]=[
    "Recurring wallet/merchant/agent counterparty pools added to reduce unrealistic new-recipient rates.",
    "Opening wallet liquidity increased to reduce artificial insufficient-funds failures.",
    "Non-airtime amount distributions rescaled toward the CBN-calibrated overall mean target.",
    "Legitimate trader/cash-heavy velocity bursts injected to create overlap with fraud behaviour.",
    "Unusual-hour threshold loosened to reduce over-labelling routine transactions."
]
params_path.write_text(json.dumps(params,indent=2),encoding="utf-8")

zip_path = OUT.parent / "NSMMF_250K_generated_bundle.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        z.write(p,arcname=p.name)

print(json.dumps(validation,indent=2))
print(f"\nBundle: {zip_path} ({zip_path.stat().st_size/1024/1024:.2f} MB)")
