import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 5_000_000  # 5 million rows

# ================================================================
# 1. USER PROFILES (high realism)
# ================================================================

user_ids = np.random.randint(1, 200000, size=N)

# Assign user types
user_type = np.random.choice(
    ["normal", "power_user", "risky_user", "new_user"],
    size=N,
    p=[0.75, 0.15, 0.05, 0.05]
)

# Base spend levels depending on user type
base_spend_map = {
    "normal": 300,
    "power_user": 1500,
    "risky_user": 600,
    "new_user": 200
}
base_spend = np.array([base_spend_map[u] for u in user_type])

# ================================================================
# 2. TRANSACTION AMOUNT (realistic distributions)
# ================================================================

amount = np.random.lognormal(mean=np.log(base_spend), sigma=0.6)

# Occasional very high transactions
high_value_mask = np.random.rand(N) < 0.02
amount[high_value_mask] *= np.random.uniform(5, 20, size=high_value_mask.sum())

amount = np.clip(amount, 1, 500000)  # clamp unrealistic values

# ================================================================
# 3. TIME FEATURES
# ================================================================

hour_of_day = np.random.randint(0, 24, size=N)
is_night = (hour_of_day <= 5).astype(int)
day_of_week = np.random.randint(0, 7, size=N)

# Night has higher fraud probability
night_fraud_boost = is_night * np.random.uniform(0.5, 1.5, size=N)

# ================================================================
# 4. DEVICE BEHAVIOR
# ================================================================

device_change_flag = (np.random.rand(N) < 0.02).astype(int)

# Risky users change devices more often
device_change_flag[user_type == "risky_user"] = \
    (np.random.rand((user_type == "risky_user").sum()) < 0.06)

# ================================================================
# 5. LOCATION FEATURES
# ================================================================

location_risk_level = np.random.choice(
    [0, 1, 2, 3],
    size=N,
    p=[0.85, 0.10, 0.04, 0.01]
)

# Higher risk for risky_user type
location_risk_level[user_type == "risky_user"] += (np.random.rand((user_type == "risky_user").sum()) < 0.3)

location_risk_level = np.clip(location_risk_level, 0, 3)

location_velocity_flag = (np.random.rand(N) < 0.003).astype(int)

# ================================================================
# 6. BANK FEATURES
# ================================================================

sender_bank_code = np.random.randint(1, 6, size=N)
receiver_bank_code = np.random.randint(1, 6, size=N)

# ================================================================
# 7. USER BEHAVIOR (velocity, failed attempts)
# ================================================================

velocity_1h = np.random.poisson(lam=0.5, size=N)
failed_attempts_24h = np.random.poisson(lam=0.1, size=N)

# Risky users have more failed attempts
failed_attempts_24h[user_type == "risky_user"] += np.random.poisson(lam=1.2, size=(user_type == "risky_user").sum())

fraud_history_flag = (np.random.rand(N) < 0.005).astype(int)

is_new_beneficiary = (np.random.rand(N) < 0.04).astype(int)
is_new_beneficiary[user_type == "new_user"] = 1

# ================================================================
# 8. AMOUNT Z-SCORE (realistic)
# ================================================================

median_amount = np.median(amount)
mad = np.median(np.abs(amount - median_amount)) + 1e-6

amount_zscore = (amount - median_amount) / mad
amount_zscore = np.clip(amount_zscore, -10, 12)

# ================================================================
# 9. FRAUD SCORE (high realism)
# ================================================================

w = {
    "amount_z": 0.2,
    "night": 0.6,
    "device_change": 1.8,
    "loc_risk": 0.8,
    "loc_vel": 3.0,
    "velocity": 0.25,
    "failed": 0.6,
    "fraud_hist": 2.5,
    "new_ben": 1.4,
    "user_risky": 2.2,
}

user_risky_flag = (user_type == "risky_user").astype(int)

# Linear risk model with noise injection
linear_score = (
    w["amount_z"] * (amount_zscore / (np.std(amount_zscore) + 1e-6)) +
    w["night"] * is_night +
    w["device_change"] * device_change_flag +
    w["loc_risk"] * (location_risk_level / 3) +
    w["loc_vel"] * location_velocity_flag +
    w["velocity"] * velocity_1h +
    w["failed"] * failed_attempts_24h +
    w["fraud_hist"] * fraud_history_flag +
    w["new_ben"] * is_new_beneficiary +
    w["user_risky"] * user_risky_flag +
    np.random.normal(0, 0.5, size=N)  # add noise = realism
)

# Convert to probability (target ~3% fraud)
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

prob = sigmoid(linear_score - 5.0)  # tuned for ~3%

fraud_label = (np.random.rand(N) < prob).astype(int)

# ================================================================
# BUILD FINAL DATAFRAME
# ================================================================

df = pd.DataFrame({
    "transaction_amount": np.round(amount, 2),
    "amount_zscore": np.round(amount_zscore, 3),
    "hour_of_day": hour_of_day,
    "is_night": is_night,
    "day_of_week": day_of_week,
    "device_change_flag": device_change_flag,
    "location_risk_level": location_risk_level,
    "location_velocity_flag": location_velocity_flag,
    "sender_bank_code": sender_bank_code,
    "receiver_bank_code": receiver_bank_code,
    "velocity_1h": velocity_1h,
    "failed_attempts_24h": failed_attempts_24h,
    "fraud_history_flag": fraud_history_flag,
    "is_new_beneficiary": is_new_beneficiary,
    "fraud_label": fraud_label
})

fraud_rate = df["fraud_label"].mean()

print(f"Synthetic Fraud Dataset Generated:")
print(f"- Rows: {len(df):,}")
print(f"- Fraud Rate: {fraud_rate:.3%}")

df.to_csv("synthetic_training_realistic.csv", index=False)
