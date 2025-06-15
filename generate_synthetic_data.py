import pandas as pd
import numpy as np
from datetime import timedelta
#from ace_tools import display_dataframe_to_user

# Create a synthetic one‑day, 5‑minute‑granularity data set for a single host/service
rng = pd.date_range("2025-05-14 00:00", periods=288*365*5, freq="5min")  # 24h * 12 = 288 rows, 365 days, 5 years
df = pd.DataFrame({
    "timestamp_utc": rng,
    "host_id": "app-01",
    "service": "payments-api"
})

# Baseline utilisation
np.random.seed(42)
df["cpu_pct"] = np.clip(np.random.normal(loc=48, scale=10, size=len(df)), 5, 98)
df["mem_pct"] = np.clip(np.random.normal(loc=55, scale=8, size=len(df)), 10, 98)
df["disk_used_pct"] = np.linspace(70, 92, len(df)) + np.random.normal(0, 1, len(df))  # slow growth
df["error_rate"] = np.round(np.random.exponential(scale=0.02, size=len(df)), 3)

# Inject a 2‑hour sustained CPU+Memory spike (rows 100‑143 = 44 rows ≈ 220 min)
spike_idx = df.index[100:144]
df.loc[spike_idx, ["cpu_pct", "mem_pct"]] = np.random.uniform(90, 97, size=(len(spike_idx), 2))

# Inject a change deployment 30 min before the spike window
change_flag = (df["timestamp_utc"] >= df.loc[spike_idx[0], "timestamp_utc"] - timedelta(minutes=30)) & \
              (df["timestamp_utc"] < df.loc[spike_idx[0], "timestamp_utc"])
df["change_deployed_prev_1h"] = change_flag.astype(int)

# Calendar / holiday flag (assume not a holiday)
df["is_holiday"] = 0

# Labels:
#  - incident_open_in_next_2h: 1 if within 2 h before an outage window we hand‑label
#  - p1_or_p2_next_incident: "P1" if spike, else "None"
df["incident_open_in_next_2h"] = 0
df.loc[spike_idx[0]-4:spike_idx[0], "incident_open_in_next_2h"] = 1  # 20 min ahead
df["p1_or_p2_next_incident"] = "None"
df.loc[spike_idx[0], "p1_or_p2_next_incident"] = "P1"

# Reorder columns for readability
df = df[[
    "timestamp_utc", "host_id", "service", "cpu_pct", "mem_pct", "disk_used_pct",
    "error_rate", "is_holiday", "change_deployed_prev_1h",
    "incident_open_in_next_2h", "p1_or_p2_next_incident"
]]

# Save to CSV for user download
file_path = "/Users/saidulislam/MyExpm/Machine_Learning/app-issue-prediction/sample_operational_metrics.csv"
df.to_csv(file_path, index=False)

# Show the first 30 rows for quick inspection
#display_dataframe_to_user("Sample Operational Metrics (first 30 rows)", df.head(30))


