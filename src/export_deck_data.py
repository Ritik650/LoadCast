"""Emit the compact numbers the pitch deck quotes, so every figure on a slide
traces back to a real run rather than a guess."""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

m = json.load(open(os.path.join(OUT, "metrics.json")))
print("METRICS")
for k, v in m.items():
    if k != "top_features":
        print("  %-22s %s" % (k, round(v, 4) if isinstance(v, float) else v))

print("\nTOP FEATURES")
for r in m["top_features"][:12]:
    print("  %-24s %.0f" % (r["feature"], r["gain"]))

sw = pd.read_csv(os.path.join(OUT, "threshold_sweep.csv"))
print("\nSWEEP (every 5th row)")
print(sw.iloc[::5].round(3).to_string(index=False))

# load trajectory around true onset - the pre-injury ramp, from train risk window
DA = pd.read_csv(os.path.join(ROOT, "Dataset", "dailyActivity_merged.csv"),
                 parse_dates=["ActivityDate"],
                 usecols=["Id", "ActivityDate", "VeryActiveMinutes",
                          "FairlyActiveMinutes", "LightlyActiveMinutes"])
L = pd.read_csv(os.path.join(ROOT, "Dataset", "train_labels.csv"))
DA["day"] = (DA.ActivityDate - pd.Timestamp("2026-01-05")).dt.days + 1
DA["load"] = (DA.VeryActiveMinutes * 3 + DA.FairlyActiveMinutes * 2
              + DA.LightlyActiveMinutes)
d = DA.merge(L, left_on="Id", right_on="athlete_id")
inj = d[d.injured_in_risk_window == 1].copy()
inj["rel"] = inj.day - (30 + inj.onset_day_offset)
ramp = inj[inj.rel.between(-14, 14)].groupby("rel").load.mean().round(1)
print("\nLOAD RAMP AROUND ONSET")
print(json.dumps({int(k): float(v) for k, v in ramp.items()}))

hist_on = np.histogram(L[L.injured_in_risk_window == 1].onset_day_offset,
                       bins=6, range=(0, 30))[0]
hist_rc = np.histogram(L[L.injured_in_risk_window == 1].recovery_duration,
                       bins=8, range=(4, 20))[0]
print("\nonset hist (5-day bins):", hist_on.tolist())
print("recovery hist (2-day bins from 4):", hist_rc.tolist())
