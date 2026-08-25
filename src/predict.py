"""
PLAYHACK / ML  -  Inference from the saved models.

    python src/predict.py --data "Dataset/Test data" --out output/submission.csv

Rebuilds the observation-window features for the given folder, loads
models/loadcast_models.pkl and writes a submission in the required format.
Use --threshold f1 to switch to the F1-optimal operating point.
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib

from build_features import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "Dataset", "Test data"),
                    help="folder holding the CSVs to score")
    ap.add_argument("--models", default=os.path.join(ROOT, "models",
                                                     "loadcast_models.pkl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "submission.csv"))
    ap.add_argument("--threshold", default="default", choices=["default", "f1"],
                    help="'default' = recall-first, 'f1' = F1-optimal")
    ap.add_argument("--features", default=None,
                    help="reuse a prebuilt feature CSV instead of rebuilding")
    a = ap.parse_args()

    if a.features and os.path.exists(a.features):
        F = pd.read_csv(a.features)
        print("loaded features from " + a.features)
    else:
        tmp = os.path.join(ROOT, "output", "_predict_features.csv")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        F = build(a.data, tmp)

    b = joblib.load(a.models)
    thr = b["threshold_f1"] if a.threshold == "f1" else b["threshold"]

    # restore the exact training-time categorical encoding
    for c in b["cat_cols"]:
        F[c] = pd.Categorical(F[c].astype(str), categories=b["categories"][c])

    X = F[b["features"]]
    Xn = F[b["num_features"]]
    w = b["w_lgb"]

    # average across the saved folds - identical to what produced the
    # cross-validated numbers, so the tuned threshold stays calibrated
    P, ON, RC = [], [], []
    for f in b["folds"]:
        P.append(f["clf"].predict_proba(X)[:, 1])
        ON.append(w * f["onset_lgb"].predict(X) + (1 - w) * f["onset_rf"].predict(Xn))
        RC.append(w * f["recovery_lgb"].predict(X) + (1 - w) * f["recovery_rf"].predict(Xn))
    p, onset, recov = np.mean(P, 0), np.mean(ON, 0), np.mean(RC, 0)

    sub = pd.DataFrame({
        "athlete_id": F.athlete_id.values,
        "injured_in_risk_window": (p >= thr).astype(int),
        "onset_day_offset": np.clip(np.rint(onset), 1, 30).astype(int),
        "recovery_duration": np.clip(np.rint(recov), 1, None).astype(int),
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sub.to_csv(a.out, index=False)
    print("wrote %s  rows=%d  threshold=%.2f  positives=%d"
          % (a.out, len(sub), thr, sub.injured_in_risk_window.sum()))


if __name__ == "__main__":
    main()
