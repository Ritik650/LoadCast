"""
PLAYHACK / ML  -  Model selection evidence.

Same folds, same features, same metric for every candidate, so the choice of
LightGBM is an argued result rather than a default. Classification is scored by
ROC-AUC and best-threshold F1; onset regression by MAE against the brief's
mean-prediction baseline.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
SEED, N_FOLDS = 42, 5
CAT = ["sport", "gender", "dominant_side", "position", "team_id"]


def load():
    tr = pd.read_csv(os.path.join(OUT, "train_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "Dataset", "train_labels.csv"))
    tr = tr.merge(lab, on="athlete_id")
    y = tr.injured_in_risk_window.values
    onset = tr.onset_day_offset.values
    drop = ["athlete_id", "injured_in_risk_window", "onset_day_offset",
            "recovery_duration"]
    X = tr.drop(columns=drop)
    Xn = X.drop(columns=CAT)                      # numeric-only for sklearn models
    Xc = X.copy()
    for c in CAT:
        Xc[c] = Xc[c].astype("category")
    return Xn, Xc, y, onset


def best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int))
               for t in np.arange(0.05, 0.95, 0.01))


def cv_clf(name, make, X, y, native_cat=False):
    oof = np.zeros(len(X))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for a, b in skf.split(X, y):
        m = make()
        m.fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    r = {"model": name, "auc": roc_auc_score(y, oof), "f1": best_f1(y, oof)}
    print("  %-22s AUC %.4f   best-F1 %.4f" % (name, r["auc"], r["f1"]))
    return r


def cv_reg(name, make, X, t):
    oof = np.zeros(len(X))
    for a, b in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        m = make()
        m.fit(X.iloc[a], t[a])
        oof[b] = m.predict(X.iloc[b])
    mae = float(np.abs(np.clip(np.rint(oof), 1, 30) - t).mean())
    base = float(np.abs(t - t.mean()).mean())
    r = {"model": name, "mae": mae, "skill": max(0.0, 1 - mae / base)}
    print("  %-22s MAE %.3f   skill %.3f" % (name, mae, r["skill"]))
    return r


def main():
    Xn, Xc, y, onset = load()
    res = {}

    print("TASK A - classification (5-fold OOF)")
    res["classification"] = [
        cv_clf("Logistic regression", lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5)), Xn, y),
        cv_clf("Decision tree", lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=6, random_state=SEED)), Xn, y),
        cv_clf("Random forest", lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   n_jobs=-1, random_state=SEED)), Xn, y),
        cv_clf("LightGBM (numeric)", lambda: lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED), Xn, y),
        cv_clf("LightGBM + categoricals", lambda: lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED), Xc, y),
    ]

    inj = y == 1
    Xni, Xci, ti = Xn[inj].reset_index(drop=True), Xc[inj].reset_index(drop=True), onset[inj]
    print("\nTASK B - onset regression, injured athletes only (5-fold OOF)")
    print("  %-22s MAE %.3f   skill %.3f"
          % ("Mean baseline", np.abs(ti - ti.mean()).mean(), 0.0))
    res["onset"] = [
        {"model": "Mean baseline",
         "mae": float(np.abs(ti - ti.mean()).mean()), "skill": 0.0},
        cv_reg("Ridge regression", lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            Ridge(alpha=10.0)), Xni, ti),
        cv_reg("Random forest", lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                  n_jobs=-1, random_state=SEED)), Xni, ti),
        cv_reg("LightGBM L2", lambda: lgb.LGBMRegressor(
            objective="l2", n_estimators=800, learning_rate=0.03, num_leaves=15,
            min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED), Xci, ti),
        cv_reg("LightGBM L1 (chosen)", lambda: lgb.LGBMRegressor(
            objective="l1", n_estimators=800, learning_rate=0.03, num_leaves=15,
            min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED), Xci, ti),
    ]

    json.dump(res, open(os.path.join(OUT, "model_benchmark.json"), "w"), indent=2)
    print("\n-> output/model_benchmark.json")


if __name__ == "__main__":
    main()
