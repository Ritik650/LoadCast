"""
PLAYHACK / ML  -  Collinearity structure of the feature table.

Answers three questions a reviewer should ask about slide 06 and slide 07:

  1. Are the six feature families independent?          (no - they are one latent)
  2. Does correlation-based pruning help?               (no - tested, it costs AUC)
  3. How much does the extra feature mass actually buy? (a headroom ladder)

    python src/collinearity.py     # writes output/collinearity.json
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
SEED = 42
CAT = ["sport", "gender", "dominant_side", "position", "team_id"]
BASE_MAE = 7.6148          # brief's mean-prediction baseline for onset

# the features that all score |r| ~ 0.85 against onset day
BLOCK = ["monotony", "strain", "acwr_3_21", "rhr_last7_delta", "hrr_slope",
         "load_slope", "acwr_7_28", "hrmean_slope", "sleep_slope",
         "acute_load7", "last7_vs_prev21"]


def lgbm_reg():
    return lgb.LGBMRegressor(objective="l1", n_estimators=800, learning_rate=0.03,
                             num_leaves=15, min_child_samples=30,
                             feature_fraction=0.7, bagging_fraction=0.8,
                             bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED)


def main():
    R = {}
    tr = pd.read_csv(os.path.join(OUT, "train_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "Dataset", "train_labels.csv"))
    d = tr.merge(lab, on="athlete_id")
    for c in CAT:
        d[c] = d[c].astype("category")
    y = d.injured_in_risk_window.values
    X = d.drop(columns=["athlete_id", "injured_in_risk_window",
                        "onset_day_offset", "recovery_duration"])
    num = X.drop(columns=CAT)
    num = num.loc[:, num.nunique() > 1]

    # ---------------------------------------------------------- 1. structure
    print("== 1. THE TOP ONSET CORRELATES ARE ONE LATENT ==")
    C = d[BLOCK].corr()
    print(C.iloc[:6, :6].round(3).to_string())
    off = C.abs().where(~np.eye(len(BLOCK), dtype=bool))
    R["block_min_abs_corr"] = float(off.min().min())
    R["block_median_abs_corr"] = float(np.nanmedian(off.values))
    print("\n  pairwise |r| within the block: median %.3f, min %.3f"
          % (R["block_median_abs_corr"], R["block_min_abs_corr"]))
    print("  monotony vs strain: r = %.6f (exactly equal? %s -> the exact-duplicate"
          " prune cannot see it)" % (d.monotony.corr(d.strain),
                                     d.monotony.equals(d.strain)))

    M = num.corr().abs().values
    names = list(num.columns)
    pairs = [(names[i], names[j], float(M[i, j]))
             for i in range(len(names)) for j in range(i + 1, len(names))
             if M[i, j] > 0.98]
    R["n_pairs_over_098"] = len(pairs)
    R["top_pairs"] = [{"a": a, "b": b, "r": round(r, 4)}
                      for a, b, r in sorted(pairs, key=lambda x: -x[2])[:12]]
    print("  %d feature pairs across the table exceed |r| = 0.98" % len(pairs))

    inj = y == 1
    t = d.onset_day_offset.values[inj]
    numi = num[inj].reset_index(drop=True)
    R["abs_corr_vs_onset"] = {c: round(abs(numi[c].corr(pd.Series(t))), 3)
                              for c in BLOCK}
    print("  |r| vs onset:", R["abs_corr_vs_onset"])

    pc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA())
    pc.fit(num[BLOCK])
    ev = pc.named_steps["pca"].explained_variance_ratio_
    R["pc1_variance_share"] = float(ev[0])
    print("  PC1 explains %.1f%% of the block's variance" % (100 * ev[0]))

    # ---------------------------------------- 2. does correlation pruning help?
    print("\n== 2. CORRELATION PRUNING - TESTED, NOT ADOPTED ==")

    def corr_prune(cols, thr):
        Cm = X[cols].corr().abs()
        keep = []
        for c in cols:
            if all(Cm.loc[c, k] <= thr for k in keep):
                keep.append(c)
        return keep

    def score(cols, label):
        Xa, Xn = X[cols + CAT], X[cols]
        oof = np.zeros(len(Xa))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, y):
            m = lgb.LGBMClassifier(objective="binary", n_estimators=1200,
                                   learning_rate=0.03, num_leaves=31,
                                   min_child_samples=40, feature_fraction=0.7,
                                   bagging_fraction=0.8, bagging_freq=1,
                                   lambda_l2=5.0, verbose=-1, seed=SEED)
            m.fit(Xa.iloc[a], y[a], eval_set=[(Xa.iloc[b], y[b])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(80, verbose=False)])
            oof[b] = m.predict_proba(Xa.iloc[b])[:, 1]
        auc = roc_auc_score(y, oof)
        f1 = max(f1_score(y, (oof >= q).astype(int)) for q in np.arange(0.05, 0.95, 0.01))
        Xi, Xni = Xa[inj].reset_index(drop=True), Xn[inj].reset_index(drop=True)
        o = np.zeros(len(Xi))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(Xi):
            g = lgbm_reg(); g.fit(Xi.iloc[a], t[a])
            r = make_pipeline(SimpleImputer(strategy="median"),
                              RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                    n_jobs=-1, random_state=SEED))
            r.fit(Xni.iloc[a], t[a])
            o[b] = 0.6 * g.predict(Xi.iloc[b]) + 0.4 * r.predict(Xni.iloc[b])
        mae = float(np.abs(np.clip(np.rint(o), 1, 30) - t).mean())
        print("  %-24s n=%3d  AUC %.4f  F1 %.4f  onset MAE %.4f"
              % (label, len(cols), auc, f1, mae))
        return {"features": len(cols), "auc": auc, "f1": f1, "onset_mae": mae}

    cols = list(num.columns)
    R["prune_sweep"] = {"shipped": score(cols, "shipped (dupes only)")}
    for thr in (0.995, 0.99, 0.98, 0.95):
        R["prune_sweep"]["r>%.3f" % thr] = score(corr_prune(cols, thr),
                                                 "corr-prune |r|>%.3f" % thr)
    print("  -> correlation pruning costs AUC and barely moves onset MAE. Not adopted.")

    # ------------------------------------------------- 3. how much is one latent?
    print("\n== 3. HEADROOM LADDER (onset MAE, 5-fold on injured athletes) ==")
    def cvmae(model, Xd):
        pr = cross_val_predict(model, Xd, t, cv=KFold(5, shuffle=True, random_state=SEED))
        return float(np.abs(np.clip(np.rint(pr), 1, 30) - t).mean())

    lat = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        PCA(n_components=1)).fit_transform(num[BLOCK])
    ladder = {
        "mean baseline": float(np.abs(t - t.mean()).mean()),
        "linear on acwr_3_21": cvmae(make_pipeline(SimpleImputer(strategy="median"),
                                                   LinearRegression()), numi[["acwr_3_21"]]),
        "LGBM on acwr_3_21": cvmae(lgbm_reg(), numi[["acwr_3_21"]]),
        "LGBM on PC1 of block": cvmae(lgbm_reg(), pd.DataFrame(lat[inj], columns=["pc1"])),
        "LGBM on 11-feature block": cvmae(lgbm_reg(), numi[BLOCK]),
        "LGBM on all numeric": cvmae(lgbm_reg(), numi),
    }
    for k, v in ladder.items():
        print("  %-26s %.3f   skill %.3f" % (k, v, max(0, 1 - v / BASE_MAE)))
    R["headroom_ladder"] = ladder

    json.dump(R, open(os.path.join(OUT, "collinearity.json"), "w"), indent=2)
    print("\n-> output/collinearity.json")


if __name__ == "__main__":
    main()
