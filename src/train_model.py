"""
PLAYHACK / ML  -  Three-head model, competition-faithful validation, artefacts.

Head A : LightGBM classifier                 -> injured_in_risk_window   (F1)
Head B : LightGBM(L1) + RandomForest blend   -> onset_day_offset         (skill)
Head C : LightGBM(L1) + RandomForest blend   -> recovery_duration        (skill)

Scoring replicates the brief exactly: timing is graded on every TRULY injured
athlete, and a missed injury (pred 0, true 1) incurs PENALTY = n_risk = 30 on
BOTH timing errors. That asymmetry is why the operating threshold is tuned on
the composite score rather than on F1 alone.

Outputs
  output/submission.csv                 recall-first (default)
  output/submission_f1_balanced.csv     F1-optimal alternative
  output/metrics.json  threshold_sweep.csv  feature_importance.csv
  models/*.pkl                          saved fold ensemble for inference
"""
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
MODELS = os.path.join(ROOT, "models")
PENALTY = 30.0
SEED = 42
N_FOLDS = 5
W_LGB = 0.6                     # blend weight, chosen on the OOF sweep

CAT_COLS = ["sport", "gender", "dominant_side", "position", "team_id"]
DROP = ["athlete_id", "injured_in_risk_window", "onset_day_offset",
        "recovery_duration"]

CLF_P = dict(objective="binary", learning_rate=0.03, num_leaves=31,
             min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=5.0, n_estimators=1200, verbose=-1,
             seed=SEED)
REG_P = dict(objective="l1", learning_rate=0.03, num_leaves=15,
             min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=5.0, n_estimators=800, verbose=-1,
             seed=SEED)


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=400,
                                               min_samples_leaf=3, n_jobs=-1,
                                               random_state=SEED))


def prune(train_X):
    """Drop zero-variance and exactly duplicated columns found on the train set."""
    num = train_X.drop(columns=CAT_COLS)
    const = [c for c in num.columns if num[c].nunique(dropna=False) <= 1]
    rest, dup = [c for c in num.columns if c not in const], []
    for i, a in enumerate(rest):
        for b in rest[i + 1:]:
            if b not in dup and num[a].equals(num[b]):
                dup.append(b)
    return const, dup


def prep(train, test, do_prune=True):
    """Align columns, encode categoricals identically, drop dead features."""
    for c in CAT_COLS:
        cats = pd.Categorical(pd.concat([train[c], test[c]]).astype(str)).categories
        train[c] = pd.Categorical(train[c].astype(str), categories=cats)
        test[c] = pd.Categorical(test[c].astype(str), categories=cats)
    feats = [c for c in train.columns if c not in DROP]
    if do_prune:
        const, dup = prune(train[feats])
        feats = [c for c in feats if c not in const + dup]
        print("pruned %d dead columns (%d constant, %d duplicate)"
              % (len(const) + len(dup), len(const), len(dup)))
        print("  constant:", const)
        print("  duplicate:", dup)
    else:
        print("PRUNING DISABLED (ablation run)")
    return train[feats], test[feats], feats


def skill(mae_model, mae_base):
    return max(0.0, 1.0 - mae_model / mae_base)


def timing_mae(pred_flag, pred_val, true_val, true_flag):
    """MAE over truly-injured athletes; misses take the fixed 30-day penalty."""
    m = true_flag == 1
    err = np.where(pred_flag[m] == 1, np.abs(pred_val[m] - true_val[m]), PENALTY)
    return float(err.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-prune", action="store_true",
                    help="ablation: keep zero-variance and duplicate columns")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, so a run cannot overwrite another")
    ap.add_argument("--no-save", action="store_true",
                    help="skip writing models/ and submissions (ablation runs)")
    args = ap.parse_args()
    tag = args.tag
    os.makedirs(MODELS, exist_ok=True)
    tr = pd.read_csv(os.path.join(OUT, "train_features.csv"))
    te = pd.read_csv(os.path.join(OUT, "test_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "Dataset", "train_labels.csv"))
    tr = tr.merge(lab, on="athlete_id", how="inner")

    y = tr.injured_in_risk_window.values
    onset = tr.onset_day_offset.values
    recov = tr.recovery_duration.values
    Xtr, Xte, feats = prep(tr.copy(), te.copy(), do_prune=not args.no_prune)
    Xtr_n, Xte_n = Xtr.drop(columns=CAT_COLS), Xte.drop(columns=CAT_COLS)
    print("features: %d | train %d | test %d | pos-rate %.3f"
          % (len(feats), len(Xtr), len(Xte), y.mean()))

    oof_p = np.zeros(len(Xtr))
    oof_on = np.zeros(len(Xtr))
    oof_rc = np.zeros(len(Xtr))
    te_p, te_on, te_rc = [], [], []
    imp = np.zeros(len(feats))
    fold_models = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (i_tr, i_va) in enumerate(skf.split(Xtr, y)):
        # ---- Head A: classification on all athletes
        m = lgb.LGBMClassifier(**CLF_P)
        m.fit(Xtr.iloc[i_tr], y[i_tr],
              eval_set=[(Xtr.iloc[i_va], y[i_va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof_p[i_va] = m.predict_proba(Xtr.iloc[i_va])[:, 1]
        te_p.append(m.predict_proba(Xte)[:, 1])
        imp += m.booster_.feature_importance("gain")
        fold = {"clf": m}

        # ---- Heads B & C: blend, fit on injured athletes only
        inj = i_tr[y[i_tr] == 1]
        for tgt, store, tstore in ((onset, oof_on, te_on), (recov, oof_rc, te_rc)):
            g = lgb.LGBMRegressor(**REG_P)
            g.fit(Xtr.iloc[inj], tgt[inj])
            r = make_rf()
            r.fit(Xtr_n.iloc[inj], tgt[inj])
            store[i_va] = (W_LGB * g.predict(Xtr.iloc[i_va])
                           + (1 - W_LGB) * r.predict(Xtr_n.iloc[i_va]))
            tstore.append(W_LGB * g.predict(Xte) + (1 - W_LGB) * r.predict(Xte_n))
            tname = "onset" if tgt is onset else "recovery"
            fold[tname + "_lgb"], fold[tname + "_rf"] = g, r
        fold_models.append(fold)
        print("  fold %d  auc=%.4f" % (k + 1, roc_auc_score(y[i_va], oof_p[i_va])))

    te_p = np.mean(te_p, axis=0)
    te_on = np.mean(te_on, axis=0)
    te_rc = np.mean(te_rc, axis=0)

    auc = roc_auc_score(y, oof_p)
    ap = average_precision_score(y, oof_p)

    # ---- baselines exactly as the brief defines them
    base_on, base_rc = float(np.mean(onset[y == 1])), float(np.mean(recov[y == 1]))
    mae_base_on = float(np.abs(onset[y == 1] - base_on).mean())
    mae_base_rc = float(np.abs(recov[y == 1] - base_rc).mean())

    # ---- threshold sweep on the composite objective
    rows = []
    for t in np.arange(0.00, 0.96, 0.01):
        flag = (oof_p >= t).astype(int)
        f1 = f1_score(y, flag)
        m_on = timing_mae(flag, np.clip(np.rint(oof_on), 1, 30), onset, y)
        m_rc = timing_mae(flag, np.clip(np.rint(oof_rc), 1, None), recov, y)
        s_on, s_rc = skill(m_on, mae_base_on), skill(m_rc, mae_base_rc)
        rows.append((t, f1, m_on, m_rc, s_on, s_rc, (f1 + s_on + s_rc) / 3.0,
                     float((flag[y == 1] == 1).mean())))
    sweep = pd.DataFrame(rows, columns=["thr", "f1", "mae_onset", "mae_recov",
                                        "skill_onset", "skill_recov",
                                        "composite", "recall"])
    sweep.to_csv(os.path.join(OUT, "threshold_sweep%s.csv" % tag), index=False)
    best_f1 = sweep.loc[sweep.f1.idxmax()]
    best_cp = sweep.loc[sweep.composite.idxmax()]
    zero = sweep[sweep.skill_onset <= 0]
    cliff = sweep.loc[zero.index[0]] if len(zero) else None

    hit = y == 1
    mae_on_hit = float(np.abs(np.clip(np.rint(oof_on[hit]), 1, 30) - onset[hit]).mean())
    mae_rc_hit = float(np.abs(np.clip(np.rint(oof_rc[hit]), 1, None) - recov[hit]).mean())

    print("\n=== OOF (5-fold, %d athletes) ===" % len(Xtr))
    print("AUC %.4f | PR-AUC %.4f" % (auc, ap))
    print("best-F1   thr=%.2f  F1=%.4f  recall=%.3f  composite=%.4f"
          % (best_f1.thr, best_f1.f1, best_f1.recall, best_f1.composite))
    print("best-COMP thr=%.2f  F1=%.4f  recall=%.3f  composite=%.4f"
          % (best_cp.thr, best_cp.f1, best_cp.recall, best_cp.composite))
    print("onset : MAE(hits) %.3f  vs baseline %.3f  -> skill %.3f"
          % (mae_on_hit, mae_base_on, skill(mae_on_hit, mae_base_on)))
    print("recov : MAE(hits) %.3f  vs baseline %.3f  -> skill %.3f"
          % (mae_rc_hit, mae_base_rc, skill(mae_rc_hit, mae_base_rc)))
    print("SCORED (penalty included) @thr=%.2f: onset skill %.3f | recovery skill %.3f"
          % (best_cp.thr, best_cp.skill_onset, best_cp.skill_recov))
    if cliff is not None:
        print("onset-skill cliff at thr=%.2f (recall %.3f, scored MAE %.3f vs baseline %.3f)"
              % (cliff.thr, cliff.recall, cliff.mae_onset, mae_base_on))

    fi = pd.DataFrame({"feature": feats, "gain": imp / N_FOLDS}) \
           .sort_values("gain", ascending=False)
    fi.to_csv(os.path.join(OUT, "feature_importance%s.csv" % tag), index=False)
    print("\ntop 12 features by gain:")
    print(fi.head(12).to_string(index=False))

    # ---- submissions at both defensible operating points
    def write(thr, name):
        s = pd.DataFrame({
            "athlete_id": te.athlete_id.values,
            "injured_in_risk_window": (te_p >= thr).astype(int),
            "onset_day_offset": np.clip(np.rint(te_on), 1, 30).astype(int),
            "recovery_duration": np.clip(np.rint(te_rc), 1, None).astype(int),
        })
        s.to_csv(os.path.join(OUT, name), index=False)
        print("%-28s thr=%.2f  positives=%d/%d (%.1f%%)"
              % (name, thr, s.injured_in_risk_window.sum(), len(s),
                 100 * s.injured_in_risk_window.mean()))
        return s

    thr = float(best_cp.thr)
    print()
    if not args.no_save:
        write(thr, "submission.csv")
        write(float(best_f1.thr), "submission_f1_balanced.csv")

    # ---- persist the fold ensemble itself.
    # The thresholds above were tuned on fold-averaged probabilities, so a
    # single full-data refit would be differently calibrated and would NOT
    # reproduce these submissions. Saving the folds keeps inference identical
    # to what was validated.
    bundle = {"folds": fold_models, "features": feats, "cat_cols": CAT_COLS,
              "num_features": [c for c in feats if c not in CAT_COLS],
              "threshold": thr, "threshold_f1": float(best_f1.thr),
              "w_lgb": W_LGB, "n_folds": N_FOLDS,
              "categories": {c: list(Xtr[c].cat.categories) for c in CAT_COLS}}
    if args.no_save:
        print("\n--no-save: skipping models/ and submissions")
    else:
        print("\nsaving fold ensemble to models/ ...")
        path = os.path.join(MODELS, "loadcast_models.pkl")
        joblib.dump(bundle, path, compress=3)
        print("  -> models/loadcast_models.pkl (%.1f MB)"
              % (os.path.getsize(path) / 1e6))

    json.dump({
        "n_features": len(feats), "auc": auc, "pr_auc": ap,
        "f1_at_best_f1": float(best_f1.f1), "thr_best_f1": float(best_f1.thr),
        "f1_at_composite": float(best_cp.f1), "thr_composite": thr,
        "recall_composite": float(best_cp.recall),
        "mae_onset_hits": mae_on_hit, "mae_recov_hits": mae_rc_hit,
        "mae_base_onset": mae_base_on, "mae_base_recov": mae_base_rc,
        "skill_onset_clean": skill(mae_on_hit, mae_base_on),
        "skill_recov_clean": skill(mae_rc_hit, mae_base_rc),
        "skill_onset_scored": float(best_cp.skill_onset),
        "skill_recov_scored": float(best_cp.skill_recov),
        "composite": float(best_cp.composite),
        "cliff_thr": (None if cliff is None else float(cliff.thr)),
        "cliff_recall": (None if cliff is None else float(cliff.recall)),
        "cliff_mae_onset": (None if cliff is None else float(cliff.mae_onset)),
        "pruning_enabled": (not args.no_prune),
        "pos_rate_train": float(y.mean()),
        "top_features": fi.head(20).to_dict("records"),
    }, open(os.path.join(OUT, "metrics%s.json" % tag), "w"), indent=2)


if __name__ == "__main__":
    main()
