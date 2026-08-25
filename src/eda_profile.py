"""
PLAYHACK / ML  -  EDA and data-quality profile.

Regenerates every number quoted on deck slides 03 (EDA) and 05 (preprocessing),
so each figure on a slide traces back to a command a judge can rerun.

    python src/eda_profile.py            # prints the report, writes output/eda_profile.json
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "Dataset")
T = os.path.join(D, "Test data")
OUT = os.path.join(ROOT, "output")
OBS_START, OBS_END = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-03")


def p(*a):
    return os.path.join(*a)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {}

    # ---------------------------------------------------------------- scale
    meta = pd.read_csv(p(D, "athlete_metadata.csv"))
    meta_t = pd.read_csv(p(T, "athlete_metadata.csv"))
    lab = pd.read_csv(p(D, "train_labels.csv"))
    hourly = ["hourlyHeartrate", "hourlySteps", "hourlyIntensities", "hourlyCalories"]
    n_hourly = sum(sum(1 for _ in open(p(d, f + "_merged.csv"))) - 1
                   for d in (D, T) for f in hourly)
    R["n_train_athletes"] = int(len(meta))
    R["n_test_athletes"] = int(len(meta_t))
    R["n_hourly_rows"] = int(n_hourly)
    print("== SCALE ==")
    print("  train athletes %d | test athletes %d | hourly rows %s"
          % (len(meta), len(meta_t), f"{n_hourly:,}"))

    # ---------------------------------------------------------------- targets
    inj = lab[lab.injured_in_risk_window == 1]
    R["injury_rate"] = float(lab.injured_in_risk_window.mean())
    R["n_injured"] = int(len(inj))
    R["onset_mean"] = float(inj.onset_day_offset.mean())
    R["recovery_mean"] = float(inj.recovery_duration.mean())
    R["recovery_std"] = float(inj.recovery_duration.std())
    R["onset_hist_5day"] = np.histogram(inj.onset_day_offset, bins=6,
                                        range=(0, 30))[0].tolist()
    R["recovery_hist_2day"] = np.histogram(inj.recovery_duration, bins=8,
                                           range=(4, 20))[0].tolist()
    print("\n== TARGETS ==")
    print("  injury rate %.3f (%d of %d)" % (R["injury_rate"], len(inj), len(lab)))
    print("  onset  mean %.2f  hist(5-day bins) %s" % (R["onset_mean"], R["onset_hist_5day"]))
    print("  recovery mean %.2f sd %.2f  hist(2-day bins from 4) %s"
          % (R["recovery_mean"], R["recovery_std"], R["recovery_hist_2day"]))

    # --------------------------------------------- static factors carry little
    d = meta.merge(lab, on="athlete_id")
    by_sport = d.groupby("sport").injured_in_risk_window.mean().round(3)
    by_prior = d.groupby("prior_season_injury_count").injured_in_risk_window \
                .agg(["mean", "size"]).round(3)
    R["injury_rate_by_sport"] = by_sport.to_dict()
    R["injury_rate_by_prior_injury"] = {
        int(k): {"rate": float(v["mean"]), "n": int(v["size"])}
        for k, v in by_prior.iterrows()}
    print("\n== STATIC RISK FACTORS ==")
    print("  by sport:", by_sport.to_dict())
    print("  by prior injuries (rate, n):",
          {k: (v["mean"], int(v["size"])) for k, v in by_prior.iterrows()})

    # ---------------------------------------------------------------- coverage
    print("\n== COVERAGE (train) ==")
    cov = {}
    for f, c in [("dailyActivity_merged.csv", "Id"), ("sleepDay_merged.csv", "Id"),
                 ("weightLogInfo_merged.csv", "Id"),
                 ("training_sessions.csv", "athlete_id"),
                 ("hourlyHeartrate_merged.csv", "Id")]:
        n = pd.read_csv(p(D, f), usecols=[c])[c].nunique()
        cov[f] = round(100 * n / len(meta), 1)
        print("  %-32s %5.1f%% of athletes" % (f, cov[f]))
    R["coverage_pct"] = cov

    # ---------------------------------------------------------- quality audit
    da = pd.read_csv(p(D, "dailyActivity_merged.csv"))
    sl = pd.read_csv(p(D, "sleepDay_merged.csv"))
    w = pd.read_csv(p(D, "weightLogInfo_merged.csv"))
    R["sedentary_unique_values"] = int(da.SedentaryMinutes.nunique())
    R["sedentary_value"] = int(da.SedentaryMinutes.iloc[0])
    R["tracker_equals_total_pct"] = float(100 * (da.TrackerDistance == da.TotalDistance).mean())
    R["sleep_records_unique"] = sorted(int(v) for v in sl.TotalSleepRecords.unique())
    R["daily_nulls"] = int(da.isna().sum().sum())
    R["daily_dupes"] = int(da.duplicated(["Id", "ActivityDate"]).sum())
    R["sleep_nulls"] = int(sl.isna().sum().sum())
    R["weight_fat_null_pct"] = float(100 * w.Fat.isna().mean())
    R["weight_logs_median"] = float(w.groupby("Id").size().median())
    R["daily_rows"] = int(len(da))
    print("\n== QUALITY AUDIT ==")
    print("  SedentaryMinutes: %d unique value(s) = %d  across %s rows"
          % (R["sedentary_unique_values"], R["sedentary_value"], f"{len(da):,}"))
    print("  TrackerDistance == TotalDistance in %.1f%% of rows" % R["tracker_equals_total_pct"])
    print("  TotalSleepRecords unique values: %s" % R["sleep_records_unique"])
    print("  dailyActivity nulls %d, duplicate athlete-days %d; sleep nulls %d"
          % (R["daily_nulls"], R["daily_dupes"], R["sleep_nulls"]))
    print("  weightLogInfo: Fat %.1f%% null, median %d logs/athlete"
          % (R["weight_fat_null_pct"], R["weight_logs_median"]))

    # ------------------------------------------------------------- the ramp
    da["day"] = (pd.to_datetime(da.ActivityDate) - OBS_START).dt.days + 1
    da["load"] = (da.VeryActiveMinutes * 3 + da.FairlyActiveMinutes * 2
                  + da.LightlyActiveMinutes)
    m = da.merge(lab, left_on="Id", right_on="athlete_id")
    i = m[m.injured_in_risk_window == 1].copy()
    i["rel"] = i.day - (30 + i.onset_day_offset)
    ramp = i[i.rel.between(-14, 14)].groupby("rel").load.mean().round(1)
    R["load_ramp_around_onset"] = {int(k): float(v) for k, v in ramp.items()}
    print("\n== LOAD RAMP AROUND ONSET ==")
    print("  day -14 %.0f  ->  day -1 %.0f  ->  onset %.0f"
          % (ramp[-14], ramp[-1], ramp[0]))

    # ACWR inside the observation window, injured vs healthy
    obs = m[m.day <= 30].sort_values("day")
    acwr = obs.groupby(["athlete_id", "injured_in_risk_window"]).load.apply(
        lambda s: s.values[-7:].mean() / s.values[-28:].mean()).reset_index()
    R["acwr_injured"] = float(acwr[acwr.injured_in_risk_window == 1].load.mean())
    R["acwr_healthy"] = float(acwr[acwr.injured_in_risk_window == 0].load.mean())
    print("  observation-window ACWR: injured %.3f vs healthy %.3f"
          % (R["acwr_injured"], R["acwr_healthy"]))

    # ------------------------------------- correlation ceiling of each target
    fp = p(OUT, "train_features.csv")
    if os.path.exists(fp):
        F = pd.read_csv(fp).merge(lab, on="athlete_id")
        F = F[F.injured_in_risk_window == 1].select_dtypes("number") \
             .drop(columns=["athlete_id", "injured_in_risk_window"])
        for tgt in ("onset_day_offset", "recovery_duration"):
            c = F.corr()[tgt].drop(["onset_day_offset", "recovery_duration"]).abs()
            R["max_abs_corr_" + tgt] = float(c.max())
            R["top_corr_" + tgt] = c.sort_values(ascending=False).head(5).round(3).to_dict()
            print("\n  max |r| vs %-18s %.3f   top: %s"
                  % (tgt, c.max(), list(c.sort_values(ascending=False).head(3).index)))
    else:
        print("\n  (run build_features.py first for the correlation ceilings)")

    json.dump(R, open(p(OUT, "eda_profile.json"), "w"), indent=2, default=str)
    print("\n-> output/eda_profile.json")


if __name__ == "__main__":
    main()
