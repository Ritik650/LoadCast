"""
PLAYHACK / ML  -  Observation-window feature builder.

Strict rule: every feature is computed from days 1-30 only (2026-01-05 .. 2026-02-03).
Train files carry 60 days; days 31-60 are DROPPED here so train and test see
identical information. No leakage from the risk window.
"""
import os
import numpy as np
import pandas as pd

OBS_START = pd.Timestamp("2026-01-05")
OBS_END = pd.Timestamp("2026-02-03")   # inclusive -> 30 days
HOURFMT = "%m/%d/%Y %I:%M:%S %p"


def _slope(y):
    """Least-squares slope of a daily series (units/day)."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return np.nan
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])


def _tail(y, k):
    y = np.asarray(y, dtype=float)
    return np.nanmean(y[-k:]) if len(y) else np.nan


def _ratio(a, b):
    return a / b if (b is not None and b == b and b > 1e-6) else np.nan


# --------------------------------------------------------------------- daily
def daily_features(path):
    df = pd.read_csv(path, parse_dates=["ActivityDate"])
    df = df[(df.ActivityDate >= OBS_START) & (df.ActivityDate <= OBS_END)]
    df = df.sort_values(["Id", "ActivityDate"])

    # Foster-style internal load proxy from activity-minute bands
    df["load"] = (df.VeryActiveMinutes * 3.0 + df.FairlyActiveMinutes * 2.0
                  + df.LightlyActiveMinutes * 1.0)
    df["active_min"] = (df.VeryActiveMinutes + df.FairlyActiveMinutes
                        + df.LightlyActiveMinutes)
    df["hi_frac"] = df.VeryActiveMinutes / df.active_min.replace(0, np.nan)

    rows = []
    for aid, g in df.groupby("Id", sort=True):
        load = g.load.values
        r = {"athlete_id": aid}

        for col in ["TotalSteps", "TotalDistance", "VeryActiveMinutes",
                    "FairlyActiveMinutes", "LightlyActiveMinutes",
                    "SedentaryMinutes", "Calories", "load"]:
            v = g[col].values.astype(float)
            r[col + "_mean"] = np.nanmean(v)
            r[col + "_std"] = np.nanstd(v)
            r[col + "_max"] = np.nanmax(v)
        r["hi_frac_mean"] = np.nanmean(g.hi_frac.values)

        # acute:chronic workload ratio - the core overuse signal
        acute7 = _tail(load, 7)
        chronic28 = np.nanmean(load[-28:])
        r["acwr_7_28"] = _ratio(acute7, chronic28)
        r["acwr_3_21"] = _ratio(_tail(load, 3), np.nanmean(load[-21:]))
        r["acute_load7"] = acute7
        r["chronic_load28"] = chronic28

        # monotony & strain (Foster)
        sd = np.nanstd(load)
        r["monotony"] = _ratio(np.nanmean(load), sd)
        r["strain"] = (np.nansum(load) * r["monotony"]
                       if r["monotony"] == r["monotony"] else np.nan)

        # ramp / trend
        r["load_slope"] = _slope(load)
        r["steps_slope"] = _slope(g.TotalSteps.values.astype(float))
        r["sed_slope"] = _slope(g.SedentaryMinutes.values)
        w = [np.nanmean(load[i * 7:(i + 1) * 7]) for i in range(4)]
        r["load_w4"], r["load_w1"] = w[3], w[0]
        r["load_w4_over_w1"] = _ratio(w[3], w[0])
        r["load_wow_max_jump"] = float(np.nanmax(np.diff(w)))

        # spikes vs personal baseline
        base = np.nanmedian(load)
        r["spike_days"] = int(np.nansum(load > 1.5 * base)) if base > 0 else 0
        r["rest_days"] = int(np.nansum(load < 0.5 * base)) if base > 0 else 0
        r["max_day_z"] = _ratio(np.nanmax(load) - np.nanmean(load), sd)
        r["last7_vs_prev21"] = _ratio(acute7, np.nanmean(load[:-7]))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- sleep
def sleep_features(path):
    df = pd.read_csv(path, parse_dates=["SleepDay"])
    df = df[(df.SleepDay >= OBS_START) & (df.SleepDay <= OBS_END)]
    df = df.sort_values(["Id", "SleepDay"])
    df["eff"] = df.TotalMinutesAsleep / df.TotalTimeInBed.replace(0, np.nan)
    df["debt"] = (480.0 - df.TotalMinutesAsleep).clip(lower=0)   # vs 8h reference

    g = df.groupby("Id")
    out = pd.DataFrame({
        "sleep_mean": g.TotalMinutesAsleep.mean(),
        "sleep_std": g.TotalMinutesAsleep.std(),
        "sleep_min": g.TotalMinutesAsleep.min(),
        "sleep_eff": g.eff.mean(),
        "sleep_debt": g.debt.sum(),
        "short_nights": g.TotalMinutesAsleep.apply(lambda s: int((s < 360).sum())),
        "sleep_slope": g.TotalMinutesAsleep.apply(_slope),
        "sleep_last7": g.TotalMinutesAsleep.apply(lambda s: _tail(s.values, 7)),
    })
    out["sleep_last7_delta"] = out.sleep_last7 - out.sleep_mean
    return out.reset_index().rename(columns={"Id": "athlete_id"})


# ---------------------------------------------------------------- heart rate
def hr_features(path):
    df = pd.read_csv(path, parse_dates=["ActivityHour"])
    df = df[(df.ActivityHour >= OBS_START)
            & (df.ActivityHour < OBS_END + pd.Timedelta(days=1))]
    df["date"] = df.ActivityHour.dt.normalize()
    df["hour"] = df.ActivityHour.dt.hour

    # nocturnal window = resting-HR proxy; rising RHR is a fatigue marker
    night = df[df.hour.between(1, 5)].groupby(["Id", "date"]).AvgHeartRate.mean()
    night = night.rename("rhr").reset_index().sort_values(["Id", "date"])
    dayg = df.groupby(["Id", "date"]).agg(hr_mean=("AvgHeartRate", "mean"),
                                          hr_max=("MaxHeartRate", "max"),
                                          hr_min=("MinHeartRate", "min")).reset_index()
    dayg["hrr"] = dayg.hr_max - dayg.hr_min

    g1 = night.groupby("Id").rhr
    out = pd.DataFrame({
        "rhr_mean": g1.mean(),
        "rhr_std": g1.std(),
        "rhr_slope": g1.apply(_slope),
        "rhr_last7": g1.apply(lambda s: _tail(s.values, 7)),
    })
    out["rhr_last7_delta"] = out.rhr_last7 - out.rhr_mean

    g2 = dayg.groupby("Id")
    out = out.join(pd.DataFrame({
        "hrmean_mean": g2.hr_mean.mean(),
        "hrmean_slope": g2.hr_mean.apply(_slope),
        "hrmax_mean": g2.hr_max.mean(),
        "hrmax_max": g2.hr_max.max(),
        "hrr_mean": g2.hrr.mean(),
        "hrr_slope": g2.hrr.apply(_slope),
    }), how="outer")
    return out.reset_index().rename(columns={"Id": "athlete_id"})


# ------------------------------------------------------ hourly intensity/steps
def hourly_features(int_path, step_path):
    it = pd.read_csv(int_path)
    it["ActivityHour"] = pd.to_datetime(it.ActivityHour, format=HOURFMT)
    it = it[(it.ActivityHour >= OBS_START)
            & (it.ActivityHour < OBS_END + pd.Timedelta(days=1))]
    it["date"] = it.ActivityHour.dt.normalize()
    it["hour"] = it.ActivityHour.dt.hour

    di = it.groupby(["Id", "date"]).TotalIntensity.agg(["sum", "max"]).reset_index()
    gi = di.groupby("Id")
    o1 = pd.DataFrame({
        "int_day_mean": gi["sum"].mean(),
        "int_day_slope": gi["sum"].apply(_slope),
        "int_peak_mean": gi["max"].mean(),
        "int_peak_max": gi["max"].max(),
    })

    # circadian: share of intensity accrued late at night
    late = it.assign(late=it.hour.isin([22, 23, 0, 1]) * it.TotalIntensity) \
             .groupby("Id").agg(late=("late", "sum"), tot=("TotalIntensity", "sum"))
    o1["late_intensity_share"] = late.late / late.tot.replace(0, np.nan)

    st = pd.read_csv(step_path)
    st["ActivityHour"] = pd.to_datetime(st.ActivityHour, format=HOURFMT)
    st = st[(st.ActivityHour >= OBS_START)
            & (st.ActivityHour < OBS_END + pd.Timedelta(days=1))]
    gs = st.groupby("Id").StepTotal
    o1["hour_steps_max"] = gs.max()
    o1["hour_steps_p95"] = gs.quantile(0.95)
    o1["active_hours"] = st.assign(a=st.StepTotal > 250).groupby("Id").a.sum() / 30.0
    return o1.reset_index().rename(columns={"Id": "athlete_id"})


# ------------------------------------------------------------------ sessions
def session_features(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df.date >= OBS_START) & (df.date <= OBS_END)]
    df["dur"] = (df.end_hour - df.start_hour).clip(lower=0)

    piv = df.pivot_table(index="athlete_id", columns="sport_session_type",
                         values="session_id", aggfunc="count").fillna(0)
    piv.columns = ["n_" + str(c) for c in piv.columns]

    g = df.groupby("athlete_id")
    out = pd.DataFrame({
        "n_sessions": g.session_id.count(),
        "hours_total": g.dur.sum(),
        "hours_mean": g.dur.mean(),
        "hours_max": g.dur.max(),
        "start_hour_mean": g.start_hour.mean(),
        "start_hour_std": g.start_hour.std(),
        "active_days": g.date.nunique(),
    }).join(piv, how="left")
    for c in piv.columns:
        out[c] = out[c].fillna(0)

    out["sessions_per_active_day"] = out.n_sessions / out.active_days.replace(0, np.nan)
    out["rest_days_total"] = 30 - out.active_days
    if "n_scrimmage" in out:
        out["scrimmage_ratio"] = out.n_scrimmage / out.n_sessions.replace(0, np.nan)
    if "n_gym" in out:
        out["gym_ratio"] = out.n_gym / out.n_sessions.replace(0, np.nan)

    # longest consecutive training streak without a rest day
    def streak(dates):
        d = np.sort(pd.unique(dates))
        best = cur = 1
        for i in range(1, len(d)):
            cur = cur + 1 if (d[i] - d[i - 1]) == np.timedelta64(1, "D") else 1
            best = max(best, cur)
        return best
    out["max_streak"] = g.date.apply(lambda s: streak(s.values))

    df["wk"] = ((df.date - OBS_START).dt.days // 7).clip(0, 3)
    wk = df.pivot_table(index="athlete_id", columns="wk", values="dur",
                        aggfunc="sum").fillna(0)
    if 3 in wk and 0 in wk:
        out["sess_hours_w4_over_w1"] = wk[3] / wk[0].replace(0, np.nan)
    return out.reset_index()


# ------------------------------------------------------------ weight + meta
def weight_features(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[(df.Date >= OBS_START) & (df.Date <= OBS_END)].sort_values(["Id", "Date"])
    g = df.groupby("Id")
    out = pd.DataFrame({
        "wt_mean": g.WeightKg.mean(),
        "wt_bmi": g.BMI.mean(),
        "wt_n_logs": g.WeightKg.count(),
        "wt_change": g.WeightKg.apply(
            lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else np.nan),
    })
    return out.reset_index().rename(columns={"Id": "athlete_id"})


def meta_features(path):
    m = pd.read_csv(path)
    m["bmi_base"] = m.weight_kg_baseline / (m.height_cm / 100.0) ** 2
    m["exp_per_age"] = m.years_playing / m.age.replace(0, np.nan)
    m["prior_rate"] = m.prior_season_injury_count / m.years_playing.replace(0, np.nan)
    return m


# -------------------------------------------------------------------- driver
def build(data_dir, out_csv):
    def p(f):
        return os.path.join(data_dir, f)

    print("[" + out_csv + "] daily...", flush=True)
    F = daily_features(p("dailyActivity_merged.csv"))
    print("  sleep...", flush=True)
    F = F.merge(sleep_features(p("sleepDay_merged.csv")), on="athlete_id", how="left")
    print("  heartrate...", flush=True)
    F = F.merge(hr_features(p("hourlyHeartrate_merged.csv")), on="athlete_id", how="left")
    print("  hourly...", flush=True)
    F = F.merge(hourly_features(p("hourlyIntensities_merged.csv"),
                                p("hourlySteps_merged.csv")), on="athlete_id", how="left")
    print("  sessions...", flush=True)
    F = F.merge(session_features(p("training_sessions.csv")), on="athlete_id", how="left")
    print("  weight...", flush=True)
    F = F.merge(weight_features(p("weightLogInfo_merged.csv")), on="athlete_id", how="left")
    print("  metadata...", flush=True)
    F = F.merge(meta_features(p("athlete_metadata.csv")), on="athlete_id", how="left")

    # interactions grounded in load-vs-recovery balance
    F["load_per_sleep"] = F.load_mean / F.sleep_mean.replace(0, np.nan)
    F["acwr_x_priorinj"] = F.acwr_7_28 * (1 + F.prior_season_injury_count)
    F["strain_per_sleep"] = F.strain / F.sleep_mean.replace(0, np.nan)
    F["rhr_rise_x_load"] = F.rhr_slope * F.load_mean
    F["load_per_kg"] = F.load_mean / F.weight_kg_baseline.replace(0, np.nan)

    F.to_csv(out_csv, index=False)
    print("  -> " + out_csv + "  shape=" + str(F.shape), flush=True)
    return F


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(root, "Dataset"),
          os.path.join(root, "output", "train_features.csv"))
    build(os.path.join(root, "Dataset", "Test data"),
          os.path.join(root, "output", "test_features.csv"))
