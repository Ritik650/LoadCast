"""
PLAYHACK / ML  -  Traceability check for the deck.

The pitch claims every figure on a slide comes from a command a judge can rerun.
This asserts it: each entry below pairs a string that must appear in
deck/loadcast.html with the artefact value it is supposed to come from.

    python src/verify_deck.py        # exits non-zero if any figure drifts
"""
import os
import re
import sys
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
HTML = open(os.path.join(ROOT, "deck", "loadcast.html"), encoding="utf-8").read()

M = json.load(open(os.path.join(OUT, "metrics.json")))
MN = json.load(open(os.path.join(OUT, "metrics_noprune.json")))
E = json.load(open(os.path.join(OUT, "eda_profile.json")))
C = json.load(open(os.path.join(OUT, "collinearity.json")))
B = json.load(open(os.path.join(OUT, "model_benchmark.json")))
SW = pd.read_csv(os.path.join(OUT, "threshold_sweep.csv"))


def at(thr, col):
    return float(SW.loc[(SW.thr - thr).abs().idxmin(), col])


def bench(section, name, field):
    return [r for r in B[section] if r["model"] == name][0][field]


ps = C["prune_sweep"]
sh = ps["shipped"]
checks = [
    # ---- slide 03: EDA
    ("3,000",  E["n_train_athletes"],            "{:,}"),
    ("1,100",  E["n_test_athletes"],             "{:,}"),
    ("35.0%",  E["injury_rate"] * 100,           "{:.1f}%"),
    ("15.3",   E["onset_mean"],                  "{:.1f}"),
    ("11.5",   E["recovery_mean"],               "{:.1f}"),
    ("0.322",  min(E["injury_rate_by_sport"].values()), "{:.3f}"),
    ("0.365",  max(E["injury_rate_by_sport"].values()), "{:.3f}"),
    ("83.5%",  E["coverage_pct"]["weightLogInfo_merged.csv"], "{:.1f}%"),
    ("69.4%",  E["weight_fat_null_pct"],         "{:.1f}%"),
    ("1.051",  E["acwr_injured"],                "{:.3f}"),
    ("1.000",  E["acwr_healthy"],                "{:.3f}"),
    ("0.058",  E["max_abs_corr_recovery_duration"], "{:.3f}"),
    ("0.855",  E["max_abs_corr_onset_day_offset"],  "{:.3f}"),

    # ---- slides 07 / 09: model metrics
    ("0.767",  M["auc"],                         "{:.3f}"),
    ("0.754",  M["pr_auc"],                      "{:.3f}"),
    ("0.657",  M["f1_at_best_f1"],               "{:.3f}"),
    ("2.61",   M["mae_onset_hits"],              "{:.2f}"),
    ("3.06",   M["mae_recov_hits"],              "{:.2f}"),
    ("7.61",   M["mae_base_onset"],              "{:.2f}"),
    ("3.24",   M["mae_base_recov"],              "{:.2f}"),
    ("0.411",  M["composite"],                   "{:.3f}"),
    ("0.055",  M["skill_recov_clean"],           "{:.3f}"),

    # ---- slide 08: operating points and the cliff
    ("0.21",   M["cliff_thr"],                   "{:.2f}"),
    ("0.805",  M["cliff_recall"],                "{:.3f}"),
    ("7.643",  M["cliff_mae_onset"],             "{:.3f}"),
    ("0.06",   M["thr_composite"],               "{:.2f}"),
    ("0.37",   M["thr_best_f1"],                 "{:.2f}"),
    ("0.519",  at(0.06, "f1"),                   "{:.3f}"),
    ("0.520",  at(0.37, "recall"),               "{:.3f}"),
    ("0.219",  at(0.37, "composite"),            "{:.3f}"),

    # ---- slide 07: benchmark table
    ("0.735",  bench("classification", "Decision tree", "auc"),           "{:.3f}"),
    ("0.752",  bench("classification", "LightGBM + categoricals", "auc"), "{:.3f}"),
    ("0.760",  bench("classification", "Random forest", "auc"),           "{:.3f}"),
    ("0.761",  bench("classification", "Logistic regression", "auc"),     "{:.3f}"),
    ("0.762",  bench("classification", "LightGBM (numeric)", "auc"),      "{:.3f}"),
    ("2.971",  bench("onset", "Ridge regression", "mae"),                 "{:.3f}"),
    ("2.760",  bench("onset", "LightGBM L2", "mae"),                      "{:.3f}"),
    ("2.658",  bench("onset", "LightGBM L1 (chosen)", "mae"),             "{:.3f}"),
    ("2.623",  bench("onset", "Random forest", "mae"),                    "{:.3f}"),
    ("7.615",  bench("onset", "Mean baseline", "mae"),                    "{:.3f}"),

    # ---- slide 06: collinearity + headroom ladder
    ("92.8%",  C["pc1_variance_share"] * 100,    "{:.1f}%"),
    ("0.92",   C["block_median_abs_corr"],       "{:.2f}"),
    ("70 pairs", C["n_pairs_over_098"],          "{:d} pairs"),
    ("7.61",   C["headroom_ladder"]["mean baseline"],            "{:.2f}"),
    ("3.43",   C["headroom_ladder"]["LGBM on acwr_3_21"],        "{:.2f}"),
    ("2.92",   C["headroom_ladder"]["LGBM on PC1 of block"],     "{:.2f}"),
    ("2.64",   C["headroom_ladder"]["LGBM on all numeric"],      "{:.2f}"),

    # ---- slide 05 + 09: pruning ablation deltas
    ("+0.005", M["auc"] - MN["auc"],                             "{:+.3f}"),
    ("+0.010", M["f1_at_best_f1"] - MN["f1_at_best_f1"],         "{:+.3f}"),
    ("&minus;0.006", M["skill_recov_clean"] - MN["skill_recov_clean"], "&minus;{:.3f}"),
    ("&minus;0.002", M["composite"] - MN["composite"],           "&minus;{:.3f}"),

    # ---- slide 10: the rejected correlation prune
    ("0.011 AUC", sh["auc"] - min(v["auc"] for v in ps.values()), "{:.3f} AUC"),
]

fails = []
for literal, value, fmt in checks:
    expected = fmt.format(abs(value) if fmt.startswith("&minus;") else value)
    if expected != literal:
        fails.append("  MISMATCH  deck says %-12s  artefact gives %-12s" % (literal, expected))
    elif literal not in HTML:
        fails.append("  ABSENT    %-12s traces correctly but is not in the deck" % literal)

# feature count: the deck's title and slide 05 must match metrics.json
if M["n_features"] != 100 or "108 &rarr; 100" not in HTML:
    fails.append("  FEATURE COUNT  metrics.json says %d; deck must say 108 -> 100"
                 % M["n_features"])

# no correlation-pruned config may beat the shipped onset MAE (slide 10 claim)
best = min(v["onset_mae"] for k, v in ps.items() if k != "shipped")
if best < sh["onset_mae"] - 1e-9:
    fails.append("  CLAIM BROKEN  a pruned config improves onset MAE (%.4f < %.4f)"
                 % (best, sh["onset_mae"]))

print("checked %d figures against output/*.json" % len(checks))
if fails:
    print("\n".join(fails))
    print("\n%d PROBLEM(S)" % len(fails))
    sys.exit(1)
print("all figures trace to the artefacts")
