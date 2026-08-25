# LOADCAST — PlayHack / ML Track

**Team Isometric** — Ritik Yadav (Team Leader) · Manish Bisla · Abhihar Singh Rathore · Dipankar Ghosh

Athlete injury forecasting from a 30-day observation window. Predicts **whether**
an injury onset occurs in the hidden risk window (days 31–60), **when** it starts,
and **how long** it sidelines the athlete.

Presentation: `deck/Loadcast_PlayHack_ML.pptx` and `deck/Loadcast_PlayHack_ML.pdf`
(source: `deck/loadcast.html`).

---

## Quick start

> **Version pin matters.** The shipped `models/loadcast_models.pkl` was written
> under scikit-learn **1.6.1**. scikit-learn pickles are not forward-compatible —
> on a newer scikit-learn (observed on 1.8) loading fails with
> `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`.
> Install the pinned versions, or just retrain
> (`train_model.py` rebuilds the bundle in ~3 min).

```bash
pip install -r requirements.txt      # pinned: sklearn 1.6.1, lightgbm 4.6.0

python src/build_features.py         # train + test feature tables   (~3 min)
python src/train_model.py            # CV, metrics, submissions, models (~3 min)
```

Scoring new athletes from the saved models — no retraining:

```bash
python src/predict.py --data "Dataset/Test data" --out output/submission.csv
python src/predict.py --data "Dataset/Test data" --threshold f1 \
                      --out output/submission_f1_balanced.csv
```

Evidence runs (every figure quoted in the deck):

```bash
python src/eda_profile.py                                  # slides 03 + 05
python src/benchmark_models.py                             # slide 07 tables
python src/collinearity.py                                 # slide 06 latent analysis
python src/verify_deck.py                                  # asserts every deck figure traces
python src/train_model.py --no-prune --tag _noprune --no-save   # slide 05 ablation
python src/export_deck.py                                  # deck -> PDF + PPTX
```

Expected layout — `Dataset/` ships with the competition data:

```
Dataset/                     training athletes (60 days present; days 31–60 unused)
Dataset/Test data/           test athletes (30 days)
```

---

## Approach

**Hypothesis.** Overuse injury is a *process*, not an event. Load accumulates
faster than the body adapts, and that imbalance is visible before the breakdown.
So we model the **shape** of the 30 days — ramps, ratios, drift — not 30-day averages.

The data agrees. Averaging daily load across the 1,050 injured athletes, aligned
on each athlete's true onset day, shows a ramp running at least two weeks
(324 → 411) then a cliff to 249 at onset.

### Pipeline

| Stage | File | What happens |
|---|---|---|
| Window clip | `build_features.py` | Every source hard-clipped to days 1–30 (2026-01-05 → 2026-02-03) |
| Feature build | `build_features.py` | 6 families → one row per athlete (108 columns) |
| Pruning | `train_model.py` | 7 zero-variance + 1 duplicate column dropped → 100 |
| Three heads | `train_model.py` | Classifier + two blended regressors, 5-fold |
| Scoring strategy | `train_model.py` | Threshold swept 0.00–0.95 against the brief's own metric |

### Leakage guard

The training folder contains **60 days** per athlete; the test folder contains
**30**. Days 31–60 are the answer key. We measured the trap rather than assuming
it: six crude load statistics over all 60 days score **AUC 1.000**; the same six
over days 1–30 score **0.709**.

A single date filter therefore sits at the top of every extractor, and train and
test run through **the same function**.

### Feature families (95 numeric + 5 categorical)

| Family | Examples | Mechanism |
|---|---|---|
| Load & ACWR | `acwr_7_28`, `acwr_3_21`, `load_w4_over_w1` | Acute:chronic imbalance |
| Monotony & strain | `monotony`, `strain`, `max_day_z` | Foster: under-varied training |
| Autonomic | `rhr_slope`, `rhr_last7_delta`, `hrr_slope` | Rising resting HR = non-adaptation |
| Recovery | `sleep_debt`, `sleep_eff`, `short_nights` | Sleep shortfall |
| Congestion | `max_streak`, `start_hour_std` | Schedule density, circadian disruption |
| Context | sport, position, prior injuries, interactions | Athlete baseline |

Resting HR is reconstructed from the 01:00–05:00 hourly band. Internal load proxy
is `3×very + 2×fairly + 1×lightly` active minutes.

---

## Results (5-fold out-of-fold, 3,000 athletes)

| Metric | Value |
|---|---|
| ROC-AUC | 0.7668 |
| PR-AUC | 0.7545 |
| Best F1 (thr 0.37) | 0.6570 |
| Onset MAE / skill (hits) | 2.611 / 0.657 |
| Recovery MAE / skill (hits) | 3.062 / 0.055 |
| Composite @ default thr 0.06 | 0.4106 |

Timing is reported **twice** — hits-only and with the 30-day miss penalty applied.
At the default threshold recall is 1.000, so nothing is penalised and the two
coincide. At the F1-optimal threshold half the injuries are missed and **both
skills collapse to zero**.

### Pruning ablation (one variable, `--no-prune`)

| | pruned (100) | unpruned (108) | delta |
|---|---|---|---|
| ROC-AUC | 0.7668 | 0.7619 | **+0.0049** |
| Best F1 | 0.6570 | 0.6467 | **+0.0103** |
| Onset skill | 0.6572 | 0.6582 | −0.0010 |
| Recovery skill | 0.0554 | 0.0613 | −0.0059 |
| Composite | 0.4106 | 0.4128 | −0.0022 |

Stated plainly: pruning **trades 0.002 of our composite for 0.010 of F1**. We keep
it because F1 is an officially scored target and the composite is our own
construct — but it is a trade, not a free win.

### Collinearity — six mechanisms, one latent

`collinearity.py` reports the structure behind the feature table. The six
"independent" families are not independent:

- Monotony, ACWR, RHR drift and load slope inter-correlate at |r| = 0.80–0.99,
  median **0.92**; `monotony` vs `strain` is **0.9995** (not exactly equal, so the
  exact-duplicate prune cannot see it). **70 pairs** table-wide exceed |r| = 0.98.
- One principal component holds **92.8%** of that block's variance — which is why
  four physiologically unrelated features all land at |r| ≈ 0.855 against onset.
- That, not feature brilliance, is why logistic regression ties LightGBM: there is
  one signal and every model finds it.

Headroom ladder (onset MAE, 5-fold on injured athletes):

| Predictors | MAE | Skill |
|---|---|---|
| Mean baseline | 7.615 | 0.000 |
| Linear on `acwr_3_21` | 3.450 | 0.547 |
| LightGBM on `acwr_3_21` | 3.432 | 0.549 |
| LightGBM on PC1 of the block | 2.923 | 0.616 |
| LightGBM on the 11-feature block | 2.709 | 0.644 |
| LightGBM on all numeric | 2.635 | 0.654 |

**Correlation pruning was tested and rejected.** It costs up to **0.011 AUC**
(0.7654 → 0.7541 at |r| > 0.98, 61 features) and **never improves onset MAE** —
the best case is unchanged at 2.617, and |r| > 0.95 makes it materially worse
(2.706). LightGBM handles the collinearity better than removing it does.

| Config | n | AUC | Onset MAE |
|---|---|---|---|
| shipped (dupes only) | 96 | 0.7654 | 2.6171 |
| corr-prune \|r\| > 0.995 | 77 | 0.7634 | 2.6171 |
| corr-prune \|r\| > 0.990 | 74 | 0.7596 | 2.6190 |
| corr-prune \|r\| > 0.980 | 61 | 0.7541 | 2.6181 |
| corr-prune \|r\| > 0.950 | 43 | 0.7581 | 2.7057 |

---

## Scoring strategy

The brief applies **PENALTY = 30** to *both* timing predictions whenever a truly
injured athlete is missed. That inverts the usual precision/recall trade-off.

Measured, not illustrated: at threshold **0.21** recall is **0.805** and the
scored onset MAE is **7.643** against a 7.615 baseline — skill is already exactly
zero. A fifth of the injuries missed erases every correct timing prediction.

| File | Threshold | Recall | F1 | Composite |
|---|---|---|---|---|
| `submission.csv` (default) | 0.06 | 1.000 | 0.519 | **0.411** |
| `submission_f1_balanced.csv` | 0.37 | 0.520 | 0.657 | 0.219 |

**Full disclosure:** at threshold 0.06 the composite optimum is degenerate — it
flags every athlete, so the classifier contributes nothing to the score. We ship
it because it maximises the metric as written, and ship the F1-balanced file
alongside in case Task A is graded on its own.

---

## Files

```
src/build_features.py     leak-guarded feature extractor
src/train_model.py        three heads, CV, official metric, sweep, saves models
                          flags: --no-prune  --tag <suffix>  --no-save
src/predict.py            inference from the saved fold ensemble
src/benchmark_models.py   model-selection comparison (slide 07)
src/eda_profile.py        regenerates every slide 03 / 05 figure
src/collinearity.py       latent structure, prune sweep, headroom ladder (slide 06)
src/verify_deck.py        asserts all 52 deck figures match the artefacts (CI-able)
src/export_deck.py        renders the deck to PDF + PPTX
src/export_deck_data.py   chart figures quoted in the deck

models/loadcast_models.pkl        saved 5-fold ensemble + features + thresholds
output/submission.csv             primary submission
output/submission_f1_balanced.csv alternative
output/metrics.json               all validated numbers
output/metrics_noprune.json       ablation run
output/threshold_sweep.csv        full operating-point sweep
output/feature_importance.csv     gain ranking
output/model_benchmark.json       candidate comparison
output/eda_profile.json           EDA + data-quality profile
output/collinearity.json          correlation structure + headroom ladder
deck/loadcast.html                deck source
deck/Loadcast_PlayHack_ML.pptx    presentation
deck/Loadcast_PlayHack_ML.pdf     presentation
```

The saved ensemble is the **fold ensemble**, not a full-data refit: the
thresholds were tuned on fold-averaged probabilities, so a refit would be
differently calibrated. Verified that `predict.py` reproduces `submission.csv`
exactly (100% flag agreement, identical timing values).

---

## Known limits

- **Recovery duration is close to unlearnable** from this window. The strongest
  correlate anywhere in 100 features reaches |r| = 0.058, so 0.055 skill is near
  the ceiling — severity signal is not present in the wearable data.
- The default submission flags every test athlete (see disclosure above).
- No test labels exist, so every figure reported is cross-validated.
- **Open question for the organisers:** are Tasks A and B combined into one score
  or reported separately? Combined, the recall-first file wins; separately,
  `submission_f1_balanced.csv` should be primary. The answer decides which file
  we submit.

Python 3.9+. CPU only — no GPU required.
