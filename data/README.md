# Data Directory

## `Cardiovascular_Disease_Dataset.csv`

The dataset behind the cardiovascular ML model (`Agents/plugins/cardio_model.py`).

| | |
|---|---|
| Rows | 1000 |
| Columns | 14 |
| Missing values | none |
| Duplicate rows | none (checked on all feature columns, ignoring `patientid`) |
| Target | `target` — 580 positive (58.0%), 420 negative |

### Provenance and licensing

This file was supplied locally for this project; **its licence could not be
independently verified for this copy** and no licence file accompanies it.
Treat it as internal-use-only until the upstream terms are confirmed.

The column names, encodings and value ranges are those of the **UCI Heart
Disease dataset** (Cleveland variant), published by the UCI Machine Learning
Repository under a **CC BY 4.0** licence, with `age` and `gender` added and the
sample expanded from 303 to 1000 rows under new `patientid` values. The
expansion's author and licence terms are not recorded in the file itself.
Before redistributing this file or the trained model, confirm the terms with
the party that supplied it.

The upstream UCI attribution, if the data is redistributed:

> UCI Machine Learning Repository — Heart Disease. https://archive.ics.uci.edu/dataset/45/heart+disease
> Licensed under CC BY 4.0.

### The label is rule-derived, not an observed outcome

`target` behaves like a deterministic function of the other columns rather
than like a recorded clinical outcome:

- 204 of the 230 groups formed by `slope`, `noofmajorvessels`, `chestpain`,
  `restingrelectro` and `fastingbloodsugar` are **pure** (every member shares
  the same label).
- An unpruned decision tree on just those five columns reaches **96.7%**
  training accuracy.
- A logistic regression on all twelve feature columns reaches a held-out
  **ROC-AUC of 0.998**.

Held-out performance therefore measures how well the model recovers the rule
that generated the labels — **not clinical validity**. A model fitted on this
data can be shown to work on this data, and nothing more. The evaluation
numbers in `Agents/plugins/cardio_model_metadata.json` must be read with that
in mind, and the model's confidence is capped accordingly.

### Target semantics — this is not a Framingham score

The previous model was trained on `data/framingham.csv`, whose label
`TenYearCHD` is a **10-year incident coronary heart disease** outcome.

This dataset's `target` is **cross-sectional**: the patient is recorded as
having cardiovascular disease in the dataset, with no time horizon. The model's
output is therefore a **disease likelihood**, and must not be presented as a
"10-year risk" or any other time-to-event estimate. The plugin's summary text
and the API response carry that qualification.

### Feature dictionary

| Column | Meaning | Unit / encoding | Maps to API field |
|---|---|---|---|
| `patientid` | Row identifier | integer, unique | *not used* |
| `age` | Age | years | `age` |
| `gender` | Sex | 1 = male, 0 = female | *not accepted* |
| `chestpain` | Chest pain type | 0–3, ordinal | *not accepted* |
| `restingBP` | Resting blood pressure (systolic) | mmHg | `systolic_bp` |
| `serumcholestrol` | Serum cholesterol | mg/dL | `total_cholesterol` |
| `fastingbloodsugar` | Fasting glucose **above 120 mg/dL** | 0/1 flag | *not accepted* |
| `restingrelectro` | Resting ECG result | 0–2, categorical | *not accepted* |
| `maxheartrate` | Maximum heart rate | bpm | *not accepted* |
| `exerciseangia` | Exercise-induced angina | 0/1 | *not accepted* |
| `oldpeak` | ST depression | continuous | *not accepted* |
| `slope` | Peak-exercise ST segment slope | 0–3, ordinal | *not accepted* |
| `noofmajorvessels` | Major vessels coloured by fluoroscopy | 0–3 | *not accepted* |
| `target` | Cardiovascular disease present | 0/1 | *label* |

### The feature mapping the model actually uses

The model consumes **only** fields the API accepts. Two variants are trained:

| Model feature | Dataset column | Unit | In which variant |
|---|---|---|---|
| `age` | `age` | years | base and extended |
| `systolic_bp` | `restingBP` | mmHg | base and extended |
| `total_cholesterol` | `serumcholestrol` | mg/dL | base and extended |
| `chest_pain_type` | `chestpain` | ordinal code 0-3 | extended only |
| `st_slope` | `slope` | ordinal code 0-3 | extended only |

The **base** variant uses the three features every cardiovascular case has,
which are exactly the plugin's required fields. The **extended** variant adds
the two optional category codes. A case is scored by the extended model only
when *both* category codes are present; otherwise it is scored by the base
model. A partially-supplied case is never handed to a model that expects a
feature it does not have, and the prediction reports which variant was used
and which optional features were missing.

### The category codes cannot be given clinical names

`chestpain` and `slope` are **ordinal codes whose clinical meaning could not
be verified for this copy of the dataset**:

- The upstream UCI variable `slope` has three levels (up / flat / down); this
  file has four, so it is not a re-encoding of the published categories.
- The observed outcome rates run in the opposite order to the published UCI
  ones for `chestpain`, so the codes cannot be mapped onto the published
  clinical categories by inference either.

Their measured relationship to the label in this file:

| `chestpain` code | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| share labelled positive | 25.7% | 68.8% | 89.4% | 88.6% |

| `slope` code | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| share labelled positive | 0.0% | 23.7% | 96.3% | 100.0% |

The codes are therefore reported as **the dataset's own ordinal codes** — "chest
pain category 2" — and never under invented clinical names such as "typical
angina" or "downsloping ST segment". Assigning clinical labels that cannot be
verified would be worse than leaving the code visible. The two columns are
also near-deterministic separators of the label, which is further evidence
that the outcome was generated from them.

### Features deliberately excluded, and why

| Field | Reason it is not in the model |
|---|---|
| `smoker` | **The dataset has no smoking column.** The value is not fabricated and not assumed to be zero. The model is therefore independent of smoking status; the rule-based scorer still uses it, and reports an *unknown* smoking status differently from a confirmed non-smoker. |
| `diastolic_bp` | Not present in the dataset. |
| `ldl`, `hdl` | Not present in the dataset. |
| `bmi` | Not present in the dataset. |
| `glucose` | `fastingbloodsugar` is a **binary flag** for "fasting glucose above 120 mg/dL", not a glucose concentration. Mapping it onto the API's `glucose` (mg/dL) would compare different quantities, so it is excluded. |
| `gender`, `restingrelectro`, `maxheartrate`, `exerciseangia`, `oldpeak`, `noofmajorvessels` | Present in the dataset but **not accepted by the API**. Adding them would require new API fields, database columns, validation ranges and frontend inputs. They are the columns that carry the rest of the label's signal (see below). |

### Measured cost of the excluded columns

Held-out ROC-AUC on a stratified 80/20 split (`random_state=42`):

| Feature set | ROC-AUC | CV ROC-AUC (5-fold) | Accuracy |
|---|---|---|---|
| Base variant (`age`, `systolic_bp`, `total_cholesterol`) | 0.825 | 0.800 | 0.770 |
| Extended variant (base + `chestpain` + `slope`) | 0.989 | 0.984 | 0.955 |
| Base + `chestpain` only | 0.893 | 0.899 | 0.800 |
| Base + `slope` only | 0.988 | 0.979 | 0.940 |
| All twelve dataset columns | 0.998 | 0.993 | 0.985 |
| Always-positive baseline | 0.500 | — | 0.580 |

Reproduce with `python eval/compare.py`; the full comparison table is in `eval/results/feature_set_comparison.csv`.

---

## `framingham.csv`

**Retired and unused.** This file is a 10-row stub — the first and last ten
rows of the real Framingham dataset, with `TenYearCHD` appended. The model it
supported was flagged `confidence: low` and could not be validated at all.

Nothing reads it any more: the legacy parsing paths in `app/serializers.py`
are covered by inline fixtures in `tests/test_api_contracts.py`. **No model is
trained on it.** It is kept only so the change of dataset is visible in the
repository history; it can be deleted without affecting any code path.
