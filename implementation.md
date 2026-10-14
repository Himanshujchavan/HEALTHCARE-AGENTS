# Implementation Plan — Major Project Extension

Extends the existing five-agent diabetes pipeline (FastAPI + LangGraph + SQLAlchemy + React/TypeScript) into a multi-disease decision-support platform.

**Scope (5 items):** Disease Registry + Cardiovascular module, Differential Agent, Trend Agent, Explainability Agent, Comparative Evaluation.
**Deferred (future work):** Clinician review workflow, vector DB, Celery/Redis, MLflow, Docker Compose, tracing, clinician dashboard, FHIR/HL7.

File references use `file:line` from the current codebase. Items marked **[verify]** were not fully traced and must be checked before relying on them.

---

## 0. Verified baseline (what the code actually does today)

| Area | Fact | Where |
|---|---|---|
| Orchestration | LangGraph `StateGraph`. Nodes: `extract_pdf → analyze → predict_risk → check_symptoms → merge_results → generate_alert → finalize`, with `_route_after_*` guards | `Agents/masterhealth.py:41, 485-533, 600-644` |
| Fusion | `final = ml*0.6 + min(abnormal/3,1)*0.3 + symptom*0.1`; levels at 0.40 / 0.65 / 0.80 | `masterhealth.py:303-362, 561-568` |
| "ML probability" source | Comes from `risk_result` (`RiskPredictorAgent`, falling back to rules). `_llm_risk_prediction()` no longer calls an LLM; it delegates to `_rule_based_risk()` | `masterhealth.py:147-200, 376-386` |
| Pima model | Runs inside `analyze_symptoms()` (`symptomchecker.py:1192, 1520, 1529`). `masterhealth.py` never references it, so its probability is **not** in the fusion score | grep: no `pima` in `masterhealth.py` |
| Validator | `validate_parameters()` loops over any keys with a range; only the `endocrine_flags` and `NOTES` are diabetes-specific | `reportanalyzer.py:28-135` |
| Rule scoring | Weights hba1c .35, glucose .35, bmi .20, age .10; levels `LOW/MEDIUM/HIGH/CRITICAL` | `riskpredictor.py:128-189` |
| Alerts | `AlertThresholds` (prob > 0.70, HbA1c > 7.0, abnormal ≥ 3). `_extract_metrics()` does not read `symptom_score` | `alertsystem.py:40, 126` |
| DB | `HealthRecord`: `hba1c, glucose, bmi, age` all `nullable=False`; `analysis_result` is a JSON **string** | `database/models.py` |
| CRUD | `get_user_health_records`, `get_latest_health_record`, `update_health_record_analysis` already exist | `database/crud.py` |
| API | Route builds a 4-key dict, calls `master_agent.process_health_data()`, stores `json.dumps(result)` | `app/healthroutes.py:149-225`; also `reanalyze_record` at `:614`, `upload-report` at `:67` |
| Schema | `HealthInput` requires all 4 fields | `schemas/health_schema.py` |
| Frontend | React 18, TypeScript, Vite, Tailwind, Redux Toolkit, React Query, Axios, React Hook Form, React Router, Framer Motion. **No Recharts, Zustand or lucide-react** | `frontend/package.json` |
| Tests | 5 modules, 12 tests, all `use_llm=False`; pipeline test accepts any of the 4 levels | `tests/` |

---

## Phase 0 — Cleanup before building (≈ 0.5 day)

- [ ] **Convert `requirements.txt` to UTF-8.** It is UTF-16 LE with CRLF, which breaks `pip` on many setups.
  `python -c "open('requirements.txt','w',encoding='utf-8').write(open('requirements.txt',encoding='utf-16').read())"`
- [ ] **Unify risk labels.** Three sets exist: `utils/constants.py RISK_LEVELS` (Very Low…Very High), `riskpredictor.py` (`LOW/MEDIUM/HIGH/CRITICAL`), master/alert (`Low/Moderate/High/Critical`). Add `normalize_level()` to `utils/helpers.py`, map `MEDIUM → Moderate`, and apply it where levels cross agent boundaries. Grep for `RISK_LEVELS` usage before deleting it.
- [ ] **Decide the Pima question (recommended: do not blend).** The standard Pima features do not include HbA1c, and blending would shift the probabilities the current tests assert on. Expose it as a supplementary `pima_probability` field instead. Real ML enters the fusion through the cardiovascular model (Phase 2). In the paper, call the diabetes term "risk probability (Risk Prediction Agent)" rather than "ML probability". Leave the import-time `_pima_model = PimaModel()` (`symptomchecker.py:1192`) as it is; refactor later if needed.
- [ ] Add `Agents/__init__.py` and `Agents/plugins/__init__.py`.
- [ ] Add Alembic: `pip install alembic`, `alembic init migrations`, set `target_metadata = Base.metadata`, and `render_as_batch=True` (required for SQLite column changes). **[verify]** how tables are currently created at startup (`create_all` in `main.py`/`database/config.py`) and replace that with `alembic upgrade head`.
- [ ] **Fix `/analyze/{id}` (`healthroutes.py:614`).** `reanalyze_record` rebuilds the payload from the 4 stored columns and calls `process_health_data(health_data_dict)`, so symptoms, `manual_text` and `use_llm` are dropped and a re-analysis can score differently from the original. Symptoms and manual text are not stored anywhere today, so do all of this: add `symptoms` (JSON text) and `manual_text` columns to `HealthRecord`; persist them in `create_health_record`; pass them (plus `use_llm`, default `False`) in `reanalyze_record`; generate an Alembic revision for the two columns. Add a test: submit → re-analyze → identical `final_assessment.score`. Do this right after the Alembic setup above.
- [ ] `cd frontend && npm i recharts lucide-react`.
- [ ] Delete or implement the empty `tools/validation-tool.py`.
- [ ] Make LLM model names consistent in code and docs. Currently `qwen` appears in `alertsystem.py` and `symptomchecker.py`, `mistral` in `tools/labparse.py`, and `mistral`/`deepseek` in the README.
- [ ] Add a pytest regression snapshot of the current diabetes output for `hba1c=6.8, glucose=148, bmi=29, age=45` (exact `risk_level` and `score`). This is the safety net for Phases 1–3.

**Done when:** all 12 existing tests plus the snapshot test pass on a fresh clone.

---

## Phase 1 — Disease Registry (≈ 1–1.5 days)

Create `Agents/plugins/`:

```
Agents/plugins/
  __init__.py
  base.py          # DiseasePlugin interface
  registry.py      # register / get / applicable
  diabetes.py      # DiabetesPlugin: wraps existing logic, behaviour unchanged
```

`base.py`:

```python
class DiseasePlugin:
    name: str
    required_fields: list[str]
    optional_fields: list[str]
    reference_ranges: dict
    trend_scales: dict            # param -> delta considered "full" trend (Phase 4)
    symptom_profiles: dict
    alert_thresholds: AlertThresholds
    def flags(self, data) -> list[str]: ...
    def notes(self, param, status) -> str: ...
    def rule_risk(self, data, analysis) -> dict: ...   # {risk_level, risk_probability, ...}
    def score_symptoms(self, symptoms) -> dict: ...    # {severity_score 0-1, top_hypothesis}
```

`registry.applicable(health_data)` returns the plugins whose `required_fields` are all present.

Steps:
- [ ] Write `DiabetesPlugin(DiseasePlugin)` as a **wrapper** around the existing diabetes logic instead of moving code: it exposes the ranges from `utils/constants.py`, the `DIABETES CRITERIA MET` / `INSULIN RESISTANCE` flags and `NOTES` from `reportanalyzer.py`, and the `AlertThresholds` values, and its `rule_risk()` calls the existing risk function. Leave the original functions in place until all tests are green; remove duplication afterwards.
- [ ] `validate_parameters(data, plugin=None)`: default to the diabetes plugin so existing callers and tests keep working.
- [ ] Two parallel rule scorers exist: `_rule_based_risk` (`masterhealth.py:388`, the fallback path) and `_compute_lab_risk` (`riskpredictor.py:128`). **[verify]** that they agree on the same inputs and add a test comparing them; then point `DiabetesPlugin.rule_risk()` at one of them and have the other delegate to it.

**Done when:** all existing tests and the Phase 0 snapshot pass unchanged, with diabetes running through the plugin.

---

## Phase 2 — Cardiovascular module (≈ 2–3 days)

**Data and schema**
- [ ] `schemas/health_schema.py`: make the four diabetes fields `Optional`; add optional `systolic_bp, diastolic_bp, total_cholesterol, ldl, hdl, smoker`; add a `model_validator` requiring at least one plugin's `required_fields` to be complete.
- [ ] `database/models.py`: add the cardiovascular columns (nullable) and relax `hba1c/glucose/bmi/age` to nullable. Generate an Alembic revision (batch mode).
- [ ] `database/crud.py::create_health_record`: persist the new fields.
- [ ] `healthroutes.py`: build `health_data_dict` from every non-null field in `submit_health_data` **and** `reanalyze_record`. **[verify]** `upload-report` and `tools/labparse.py`: cardiovascular PDF extraction is optional; manual entry is enough for this phase.

**Plugin** `plugins/cardiovascular.py`
- [ ] Required: `age, systolic_bp, total_cholesterol, smoker`. Optional: `diastolic_bp, ldl, hdl, glucose, bmi`.
- [ ] Ranges, e.g. SBP 90–119, total cholesterol < 200, LDL < 100, HDL ≥ 40; flags such as `HYPERTENSION RANGE`, `HIGH LDL`.
- [ ] `rule_risk()`: weighted additive score in the same style as `_compute_lab_risk`.
- [ ] `symptom_profiles` (chest pain, breathlessness, palpitations, dizziness) plus red flags (crushing chest pain, left-arm pain). `score_symptoms()` uses the same 0–1 definition as `_derive_symptom_metrics` (`masterhealth.py:571`). Do not refactor the 1,746-line `symptomchecker.py`; diabetes keeps using `analyze_symptoms()`.

**ML model**
- [ ] Place the Framingham CSV at `data/framingham.csv`. It is not in the repo. **[verify]** column names after download; the label is typically `TenYearCHD`.
- [ ] `plugins/cardio_model.py`: Logistic Regression on columns that map to plugin fields (age, systolic BP, diastolic BP, total cholesterol, smoker, glucose, BMI), with median imputation and scaling, cached to a `.pkl` as `PimaModel` does. Use a stratified 80/20 split with a fixed `random_state`.
- [ ] `predict_risk()` = ML probability when the model is available, otherwise `rule_risk()`.

**Done when:** a cardiovascular-only submission returns a stored record, a validated analysis, a risk level and an alert, and a diabetes-only submission is unchanged.

---

## Phase 3 — Workflow changes in `masterhealth.py` (≈ 1.5 days)

Design rule: **keep `analysis_result` and `risk_result` for the primary disease** so routes, tests and the frontend keep working; add multi-disease data alongside.

- [ ] `HealthWorkflowState` (`:41`): add `disease_results: Dict[str, Dict]`, `history: List[Dict]`, `trend_result`, `differential_result`, `explanation`.
- [ ] `analyze_report_node`: loop over `registry.applicable(health_data)`; store each in `disease_results[name]["analysis"]`; set `analysis_result` to the primary one.
- [ ] `predict_risk_node`: same loop for `risk`. Diabetes keeps the `RiskPredictorAgent → rule fallback` chain.
- [ ] `check_symptoms_node`: diabetes via `analyze_symptoms`; other plugins via `plugin.score_symptoms`.
- [ ] `merge_results_node`: compute a `final_assessment` per disease; the overall `final_assessment` is the highest-scoring disease, with `disease` and `all_diseases` keys added.
- [ ] New nodes and routing:

```
START → [extract_pdf] → analyze → predict_risk → check_symptoms
      → trend → differential → merge_results → explain → generate_alert → finalize
```

Add a `_route_after_*` function per new node with the same `failed → finalize` guard.
- [ ] `MasterHealthAgent.process_health_data(..., history=None)`: pass `history` into the initial state.

**Done when:** the Phase 0 snapshot still matches for diabetes-only input, and a two-disease input returns both results.

---

## Phase 4 — Trend Agent (≈ 1.5 days)

- [ ] Route: fetch `get_user_health_records(db, user.id, limit=5)`, drop the record just created, pass it as `history`. Agents stay database-free.
- [ ] Risk history is free: each past `analysis_result` is JSON, so `json.loads(rec.analysis_result)["final_assessment"]["score"]` gives a score series.
- [ ] `Agents/trendagent.py`: per parameter, `delta = latest − oldest` over the window, `param_trend = clamp(delta / plugin.trend_scales[param], 0, 1)` (e.g. HbA1c scale 0.5). `trend_score = max(param_trend)`. Return `{trend_score, direction, rising_but_normal: [params], visits_used}`; return `insufficient_history` for fewer than 2 prior records.
- [ ] Fusion with history: `0.50·risk + 0.25·abnormal + 0.10·symptom + 0.15·trend`. **With fewer than 2 records, keep the original 0.6 / 0.3 / 0.1** so existing behaviour and tests are unchanged. Record the formula used in `final_assessment["formula"]` and add `trend_component` to `components`. The weights are a design choice you must justify in the paper.

**Done when:** a synthetic history of rising HbA1c (5.4 → 5.6 → 5.9) with a normal latest-visit pattern yields a higher score than the same latest values with no history.

---

## Phase 5 — Differential Agent (≈ 1 day)

- [ ] `Agents/differentialagent.py`, rule-based first. Input: `disease_results` and symptom mappings. Output: `{primary, alternatives: [{disease, score}], shared_symptoms}`, where `shared_symptoms` is the intersection of plugin `symptom_profiles` with the user's symptoms.
- [ ] Skip the node (return `None`) when only one disease is applicable.
- [ ] Optional LLM re-ranking only if time allows, with a rule-based fallback.

---

## Phase 6 — Explainability Agent + Alert fixes (≈ 1 day)

- [ ] `Agents/explainer.py`: template explanation built from `final_assessment["components"]`, `trend_result`, and alert triggers, e.g. *"Score 0.71 (High): risk model 0.30, abnormal labs 0.30, symptoms 0.05, trend 0.06."* Optional Qwen paraphrase constrained to those numbers; always fall back to the template.
- [ ] Store in `final_assessment["explanation"]` and include it in the alert report.
- [ ] `alertsystem.py`: read thresholds from the plugin; extract fields by disease instead of `hba1c` only (`_extract_metrics`, `:126`); add `symptom_score` as a standalone trigger (closes the gap in your report §4.4.2).

---

## Phase 7 — API and frontend (≈ 2 days)

**Backend**
- [ ] `HealthDataSubmitResponse`: add `disease`, `explanation`, `trend`, `differential`. **[verify]** `GET /health-data/{id}/analysis` returns the stored workflow JSON so new keys appear without schema changes.

**Frontend** (`frontend/src`)
- [ ] `models/health.ts`, `api/health.ts`: new fields and response types.
- [ ] `components/analyzer/ManualEntryForm.tsx`: disease tabs (Diabetes / Cardiovascular) with lucide icons; send only the fields for the selected tab(s).
- [ ] `components/results/ResultsOverview.tsx`: three cards: Explanation, Trend (Recharts `LineChart`), Differential list.
- [ ] `components/history/HistoryList.tsx`: show the disease per record.

---

## Phase 8 — Tests and comparative evaluation (≈ 2 days)

**Tests**
- [ ] New: `test_disease_registry.py`, `test_cardiovascular.py`, `test_trend_agent.py`, `test_differential_agent.py`, `test_explainer.py`.
- [ ] Tighten `test_full_agents_pipeline`: assert the expected level for `hba1c=6.8, glucose=148, bmi=29, age=45` instead of "any valid level".
- [ ] Add one two-disease end-to-end test and one API test for `/health-data` with JWT (auth/DB/routes currently untested).

**Evaluation** (`eval/compare.py`)
- [ ] Hold-out: stratified 80/20 Framingham split, fixed seed, never used for training.
- [ ] Four configurations on the same test set: rule-only, ML-only, LLM-only (Qwen, run on a sampled subset such as 200 rows because of CPU inference), and hybrid. The headline comparison for the paper is **hybrid vs the rules-only baseline**; the other two are ablations.
- [ ] Report a **confusion matrix**, accuracy, precision, recall, F1 and ROC-AUC. The positive class is a minority (roughly 15%, **[verify]**), so lead with recall/F1, not accuracy. Save one confusion-matrix plot per configuration (Matplotlib, optionally Seaborn `heatmap`) to `eval/results/`. `matplotlib` is not in `requirements.txt` (numpy, pandas and scikit-learn are), so add it.
- [ ] Pima is a poor fit for the hybrid: it has no HbA1c, which the diabetes plugin requires. Use it only as a secondary evaluation of the Pima Logistic Regression against a glucose/BMI/age rule baseline, on your own held-out split (**[verify]** whether the cached model's training split is exposed).
- [ ] Limitation to state in the paper: Framingham has no symptoms or visit history, so on it the hybrid reduces to risk + abnormal-count terms; the symptom and trend terms are evaluated through unit tests and synthetic cases instead.

---

## Build order if time is short

1. Phase 0 → 1 → 2 → 3 (the core: registry, second disease, workflow)
2. Phase 6 (explainer + alert fix; quick win)
3. Phase 4 (trend), then Phase 5 (differential)
4. Phase 7 (frontend)
5. Phase 8 (evaluation last, but reserve time for it: it supplies the paper's results section)

Rough total: 12–15 working days for one developer; parallelise Phase 7 with Phases 4–6 across a team.

---

## Definition of done

- [ ] Diabetes-only input behaves exactly as before (snapshot test).
- [ ] Re-analysing a record reproduces the original score (symptoms and notes are persisted and passed).
- [ ] Cardiovascular-only and two-disease inputs run end to end through the API and UI.
- [ ] Each new agent has a rule-based fallback and a unit test.
- [ ] DB changes are Alembic migrations, not manual edits.
- [ ] Evaluation table with a confusion matrix exists for the cardiovascular module.
- [ ] README and report use the final label set (`Low/Moderate/High/Critical`) and the correct frontend stack.