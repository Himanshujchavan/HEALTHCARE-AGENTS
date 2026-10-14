# Team Assignment — Major Project Extension

Companion to `implementation.md`. Phase numbers refer to that file.

**Team:** Himanshu Chavan, Shreya Thakur, Vaibhavi Tumsare, Yogesh Nagrare, Arpit Dhumane.

**Ownership rule:** each person owns the files they built; new work goes to whoever's module it touches. **Only the file owner edits that file.** Everyone else delivers new agents as standalone functions with unit tests, and Himanshu wires them into `masterhealth.py`. This keeps the critical path out of merge conflicts.

---

## At a glance

| Person | Existing ownership | New responsibility | Est. effort |
|---|---|---|---|
| **Himanshu** | Master Health Agent, Report Analyzer, Alert Agent | Disease Registry, workflow rewiring, fusion, alert changes, integration | ≈ 5 days |
| **Shreya** | Risk Predictor | Cardiovascular plugin and ML model, comparative evaluation | ≈ 5 days |
| **Vaibhavi** | Symptom Checker | Symptom scoring per disease, Differential Agent, Pima secondary check | ≈ 3.75 days |
| **Yogesh** | Frontend | Disease tabs, Explanation / Trend / Differential cards, API tests | ≈ 3 days (spare capacity) |
| **Arpit** | Backend, database and API support | Alembic, schema and routes, Trend Agent, Explainer | ≈ 4.5 days |

| Phase | Lead | Support |
|---|---|---|
| 0 Cleanup | Arpit | Shreya, Vaibhavi, Yogesh, Himanshu |
| 1 Disease Registry | Himanshu | Shreya, Vaibhavi |
| 2 Cardiovascular module | Shreya | Arpit, Vaibhavi, Himanshu |
| 3 Workflow changes | Himanshu | — |
| 4 Trend Agent | Arpit | Himanshu |
| 5 Differential Agent | Vaibhavi | Himanshu |
| 6 Explainer and alert fixes | Himanshu | Arpit |
| 7 API and frontend | Yogesh | Arpit |
| 8 Tests and evaluation | Shreya | everyone |

---

## Himanshu — Master, Analyzer, Alert

**Files:** `Agents/masterhealth.py`, `Agents/reportanalyzer.py`, `Agents/alertsystem.py`, `Agents/plugins/base.py`, `Agents/plugins/registry.py`

**Tasks (in order)**

1. **Day 1 — Snapshot test.** Add a regression test for `hba1c=6.8, glucose=148, bmi=29, age=45` asserting the exact `risk_level` and `score`. This protects everything that follows.
2. **Day 1 — Registry interface.** Write `DiseasePlugin` (`name`, `required_fields`, `optional_fields`, `reference_ranges`, `trend_scales`, `symptom_profiles`, `alert_thresholds`, `flags()`, `notes()`, `rule_risk()`, `score_symptoms()`) and `registry.py` (`register`, `get`, `applicable`). **Publish by end of day 1** so others can code against it.
3. **Days 1–3 — Plugin-aware analyzer.** Change `validate_parameters(data, plugin=None)` to read ranges, flags and notes from the plugin, defaulting to diabetes. All 12 existing tests must still pass.
4. **Days 3–6 — Workflow rewiring (`masterhealth.py`).**
   - Add `disease_results`, `history`, `trend_result`, `differential_result`, `explanation` to `HealthWorkflowState`.
   - Loop `analyze_report_node`, `predict_risk_node` and `check_symptoms_node` over `registry.applicable(health_data)`; keep `analysis_result` and `risk_result` pointing at the primary disease.
   - Compute a `final_assessment` per disease; the overall one is the highest-scoring disease, with `disease` and `all_diseases` keys.
   - Add nodes `trend → differential → merge_results → explain → generate_alert`, each with a `_route_after_*` guard (`failed → finalize`).
   - Add `history` to `MasterHealthAgent.process_health_data()`.
5. **Days 6–8 — Integration.**
   - Add the trend term to fusion: `0.50·risk + 0.25·abnormal + 0.10·symptom + 0.15·trend`; with fewer than 2 history records, keep `0.6 / 0.3 / 0.1`. Record the formula used in `final_assessment["formula"]`.
   - Connect Vaibhavi's differential function and Arpit's trend and explainer functions as nodes.
6. **Days 7–8 — Alert Agent.** Read thresholds from the plugin, make `_extract_metrics` disease-aware, add `symptom_score` as a standalone trigger, include the explanation in the report.
7. **Throughout.** Review merges and run the integration tests.

**Done when:** diabetes-only input matches the snapshot, and a two-disease input returns both results.

---

## Shreya — Risk Predictor

**Files:** `Agents/riskpredictor.py`, `Agents/plugins/diabetes.py` (`rule_risk` only), `Agents/plugins/cardiovascular.py`, `Agents/plugins/cardio_model.py`, `data/framingham.csv`, `eval/`

**Tasks (in order)**

1. **Phase 0 — Labels and scorers.**
   - Add `normalize_level()` to `utils/helpers.py` and apply it in `riskpredictor.py` (`MEDIUM → Moderate`). Himanshu applies it in master and alert.
   - Write a test that `_rule_based_risk` (`masterhealth.py:388`) and `_compute_lab_risk` (`riskpredictor.py:128`) agree on the same inputs.
2. **Phase 1.** Point `DiabetesPlugin.rule_risk()` at one of the two scorers, and make the other delegate to it.
3. **Phase 2 — Cardiovascular plugin.** Required fields `age, systolic_bp, total_cholesterol, smoker`; optional `diastolic_bp, ldl, hdl, glucose, bmi`. Define ranges (e.g. SBP, cholesterol, LDL, HDL), flags (`HYPERTENSION RANGE`, `HIGH LDL`) and a weighted `rule_risk()` in the style of `_compute_lab_risk`.
4. **Phase 2 — ML model.** Download Framingham to `data/framingham.csv` and verify the column names (the label is typically `TenYearCHD`). Train Logistic Regression on features that map to plugin fields, with median imputation, scaling and a cached `.pkl` like `PimaModel`. Use a stratified 80/20 split with a fixed `random_state`. `predict_risk()` uses the model and falls back to `rule_risk()`.
5. **Phase 8 — Evaluation (`eval/compare.py`).** Four configurations on the same hold-out set: rule-only, ML-only, LLM-only (Qwen, a sample of about 200 rows because of CPU inference) and hybrid. Headline comparison: **hybrid vs rules-only**; the other two are ablations. Compute accuracy, precision, recall, F1 and ROC-AUC, and save one confusion-matrix plot per configuration to `eval/results/`. Add `matplotlib` to requirements. Lead with recall and F1 because the positive class is a minority.
6. **Tip.** Build the rule and ML parts of the harness before Phase 3 finishes; only the hybrid run needs the workflow.

**Done when:** a cardiovascular input gets a probability from the model, and the evaluation table and plots exist.

---

## Vaibhavi — Symptom Checker

**Files:** `Agents/symptomchecker.py` (do not refactor it), the symptom parts of the plugins, `Agents/differentialagent.py`

**Tasks (in order)**

1. **Phase 0 — Pima decision.** Recommended: expose `pima_probability` as a supplementary field and do not blend it into the fusion score (the standard Pima features have no HbA1c, and blending would shift outputs the current tests assert on). Write the decision up for the paper.
2. **Phase 1.** Write `DiabetesPlugin.score_symptoms()` as a wrapper over the existing `analyze_symptoms` result. It returns `{severity_score (0–1), top_hypothesis}`.
3. **Phase 2 — Cardiovascular symptoms.** Write the profiles (chest pain, breathlessness, palpitations, dizziness) and red flags (e.g. crushing chest pain, left-arm pain), plus `score_symptoms()` using the same 0–1 definition as `_derive_symptom_metrics`.
4. **Phase 5 — Differential Agent.** Write `run_differential(disease_results, symptoms)`, rule-based first. It returns `{primary, alternatives: [{disease, score}], shared_symptoms}`, where `shared_symptoms` is the intersection of plugin profiles with the user's symptoms. It returns `None` when only one disease is applicable. An LLM re-rank is optional, with a rule-based fallback.
5. **Phase 8.** Run the Pima secondary evaluation: the Pima Logistic Regression against a glucose/BMI/age rule baseline, on your own held-out split. The hybrid cannot run on Pima because it has no HbA1c. Write unit tests for the symptom scorers and the differential agent.

**Done when:** a patient with chest pain and fatigue gets cross-disease alternatives listed.

---

## Yogesh — Frontend

**Files:** `frontend/src/models/health.ts`, `api/health.ts`, `api/hooks.ts`, `components/analyzer/ManualEntryForm.tsx`, `components/results/ResultsOverview.tsx`, `components/history/HistoryList.tsx`

**Tasks (in order)**

1. **Phase 0.** Run `npm i recharts lucide-react` (neither is installed today; the current stack is React 18, TypeScript, Redux Toolkit, React Query, Axios, React Hook Form, Framer Motion).
2. **Days 1–5 — Build on mock JSON** using the agreed shapes, so you are not waiting for the backend.
   - Add types for `disease`, `explanation`, `trend`, `differential`.
   - Add Diabetes / Cardiovascular tabs to `ManualEntryForm.tsx` with lucide icons. Send only the selected disease's fields, and validate cardiovascular ranges to match the backend.
3. **Days 5–9 — Results.** Add three cards to `ResultsOverview.tsx`: Explanation, Trend (Recharts `LineChart`) and Differential list. Make `HistoryList.tsx` show the disease per record.
4. **Days 5–9 — Edge cases.** Handle a single disease and fewer than 2 records ("insufficient history") without breaking the layout.
5. **Days 9–10.** Switch from mocks to the real API and fix mismatches.
6. **Phase 8.** Write API tests with JWT for `/health-data`, `/analyze/{id}` and the auth routes. Correct the README's frontend stack and risk labels.

**Done when:** both diseases can be submitted from the UI and all three cards render with real data.

---

## Arpit — Backend, Database and API

**Files:** `database/models.py`, `database/crud.py`, `schemas/health_schema.py`, `app/healthroutes.py`, `migrations/`, `Agents/trendagent.py`, `Agents/explainer.py`, `requirements.txt`

**Tasks (in order)**

1. **Phase 0 — Requirements.** Convert `requirements.txt` to UTF-8 (it is UTF-16 LE with CRLF today).
2. **Phase 0 — Alembic.** Set it up with `target_metadata = Base.metadata` and `render_as_batch=True` (required for SQLite column changes). Check how tables are created at startup and replace that with `alembic upgrade head`.
3. **Phase 0 — Fix `/analyze/{id}` (`healthroutes.py:614`).** It currently drops symptoms, manual text and `use_llm`. Add `symptoms` (JSON text) and `manual_text` columns to `HealthRecord` with a migration, persist them in `create_health_record`, pass them (plus `use_llm`, default `False`) in `reanalyze_record`, and add a test that submit then re-analyze gives the same score.
4. **Phase 2 — Schema and DB.**
   - Make the four diabetes fields in `HealthInput` Optional.
   - Add `systolic_bp, diastolic_bp, total_cholesterol, ldl, hdl, smoker`.
   - Add a `model_validator` requiring at least one plugin's required fields.
   - Add the matching `HealthRecord` columns (diabetes fields nullable), a migration, and the CRUD changes.
5. **Phase 2 — Routes.** Make `submit_health_data` and `reanalyze_record` build `health_data_dict` from every non-null field. Check whether `upload-report` needs a change.
6. **Phase 4 — Trend Agent.** In the route, fetch history with `get_user_health_records(limit=5)`, drop the current record, and also pass each past `final_assessment.score` parsed from its stored `analysis_result` JSON. Write `run_trend(history, plugin, current)` returning `{trend_score, direction, rising_but_normal, visits_used}`. The score is `clamp(delta / plugin.trend_scales[param], 0, 1)` per parameter, with `trend_score` the maximum. Return `insufficient_history` for fewer than 2 prior records.
7. **Phase 6 — Explainer.** Write `explain(final_assessment, trend, triggers)`: a template first, plus an optional Qwen paraphrase restricted to those numbers.
8. **Phase 7.** Add `disease`, `explanation`, `trend` and `differential` to `HealthDataSubmitResponse`. Check that `GET /health-data/{id}/analysis` returns the new keys.
9. **Phase 8.** Write tests for the trend agent, the explainer and the migrations.

**Done when:** a fresh database builds from migrations alone, and re-analysing a record reproduces its original score.

---

## Day-1 agreements (all five)

| Contract | Owner |
|---|---|
| `DiseasePlugin` interface and registry | Himanshu |
| `run_trend(history, plugin, current) -> dict` | Arpit |
| `run_differential(disease_results, symptoms) -> dict` | Vaibhavi |
| `explain(final_assessment, trend, triggers) -> str` | Arpit |
| Response and result JSON shapes (`disease`, `explanation`, `trend`, `differential`) | Himanshu and Yogesh, agreed together |

Write each contract as a stub function or TypeScript type and commit it on day 1.

---

## Timeline (about 12 working days)

| Days | Work |
|---|---|
| 1–2 | Phase 0 (Arpit leads); Himanshu publishes the registry interface and the wrappers |
| 3–6 | Phase 2 in parallel across Shreya, Arpit and Vaibhavi; Himanshu starts Phase 3 on stub plugins |
| 6–8 | Phases 4, 5 and 6 in parallel |
| 7–10 | Phase 7 integration |
| 9–12 | Phase 8 and the paper |

**If time runs short:** cut Phase 5 (differential) first, then the Pima secondary check.

## Dependencies to watch

- Phase 3 waits on the interface from Himanshu's day-1 work; Shreya, Vaibhavi and Arpit also wait on it.
- Yogesh can start immediately on mocks, so he is not blocked.
- The hybrid evaluation run waits on Phase 3; the rule and ML runs do not.
- Himanshu is the bottleneck (all of `masterhealth.py`), so everyone else should deliver standalone functions with tests rather than editing it.