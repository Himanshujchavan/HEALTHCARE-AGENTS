# AI Health System

AI Health System is a multi-agent healthcare analysis platform built with FastAPI, LangChain, LangGraph and a React frontend.

The project combines structured health data, symptom interpretation, risk reasoning, and alert/report generation into a coordinated agent workflow focused on diabetes and metabolic risk screening.

## Why This Project Is Agentic AI

This project is agentic AI because it is not a single prompt or single model call. It uses multiple specialized agents with distinct responsibilities, a controller that routes work between them, intermediate state passing, conditional execution, and fallback behavior when an LLM is unavailable.

Agentic behavior in this codebase includes:

- Specialized agents with clear responsibilities instead of one monolithic function.
- A master orchestrator built with LangGraph in `Agents/masterhealth.py`.
- Step-by-step execution where one agent's output becomes another agent's input.
- Conditional routing based on workflow state and enabled stages.
- Hybrid reasoning that combines LLM calls with deterministic medical rules.
- Final decision synthesis from multiple signals instead of a single raw prediction.

In short: the system behaves like a coordinated team of agents rather than a single chatbot.

## Work Done Report (Detailed)

This section provides a complete project execution summary, written in report style, covering architecture design, implementation details, validation approach, and development methodology for the AI Health System.

### 1. System Design and Architecture

The project was designed as a multi-agent healthcare intelligence platform instead of a single monolithic model call. This architectural decision was made to improve modularity, interpretability, maintainability, and execution reliability in production-like conditions.

At the system design level, the core idea was to separate health analysis into specialized responsibilities and then combine their outputs using an orchestrator. Each agent performs one well-defined task and passes structured output to downstream stages. This allows easier debugging, targeted improvements, and controlled fallbacks when a specific LLM-dependent stage is unavailable.

Five agents were defined and integrated:

- Report Analyzer
- Risk Predictor
- Symptom Checker
- Master Health Agent
- Alert Agent

The interaction pipeline was structured as:

1. Input ingestion from API payload or PDF report.
2. Parameter validation and abnormality detection.
3. Risk score prediction and probability normalization.
4. Symptom-context interpretation and hypothesis mapping.
5. Weighted fusion of all intermediate signals.
6. Alert decision and patient-friendly report generation.

The Master Health Agent was implemented with a stateful graph-based workflow so that each stage can be conditionally executed. This means the pipeline can skip or re-route stages based on data availability and runtime status while preserving clear, auditable execution steps.

### 2. Frontend Development

The product scope covers three modules:

- Health data input form for numeric and structured clinical parameters.
- Symptom input interface for natural-language symptom capture.
- Dashboard for showing analysis outcomes, risk level, and report summary.

That scope is now implemented. The frontend under `frontend/` is a React + Vite application wired to the live API: the report panel reads a PDF through `/upload-report/preview`, shows every extracted value for correction, and confirms the corrected values through `/upload-report/confirm`; the manual form is rendered from `/input-schema`, so it always offers exactly the parameters the disease plugins require; and the history page lists the assessments and charts the series from `/history` and `/history/summary`. See [Frontend](#frontend) for the details.

### 3. Backend Development

Backend services were developed using FastAPI with a production-oriented API structure and modular routing. The backend implementation includes application initialization, logging, database setup, authentication, health data management, and orchestration invocation.

Implemented backend features include:

- Authentication system with registration, login, and token endpoints.
- Secure route protection through token-based user context.
- Health data submission endpoints with request validation.
- Health record retrieval, latest-record queries, and re-analysis endpoints.
- PDF upload endpoint for report-driven analysis.
- Integration with master pipeline for end-to-end agent execution.

The backend flow for health submission is:

1. Validate request with schema enforcement.
2. Authenticate user identity.
3. Store health input in the database.
4. Invoke master multi-agent workflow.
5. Persist analysis output.
6. Return consolidated result to client.

This creates a complete API-to-agent lifecycle with traceable records and consistent response structure for downstream consumers.

### 4. Report Analyzer Implementation

The Report Analyzer was implemented as the first analytical stage in the workflow. Its core objective is deterministic validation of clinical parameters before higher-level risk reasoning.

The module performs:

- Reference-range based validation for key parameters.
- Structured extraction of normalized parameter status.
- Abnormal parameter counting.
- Generation of abnormal parameter lists.
- Endocrine-relevant flag creation for downstream interpretation.

The analyzer supports both direct structured input and PDF-driven extraction workflow. When using PDF reports, extracted values are filtered and then passed through the same validation logic to maintain consistency between input channels.

This design ensures that all downstream reasoning stages consume uniform, normalized data regardless of whether input originated from manual entry or document parsing.

### 5. Risk Prediction Module

The Risk Prediction layer was implemented as a hybrid module with compatibility across conversational and orchestration contexts.

Primary risk indicators include:

- HbA1c
- Glucose
- BMI
- Age

The implementation supports:

- Risk-level extraction from agent-generated text outputs.
- Mapping of risk level to normalized probability.
- Rule-based fallback scoring when external model dependencies are unavailable.
- Session-aware conversational risk interaction in standalone mode.

To maintain orchestration stability, the module includes fallback behavior that guarantees output even when LLM services or external dependencies fail. This provides dependable runtime characteristics while still allowing richer reasoning in model-enabled environments.

### 6. Symptom Checker (LangChain)

The Symptom Checker was implemented using a LangChain-oriented approach with deterministic symptom mapping and optional LLM-supported reasoning.

Key capabilities include:

- Processing natural-language symptom context.
- Taxonomy-driven symptom-to-condition mapping.
- Ranked condition hypothesis generation.
- Identification of unmatched symptoms.
- Clinical-style reasoning synthesis with rule-based fallback.

The module is designed for diabetes-related symptom intelligence and supports integration of upstream context from lab analysis and risk prediction. This allows symptom interpretation to be context-aware, not isolated.

When LLM reasoning is unavailable, the component still returns useful structured interpretation and recommendations through a deterministic rule-based path, ensuring predictable behavior.

### 7. Master Health Agent

The Master Health Agent serves as the orchestration backbone and was implemented as a state graph workflow with explicit nodes and conditional routing.

Orchestration responsibilities include:

- Coordinating each specialized agent in sequence.
- Managing execution state and intermediate artifacts.
- Handling conditional transitions and failure paths.
- Merging outputs into a single final assessment.
- Delegating communication output to the Alert Agent.

Final decision logic uses weighted scoring to combine major evidence sources:

- Risk probability contribution.
- Abnormal lab count contribution.
- Symptom severity/alignment contribution.

This weighted fusion strategy improves interpretability because each component impact is visible and traceable, and it improves robustness because no single subsystem fully determines the final result.

### 8. Alert and Report Generation

The Alert Agent was implemented as the final communication and decision layer. It transforms technical analysis outputs into actionable and user-friendly messages.

Implemented outputs include:

- Risk alert decision.
- Risk level classification.
- Trigger explanation.
- Plain-language health report.
- Notification text for high-risk and non-alert scenarios.

Threshold-driven alerting is deterministic and configurable. The agent can also use an LLM-generated report style when available, with deterministic fallback text generation when model inference is unavailable. This ensures both quality and reliability in output generation.

### 9. Testing and Validation

Testing was performed with pytest using both unit-level and end-to-end coverage patterns across agent modules.

`tests/test_user_journey.py` drives the whole journey the frontend drives —
upload, preview, correct, confirm, history, summary, re-analyse — against a
throwaway database, and asserts the shape of every response the UI reads.
`frontend/src/test/journey.test.tsx` drives the same journey through the React
components, and `frontend/src/test/states.test.tsx` covers the four
assessment states and the status vocabulary.

Validation focus areas include:

- Report analyzer behavior for abnormal and normal cases.
- Risk prediction behavior for high-risk and low-risk scenarios.
- Symptom checker response integrity with and without symptoms.
- Alert agent triggering and non-triggering behavior.
- Full pipeline execution through fused scoring and alert generation.

Edge-case coverage includes:

- Missing extractable parameters in PDF flows.
- Optional LLM stage disablement.
- Fallback execution paths in agent modules.

This testing strategy verifies not only component correctness but also orchestration compatibility between modules.

### 10. Methodology Summary

A hybrid approach combining rule-based analysis, machine learning, and LLM-based reasoning was adopted to ensure accuracy, interpretability, and scalability.

Methodology principles used in this project:

- Deterministic first-pass validation for clinical safety and traceability.
- LLM-assisted reasoning for richer context understanding where beneficial.
- Rule-based fallback to preserve service availability.
- Modular agent boundaries to simplify upgrades and testing.
- Unified orchestration to produce a coherent final outcome.

Overall, the implementation balances engineering reliability with AI flexibility. Deterministic logic provides explainable foundations, while model-assisted stages add semantic intelligence where required. This makes the system suitable for iterative scaling and practical real-world deployment workflows.

## Full Multi-Agent Workflow

### Structured Input Example

```json
{
  "hba1c": 6.8,
  "glucose": 148,
  "bmi": 29,
  "age": 45,
  "symptoms": ["Fatigue / Low energy", "Polyuria (frequent urination)"]
}
```

### Actual Processing Flow

```text
User Input
   -> Master Health Agent
   -> Report Analyzer Agent
   -> Risk Prediction Agent
   -> Symptom Checker Agent
   -> Master Agent result fusion
   -> Alert Agent
   -> Final health report
```

### Result Fusion Formula

The master agent combines multiple signals into a final score:

```text
final_score =
  (ml_probability * 0.6)
  + (abnormal_count_weight * 0.3)
  + (symptom_score * 0.1)
```

Where:

- `ml_probability` is derived from the Risk Prediction Agent
- `abnormal_count_weight` is the normalized abnormal lab count
- `symptom_score` is the normalized symptom severity/alignment score

The master agent then maps the score to a final risk level and passes that to the Alert Agent.

## Example Final Output

```json
{
  "workflow_status": "completed",
  "risk_level": "Critical",
  "alert": true,
  "report": "Your glucose value is elevated and needs attention. Your BMI is in the overweight range. Estimated diabetes risk probability is about 91%. Risk Level: Critical. Please consult a healthcare professional soon for a complete evaluation.",
  "final_assessment": {
    "risk_level": "Critical",
    "score": 0.91,
    "ml_probability": 0.95,
    "abnormal_count": 3,
    "symptom_score": 0.5
  }
}
```

## Project Architecture

```text
AI-HEALTH/
├── Agents/
│   ├── alertsystem.py
│   ├── masterhealth.py
│   ├── reportanalyzer.py
│   ├── riskpredictor.py
│   ├── symptomchecker.py
│   └── health_agent/
│       ├── agent/
│       ├── chains/
│       ├── knowledge_base/
│       ├── data/
│       ├── tests/
│       ├── main.py
│       └── requirements.txt
├── app/
│   ├── auth.py
│   ├── authroutes.py
│   └── healthroutes.py
├── database/
│   ├── config.py
│   ├── crud.py
│   └── models.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── schemas/
├── tools/
├── utils/
├── main.py
└── requirements.txt
```

## Backend Components

### FastAPI App

File: `main.py`

Responsibilities:

- application startup and shutdown
- logging configuration
- database initialization
- global exception handling
- auth and health route registration

### Authentication Layer

Files:

- `app/auth.py`
- `app/authroutes.py`

Features:

- user registration
- login with JWT token generation
- password hashing with bcrypt
- protected routes via dependency injection

### Database Layer

Files:

- `database/config.py`
- `database/models.py`
- `database/crud.py`

Current storage model:

- `User`
- `HealthRecord`

The database currently stores structured health records and the analysis JSON produced by the health workflow. The `analysis_result` field may contain either legacy analyzer output or the full master-agent orchestration result.

`HealthRecord` also stores:

- `source` - `manual` or `report`
- `extracted_parameters` - JSON map of every parameter with its value, clinical status and origin
- `assessment` - the normalized assessment snapshot (status, risk level, score, disease, components, coverage, agent statuses)
- `homa_ir` - insulin resistance value

The snapshot means the history endpoints never have to re-parse the full workflow JSON. Legacy records that predate the snapshot are normalized on read by `app/serializers.py`, which reports them as `pending` (no assessment yet) or `unreadable` (analysed but the payload cannot be parsed).

### Validation Layer

File: `schemas/health_schema.py`

Current request validation for the main health API includes:

- HbA1c
- glucose
- BMI
- HOMA-IR
- age
- systolic / diastolic blood pressure
- total cholesterol, LDL, HDL
- smoker flag (optional — see below)
- chest pain category and ST segment slope category (optional, ordinal codes 0-3)
- optional symptoms list
- optional manual clinical note text

The required and optional fields per condition are not hard-coded here: they are read from the disease plugin registry (`Agents/plugins/registry.py`), so the API contract always matches what the agents actually require. A request is accepted when it fully supports at least one condition; a request that supports none is rejected with the per-condition list of missing parameters.

Validation also covers:

- out-of-range values (every parameter has a `min`/`max` in `PARAMETER_RANGES`)
- non-numeric or non-boolean values
- clinically inconsistent combinations (diastolic BP above systolic, LDL above total cholesterol, HDL + LDL far above total cholesterol)

Important note:

The FastAPI `POST /api/v1/health/health-data` endpoint now validates input, stores the health record, triggers the full Master Health Agent workflow, stores the orchestration result, and returns the final risk, alert flag, and patient-facing report.

### Cardiovascular ML Model

Files: `Agents/plugins/cardio_model.py`, `Agents/plugins/cardio_model.pkl`, `Agents/plugins/cardio_model_metadata.json`, `data/Cardiovascular_Disease_Dataset.csv`, `data/README.md`

A logistic regression over the cardiovascular disease dataset (1000 rows). Retrain and re-evaluate it with:

```bash
python -m Agents.plugins.cardio_model   # refits and rewrites the metadata
python eval/compare.py                  # writes eval/results/
```

**Two model variants.** The **base** model's features are exactly the plugin's required fields — `age`, `systolic_bp`, `total_cholesterol` — so a case the plugin accepts can always be scored and a case it rejects can never be scored. The **extended** model adds two optional fields the API also accepts, `chest_pain_type` and `st_slope`. A case is scored by the extended model only when *both* are present; otherwise it is scored by the base model. A partially-supplied case is never handed to a model that expects a feature it does not have, and every prediction reports which variant scored it and which optional features were missing.

**Feature mapping.** The dataset columns map onto the API's field names: `age` → `age`, `restingBP` → `systolic_bp`, `serumcholestrol` → `total_cholesterol`, `chestpain` → `chest_pain_type`, `slope` → `st_slope`. Every other column is either unavailable or not accepted by the API; the full mapping table and the reason for each exclusion are in `data/README.md`.

**The two category codes carry no clinical names.** `chestpain` and `slope` are ordinal codes whose clinical meaning could not be verified for this copy of the dataset — the upstream UCI `slope` variable has three levels where this file has four, and the outcome rates run in the opposite order to the published UCI ones. They are therefore reported as the dataset's own codes ("chest pain category 2"), never under invented names such as "typical angina".

**`smoker` is not in the model.** The dataset has no smoking column. The value is not fabricated and not assumed to be zero, so the model score is independent of smoking status. The rule-based scorer does use it, and treats an absent smoking status as *unknown* rather than as a non-smoker: the smoking component is dropped from the score, its weight is redistributed across the known components, and the result records the assumption. A confirmed non-smoker and an unknown status therefore produce different scores.

**The label is a disease likelihood, not a 10-year risk.** The previous model used Framingham's `TenYearCHD`, a 10-year incident outcome. This dataset's `target` is cross-sectional — the patient is recorded as having cardiovascular disease, with no time horizon. The plugin's summary, its `disclaimer` field and the model's metadata all say so, and the API never labels the output as a 10-year risk.

**Measured performance** (stratified 80/20 split, `random_state=42`):

| | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| ML model, extended features | 0.955 | 0.965 | 0.957 | 0.961 | **0.989** | 0.036 |
| ML model, base features | 0.770 | 0.792 | 0.819 | 0.805 | **0.825** | 0.166 |
| Rule scorer | 0.640 | 0.629 | 0.922 | 0.748 | 0.763 | 0.203 |
| Always-positive baseline | 0.580 | 0.580 | 1.000 | 0.734 | 0.500 | 0.244 |

5-fold cross-validated ROC-AUC: **0.984** for the extended model, **0.800** for the base model.

**What these numbers do and do not mean.** The dataset's labels are rule-derived rather than observed clinical outcomes — an unpruned decision tree on five of its columns reaches 96.7% training accuracy, and 204 of the 230 groups they form are perfectly pure. Held-out performance therefore measures how well the model recovers the generating rule; it does **not** establish clinical validity. The model's confidence is capped at `moderate` for that reason, and the reasoning travels with the prediction in `model_confidence`, `model_target` and `model_limitations`.

The remaining gap is the feature set: the columns the model still cannot use (resting ECG, maximum heart rate, exercise angina, ST depression, vessels coloured) carry the rest of the label's signal. Measured on the same split, using all twelve dataset columns reaches ROC-AUC 0.998. Closing that gap means adding those fields to the API contract, the database and the frontend — a deliberate decision, not an oversight.

The extended model's much higher score is itself a warning: `chestpain` and `slope` separate the label almost perfectly, which is further evidence that the outcome was generated from them rather than observed. A model that recovers a generating rule this precisely has not been tested against reality.

## Current API Surface

### Auth Endpoints

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/token`

### Health Endpoints

- `POST /api/v1/health/health-data`
- `GET /api/v1/health/health-data/{record_id}`
- `GET /api/v1/health/health-data/{record_id}/analysis`
- `GET /api/v1/health/health-data`
- `GET /api/v1/health/latest`
- `POST /api/v1/health/analyze/{record_id}`
- `GET /api/v1/health/input-schema`
- `POST /api/v1/health/upload-report`
- `POST /api/v1/health/upload-report/preview`
- `POST /api/v1/health/upload-report/confirm`
- `GET /api/v1/health/history`
- `GET /api/v1/health/history/summary`
- `DELETE /api/v1/health/health-data/{record_id}`

`GET /input-schema` describes every accepted parameter (min, max, unit, type) and the conditions they belong to, so the frontend can build its form from the server contract instead of hard-coding field lists.

`POST /upload-report/preview` stores the report as a pending record and returns the extracted values without running an assessment. `POST /upload-report/confirm` takes the (optionally corrected) values, validates them exactly like a manual submission, runs the workflow and stores the assessment. The original extraction is kept for audit.

`GET /history` returns paginated assessment history (newest first) with the normalized snapshot for each record, plus the missing parameters per condition. `GET /history/summary` returns parameter and score time series and a risk-level tally.

### Error Format

All errors use one envelope:

```json
{
  "status": "error",
  "error": "ValidationError",
  "message": "Request validation failed",
  "details": [{ "field": "hba1c", "message": "Input should be less than or equal to 20", "code": "less_than_equal" }]
}
```

### Root / Utility Endpoints

- `GET /`
- `GET /health`

## Frontend

The project includes a React + Vite frontend under `frontend/`.

Current frontend stack:

- React 18 + TypeScript
- Vite
- React Router
- TanStack Query
- React Hook Form
- Axios
- Tailwind CSS
- Recharts
- Vitest + React Testing Library

### The report flow

A report is read before it is assessed, so a value the parser got wrong never
reaches the agents:

1. `POST /api/v1/health/upload-report/preview` stores the PDF, extracts the
   values and creates a **pending** record. No assessment runs.
2. The review screen shows each extracted value with its clinical status,
   unit and note, and lets the user correct it. Values the report did not
   contain are offered too, along with the conditions that are still missing
   a required parameter and the conditions that can already be assessed.
3. `POST /api/v1/health/upload-report/confirm` validates the confirmed values
   exactly like a manual submission, runs the master workflow and stores the
   result on the same record.

The one-shot `POST /api/v1/health/upload-report` endpoint still exists for
API clients that do not need the correction step.

### The manual form

The manual entry form is rendered from `GET /api/v1/health/input-schema`,
which is itself generated from the disease plugins. A parameter therefore
appears in the form as soon as the API accepts it, with the bounds the API
enforces, without a frontend change. The smoking status is a three-state
choice: leaving it unset sends no value at all, which the scorer reports as
*unknown* rather than as a confirmed non-smoker.

### The results screen

Every condition is shown with the state it is actually in. The four states
are kept apart, because they mean different things:

| State | What the screen shows |
|---|---|
| `completed` | The score, the contributing factors, and the model's confidence |
| `completed_with_errors` | The score, and a note that it rests on less than the full formula |
| `failed` | That the assessment failed, and the reason — no score is shown |
| `unavailable` | That there was not enough data to assess the condition |
| `not_applicable` | That no required values were submitted for the condition |

The agents that ran are listed with their own status, so a degraded
assessment is visible rather than implied.

### The history

`GET /api/v1/health/history` lists the assessments, each with its state and
the level of every condition it covered. `GET /api/v1/health/history/summary`
returns the per-parameter and per-score time series the charts are drawn
from. Re-analysing a record replaces its result rather than adding a second
record, and refreshes both.

## Standalone Conversational Health Agent

Folder: `Agents/health_agent/`

This is a second, self-contained agent subsystem used by `riskpredictor.py`.

It includes:

- tool-calling agent construction
- conversational memory
- risk and triage tools
- knowledge-base retrieval components
- its own FastAPI app in `Agents/health_agent/main.py`
- its own `requirements.txt`

Important note:

This module depends on additional packages and environment variables beyond the root project setup. In particular, the code references `langchain_groq` and an external model configuration for the conversational path.

## Technology Stack

### AI / Agent Stack

- LangChain
- LangGraph
- Ollama-based local LLM calls for some stages
- rule-based fallback logic when LLM calls fail

### Backend Stack

- FastAPI
- SQLAlchemy
- Pydantic
- JWT authentication
- SQLite by default, PostgreSQL supported by environment configuration

### Frontend Stack

- React
- Vite
- Axios

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with values such as:

```env
DATABASE_URL=sqlite:///./health_app.db
SECRET_KEY=change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
LOG_LEVEL=INFO
# Required only when explicitly enabling the hosted LLM path:
# GROQ_API_KEY=your-groq-api-key
```

Risk prediction is deterministic by default. Set `use_llm` to `true` for a
request only when the required model provider is configured; provider failures
are logged and returned as explicit workflow fallback metadata.

### 4. Start the backend

```powershell
python main.py
```

Or:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open API docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 6. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

### 7. Run the tests

```powershell
# Backend
python -m pytest

# Frontend
cd frontend
npm test
```

## Optional Local LLM Setup

Some agents attempt to use local Ollama models. If the model is unavailable or the machine does not have enough RAM, the code falls back to deterministic logic for several stages.

Examples used in the codebase:

- `qwen2.5:3b`
- `deepseek-r1:8b`

Example Ollama commands:

```powershell
ollama pull qwen2.5:3b
ollama pull deepseek-r1:8b
```

## Example Usage of the Master Agent

```python
from Agents.masterhealth import MasterHealthAgent

agent = MasterHealthAgent()

payload = {
    "hba1c": 6.8,
    "glucose": 148,
    "bmi": 29,
    "age": 45,
    "symptoms": [
        "Fatigue / Low energy",
        "Polyuria (frequent urination)"
    ]
}

result = agent.process_health_data(payload)
print(result["risk_level"])
print(result["alert"])
print(result["report"])
```

## Medical Scope in the Current Code

The current implementation is focused mainly on:

- diabetes screening signals
- glucose/HbA1c-based metabolic risk
- diabetes-oriented symptom reasoning
- patient-friendly alert/report generation

It is not a general diagnosis engine.

## Safety and Practical Notes

- The system provides risk support, not medical diagnosis.
- Some agent paths depend on local or external model availability.
- Rule-based fallbacks are intentionally used to keep the pipeline operational when LLM calls fail.
- The standalone `Agents/health_agent/` module has separate dependency requirements from the root backend.

## Current Status Summary

Implemented and present in code:

- authentication and protected health endpoints
- database persistence for users and health records
- Report Analyzer Agent
- Risk Prediction Agent wrapper
- Symptom Checker Agent
- Alert Agent
- Master Health Agent with LangGraph orchestration
- React frontend scaffold

Already validated during development:

- workspace diagnostics were clean
- master workflow executed end-to-end
- symptom stage, merge stage, and alert stage executed successfully
- a dedicated API route that exposes the complete master-agent orchestration response directly to the frontend
