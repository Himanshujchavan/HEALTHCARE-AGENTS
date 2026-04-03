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

The target product direction includes a frontend layer intended to support patient/clinician interaction through a modern web interface. The expected frontend stack is Next.js and Tailwind CSS, and the functional UI scope has been defined in three major modules:

- Health data input form for numeric and structured clinical parameters.
- Symptom input interface for natural-language symptom capture.
- Dashboard for showing analysis outcomes, risk level, and report summary.

From a product and architecture standpoint, this frontend scope is aligned with the backend contract and orchestration output. The backend already returns structured response fields such as risk level, alert state, report text, final assessment, and execution metadata, which are suitable for direct rendering on a dashboard.

In this workspace, the active implementation focus is backend and multi-agent pipeline execution. The frontend section is documented as product-ready scope and integration target, and the API contract has been designed to support this UI flow cleanly.

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

### Validation Layer

File: `schemas/health_schema.py`

Current request validation for the main health API includes:

- HbA1c
- glucose
- BMI
- age
- optional symptoms list
- optional manual clinical note text

Important note:

The FastAPI `POST /api/v1/health/health-data` endpoint now validates input, stores the health record, triggers the full Master Health Agent workflow, stores the orchestration result, and returns the final risk, alert flag, and patient-facing report.

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

### Root / Utility Endpoints

- `GET /`
- `GET /health`

## Frontend

The project also includes a React + Vite frontend under `frontend/`.

Current frontend stack:

- React 19
- Vite
- React Router
- React Hook Form
- Axios
- Zustand
- Recharts

This frontend currently provides the client-side foundation for user interaction and authentication-related flows.

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
```

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

## Optional Local LLM Setup

Some agents attempt to use local Ollama models. If the model is unavailable or the machine does not have enough RAM, the code falls back to deterministic logic for several stages.

Examples used in the codebase:

- `mistral`
- `deepseek-r1:8b`

Example Ollama commands:

```powershell
ollama pull mistral
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

