# AI Health System

AI Health System is a multi-agent healthcare analysis platform built with FastAPI, LangChain, LangGraph, SQLAlchemy, and a React frontend.

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

## Core Agents

### 1. Master Health Agent

File: `Agents/masterhealth.py`

This is the central orchestrator of the system.

Responsibilities:

- receives structured health input or PDF-derived lab input
- triggers the other agents in sequence
- stores workflow state between steps
- merges lab, risk, and symptom signals into one final assessment
- forwards the final result to the Alert Agent
- returns the final health report payload

Current LangGraph flow:

```text
START
  -> extract_pdf (optional)
  -> analyze
  -> predict_risk
  -> check_symptoms
  -> merge_results
  -> generate_alert
  -> finalize
END
```

The master agent is what makes the system agentic. It does not perform all reasoning itself. It coordinates the rest of the system.

### 2. Report Analyzer Agent

File: `Agents/reportanalyzer.py`

Purpose:

- validates lab values against reference ranges
- detects abnormal parameters
- returns structured JSON for downstream agents

Example responsibilities in code:

- HbA1c range checks
- glucose range checks
- BMI range checks
- abnormal count calculation
- abnormal parameter list generation

This agent is the structured lab-analysis layer.

### 3. Risk Prediction Agent

File: `Agents/riskpredictor.py`

Purpose:

- converts health analysis into a risk assessment
- exposes both a master-agent compatibility interface and a conversational interface

Current implementation details:

- wraps a standalone conversational health agent under `Agents/health_agent/`
- can infer a risk level from agent output
- maps risk level to normalized probability for orchestration
- preserves conversational sessions in memory for repeated interactions

The nested standalone health agent under `Agents/health_agent/` uses its own tools and memory flow and is designed more like an assistant-oriented risk triage service.

### 4. Symptom Checker Agent

File: `Agents/symptomchecker.py`

Purpose:

- interprets user symptoms in the context of diabetes-related conditions
- maps symptom patterns to possible conditions
- generates a reasoning summary
- provides a normalized symptom contribution used by the master agent

It supports:

- symptom taxonomy matching
- condition hypothesis ranking
- optional LLM reasoning with rule-based fallback
- symptom severity normalization for result fusion

This is the symptom intelligence layer of the project.

### 5. Alert Agent

File: `Agents/alertsystem.py`

Purpose:

- receives the combined output from the master workflow
- checks alert thresholds
- generates a plain-language patient-friendly report
- produces the final notification/warning message

Default alert rules in code:

- `diabetes_probability > 0.70`
- `hba1c > 7.0`
- `abnormal_parameters >= 3`

This is the final communication layer between the AI pipeline and the end user.

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

