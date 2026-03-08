# Report Analyzer Agent — How It Works

## Overview

The **Report Analyzer Agent** (`Agents/reportanalyzer.py`) checks lab values, identifies abnormal health parameters, and optionally uses a LangChain agent backed by GPT-4 for intelligent analysis. It supports both **manual data input** and **PDF lab report parsing**.

---

## Architecture

```
User Input (dict or PDF)
        │
        ▼
┌──────────────────────┐
│  ReportAnalyzerAgent │
│  (class wrapper)     │
└──────┬───────────────┘
       │
       ├── analyze_health_record()  ← dict input
       │        │
       │        ▼
       │   validate_parameters()
       │        │
       │        ▼
       │   REFERENCE_RANGES (from utils/constants.py)
       │
       └── analyze_pdf_report()     ← PDF input
                │
                ▼
          LabReportParser (tools/labparse.py)
                │
                ├── Regex extraction
                └── LLM extraction (GPT-4)
```

---

## Key Components Used for Analysis

### 1. Reference Ranges (`utils/constants.py`)

All parameter validation is done against clinically-based reference ranges:

| Parameter | Low   | High  | Unit   | Description                  |
|-----------|-------|-------|--------|------------------------------|
| HbA1c     | 4.0   | 5.6   | %      | Hemoglobin A1c               |
| Glucose   | 70.0  | 99.0  | mg/dL  | Fasting Blood Glucose        |
| BMI       | 18.5  | 24.9  | kg/m²  | Body Mass Index              |
| Age       | 0     | 120   | years  | Patient Age                  |

Risk categories are also defined for:
- **HbA1c**: Normal (≤5.6), Prediabetes (5.7–6.4), Diabetes (≥6.5)
- **Glucose**: Low (<70), Normal (70–99), Prediabetes (100–125), Diabetes (≥126)
- **BMI**: Underweight (<18.5), Normal (18.5–24.9), Overweight (25–29.9), Obese Class I/II/III (30+)

---

### 2. `validate_parameters()` — Core Validation Logic

This is the heart of the analysis. For each input parameter:
- Compares the value against its reference range
- Labels it **"Normal"**, **"Low"**, or **"High"**
- Counts total abnormal parameters and lists them

**Example input:**
```json
{ "hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45 }
```

**Example output:**
```json
{
  "parameters": {
    "hba1c":   { "value": 6.8, "status": "High" },
    "glucose": { "value": 148, "status": "High" },
    "bmi":     { "value": 29,  "status": "High" },
    "age":     { "value": 45,  "status": "Normal" }
  },
  "abnormal_count": 3,
  "abnormal_parameters": ["hba1c", "glucose", "bmi"]
}
```

---

### 3. LangChain Integration

| Component               | What It Does                                                    |
|--------------------------|-----------------------------------------------------------------|
| `@tool analyze_lab_values` | Wraps `validate_parameters()` as a LangChain tool             |
| `ChatOpenAI(model="gpt-4")` | LLM used by the LangChain agent                             |
| `initialize_agent()`      | Creates an OpenAI-tools agent wired to `analyze_lab_values`  |

The LangChain agent can accept natural-language queries like:
> "Analyze these lab values and detect abnormalities: {data}"

and will invoke the `analyze_lab_values` tool internally.

---

### 4. PDF Lab Report Parsing (`tools/labparse.py`)

The `analyze_pdf_report()` method delegates to `LabReportParser`, which uses a **two-stage extraction** approach:

#### Stage A — Regex Extraction
Regex patterns match common lab report formats for each parameter:
- **HbA1c**: patterns like `HbA1c HPLC 6.5 %`, `A1C: 6.5%`, `Glycated Hemoglobin: 6.5`
- **Glucose**: `Blood Glucose (Fasting) GOD-POD 120 mg/dL`, `Fasting Glucose: 120`
- **BMI**: `BMI Calculated 28.4`, `Body Mass Index: 28.4`
- **Age**: `Age: 45 Years`

#### Stage B — LLM Extraction (GPT-4)
A structured prompt asks GPT-4 to extract the same 4 parameters from the raw PDF text and return JSON. This acts as a fallback for non-standard report formats.

#### Merge Strategy
- **Regex results are preferred** when both methods find a value
- **LLM results fill gaps** where regex found nothing

#### PDF Libraries Used
- **pdfplumber** (primary) — better text extraction
- **PyPDF2** (fallback) — used if pdfplumber fails

---

### 5. `ReportAnalyzerAgent` Class

This is the main class consumed by `masterhealth.py` and `healthroutes.py`:

| Method                   | Purpose                                           |
|--------------------------|---------------------------------------------------|
| `analyze_health_record()` | Validates a dict of health values, returns JSON  |
| `analyze_pdf_report()`    | Extracts data from PDF, then validates it        |
| `get_summary()`           | Returns a human-readable summary of results      |

Each analysis result includes:
- Per-parameter status (Normal / Low / High)
- Abnormal count and list
- Timestamp and analyzer version

---

## Libraries & Tools Used

| Library / Tool        | Purpose                                      |
|-----------------------|----------------------------------------------|
| **LangChain**         | Agent framework, tool decorator              |
| **langchain-openai**  | GPT-4 integration via `ChatOpenAI`           |
| **pdfplumber**        | Primary PDF text extraction                  |
| **PyPDF2**            | Fallback PDF text extraction                 |
| **Python `re`**       | Regex-based parameter extraction from text   |
| **Python `json`**     | Serialization of analysis results            |
| **Python `logging`**  | Logging throughout the analysis pipeline     |

---

## Data Flow Summary

```
1. Input arrives (dict or PDF path)
2. If PDF → extract text (pdfplumber/PyPDF2) → extract values (regex + LLM) → merge
3. Validate each parameter against REFERENCE_RANGES
4. Label each as Normal / Low / High
5. Count abnormalities
6. Return structured JSON result with timestamp
```
