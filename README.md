# AI Health System

Production-ready health analysis system with multi-agent architecture.

## 🚀 Features Implemented

### ✅ Step 1: Health Data Input API
- **POST** `/api/v1/health/health-data` - Submit health data
- Input validation with Pydantic
- JWT authentication
- Database storage
- Error handling & logging
- Structured responses

### ✅ Step 2: Report Analyzer Agent
- Validates lab values
- Compares with medical reference ranges
- Detects abnormalities
- Generates structured JSON output
- Prepares features for ML models

## 📁 Project Structure

```
AI-HEALTH/
├── Agents/
│   ├── reportanalyzer.py      # Report Analyzer Agent (Step 2)
│   ├── riskpredictor.py        # Risk Prediction Agent (Future)
│   ├── symptomchecker.py       # Symptom Checker (Future)
│   ├── alertsystem.py          # Alert System (Future)
│   └── masterhealth.py         # Master Coordinator (Future)
├── app/
│   ├── auth.py                  # Authentication dependencies
│   ├── authroutes.py           # Auth endpoints (login/register)
│   └── healthroutes.py         # Health data endpoints
├── database/
│   ├── config.py               # Database configuration
│   ├── models.py               # SQLAlchemy models
│   └── crud.py                 # CRUD operations
├── schemas/
│   └── health_schema.py        # Pydantic validation schemas
├── utils/
│   ├── constants.py            # Medical reference ranges
│   └── helpers.py              # Utility functions
├── logs/                       # Application logs
├── main.py                     # FastAPI application
└── requirements.txt            # Dependencies
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Additional required packages:
```bash
pip install uvicorn python-jose[cryptography] passlib[bcrypt] python-multipart python-dotenv
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update:
```bash
cp .env.example .env
```

Edit `.env` with your database URL and secret key.

### 3. Run the Application

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 API Endpoints

### Authentication

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=john_doe&password=SecurePass123
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### Health Data

#### Submit Health Data
```http
POST /api/v1/health/health-data
Authorization: Bearer <token>
Content-Type: application/json

{
  "hba1c": 6.8,
  "glucose": 148,
  "bmi": 29,
  "age": 45
}
```

Response:
```json
{
  "status": "success",
  "message": "Health data stored and analyzed successfully",
  "record_id": 128,
  "timestamp": "2026-03-07T10:30:00"
}
```

#### Get Health Record
```http
GET /api/v1/health/health-data/{record_id}
Authorization: Bearer <token>
```

#### Get Analysis
```http
GET /api/v1/health/health-data/{record_id}/analysis
Authorization: Bearer <token>
```

Response:
```json
{
  "record_id": 128,
  "created_at": "2026-03-07T10:30:00",
  "analysis": {
    "parameters": {
      "hba1c": {
        "value": 6.8,
        "status": "High",
        "category": "diabetes"
      },
      "glucose": {
        "value": 148,
        "status": "High",
        "category": "prediabetes"
      },
      "bmi": {
        "value": 29,
        "status": "High",
        "category": "overweight"
      },
      "age": {
        "value": 45,
        "status": "Normal"
      }
    },
    "abnormal_count": 3,
    "abnormal_parameters": ["hba1c", "glucose", "bmi"],
    "ml_features": [6.8, 148, 29, 45]
  },
  "summary": "⚠️ 3 abnormal parameter(s) detected: hba1c, glucose, bmi"
}
```

#### Get Latest Record
```http
GET /api/v1/health/latest
Authorization: Bearer <token>
```

## 🏗️ Architecture

### Step 1: Health Data Input API

```
Frontend
   ↓
POST /api/health-data
   ↓
Pydantic Validation (HealthInput schema)
   ↓
JWT Authentication
   ↓
Store in Database (HealthRecord)
   ↓
Trigger Report Analyzer Agent
   ↓
Return Success Response
```

### Step 2: Report Analyzer Agent

```
Health Data Record
       ↓
Report Analyzer Agent
       ↓
Parameter Validation
       ↓
Compare with Reference Ranges
       ↓
Abnormality Detection
       ↓
Structured JSON Output
       ↓
Store Analysis Result
       ↓
Ready for Risk Prediction Agent
```

## 🔒 Security Features

- ✅ JWT-based authentication
- ✅ Password hashing (bcrypt)
- ✅ Input validation (Pydantic)
- ✅ SQL injection protection (SQLAlchemy)
- ✅ Authorization checks (user owns data)
- ✅ CORS configuration
- ✅ Secure token expiration

## 📊 Medical Reference Ranges

| Parameter | Normal Range | Unit |
|-----------|-------------|------|
| HbA1c | 4.0 - 5.6 | % |
| Glucose | 70 - 99 | mg/dL |
| BMI | 18.5 - 24.9 | kg/m² |
| Age | 0 - 120 | years |

## 🧪 Testing the System

### Test the Report Analyzer Agent
```bash
cd Agents
python reportanalyzer.py
```

### Test API with curl

1. Register:
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test_user","email":"test@example.com","password":"password123"}'
```

2. Login:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=test_user&password=password123"
```

3. Submit Health Data:
```bash
curl -X POST http://localhost:8000/api/v1/health/health-data \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"hba1c":6.8,"glucose":148,"bmi":29,"age":45}'
```

## 📝 Logging

Application logs are stored in `logs/app.log`:
- User authentication events
- Health data submissions
- Analysis results
- Errors and exceptions

## 🔄 Next Steps

### Future Agents (Step 3+):
1. **Risk Prediction Agent** - ML-based risk scoring
2. **Symptom Checker Agent** - AI symptom analysis
3. **Alert System Agent** - Critical value notifications
4. **Master Health Coordinator** - Orchestrates all agents

### Production Enhancements:
- Add database migrations (Alembic)
- Implement rate limiting
- Add caching (Redis)
- Set up monitoring (Prometheus)
- Add comprehensive tests
- Docker containerization
- CI/CD pipeline

## 🐛 Troubleshooting

### Database Issues
```bash
# Delete existing database and reinitialize
rm health_app.db
python main.py
```

### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install uvicorn python-jose[cryptography] passlib[bcrypt] python-multipart python-dotenv
```

## 📄 License

MIT License

## 👤 Author

AI Health System Development Team
