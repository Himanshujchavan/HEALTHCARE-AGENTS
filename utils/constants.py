"""
Medical reference ranges and constants
Based on standard clinical guidelines
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Medical Reference Ranges
REFERENCE_RANGES = {
    "hba1c": {
        "low": 4.0,
        "high": 5.6,
        "unit": "%",
        "description": "Hemoglobin A1c (Normal: 4.0-5.6%)"
    },
    "glucose": {
        "low": 70.0,
        "high": 99.0,
        "unit": "mg/dL",
        "description": "Fasting Blood Glucose (Normal: 70-99 mg/dL)"
    },
    "bmi": {
        "low": 18.5,
        "high": 24.9,
        "unit": "kg/m²",
        "description": "Body Mass Index (Normal: 18.5-24.9)"
    },
    "age": {
        "low": 0,
        "high": 120,
        "unit": "years",
        "description": "Patient Age"
    }
}


# HbA1c Risk Categories
HBA1C_CATEGORIES = {
    "normal": (0, 5.6),
    "prediabetes": (5.7, 6.4),
    "diabetes": (6.5, 20.0)
}


# Glucose Risk Categories (Fasting)
GLUCOSE_CATEGORIES = {
    "low": (0, 70),
    "normal": (70, 99),
    "prediabetes": (100, 125),
    "diabetes": (126, 400)
}


# BMI Risk Categories
BMI_CATEGORIES = {
    "underweight": (0, 18.5),
    "normal": (18.5, 24.9),
    "overweight": (25.0, 29.9),
    "obese_class1": (30.0, 34.9),
    "obese_class2": (35.0, 39.9),
    "obese_class3": (40.0, 60.0)
}


# Risk Levels
RISK_LEVELS = {
    0: "Very Low",
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High"
}


# Parameter Status
STATUS_NORMAL = "Normal"
STATUS_LOW = "Low"
STATUS_HIGH = "High"
STATUS_CRITICAL = "Critical"


# Database Configuration
DB_CONFIG = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600
}


# API Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-dev-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
