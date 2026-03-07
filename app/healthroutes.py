"""
HEALTH DATA API ROUTES - Step 1
Handles health data submission and retrieval
Includes validation, authentication, logging
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
import logging
import json

# Imports
from database.config import get_db
from database.models import User, HealthRecord
from database.crud import (
    create_health_record,
    get_health_record,
    get_user_health_records,
    get_latest_health_record,
    update_health_record_analysis
)
from schemas.health_schema import (
    HealthInput,
    HealthRecordResponse,
    HealthDataSubmitResponse
)
from app.auth import get_current_user
from Agents.reportanalyzer import ReportAnalyzerAgent

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health Data"]
)

# Initialize Report Analyzer Agent
analyzer_agent = ReportAnalyzerAgent()


@router.post("/health-data", response_model=HealthDataSubmitResponse)
async def submit_health_data(
    data: HealthInput,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    POST /api/v1/health/health-data
    
    Submit health data for analysis
    
    Flow:
    1. Validate input (Pydantic)
    2. Authenticate user (JWT)
    3. Store in database
    4. Trigger Report Analyzer Agent
    5. Return success response
    
    Example Request:
    {
        "hba1c": 6.8,
        "glucose": 148,
        "bmi": 29,
        "age": 45
    }
    """
    try:
        logger.info(f"Health data submission from user {user.id} ({user.username})")
        
        # Step 1 & 2: Already done by dependencies (Pydantic validation + JWT auth)
        
        # Step 3: Store in database
        record = create_health_record(db, user.id, data)
        
        # Step 4: Trigger Report Analyzer Agent
        health_data_dict = {
            "hba1c": data.hba1c,
            "glucose": data.glucose,
            "bmi": data.bmi,
            "age": data.age
        }
        
        # Analyze the data
        analysis_result = analyzer_agent.analyze_health_record(health_data_dict)
        
        # Store analysis result
        update_health_record_analysis(db, record.id, analysis_result)
        
        logger.info(f"Health record created successfully: ID={record.id}")
        
        # Step 5: Return success response
        return HealthDataSubmitResponse(
            status="success",
            message="Health data stored and analyzed successfully",
            record_id=record.id,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Error submitting health data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process health data: {str(e)}"
        )


@router.get("/health-data/{record_id}", response_model=HealthRecordResponse)
async def get_health_data(
    record_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET /api/v1/health/health-data/{record_id}
    
    Retrieve a specific health record with analysis
    """
    try:
        record = get_health_record(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found"
            )
        
        # Security check: ensure user owns the record
        if record.user_id != user.id:
            logger.warning(
                f"User {user.id} attempted to access record {record_id} owned by {record.user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return record
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving health record: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health record"
        )


@router.get("/health-data", response_model=List[HealthRecordResponse])
async def get_user_health_data(
    skip: int = 0,
    limit: int = 100,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET /api/v1/health/health-data
    
    Retrieve all health records for current user
    Supports pagination
    """
    try:
        records = get_user_health_records(db, user.id, skip, limit)
        logger.info(f"Retrieved {len(records)} health records for user {user.id}")
        return records
    
    except Exception as e:
        logger.error(f"Error retrieving health records: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health records"
        )


@router.get("/health-data/{record_id}/analysis")
async def get_health_data_analysis(
    record_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET /api/v1/health/health-data/{record_id}/analysis
    
    Get detailed analysis for a health record
    """
    try:
        record = get_health_record(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found"
            )
        
        # Security check
        if record.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        if not record.analyzed or not record.analysis_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not available for this record"
            )
        
        # Parse and return analysis
        analysis = json.loads(record.analysis_result)
        
        # Add summary
        analysis["summary"] = analyzer_agent.get_summary(analysis)
        
        return {
            "record_id": record_id,
            "created_at": record.created_at,
            "analysis": analysis
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis"
        )


@router.get("/latest")
async def get_latest_record(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET /api/v1/health/latest
    
    Get the most recent health record with analysis
    """
    try:
        record = get_latest_health_record(db, user.id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No health records found"
            )
        
        response = {
            "record": {
                "id": record.id,
                "hba1c": record.hba1c,
                "glucose": record.glucose,
                "bmi": record.bmi,
                "age": record.age,
                "created_at": record.created_at
            }
        }
        
        # Include analysis if available
        if record.analyzed and record.analysis_result:
            analysis = json.loads(record.analysis_result)
            response["analysis"] = analysis
            response["summary"] = analyzer_agent.get_summary(analysis)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest record: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve latest record"
        )


@router.post("/analyze/{record_id}")
async def reanalyze_record(
    record_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    POST /api/v1/health/analyze/{record_id}
    
    Re-run analysis on existing health record
    Useful if analyzer logic is updated
    """
    try:
        record = get_health_record(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found"
            )
        
        # Security check
        if record.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Re-analyze
        health_data_dict = {
            "hba1c": record.hba1c,
            "glucose": record.glucose,
            "bmi": record.bmi,
            "age": record.age
        }
        
        analysis_result = analyzer_agent.analyze_health_record(health_data_dict)
        update_health_record_analysis(db, record.id, analysis_result)
        
        logger.info(f"Record {record_id} re-analyzed successfully")
        
        return {
            "status": "success",
            "message": "Record re-analyzed successfully",
            "record_id": record_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-analyzing record: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to re-analyze record"
        )
