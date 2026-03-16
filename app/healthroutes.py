"""
HEALTH DATA API ROUTES - Auth-Free Version
Handles health data submission and retrieval for testing without authentication
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Dict, Any
import logging
import json

# Imports
from database.config import get_db
from database.models import HealthRecord
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
from Agents.reportanalyzer import ReportAnalyzerAgent
from Agents.masterhealth import MasterHealthAgent

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health Data"]
)

# Initialize Agents
analyzer_agent = ReportAnalyzerAgent()
master_agent = MasterHealthAgent()


def _build_analysis_summary(analysis: Dict[str, Any]) -> str:
    """Create a route-level summary for either legacy or master workflow output."""
    if analysis.get("final_assessment"):
        final_assessment = analysis.get("final_assessment", {})
        risk_level = analysis.get("risk_level") or final_assessment.get("risk_level", "Unknown")
        score = final_assessment.get("score")
        alert = analysis.get("alert")

        summary = [f"Final risk level: {risk_level}"]
        if score is not None:
            summary.append(f"score={score}")
        if alert is not None:
            summary.append("alert triggered" if alert else "no alert triggered")
        return " | ".join(summary)

    return analyzer_agent.get_summary(analysis)


@router.post("/health-data", response_model=HealthDataSubmitResponse)
async def submit_health_data(
    data: HealthInput,
    db: Session = Depends(get_db)
):
    """
    Submit health data for full multi-agent health analysis.
    Auth removed for testing.
    """
    try:
        # Use a dummy user id for database storage
        dummy_user_id = 1

        # Store in database
        record = create_health_record(db, dummy_user_id, data)
        
        # Trigger Master Health Agent
        health_data_dict = {
            "hba1c": data.hba1c,
            "glucose": data.glucose,
            "bmi": data.bmi,
            "age": data.age,
            "symptoms": data.symptoms,
        }

        workflow_result = master_agent.process_health_data(
            health_data=health_data_dict,
            manual_text=data.manual_text,
        )

        # Store workflow result
        update_health_record_analysis(db, record.id, json.dumps(workflow_result))
        
        logger.info(f"Health record created successfully: ID={record.id}")
        
        # Return final orchestration result
        return HealthDataSubmitResponse(
            status="success",
            message="Health data stored and processed through the master health workflow",
            record_id=record.id,
            timestamp=datetime.utcnow(),
            workflow_status=workflow_result.get("workflow_status"),
            risk_level=workflow_result.get("risk_level"),
            alert=workflow_result.get("alert"),
            report=workflow_result.get("report"),
            final_assessment=workflow_result.get("final_assessment"),
        )
    
    except Exception as e:
        logger.error(f"Error submitting health data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process health data: {str(e)}"
        )


@router.get("/health-data/{record_id}", response_model=HealthRecordResponse)
async def get_health_data(record_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific health record with analysis."""
    try:
        record = get_health_record(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found"
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
async def get_all_health_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve all health records (auth removed)."""
    try:
        # For testing, fetch all records without filtering by user
        records = get_user_health_records(db, user_id=None, skip=skip, limit=limit)
        logger.info(f"Retrieved {len(records)} health records (auth-free)")
        return records
    
    except Exception as e:
        logger.error(f"Error retrieving health records: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health records"
        )


@router.get("/health-data/{record_id}/analysis")
async def get_health_data_analysis(record_id: int, db: Session = Depends(get_db)):
    """Get detailed analysis for a health record (auth removed)."""
    try:
        record = get_health_record(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found"
            )
        
        if not record.analyzed or not record.analysis_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not available for this record"
            )
        
        # Parse and return analysis
        analysis = json.loads(record.analysis_result)
        analysis["summary"] = _build_analysis_summary(analysis)
        
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
async def get_latest_record(db: Session = Depends(get_db)):
    """Get the most recent health record (auth removed)."""
    try:
        # For testing, pick the latest record regardless of user
        record = get_latest_health_record(db, user_id=None)
        
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
        
        if record.analyzed and record.analysis_result:
            analysis = json.loads(record.analysis_result)
            response["analysis"] = analysis
            response["summary"] = _build_analysis_summary(analysis)
        
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
async def reanalyze_record(record_id: int, db: Session = Depends(get_db)):
    """Re-run master-agent analysis on an existing health record (auth removed)."""
    try:
        record = get_health_record(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Health record not found"
            )
        
        # Re-run master workflow
        health_data_dict = {
            "hba1c": record.hba1c,
            "glucose": record.glucose,
            "bmi": record.bmi,
            "age": record.age
        }

        workflow_result = master_agent.process_health_data(health_data_dict)
        update_health_record_analysis(db, record.id, json.dumps(workflow_result))
        
        logger.info(f"Record {record_id} re-analyzed successfully")
        
        return {
            "status": "success",
            "message": "Record re-analyzed through the master health workflow",
            "record_id": record_id,
            "workflow_status": workflow_result.get("workflow_status"),
            "risk_level": workflow_result.get("risk_level"),
            "alert": workflow_result.get("alert"),
            "report": workflow_result.get("report"),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-analyzing record: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to re-analyze record"
        )