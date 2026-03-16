"""HEALTH DATA API ROUTES

Handles health data submission, PDF uploads, retrieval,
and orchestration through the master health workflow.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
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
from Agents.masterhealth import MasterHealthAgent

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/health",
    tags=["Health Data"]
)

# Initialize Report Analyzer Agent
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


@router.post("/upload-report")
async def upload_health_report(
    file: UploadFile = File(...),
    use_llm: bool = False,
    user: User = Depends(get_current_user),
):
    """Upload a PDF lab report and run it through the master health workflow.

    Endpoint: POST /api/v1/health/upload-report

    Workflow:
    1. User uploads PDF
    2. Backend saves to local uploads/ folder
    3. MasterHealthAgent.process_pdf_report runs:
       - tools.labparse.extract_health_data_from_pdf → raw parameters
       - ReportAnalyzerAgent.analyze_health_record → validated parameters
       - downstream risk, fusion, and alert steps
    4. API returns both extracted parameters and full workflow result.
    """
    try:
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported",
            )

        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename).name or "report.pdf"
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        stored_path = uploads_dir / f"{user.id}_{timestamp}_{safe_name}"

        # Stream file to disk in chunks
        with stored_path.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

        logger.info("PDF report uploaded by user %s: %s", user.id, stored_path)

        workflow_result = master_agent.process_pdf_report(
            pdf_path=str(stored_path),
            use_llm=use_llm,
        )

        steps = workflow_result.get("steps", {})
        pdf_step = steps.get("0_pdf_extraction", {})
        structured_step = steps.get("3_structured_output", {})
        analysis_result = structured_step.get("analysis_result") or {}

        extracted_parameters = (
            pdf_step.get("extracted_parameters")
            or list(analysis_result.get("parameters", {}).keys())
        )

        return {
            "status": "success",
            "message": "PDF processed through the master health workflow",
            "filename": file.filename,
            "stored_path": str(stored_path),
            "extracted_parameters": extracted_parameters,
            "analysis": analysis_result,
            **workflow_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing uploaded PDF: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process PDF report",
        )


@router.post("/health-data", response_model=HealthDataSubmitResponse)
async def submit_health_data(
    data: HealthInput,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    POST /api/v1/health/health-data
    
    Submit health data for full multi-agent health analysis
    
    Flow:
    1. Validate input (Pydantic)
    2. Authenticate user (JWT)
    3. Store in database
    4. Trigger Master Health Agent
    5. Store workflow result
    6. Return final risk, alert, and report
    
    Example Request:
    {
        "hba1c": 6.8,
        "glucose": 148,
        "bmi": 29,
        "age": 45,
        "symptoms": ["Fatigue / Low energy", "Polyuria (frequent urination)"],
        "manual_text": "Family history of diabetes"
    }
    """
    try:
        logger.info(f"Health data submission from user {user.id} ({user.username})")
        
        # Step 1 & 2: Already done by dependencies (Pydantic validation + JWT auth)
        
        # Step 3: Store in database
        record = create_health_record(db, user.id, data)
        
        # Step 4: Trigger Master Health Agent
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

        # Step 5: Store workflow result
        update_health_record_analysis(db, record.id, json.dumps(workflow_result))
        
        logger.info(f"Health record created successfully: ID={record.id}")
        
        # Step 6: Return final orchestration result
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
        
        # Add summary for legacy or master workflow result
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
async def reanalyze_record(
    record_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    POST /api/v1/health/analyze/{record_id}
    
    Re-run the full master-agent analysis on an existing health record
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
