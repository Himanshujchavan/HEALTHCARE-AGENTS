import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
from utils.constants import (
    REFERENCE_RANGES, 
    HBA1C_CATEGORIES, 
    GLUCOSE_CATEGORIES, 
    BMI_CATEGORIES,
    STATUS_NORMAL,
    STATUS_LOW,
    STATUS_HIGH,
    STATUS_CRITICAL
)

logger = logging.getLogger(__name__)


class ReportAnalyzerAgent:
    """
    Agent responsible for analyzing health reports
    Validates parameters and detects abnormalities
    """
    
    def __init__(self):
        self.reference_ranges = REFERENCE_RANGES
        # Acceptable clinical ranges (broader than normal ranges)
        self.acceptable_ranges = {
            "hba1c": {"min": 3.0, "max": 20.0},
            "glucose": {"min": 50.0, "max": 400.0},
            "bmi": {"min": 10.0, "max": 60.0},
            "age": {"min": 0, "max": 120}
        }
        logger.info("Report Analyzer Agent initialized")
    
    def analyze_pdf_report(self, pdf_path: str, use_llm: bool = True) -> str:
        """
        Analyze health data from a PDF lab report
        
        Args:
            pdf_path: Path to PDF lab report file
            use_llm: Whether to use LLM for extraction (default: True)
        
        Returns:
            JSON string of complete analysis
        """
        try:
            # Import the lab parser
            from tools.labparse import extract_health_data_from_pdf
            
            logger.info(f"Analyzing PDF report: {Path(pdf_path).name}")
            
            # Extract health data from PDF
            health_data = extract_health_data_from_pdf(pdf_path, use_llm=use_llm)
            
            # Filter out None values
            filtered_data = {k: v for k, v in health_data.items() if v is not None}
            
            if not filtered_data:
                error_result = {
                    "error": "No health parameters could be extracted from PDF",
                    "status": "failed",
                    "file": Path(pdf_path).name
                }
                return json.dumps(error_result)
            
            # Analyze the extracted data
            return self.analyze_health_record(filtered_data)
        
        except FileNotFoundError:
            error_result = {
                "error": f"PDF file not found: {pdf_path}",
                "status": "failed"
            }
            return json.dumps(error_result)
        except Exception as e:
            logger.error(f"PDF analysis failed: {str(e)}")
            error_result = {
                "error": str(e),
                "status": "failed",
                "file": Path(pdf_path).name
            }
            return json.dumps(error_result)
    
    def validate_parameter(self, param_name: str, value: float) -> Tuple[bool, str]:
        """
        Validate a single parameter value for realistic bounds
        (Not the same as checking if it's medically normal)
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check if parameter exists
            if param_name not in self.acceptable_ranges:
                return False, f"Unknown parameter: {param_name}"
            
            # Check for numeric value
            if not isinstance(value, (int, float)):
                return False, f"Non-numeric value for {param_name}"
            
            # Check for impossible values (negative, etc.)
            if value < 0:
                return False, f"Negative value not allowed for {param_name}"
            
            # Check if within acceptable clinical range (not normal range)
            ranges = self.acceptable_ranges[param_name]
            if value < ranges["min"] or value > ranges["max"]:
                return False, f"Value out of acceptable clinical range for {param_name}"
            
            return True, ""
        
        except Exception as e:
            logger.error(f"Error validating {param_name}: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def evaluate_parameter(self, param_name: str, value: float) -> Dict[str, Any]:
        """
        Evaluate a parameter against reference ranges
        
        Returns:
            {
                "value": float,
                "status": str (Normal/Low/High/Critical),
                "category": str (specific category if applicable),
                "range": dict
            }
        """
        ranges = self.reference_ranges[param_name]
        
        result = {
            "value": value,
            "status": STATUS_NORMAL,
            "category": None,
            "range": {
                "low": ranges["low"],
                "high": ranges["high"],
                "unit": ranges["unit"]
            }
        }
        
        # Determine status
        if value < ranges["low"]:
            result["status"] = STATUS_LOW
        elif value > ranges["high"]:
            result["status"] = STATUS_HIGH
        
        # Add specific categories
        if param_name == "hba1c":
            result["category"] = self._categorize_hba1c(value)
        elif param_name == "glucose":
            result["category"] = self._categorize_glucose(value)
        elif param_name == "bmi":
            result["category"] = self._categorize_bmi(value)
        
        return result
    
    def _categorize_hba1c(self, value: float) -> str:
        """Categorize HbA1c value"""
        for category, (low, high) in HBA1C_CATEGORIES.items():
            if low <= value < high:
                return category
        return "unknown"
    
    def _categorize_glucose(self, value: float) -> str:
        """Categorize glucose value"""
        for category, (low, high) in GLUCOSE_CATEGORIES.items():
            if low <= value < high:
                return category
        return "unknown"
    
    def _categorize_bmi(self, value: float) -> str:
        """Categorize BMI value"""
        for category, (low, high) in BMI_CATEGORIES.items():
            if low <= value < high:
                return category
        return "unknown"
    
    def detect_abnormalities(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect abnormalities across all parameters
        
        Args:
            parameters: Dict with parameter names and values
        
        Returns:
            {
                "parameters": {param_name: evaluation_result},
                "abnormal_count": int,
                "abnormal_parameters": list,
                "risk_indicators": list,
                "ml_features": list
            }
        """
        try:
            analysis = {
                "parameters": {},
                "abnormal_count": 0,
                "abnormal_parameters": [],
                "risk_indicators": [],
                "ml_features": []
            }
            
            # Validate and evaluate each parameter
            for param_name, value in parameters.items():
                # Validate
                is_valid, error_msg = self.validate_parameter(param_name, value)
                if not is_valid:
                    logger.error(f"Validation failed for {param_name}: {error_msg}")
                    raise ValueError(error_msg)
                
                # Evaluate
                evaluation = self.evaluate_parameter(param_name, value)
                analysis["parameters"][param_name] = evaluation
                
                # Track abnormalities
                if evaluation["status"] != STATUS_NORMAL:
                    analysis["abnormal_count"] += 1
                    analysis["abnormal_parameters"].append(param_name)
                    
                    # Add risk indicator
                    analysis["risk_indicators"].append({
                        "parameter": param_name,
                        "value": value,
                        "status": evaluation["status"],
                        "category": evaluation["category"]
                    })
            
            # Prepare ML features (ordered list of values)
            analysis["ml_features"] = [
                parameters.get("hba1c", 0),
                parameters.get("glucose", 0),
                parameters.get("bmi", 0),
                parameters.get("age", 0)
            ]
            
            # Log results
            logger.info(
                f"Analysis complete: {analysis['abnormal_count']} abnormal parameters detected"
            )
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error during abnormality detection: {str(e)}")
            raise
    
    def analyze_health_record(self, health_data: Dict[str, float]) -> str:
        """
        Main analysis method
        Returns JSON string of analysis results
        
        Args:
            health_data: {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
        
        Returns:
            JSON string of complete analysis
        """
        try:
            logger.info("Starting health record analysis")
            
            # Perform analysis
            analysis = self.detect_abnormalities(health_data)
            
            # Add metadata
            analysis["analysis_timestamp"] = str(datetime.now())
            analysis["analyzer_version"] = "1.0"
            
            # Convert to JSON
            result_json = json.dumps(analysis, indent=2)
            
            logger.info("Analysis completed successfully")
            return result_json
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            error_result = {
                "error": str(e),
                "status": "failed"
            }
            return json.dumps(error_result)
    
    def prepare_for_risk_prediction(self, analysis_result: Dict[str, Any]) -> List[float]:
        """
        Prepare features for Risk Prediction Agent
        
        Returns:
            List of numerical features for ML model
        """
        return analysis_result.get("ml_features", [])
    
    def get_summary(self, analysis_result: Dict[str, Any]) -> str:
        """
        Generate human-readable summary
        """
        abnormal_count = analysis_result.get("abnormal_count", 0)
        
        if abnormal_count == 0:
            return "All parameters are within normal range."
        else:
            abnormal_params = ", ".join(analysis_result.get("abnormal_parameters", []))
            return f"⚠️ {abnormal_count} abnormal parameter(s) detected: {abnormal_params}"


# Helper function for standalone use
def analyze_health_data(data: Dict[str, float]) -> Dict[str, Any]:
    """
    Standalone function to analyze health data
    
    Usage:
        result = analyze_health_data({"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45})
    """
    agent = ReportAnalyzerAgent()
    json_result = agent.analyze_health_record(data)
    return json.loads(json_result)


# LangChain Tool Integration
try:
    from langchain.tools import tool
    
    @tool
    def analyze_health_report(hba1c: float, glucose: float, bmi: float, age: int) -> Dict[str, Any]:
        """
        Analyze health parameters and detect abnormalities.
        
        This tool validates health parameters, detects abnormalities, and prepares data for risk prediction.
        
        Args:
            hba1c: HbA1c percentage (Glycated Hemoglobin), normal range 4.0-5.6%
            glucose: Blood glucose level in mg/dL, normal range 70-99 mg/dL
            bmi: Body Mass Index, normal range 18.5-24.9
            age: Patient age in years
        
        Returns:
            Complete analysis with parameter evaluation, abnormality detection, and risk indicators
        """
        health_data = {
            "hba1c": hba1c,
            "glucose": glucose,
            "bmi": bmi,
            "age": age
        }
        
        agent = ReportAnalyzerAgent()
        result_json = agent.analyze_health_record(health_data)
        return json.loads(result_json)
    
    @tool
    def analyze_pdf_lab_report(pdf_path: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Analyze health data from a PDF lab report file.
        
        This tool extracts health parameters from a PDF and performs complete analysis.
        
        Args:
            pdf_path: Path to the PDF lab report file
            use_llm: Whether to use LLM for extraction (default: False, uses regex)
        
        Returns:
            Complete analysis including extracted parameters and abnormality detection
        """
        agent = ReportAnalyzerAgent()
        result_json = agent.analyze_pdf_report(pdf_path, use_llm=use_llm)
        return json.loads(result_json)
    
    # Export tools for LangChain agents
    __all__ = ["ReportAnalyzerAgent", "analyze_health_data", "analyze_health_report", "analyze_pdf_lab_report"]
    
except ImportError:
    # LangChain not available, tools won't be exported but core functionality still works
    __all__ = ["ReportAnalyzerAgent", "analyze_health_data"]


if __name__ == "__main__":
    # Test the agent
    logging.basicConfig(level=logging.INFO)
    
    test_data = {
        "hba1c": 6.8,
        "glucose": 148.0,
        "bmi": 29.0,
        "age": 45
    }
    
    print("Testing Report Analyzer Agent...")
    print("=" * 50)
    
    agent = ReportAnalyzerAgent()
    result = agent.analyze_health_record(test_data)
    
    print("\nAnalysis Result:")
    print(result)
    
    # Test summary
    result_dict = json.loads(result)
    print("\nSummary:")
    print(agent.get_summary(result_dict))
