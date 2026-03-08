"""
Lab Report PDF Parser Tool
LangChain tool for extracting health parameters from PDF lab reports
"""

import logging
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# PDF parsing imports
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("pdfplumber not available. Install with: pip install pdfplumber")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

logger = logging.getLogger(__name__)


class LabReportParser:
    """
    Parser for extracting health data from PDF lab reports
    Uses both regex extraction and LLM-based parsing
    """
    
    def __init__(self, llm_model: str = "mistral", use_llm: bool = True):
        """
        Initialize the parser
        
        Args:
            llm_model: Ollama model to use for extraction
            use_llm: Whether to use LLM for extraction (if False, uses regex only)
        """
        self.use_llm = use_llm
        if use_llm:
            self.llm = Ollama(model=llm_model)
        else:
            self.llm = None
        
        # Regex patterns for common health parameters
        # Supports both lab report format (with method names) and simple colon format
        self.patterns = {
            "hba1c": [
                r"HbA1[cC]\s+\S+\s+(\d+\.?\d*)\s*%",  # HbA1c HPLC 6.5 %
                r"HbA1[cC]\s*:?\s*(\d+\.?\d*)\s*%?",
                r"A1[cC]\s*:?\s*(\d+\.?\d*)\s*%?",
                r"Hemoglobin A1[cC]\s*:?\s*(\d+\.?\d*)\s*%?",
                r"Glycated\s+Hemoglobin\s*:?\s*(\d+\.?\d*)\s*%?"
            ],
            "glucose": [
                r"Blood\s+Glucose\s+\(Fasting\)\s+\S+\s+(\d+\.?\d*)\s*mg/d[lL]",  # Lab format
                r"Glucose\s*:?\s*(\d+\.?\d*)\s*mg/d[lL]",
                r"Blood\s+Glucose\s*:?\s*(\d+\.?\d*)\s*mg/d[lL]",
                r"Fasting\s+Glucose\s*:?\s*(\d+\.?\d*)\s*mg/d[lL]",
                r"FBS\s*:?\s*(\d+\.?\d*)\s*mg/d[lL]"
            ],
            "bmi": [
                r"BMI\s+\S+\s+(\d+\.?\d*)",  # BMI Calculated 28.4
                r"BMI\s*:?\s*(\d+\.?\d*)",
                r"Body\s+Mass\s+Index\s*:?\s*(\d+\.?\d*)"
            ],
            "age": [
                r"Age\s*:?\s*(\d+)\s*[Yy]ears?",
                r"Age\s*:?\s*(\d+)\s*[Yy]rs?",
                r"Age\s*:?\s*(\d+)"
            ]
        }
        
        logger.info(f"LabReportParser initialized (LLM: {use_llm})")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        
        # Try pdfplumber first (better text extraction)
        if PDF_AVAILABLE:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Extracted text using pdfplumber from {pdf_path.name}")
                return text
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}, trying PyPDF2")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                reader = PdfReader(str(pdf_path))
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                logger.info(f"Extracted text using PyPDF2 from {pdf_path.name}")
                return text
            except Exception as e:
                logger.error(f"PyPDF2 extraction failed: {e}")
                raise
        
        raise RuntimeError("No PDF library available. Install pdfplumber or PyPDF2")
    
    def extract_with_regex(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extract health parameters using regex patterns
        
        Args:
            text: Text content from PDF
            
        Returns:
            Dictionary of extracted parameters
        """
        results = {}
        
        for param_name, patterns in self.patterns.items():
            value = None
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            results[param_name] = value
        
        logger.info(f"Regex extraction found: {sum(1 for v in results.values() if v is not None)} parameters")
        return results
    
    def extract_with_llm(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extract health parameters using LLM
        
        Args:
            text: Text content from PDF
            
        Returns:
            Dictionary of extracted parameters
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical data extraction specialist. 
Extract the following health parameters from lab reports:
- HbA1c (%) - Glycated Hemoglobin
- Glucose (mg/dL) - Blood Glucose level
- BMI - Body Mass Index
- Age (years)

Return ONLY a valid JSON object with these exact keys: hba1c, glucose, bmi, age
If a parameter is not found, use null as the value.
Do not include any explanation, only the JSON object.

Example output:
{
  "hba1c": 6.5,
  "glucose": 120.0,
  "bmi": 25.3,
  "age": 45
}"""),
            ("user", "Extract health parameters from this lab report:\n\n{text}")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": text[:4000]})  # Limit text length
            
            # Parse JSON from response
            content = response.strip() if isinstance(response, str) else response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content)
            
            data = json.loads(content)
            
            logger.info(f"LLM extraction found: {sum(1 for v in data.values() if v is not None)} parameters")
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"hba1c": None, "glucose": None, "bmi": None, "age": None}
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return {"hba1c": None, "glucose": None, "bmi": None, "age": None}
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method to parse PDF and extract health data
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted health parameters
        """
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            # Try regex extraction first
            regex_results = self.extract_with_regex(text)
            
            # If LLM is enabled, use it as well
            if self.use_llm:
                llm_results = self.extract_with_llm(text)
                
                # Merge results (prefer regex if both found, otherwise use LLM)
                final_results = {}
                for key in ["hba1c", "glucose", "bmi", "age"]:
                    if regex_results.get(key) is not None:
                        final_results[key] = regex_results[key]
                    elif llm_results.get(key) is not None:
                        final_results[key] = llm_results[key]
                    else:
                        final_results[key] = None
            else:
                final_results = regex_results
            
            # Count successful extractions
            extracted_count = sum(1 for v in final_results.values() if v is not None)
            
            result = {
                "success": True,
                "file": Path(pdf_path).name,
                "parameters": final_results,
                "extracted_count": extracted_count,
                "message": f"Successfully extracted {extracted_count}/4 parameters"
            }
            
            logger.info(f"PDF parsing complete: {result['message']}")
            return result
        
        except Exception as e:
            logger.error(f"PDF parsing failed: {str(e)}")
            return {
                "success": False,
                "file": Path(pdf_path).name,
                "error": str(e),
                "parameters": None
            }


# LangChain Tool Definition
@tool
def parse_lab_report_pdf(pdf_path: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Extract health parameters from a PDF lab report.
    
    This tool parses medical lab reports in PDF format and extracts key health metrics:
    - HbA1c (Glycated Hemoglobin %)
    - Glucose (Blood Glucose mg/dL)
    - BMI (Body Mass Index)
    - Age (years)
    
    Args:
        pdf_path: Path to the PDF lab report file
        use_llm: Whether to use LLM-based extraction (default: True)
    
    Returns:
        Dictionary containing:
        - success: bool - Whether extraction was successful
        - parameters: dict - Extracted health parameters
        - extracted_count: int - Number of parameters found
        - message: str - Status message
    """
    parser = LabReportParser(use_llm=use_llm)
    return parser.parse_pdf(pdf_path)


# Standalone function for direct use
def extract_health_data_from_pdf(pdf_path: str, use_llm: bool = True) -> Dict[str, Optional[float]]:
    """
    Convenience function to extract health data from PDF
    
    Args:
        pdf_path: Path to PDF file
        use_llm: Whether to use LLM extraction
        
    Returns:
        Dictionary of health parameters (hba1c, glucose, bmi, age)
    """
    parser = LabReportParser(use_llm=use_llm)
    result = parser.parse_pdf(pdf_path)
    
    if result["success"]:
        return result["parameters"]
    else:
        raise ValueError(f"Failed to parse PDF: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # Test the parser
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python labparse.py <pdf_file_path>")
        print("\nExample:")
        print("  python labparse.py sample_lab_report.pdf")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    print(f"Parsing lab report: {pdf_file}")
    print("=" * 60)
    
    # Test with regex only (faster)
    print("\n1. Testing with Regex extraction only...")
    parser = LabReportParser(use_llm=False)
    result = parser.parse_pdf(pdf_file)
    print(json.dumps(result, indent=2))
    
    # Test with LLM (requires OpenAI API key)
    print("\n2. Testing with LLM extraction...")
    try:
        parser_llm = LabReportParser(use_llm=True)
        result_llm = parser_llm.parse_pdf(pdf_file)
        print(json.dumps(result_llm, indent=2))
    except Exception as e:
        print(f"LLM extraction skipped: {e}")
