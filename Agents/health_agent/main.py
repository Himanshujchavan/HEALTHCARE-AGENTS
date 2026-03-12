from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agent import build_agent
from dotenv import load_dotenv
import uuid
import re

load_dotenv()

app = FastAPI(title='Health Risk & Symptom Checker Agent')

# Session store — replace with Redis in production
sessions: dict = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    risk_level: str | None = None


def parse_risk_level(text: str) -> str | None:
    """Extract risk level from agent response text."""
    patterns = [
        r'Level:\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'risk level is\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'risk level[:\s]+(CRITICAL|HIGH|MEDIUM|LOW)',
        r'assessed as\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'risk is\s*(CRITICAL|HIGH|MEDIUM|LOW)',
        r'\b(CRITICAL|HIGH|MEDIUM|LOW)\s*risk\b',
        r'\b(CRITICAL|HIGH|MEDIUM|LOW)\b(?=\s*[-])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Fix: treat string "null" same as actual null
    raw_id = request.session_id
    if not raw_id or raw_id.lower() == 'null':
        session_id = str(uuid.uuid4())
    else:
        session_id = raw_id

    if session_id not in sessions:
        sessions[session_id] = build_agent()

    agent = sessions[session_id]

    try:
        result = agent.invoke({'input': request.message})
        output = result['output']
        risk_level = parse_risk_level(output)
        return ChatResponse(
            session_id=session_id,
            response=output,
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health_check():
    return {'status': 'ok'}