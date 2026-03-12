import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent.tools import assess_risk, triage_route, add_disclaimer
from agent.memory import create_memory
from langchain.agents import AgentExecutor, create_tool_calling_agent
def build_agent():
    SYSTEM_PROMPT = """You are a compassionate and careful AI health assistant.
You help users understand their symptoms and assess potential health risks.

WORKFLOW - Always follow this exact sequence:
1. Gather ALL symptoms from the user via conversation.
2. Collect patient context: age, symptom duration, severity (1-10), comorbidities.
3. Use symptom_lookup to find matching conditions from the knowledge base.
4. Use assess_risk with the collected patient data to determine risk level.
5. Use triage_route to recommend the appropriate care pathway.
6. Always end with add_disclaimer on your final response.

SAFETY RULES:
- NEVER diagnose. You identify possible conditions, not definitive diagnoses.
- If severity >= 9 or user reports chest pain / difficulty breathing, immediately
  recommend emergency services before any further analysis.
- Always ask clarifying questions if data is incomplete.
- Never contradict previous medical advice the user has received."""
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1)

    tools = [assess_risk, triage_route, add_disclaimer]

    prompt = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    memory = create_memory()

    executor = AgentExecutor( 
        agent=agent, 
        tools=tools,
        memory=memory,
        verbose=True,           
        max_iterations=8,
        handle_parsing_errors=True
    )
    return executor