from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model='gpt-4o', temperature=0.1)

risk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical risk assessment assistant.
    Given patient data including symptoms, age, severity, duration, and comorbidities,
    provide a brief risk summary explaining the key risk factors present.
    Be concise and factual. Do not diagnose."""),
    ("human", "Patient data: {patient_data}\nRisk score result: {risk_score}")
])

risk_chain = risk_prompt | llm | StrOutputParser() 