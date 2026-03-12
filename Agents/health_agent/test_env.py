from dotenv import load_dotenv 
import os
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", key[:10], "...")