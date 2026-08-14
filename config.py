"""
Central place for environment/config so every module imports the same
LLM instance instead of re-instantiating it.
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# temperature kept low so follow-up/memory tests are reproducible
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)
