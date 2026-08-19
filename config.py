"""
Centralized configuration for the Multi-Agent Research System.
Loads environment variables and initializes shared resources.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 3

MAX_REVISION_LOOPS = 2          # Writer ↔ Critic revision cap
RESULTS_PER_QUERY = 3           # Top N results from each search source
MIN_QUALITY_SCORE = 7           # Critic threshold for acceptance

SUBTOPIC_MIN = 4
SUBTOPIC_MAX = 6

# ---------------------------------------------------------------------------
# LLM Initialization
# ---------------------------------------------------------------------------

def get_llm(temperature: float | None = None) -> ChatGroq:
    """Return a configured ChatGroq instance."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please add it to your .env file."
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
        api_key=GROQ_API_KEY,
    )
