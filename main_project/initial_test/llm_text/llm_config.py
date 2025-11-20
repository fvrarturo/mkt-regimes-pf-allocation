"""
LLM Configuration - Supports multiple free and paid options.

Options (priority order):
1. Ollama (local/cluster) - Set USE_OLLAMA=true (FREE, no API limits)
2. Groq (FREE fast Llama API) - Set GROQ_API_KEY or GROQ_API_KEYS (comma-separated for rotation)
3. OpenAI API (paid) - Set OPENAI_API_KEY

Priority: Ollama > Groq > OpenAI
"""
import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from typing import List, Optional

# Check which service to use (priority order)
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

# Support multiple Groq API keys for rotation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEYS_STR = os.getenv("GROQ_API_KEYS", "")
if GROQ_API_KEYS_STR:
    # Parse comma-separated keys
    GROQ_API_KEYS: List[str] = [k.strip() for k in GROQ_API_KEYS_STR.split(",") if k.strip()]
elif GROQ_API_KEY:
    # Single key provided
    GROQ_API_KEYS = [GROQ_API_KEY]
else:
    GROQ_API_KEYS = []

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Track current API key index for rotation
_current_groq_key_index = 0

def get_groq_llm_model(api_key: str) -> OpenAIChatCompletionsModel:
    """Create a Groq LLM model with the given API key."""
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        ),
    )


def get_next_groq_api_key() -> Optional[str]:
    """Get the next available Groq API key, rotating through available keys."""
    global _current_groq_key_index
    if not GROQ_API_KEYS:
        return None
    if _current_groq_key_index >= len(GROQ_API_KEYS):
        return None
    key = GROQ_API_KEYS[_current_groq_key_index]
    return key


def rotate_to_next_groq_key() -> Optional[str]:
    """Rotate to the next Groq API key. Returns the new key or None if all exhausted."""
    global _current_groq_key_index
    _current_groq_key_index += 1
    if _current_groq_key_index >= len(GROQ_API_KEYS):
        return None
    print(f"🔄 Rotating to Groq API key {_current_groq_key_index + 1}/{len(GROQ_API_KEYS)}")
    return GROQ_API_KEYS[_current_groq_key_index]


def get_current_groq_key_index() -> int:
    """Get the current API key index."""
    return _current_groq_key_index


def are_all_groq_keys_exhausted() -> bool:
    """Check if all Groq API keys have been exhausted."""
    return _current_groq_key_index >= len(GROQ_API_KEYS)


if USE_OLLAMA:
    # Option 1: Ollama (local/cluster - FREE, no API limits)
    print("Using Ollama for LLM")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = OpenAIChatCompletionsModel(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        openai_client=AsyncOpenAI(
            base_url=ollama_base_url,
            api_key="ollama",
        ),
    )
elif GROQ_API_KEYS:
    # Option 2: Groq (FREE, very fast Llama inference) with multiple keys support
    print(f"Using Groq (FREE fast Llama API) for LLM with {len(GROQ_API_KEYS)} API key(s)")
    print(f"Starting with API key 1/{len(GROQ_API_KEYS)}")
    LLM_MODEL = get_groq_llm_model(GROQ_API_KEYS[0])
elif OPENAI_API_KEY:
    # Option 3: OpenAI API (paid, but reliable)
    print("Using OpenAI API for LLM")
    LLM_MODEL = OpenAIChatCompletionsModel(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY),
    )
else:
    raise ValueError(
        "No LLM configured!\n"
        "FREE options:\n"
        "  1. Ollama (recommended): export USE_OLLAMA=true (install Ollama on cluster)\n"
        "  2. Groq: export GROQ_API_KEY='your-key' or GROQ_API_KEYS='key1,key2,key3' (get free keys at https://console.groq.com)\n"
        "Paid option:\n"
        "  3. OpenAI: export OPENAI_API_KEY='sk-your-key'"
    )

