import os
import logging
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.3) -> BaseChatModel:
    """
    Returns an LLM instance based on configuration.
    Primary: Groq API (llama-3.3-70b-versatile or specified model)
    Fallback / Alternative: Ollama (local llama3)
    
    To force Ollama, set LLM_PROVIDER=ollama in .env.
    To use Groq (default), set GROQ_API_KEY in .env.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Explicitly requested Ollama
    if provider == "ollama":
        return _get_ollama_llm(temperature)

    # Primary: Use Groq if API key is provided or provider is groq
    if groq_api_key or provider == "groq":
        try:
            from langchain_groq import ChatGroq
            groq_model = os.getenv("GROQ_MODEL", "groq/compound-mini")
            logger.info(f"Using Groq LLM (model: {groq_model})")
            return ChatGroq(
                model_name=groq_model,
                api_key=groq_api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Groq LLM ({e}). Falling back to Ollama.")

    # Fallback to Ollama
    return _get_ollama_llm(temperature)


def _get_ollama_llm(temperature: float = 0.3) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.info(f"Using Ollama LLM (model: {ollama_model} at {ollama_base_url})")
    return ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=temperature,
    )
