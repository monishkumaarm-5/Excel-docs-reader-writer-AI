#config.py
"""
Gemini API configuration and validation.
"""

import google.genai as genai

# Supported models
SUPPORTED_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def validate_api_key(api_key: str, model_name: str) -> dict:
    """
    Test Gemini API key and model with a simple payload.

    Returns:
        {"success": True, "response": "..."} on success
        {"success": False, "error": "..."} on failure
    """
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents="Reply with: CONNECTION_OK",
        )

        return {
            "success": True,
            "response": response.text.strip(),
        }

    except Exception as e:
        error_msg = str(e)

        if "API_KEY_INVALID" in error_msg or "401" in error_msg:
            return {"success": False, "error": "Invalid API key. Please check and try again."}

        elif "404" in error_msg or "not found" in error_msg.lower():
            return {"success": False, "error": f"Model '{model_name}' not found. Check the model name."}

        elif "429" in error_msg:
            return {"success": False, "error": "Rate limit hit. Wait a moment and try again."}

        else:
            return {"success": False, "error": f"Connection failed: {error_msg}"}


def get_llm(api_key: str, model_name: str):
    """Create a LangChain ChatGoogleGenerativeAI instance"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )


def get_crewai_llm(api_key: str, model_name: str):
    """Create a CrewAI-native LLM instance for Gemini.

    CrewAI routes every model call through litellm, which needs the
    provider prefix ("gemini/<model>") to know to call Google's
    Gemini API and which credential to use. Passing a bare model
    name (e.g. "gemini-3.5-flash-lite") makes litellm fall back to
    the OpenAI provider, which fails immediately with an
    "LLM Failed" / "AuthenticationError" since no OpenAI key is
    configured. It's also important to pass a real crewai.LLM (or a
    plain model string) rather than a LangChain chat model instance
    — CrewAI's Agent does not know how to drive a LangChain object.

    max_tokens is set generously on purpose. The 3.x Gemini flash
    models do internal "thinking" before writing the visible answer,
    and that thinking draws from the SAME output-token budget as the
    answer itself. With no max_tokens set, litellm/Gemini falls back
    to a small default budget — thinking alone can burn through all of
    it on anything non-trivial (e.g. "calculate the average per
    subject"), leaving zero tokens for the actual reply. That's what
    "Received None or empty response from LLM call" means: the call
    succeeded, but the model's visible answer was empty because
    thinking ate the whole budget. Giving it real headroom fixes that.
    """
    from crewai import LLM

    return LLM(
        model=f"gemini/{model_name}",
        api_key=api_key,
        temperature=0,
        max_tokens=8192,
    )

def get_model_name():
    """Create a LangChain ChatGoogleGenerativeAI instance"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return