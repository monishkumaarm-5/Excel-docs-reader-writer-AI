# config.py
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
    """Test Gemini API key and model with a simple payload.

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
        return {"success": True, "response": response.text.strip()}

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
    """Create a LangChain ChatGoogleGenerativeAI instance for simple tasks."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )


def get_agentic_llm(api_key: str, model_name: str):
    """Create a LangChain ChatGoogleGenerativeAI tuned for agentic tool-calling.

    max_output_tokens is set generously because Gemini 3.x flash models do
    internal "thinking" that draws from the same output-token budget as the
    visible answer.  With the SDK default, thinking can burn through all of
    it, leaving zero tokens for the actual reply.

    temperature is 0.2 (not 0) to avoid the repetition-loop trap where
    greedy decoding locks into repeating a phrase and never reaches a tool
    call.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=8192,
    )
