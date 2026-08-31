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


def get_agentic_llm(api_key: str, model_name: str):
    """Create a LangChain ChatGoogleGenerativeAI tuned for an agentic
    tool-calling loop (the docs_agent/excel_agent specialist subgraphs).

    This carries forward tuning that used to live in get_crewai_llm
    (removed when docs_agent.py/excel_agent.py moved from CrewAI to
    LangGraph) — the underlying Gemini behavior it works around is the
    model's, not the framework's, so the same fix is still needed.

    max_output_tokens is set generously on purpose. The 3.x Gemini
    flash models do internal "thinking" before writing the visible
    answer, and that thinking draws from the SAME output-token budget
    as the answer itself. With the SDK default budget, thinking alone
    can burn through all of it on anything non-trivial (e.g.
    "calculate the average per subject"), leaving zero tokens for the
    actual reply — the call succeeds but the model's visible answer
    comes back empty because thinking ate the whole budget. Giving it
    real headroom fixes that.

    temperature is intentionally NOT 0. Greedy (temperature=0) decoding
    has no way to escape once it starts repeating a token/phrase — on a
    broad multi-line rewrite task, the model can lock into a loop like
    "gRPC, GraphQL, gRPC, GraphQL, ..." and burn the entire output
    budget on that repetition, never reaching a real tool call. That's
    a real failure this project hit: an Updater agent asked to
    refactor a paragraph produced a wall of repeated tokens as its
    final answer and never called update_line/replace_all, so the file
    was never actually written even though the agent reported success.
    A small amount of randomness (temperature=0.2) gives generation a
    way out of that trap while staying low enough that tool-call JSON
    stays reliable.

    Plain get_llm() above stays at temperature=0 on purpose — it's
    used for the single-turn, no-tools path_request_node conversation,
    where deterministic PATH_VALID:<path> parsing matters more than
    avoiding the repetition-loop failure mode above.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=8192,
    )

def get_model_name():
    """Create a LangChain ChatGoogleGenerativeAI instance"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return