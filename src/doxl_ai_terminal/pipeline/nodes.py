# nodes.py
"""
All graph nodes for the LangGraph multi-agent system.

Node Flow (simplified — no LLM path validation, no interrupt node):
  START → onboarding → test_credentials → (fail → onboarding)
                                         → (pass → path_request)
  path_request → process_file → docs_agent  ─┐
                              → excel_agent ─┤
                                             ▼
                                     continue_prompt
                                     ↓             ↓
                               (yes) ↓             ↓ (no)
                               path_request       END
"""

import os

from doxl_ai_terminal.pipeline.state import AgentState
from doxl_ai_terminal.pipeline.config import (
    validate_api_key, SUPPORTED_MODELS, get_agentic_llm,
)
from doxl_ai_terminal.pipeline.terminal_ui import (
    banner, success, error, info, warn, agent_say,
    divider, prompt_input, browse_for_file, Spinner,
    choice_prompt, console,
)
from doxl_ai_terminal.pipeline.credential_store import (
    has_credentials, load_credentials, save_credentials,
)

# Valid extensions this app supports
_VALID_EXTENSIONS = {".docx", ".xlsx"}


def _extract_text(content) -> str:
    """Extract plain text from an LLM response content.

    Handles both plain strings and structured content parts
    (list of dicts with 'type'/'text' keys) returned by newer
    langchain-google-genai versions.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


# ═══════════════════════════════════════════════════
# NODE 1: ONBOARDING
# ═══════════════════════════════════════════════════

def onboarding_node(state: AgentState) -> AgentState:
    """Welcome user and collect Gemini API key + model name.

    If credentials are already stored on disk (~/.docs-excel/config.json),
    loads them automatically and skips the prompts.
    """
    banner()

    # ── Check for stored credentials ──
    if has_credentials():
        creds = load_credentials()
        api_key = creds["api_key"]
        model_name = creds["model"]
        success("Loaded saved credentials.")
        info(f"Model: {model_name}")
        info("To reset credentials, delete ~/.docs-excel/config.json\n")

        return {
            **state,
            "state": "testing",
            "current_agent": "onboarding",
            "api_key": api_key,
            "model": model_name,
            "api_error": "",
            "onboarded": False,
        }

    # ── First-time setup ──
    agent_say("Onboarding Agent", "Welcome! Let's set up your environment.\n")

    # Show supported models as a numbered choice
    idx = choice_prompt("Select a model:", SUPPORTED_MODELS)
    model_name = SUPPORTED_MODELS[idx]
    info(f"Selected model: {model_name}")

    # Ask for API key
    console.print()
    api_key = prompt_input("Enter your Gemini API key: ")

    return {
        **state,
        "state": "testing",
        "current_agent": "onboarding",
        "api_key": api_key,
        "model": model_name,
        "api_error": "",
        "onboarded": False,
    }


# ═══════════════════════════════════════════════════
# NODE 2: TEST CREDENTIALS
# ═══════════════════════════════════════════════════

def test_credentials_node(state: AgentState) -> AgentState:
    """Test Gemini API key and model with a real request.

    On success, saves credentials to disk so they persist across sessions.
    On failure, clears any stored credentials and loops back to onboarding.
    """
    spinner = Spinner(
        "Verifying your API key...",
        done_label=f"Model '{state['model']}' is ready",
    ).start()
    result = validate_api_key(state["api_key"], state["model"])
    spinner.stop(ok=result["success"])

    if result["success"]:
        save_credentials(state["api_key"], state["model"])
        info("Credentials saved to ~/.docs-excel/config.json\n")

        return {
            **state,
            "state": "path_request",
            "current_agent": "test",
            "onboarded": True,
            "api_error": "",
        }
    else:
        error(f"Failed: {result['error']}")
        warn("Clearing credentials. Let's try again.\n")

        from doxl_ai_terminal.pipeline.credential_store import clear_credentials
        clear_credentials()

        return {
            **state,
            "state": "onboarding",
            "current_agent": "test",
            "onboarded": False,
            "api_key": "",
            "model": "",
            "api_error": result["error"],
        }


# ═══════════════════════════════════════════════════
# NODE 3: PATH REQUEST (simple Python validation — no LLM)
# ═══════════════════════════════════════════════════

def _validate_path(path: str) -> tuple[bool, str]:
    """Validate a file path. Returns (ok, message)."""
    if not path:
        return False, "No path entered."

    if not os.path.isabs(path):
        return False, f"Please enter an absolute path (got relative: '{path}')."

    if not os.path.exists(path):
        return False, f"File not found: '{path}'."

    ext = os.path.splitext(path)[1].lower()
    if ext not in _VALID_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Only .docx and .xlsx are supported."

    return True, ""


def path_request_node(state: AgentState) -> AgentState:
    """Ask the user for a file path with simple Python validation.

    No LLM call needed — just validate the path, confirm, and proceed.
    """
    divider()
    agent_say("File Agent", "Paste the path to your file, or press Enter to browse.\n")

    while True:
        user_input = prompt_input("  File path: ")

        if user_input.lower() in ("quit", "exit", "q"):
            return {
                **state,
                "state": "exit",
                "current_agent": "path_agent",
            }

        # Empty input → open a native file picker
        if not user_input:
            browsed = browse_for_file()
            if not browsed:
                warn("No file selected. Paste a path, or press Enter to try again.")
                continue
            user_input = browsed

        # Validate
        ok, msg = _validate_path(user_input)
        if not ok:
            error(msg)
            continue

        # Confirm
        success(f"Found: {user_input}")
        answer = prompt_input("  Process this file? (y/n): ")
        if answer.strip().lower() not in ("y", "yes"):
            info("Ok, enter a different path.\n")
            continue

        return {
            **state,
            "state": "process_file",
            "current_agent": "path_agent",
            "file_path": user_input,
        }


# ═══════════════════════════════════════════════════
# NODE 4: PROCESS FILE — detect type and route
# ═══════════════════════════════════════════════════

def process_file_node(state: AgentState) -> AgentState:
    """Detect the file type and set file_type in state.

    The conditional edge after this node reads file_type
    and routes to docs_agent or excel_agent accordingly.
    """
    file_path = state["file_path"]
    ext = os.path.splitext(file_path)[1].lower()

    divider()

    if ext == ".docx":
        file_type = "docx"
        success(f"Word document detected: {file_path}")
        info("Routing to Document Agent...\n")

    elif ext == ".xlsx":
        file_type = "xlsx"
        success(f"Excel spreadsheet detected: {file_path}")
        info("Routing to Excel Agent...\n")

    else:
        error(f"Unsupported file type: {ext}")
        warn("Going back to file selection.\n")
        return {
            **state,
            "state": "path_request",
            "current_agent": "processor",
            "file_type": "",
        }

    divider()

    return {
        **state,
        "state": "agent",
        "current_agent": "processor",
        "file_type": file_type,
    }


# ═══════════════════════════════════════════════════
# NODE 5: DOCS AGENT
# ═══════════════════════════════════════════════════

def docs_agent_node(state: AgentState) -> AgentState:
    """Run the Document Agent system (LangGraph router + specialist subgraphs)."""
    from doxl_ai_terminal.agents.docs_agent import DocsAgentSystem

    llm = get_agentic_llm(state["api_key"], state["model"])

    try:
        system = DocsAgentSystem(state["file_path"], llm=llm)
        system.chat()
    except Exception as e:
        error(f"Document Agent failed: {e}")
        warn("Returning to continue prompt.\n")

    return {
        **state,
        "state": "continue_prompt",
        "current_agent": "docs_agent",
    }


# ═══════════════════════════════════════════════════
# NODE 6: EXCEL AGENT
# ═══════════════════════════════════════════════════

def excel_agent_node(state: AgentState) -> AgentState:
    """Run the Excel Agent system (LangGraph router + specialist subgraphs)."""
    from doxl_ai_terminal.agents.excel_agent import ExcelAgentSystem

    llm = get_agentic_llm(state["api_key"], state["model"])

    try:
        system = ExcelAgentSystem(state["file_path"], llm=llm)
        system.chat()
    except Exception as e:
        error(f"Excel Agent failed: {e}")
        warn("Returning to continue prompt.\n")

    return {
        **state,
        "state": "continue_prompt",
        "current_agent": "excel_agent",
    }


# ═══════════════════════════════════════════════════
# NODE 7: CONTINUE PROMPT — another file?
# ═══════════════════════════════════════════════════

def continue_node(state: AgentState) -> AgentState:
    """Ask the user if they want to process another file."""
    divider()
    agent_say("Manager", "Session complete!")
    console.print()

    answer = prompt_input("Process another file? (y/n): ")
    wants = answer.strip().lower() in ("yes", "y")

    if wants:
        info("Returning to file selection...\n")
    else:
        success("Thank you for using DOXL AI. Goodbye!\n")

    return {
        **state,
        "state": "path_request" if wants else "exit",
        "current_agent": "continue_handler",
        "wants_continue": wants,
        "file_path": "",
        "file_type": "",
    }


# ═══════════════════════════════════════════════════
# ROUTING FUNCTIONS (conditional edges)
# ═══════════════════════════════════════════════════

def route_after_test(state: AgentState) -> str:
    """Route after credential testing: pass → path_request, fail → onboarding."""
    return "path_request" if state["onboarded"] else "onboarding"


def route_after_process(state: AgentState) -> str:
    """Route after process_file: based on file_type → docs_agent or excel_agent."""
    ft = state.get("file_type", "")
    if ft == "docx":
        return "docs_agent"
    elif ft == "xlsx":
        return "excel_agent"
    return "path_request"


def route_after_continue(state: AgentState) -> str:
    """Route after continue_prompt: yes → path_request, no → end."""
    return "path_request" if state.get("wants_continue", False) else "end"
