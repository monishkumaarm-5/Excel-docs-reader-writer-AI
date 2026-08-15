# nodes.py
"""
All graph nodes for the LangGraph multi-agent system.

Node Flow:
  START → onboarding → test_credentials → (fail → onboarding)
                                         → (pass → path_request)
  path_request → interrupt → (rejected → path_request)
                            → (confirmed → process_file)
  process_file → route → docs_agent  ─┐
                       → excel_agent ─┤
                                      ▼
                              continue_prompt
                              ↓             ↓
                        (yes) ↓             ↓ (no)
                        path_request       END
"""

import os
from langgraph.types import interrupt
from doxl_ai_terminal.pipeline.state import AgentState
from doxl_ai_terminal.pipeline.config import (
    validate_api_key, SUPPORTED_MODELS, get_llm, get_crewai_llm,
)
from doxl_ai_terminal.pipeline.terminal_ui import (
    banner, success, error, info, warn, agent_say,
    divider, prompt_input, interrupt_prompt,
)
from doxl_ai_terminal.pipeline.credential_store import (
    has_credentials, load_credentials, save_credentials,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


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

    # Show supported models
    info("Supported models:")
    for i, m in enumerate(SUPPORTED_MODELS, 1):
        print(f"   {i}. {m}")
    print()

    # Ask for API key
    api_key = prompt_input("Enter your Gemini API key: ")

    # Ask for model
    model_input = prompt_input("Enter model name (or number from list above): ")

    # Handle number input
    if model_input.isdigit():
        idx = int(model_input) - 1
        if 0 <= idx < len(SUPPORTED_MODELS):
            model_name = SUPPORTED_MODELS[idx]
        else:
            model_name = model_input
    else:
        model_name = model_input.strip()

    info(f"Selected model: {model_name}")

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

    agent_say("Test Agent", "Testing your credentials...")

    result = validate_api_key(state["api_key"], state["model"])

    if result["success"]:
        success(f"Connection successful! Response: {result['response']}")
        success(f"Model '{state['model']}' is ready.\n")

        # Persist credentials to disk for future sessions
        save_credentials(state["api_key"], state["model"])
        info("Credentials saved to ~/.docs-excel/config.json")

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

        # Clear any bad stored credentials
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
# NODE 3: PATH REQUEST (with LangChain + Memory)
# ═══════════════════════════════════════════════════

PATH_AGENT_PROMPT = """You are a File Path Assistant. Your ONLY job is to get a valid file path from the user.

Rules:
1. Always ask the user for a file path if not provided.
2. The path must be an absolute path (starts with / on Linux or drive letter on Windows like C:\\).
3. The file must exist on disk.
4. Only accept .docx or .xlsx files.
5. If the path is invalid, explain WHY and ask again.
6. If the path is valid, respond with exactly: PATH_VALID:<the_path>
7. Never proceed without a valid path.
8. Be helpful and patient.

Examples of valid paths:
- /home/user/documents/report.docx
- C:\\Users\\Mohan\\data.xlsx
- /tmp/files/notes.docx

If user gives a relative path like "report.docx", ask for the full absolute path.
"""


def path_request_node(state: AgentState) -> AgentState:
    """
    Agent that requests and validates file path.
    Uses LangChain with memory (chat history in state).
    """

    agent_say("Path Agent", "I need the path to your file.\n")

    llm = get_llm(state["api_key"], state["model"])

    # Build message history
    messages = [SystemMessage(content=PATH_AGENT_PROMPT)]

    # Add existing chat history
    for msg in state.get("messages", []):
        messages.append(msg)

    # If no prior messages, agent starts the conversation
    if not state.get("messages", []):
        initial = llm.invoke(messages + [
            HumanMessage(content="I want to process a file.")
        ])
        initial_text = _extract_text(initial.content)
        agent_say("Path Agent", initial_text)

        messages.append(HumanMessage(content="I want to process a file."))
        messages.append(AIMessage(content=initial_text))

    # Conversation loop with the agent
    while True:
        user_input = prompt_input("\nYou: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            return {
                **state,
                "state": "exit",
                "interrupt": False,
            }

        messages.append(HumanMessage(content=user_input))

        # Let the LLM respond
        response = llm.invoke(messages)
        response_text = _extract_text(response.content)

        messages.append(AIMessage(content=response_text))

        # Check if LLM validated the path
        if "PATH_VALID:" in response_text:
            # Extract path from response
            path = response_text.split("PATH_VALID:")[1].strip()
            path = path.strip('"').strip("'").strip("`")

            # Double-check on our end
            if os.path.exists(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in [".docx", ".xlsx"]:
                    success(f"Valid path: {path}")

                    return {
                        **state,
                        "state": "process_file",
                        "current_agent": "path_agent",
                        "file_path": path,
                        "messages": messages[1:],  # exclude system prompt
                    }

            # If our check fails, tell agent
            error(f"Path verification failed: {path}")
            correction = f"The path '{path}' does not exist or is not a .docx/.xlsx file. Ask the user again."
            messages.append(HumanMessage(content=correction))
            response2 = llm.invoke(messages)
            response2_text = _extract_text(response2.content)
            messages.append(AIMessage(content=response2_text))
            agent_say("Path Agent", response2_text)
            continue

        # Normal response from agent
        agent_say("Path Agent", response_text)


# ═══════════════════════════════════════════════════
# NODE 4: HUMAN-IN-THE-LOOP INTERRUPT (LangGraph native)
# ═══════════════════════════════════════════════════

def interrupt_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop interrupt using LangGraph's native interrupt() API.

    When this node runs, it pauses the graph and sends the interrupt
    payload back to the caller. The graph resumes when the caller
    invokes `graph.invoke(Command(resume=answer), config)`.
    """

    file_path = state.get("file_path", "")

    # --- Pause the graph here and wait for human input ---
    answer = interrupt({
        "question": f"Do you want to process this file? → {file_path}",
        "file_path": file_path,
        "options": "yes / no",
    })

    # --- Graph resumes here with the user's answer ---
    confirmed = answer.strip().lower() in ["yes", "y", "1", "true", "confirm"]

    if confirmed:
        success(f"User confirmed: {file_path}")
    else:
        warn(f"User rejected. Going back to path request.")

    return {
        **state,
        "interrupt": False,
        "interrupt_answer": answer,
        "confirmed": confirmed,
        "current_agent": "interrupt_handler",
    }


# ═══════════════════════════════════════════════════
# NODE 5: PROCESS FILE — detect type and route
# ═══════════════════════════════════════════════════

def process_file_node(state: AgentState) -> AgentState:
    """
    Detect the file type and set file_type in state.

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
        # Should not happen (path_request validates), but just in case
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
# NODE 6: DOCS AGENT (CrewAI multi-agent for .docx)
# ═══════════════════════════════════════════════════

def docs_agent_node(state: AgentState) -> AgentState:
    """
    Run the CrewAI Document Agent system.

    This node:
      1. Creates a DocsAgentSystem (reads, chunks, vectorizes the file)
      2. Runs the interactive chatbot loop
      3. When user types 'quit', control returns here
      4. Graph continues to continue_prompt
    """
    from doxl_ai_terminal.agents.docs_agent import DocsAgentSystem

    llm = get_crewai_llm(state["api_key"], state["model"])

    try:
        system = DocsAgentSystem(state["file_path"], llm=llm)
        system.chat()  # ← blocks until user types "quit"

    except Exception as e:
        error(f"Document Agent failed: {e}")
        warn("Returning to continue prompt.\n")

    return {
        **state,
        "state": "continue_prompt",
        "current_agent": "docs_agent",
    }


# ═══════════════════════════════════════════════════
# NODE 7: EXCEL AGENT (CrewAI multi-agent for .xlsx)
# ═══════════════════════════════════════════════════

def excel_agent_node(state: AgentState) -> AgentState:
    """
    Run the CrewAI Excel Agent system.

    This node:
      1. Creates an ExcelAgentSystem (reads, chunks, vectorizes the file)
      2. Runs the interactive chatbot loop
      3. When user types 'quit', control returns here
      4. Graph continues to continue_prompt
    """
    from doxl_ai_terminal.agents.excel_agent import ExcelAgentSystem

    llm = get_crewai_llm(state["api_key"], state["model"])

    try:
        system = ExcelAgentSystem(state["file_path"], llm=llm)
        system.chat()  # ← blocks until user types "quit"

    except Exception as e:
        error(f"Excel Agent failed: {e}")
        warn("Returning to continue prompt.\n")

    return {
        **state,
        "state": "continue_prompt",
        "current_agent": "excel_agent",
    }


# ═══════════════════════════════════════════════════
# NODE 8: CONTINUE PROMPT — another file?
# ═══════════════════════════════════════════════════

def continue_node(state: AgentState) -> AgentState:
    """
    Ask the user if they want to process another file.

    yes → loops back to path_request (file selection)
    no  → graph terminates at END
    """
    divider()
    agent_say("Manager", "Session complete!")
    print()

    answer = prompt_input("Process another file? (yes / no): ")
    wants = answer.strip().lower() in ["yes", "y"]

    if wants:
        info("Returning to file selection...\n")
    else:
        success("Thank you for using DOXL AI. Goodbye!\n")

    return {
        **state,
        "state": "path_request" if wants else "exit",
        "current_agent": "continue_handler",
        "wants_continue": wants,
        # Reset file-specific state for the next round
        "file_path": "",
        "file_type": "",
        "confirmed": False,
    }


# ═══════════════════════════════════════════════════
# ROUTING FUNCTIONS (conditional edges)
# ═══════════════════════════════════════════════════

def route_after_test(state: AgentState) -> str:
    """Route after credential testing: pass → path_request, fail → onboarding"""
    if state["onboarded"]:
        return "path_request"
    return "onboarding"


def route_after_interrupt(state: AgentState) -> str:
    """Route after interrupt: confirmed → process_file, rejected → path_request"""
    if state.get("confirmed", False):
        return "process_file"
    return "path_request"


def route_after_process(state: AgentState) -> str:
    """Route after process_file: based on file_type → docs_agent or excel_agent"""
    file_type = state.get("file_type", "")
    if file_type == "docx":
        return "docs_agent"
    elif file_type == "xlsx":
        return "excel_agent"
    # Fallback (unsupported type) → ask for path again
    return "path_request"


def route_after_continue(state: AgentState) -> str:
    """Route after continue_prompt: yes → path_request, no → end"""
    if state.get("wants_continue", False):
        return "path_request"
    return "end"