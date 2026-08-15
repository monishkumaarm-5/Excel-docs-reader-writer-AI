# docs_agent.py
"""
CrewAI Multi-Agent Document Handler
=====================================

Architecture:
    Manager Agent (routes tasks)
        ├── Reader Agent   → exact search, RAG semantic search, sentence extract
        ├── Writer Agent   → add new lines to document
        ├── Updater Agent  → modify existing lines, find-replace
        └── Deleter Agent  → remove lines from document

Flow:
    1. User gives filepath → file is read, chunked, vectorized
    2. Chatbot loop starts
    3. User types a request
    4. CrewAI Manager delegates to the right specialist
    5. Specialist uses tools → result returned
    6. Loop continues until user quits

Usage:
    python docs_agent.py report.docx
"""

import os
import sys
from typing import Optional, List, Dict

from crewai import Agent, Task, Crew, Process
from langchain_core.tools import tool as langchain_tool

# ---- Your existing modules ----
from doxl_ai_terminal.Frontier.ChangeLogManager_docs import LiveDocManager
from doxl_ai_terminal.Frontier.fileReader import read_word
from doxl_ai_terminal.Chunker.chunking import chunk_doc_sub_para, chunk_doc_line_by_line
from doxl_ai_terminal.data_handler.vector_config import store_in_chroma, make_collection_name
from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager
from doxl_ai_terminal.data_structure.full_sentence_extractor import (
    extract_full_sentences,
    extract_from_retrieved_chunk,
    extract_with_context,
)


# ================================================================
# SECTION 1: SHARED STATE
# ================================================================
# All tools read from this dict to access the loaded document.
# Same pattern as your existing manager_tools.py

_state: Dict = {
    "doc_manager": None,        # LiveDocManager  (edit + auto-save)
    "doc_data": None,           # Raw DocData     (for sentence extraction)
    "vector_managers": {},      # {format_name: VectorDBManager} — one per chunking strategy
    "collections": [],          # Collection names created
    "filepath": None,           # Original file path
    "_dirty": False,            # True when a write happened but the vector index hasn't caught up yet
}


# ================================================================
# SECTION 2: FILE PROCESSOR
# ================================================================
# Mimics your pipeliner.py — read → chunk → vector store
# PLUS loads LiveDocManager for live editing

def process_file(filepath: str):
    """
    Full pipeline: Read the .docx → Chunk it → Store in vector DB → Load editor.

    This is what happens ONCE when the user gives a filepath.
    After this, all tools can operate on the loaded document.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".docx":
        raise ValueError(f"Only .docx files supported. Got: {ext}")

    _state["filepath"] = filepath

    # ---- Step 1: Read the document into DocData ----
    print(f"\n  [1/4] Reading: {filepath}")
    doc_data = read_word(filepath)
    _state["doc_data"] = doc_data

    # ---- Step 2: Load LiveDocManager for editing ----
    print(f"  [2/4] Loading editor...")
    _state["doc_manager"] = LiveDocManager(filepath)

    # ---- Steps 3-4: Chunk + store in vector DB + load for search ----
    print(f"  [3/4] Chunking and vectorizing...")
    _rebuild_doc_vector_index(filepath, doc_data)
    print(f"  [4/4] Vector search ready ({len(_state['vector_managers'])} collection(s)).")

    print(f"\n  Ready! {doc_data.total_lines} lines, "
          f"{doc_data.total_paragraphs} paragraphs loaded.\n")


# ================================================================
# SECTION 2B: RE-INDEX — keep the vector store in sync with writes
# ================================================================

CHUNK_STRATEGIES = [
    ("line_by_line", chunk_doc_line_by_line),
    ("sub_para", chunk_doc_sub_para),
]


def _rebuild_doc_vector_index(filepath: str, doc_data) -> None:
    """
    (Re)build every chunking strategy's Chroma collection for this
    file and load ALL of them into _state["vector_managers"], keyed
    by format name (line_by_line / sub_para) — not just one default.
    Previously only sub_para was ever loaded, so every line_by_line
    chunk got embedded and then never searched.

    The collection name is derived from the file's full path (see
    make_collection_name), so two different files that happen to
    share a filename never collide. Any existing collection for this
    exact file+format is deleted before rebuilding, so re-embedding
    (a fresh open OR a post-write reindex) replaces stale chunks
    instead of piling duplicates on top of them.
    """
    db = VectorDBManager()
    collections = []
    vector_managers = {}

    for format_name, chunk_fn in CHUNK_STRATEGIES:
        chunks = chunk_fn(doc_data)
        if not chunks:
            continue

        collection_name = make_collection_name(filepath, format_name)
        try:
            db.delete_collection(collection_name)
        except Exception:
            pass  # nothing to delete yet, e.g. first time this file is opened

        store_in_chroma(chunks, collection_name)
        collections.append(collection_name)
        vector_managers[format_name] = VectorDBManager().load_collection(collection_name)
        print(f"         {len(chunks)} chunks -> '{collection_name}'")

    _state["collections"] = collections
    _state["vector_managers"] = vector_managers


def _refresh_doc_data() -> None:
    """
    Re-read the (now auto-saved) file from disk into _state["doc_data"].

    This is the CHEAP half of keeping things in sync — just re-parsing
    the .docx with python-docx — so find_full_sentence/find_with_context
    always see the current document, even mid-request, before the
    (much more expensive) vector re-embed below has run.
    """
    filepath = _state.get("filepath")
    if not filepath:
        return
    _state["doc_data"] = read_word(filepath)
    _state["_dirty"] = True  # vector index now needs rebuilding


def mark_doc_dirty_and_refresh() -> None:
    """Called by every mutating tool right after it changes the document."""
    try:
        _refresh_doc_data()
    except Exception as e:
        print(f"  [warn] Refreshing document data failed: {e}")


def rebuild_doc_vector_index_if_dirty() -> None:
    """
    Rebuild the (expensive) Chroma vector index, but ONLY if something
    actually changed since the last rebuild — and only once per user
    request rather than once per tool call.

    Chunking + re-embedding every collection is the slow part of a
    write (re-chunk the whole document, run it through sentence-transformers,
    rewrite Chroma). Doing that after every single add_line/update_line/
    delete_line call — which is what earlier versions did — meant a
    compound request like "find every mention of X and delete those
    lines" paid that cost once per line deleted. Calling this once,
    right after the crew finishes the whole request, keeps vector_search
    accurate for the user's NEXT message while paying the expensive
    part exactly once per turn instead of once per edit.
    """
    if not _state.get("_dirty"):
        return
    filepath = _state.get("filepath")
    doc_data = _state.get("doc_data")
    if not filepath or doc_data is None:
        return
    try:
        _rebuild_doc_vector_index(filepath, doc_data)
        _state["_dirty"] = False
    except Exception as e:
        print(f"  [warn] Vector index refresh failed: {e}")


# ================================================================
# SECTION 3: TOOLS — READER
# ================================================================
# These tools let the Reader Agent search the document. There is
# deliberately NO "view everything" tool here — see the note below.
# Each tool is a LangChain @tool (CrewAI accepts these directly).

# ─────────────────────────────────────────────────────────────────
# DISABLED: view_document used to dump every line in the document
# into one giant string. For a 100-page report that's tens of
# thousands of tokens handed to the LLM in one shot — it blows the
# context window and is painfully slow. Use search_document (exact
# keyword) or vector_search (RAG semantic search) instead: both
# return only the handful of matching lines/chunks that are actually
# relevant, so they scale to a document of any size.
#
# @langchain_tool
# def view_document(section: str = "all") -> str:
#     """View all lines in the loaded Word document.
#     Shows paragraph numbers (P) and line numbers (L) for each line.
#     Pass section='all' to see everything."""
#     mgr = _state["doc_manager"]
#     if not mgr:
#         return "ERROR: No document loaded."
#     return mgr.view()
# ─────────────────────────────────────────────────────────────────


@langchain_tool
def search_document(keyword: str) -> str:
    """Search for an exact keyword in the Word document.
    Returns all lines containing that keyword with their P and L positions.
    Use this for exact text matching."""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    results = mgr.search(keyword)
    if not results:
        return f"No matches found for '{keyword}'."
    lines = [f"P{r.paragraph} L{r.line}: {r.line_str}" for r in results]
    return "\n".join(lines)


# Chroma's default distance space is squared L2. Because embeddings are
# normalized (unit vectors), L2^2 = 2 - 2*cosine_similarity, so ranking
# by L2^2 ascending is equivalent to ranking by cosine similarity
# descending. This is that L2^2 cutoff (~ cosine similarity >= 0.35) —
# a reasonable starting default so a query with no real match in the
# document returns "no matches" instead of forcing back 4 unrelated
# chunks. Tune it if results feel too strict/loose for your data.
MAX_RELEVANT_DISTANCE = 1.3


@langchain_tool
def vector_search(query: str) -> str:
    """Semantic search — find content by MEANING, not just exact words.
    Searches across every chunking strategy for this file (line-by-line
    AND sub-paragraph) and merges the results, so nothing is missed
    just because it only surfaces well in one view of the document.
    Example: searching 'money' also finds 'revenue', 'cost', 'budget'.
    Returns the top matches — or says so plainly if nothing in the
    document is actually relevant."""
    managers = _state.get("vector_managers") or {}
    if not managers:
        return "ERROR: Vector store not loaded."

    candidates = []  # list of (Document, distance)
    for format_name, vm in managers.items():
        try:
            candidates.extend(vm.search_with_scores(query, k=4))
        except Exception as e:
            return f"Search error ({format_name}): {e}"

    # Keep only genuinely relevant hits, then take the best 4 overall
    # across BOTH chunking strategies (lower distance = more similar).
    relevant = [pair for pair in candidates if pair[1] <= MAX_RELEVANT_DISTANCE]
    relevant.sort(key=lambda pair: pair[1])
    top = relevant[:4]

    if not top:
        return f"No semantic matches for '{query}'."

    output = []
    for i, (doc, score) in enumerate(top, 1):
        para = doc.metadata.get("paragraph", "?")
        fmt = doc.metadata.get("format", "?")
        output.append(f"[{i}] (Paragraph {para}, {fmt}) {doc.page_content}")
    return "\n".join(output)


@langchain_tool
def find_full_sentence(query: str) -> str:
    """Find complete sentences that contain the query text.
    Unlike search_document which returns raw lines, this returns
    full sentences from start to full stop."""
    doc_data = _state["doc_data"]
    if not doc_data:
        return "ERROR: No document loaded."
    results = extract_full_sentences(doc_data, query)
    if not results:
        return f"No sentences containing '{query}'."
    output = []
    for r in results:
        output.append(f"P{r.paragraph} Sentence {r.sentence_index}: {r.sentence}")
    return "\n".join(output)


@langchain_tool
def find_with_context(query: str) -> str:
    """Find sentences matching the query WITH surrounding context.
    Returns: sentence before → matched sentence → sentence after.
    Useful for understanding the full context of a match."""
    doc_data = _state["doc_data"]
    if not doc_data:
        return "ERROR: No document loaded."
    results = extract_with_context(doc_data, query, context_window=1)
    if not results:
        return f"No context found for '{query}'."
    output = []
    for r in results:
        output.append(f"--- Paragraph {r['paragraph']} ---")
        if r["before"]:
            output.append(f"  Before: {' '.join(r['before'])}")
        output.append(f"  >>> {r['matched_sentence']}")
        if r["after"]:
            output.append(f"  After:  {' '.join(r['after'])}")
        output.append("")
    return "\n".join(output)


# ================================================================
# SECTION 4: TOOLS — WRITER
# ================================================================

@langchain_tool
def add_line(paragraph: int, line: int, text: str) -> str:
    """Add a new line to the document.
    Args:
        paragraph: which paragraph to add to (use search_document or
            vector_search to find the right paragraph/line numbers)
        line: line number within that paragraph
        text: the text content to add
    Auto-saves to file after adding."""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    result = mgr.add_line(paragraph, line, text)
    mark_doc_dirty_and_refresh()
    return result


# ================================================================
# SECTION 5: TOOLS — UPDATER
# ================================================================

@langchain_tool
def update_line(paragraph: int, line: int, new_text: str) -> str:
    """Update an existing line in the document.
    Args:
        paragraph: paragraph number of the line to update
        line: line number within that paragraph
        new_text: the new text to replace the old text
    IMPORTANT: Use search_document or vector_search first to find the exact P and L numbers.
    Auto-saves to file after updating."""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    result = mgr.update_line(paragraph, line, new_text)
    mark_doc_dirty_and_refresh()
    return result


@langchain_tool
def replace_all(old_text: str, new_text: str) -> str:
    """Find and replace text across ALL lines in the document.
    Args:
        old_text: the text to find
        new_text: the text to replace it with
    Every occurrence of old_text will be replaced. Auto-saves to file."""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    result = mgr.replace_all(old_text, new_text)
    mark_doc_dirty_and_refresh()
    return result


# ================================================================
# SECTION 6: TOOLS — DELETER
# ================================================================

@langchain_tool
def delete_line(paragraph: int, line: int) -> str:
    """Delete a specific line from the document.
    Args:
        paragraph: paragraph number of the line to delete
        line: line number within that paragraph
    IMPORTANT: Use search_document or vector_search first to confirm the exact line.
    Auto-saves to file after deleting."""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    result = mgr.delete_line(paragraph, line)
    mark_doc_dirty_and_refresh()
    return result


# ================================================================
# SECTION 7: TOOLS — SHARED (all agents can use)
# ================================================================

@langchain_tool
def show_history(check: str = "all") -> str:
    """Show all changes made to the document so far.
    Displays: what was changed, old value, new value, and when."""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    return mgr.history()


@langchain_tool
def save_copy(output_path: str) -> str:
    """Save the document to a NEW file path (keeps original unchanged).
    Args:
        output_path: full path for the new file (e.g., 'report_v2.docx')"""
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    return mgr.save_as(output_path)


@langchain_tool
def reload_from_disk(confirm: str = "yes") -> str:
    """Reload the document from the original file on disk.
    This DISCARDS all in-memory changes that haven't been auto-saved.
    Pass confirm='yes' to proceed."""
    if confirm.lower() != "yes":
        return "Reload cancelled. Pass confirm='yes' to proceed."
    mgr = _state["doc_manager"]
    if not mgr:
        return "ERROR: No document loaded."
    return mgr.reload()


# ================================================================
# SECTION 8: AGENT DEFINITIONS
# ================================================================
# Each agent gets ONLY the tools it needs.
# This prevents agents from doing things outside their role.

def build_agents(llm=None):
    """
    Create the 4 specialist agents + 1 manager agent.

    Hierarchy:
        Manager (routes + oversees)
            ├── Reader   (view, search, vector search, sentences)
            ├── Writer   (add lines)
            ├── Updater  (update lines, replace all)
            └── Deleter  (delete lines)
    """
    llm_config = {"llm": llm} if llm else {}

    # ---- READER AGENT ----
    reader = Agent(
        role="Document Reader",
        goal=(
            "Read and search document content with precision. "
            "Use keyword search for exact matches, vector search for "
            "meaning-based queries, and sentence extraction for full context."
        ),
        backstory=(
            "You are the team's research specialist. You know the document "
            "inside-out. When someone needs to find something, you pick the "
            "right search method: keyword search for exact text, vector "
            "search for conceptual queries, and sentence extraction when "
            "they need full sentences with context."
        ),
        tools=[
            search_document,
            vector_search,
            find_full_sentence,
            find_with_context,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- WRITER AGENT ----
    writer = Agent(
        role="Document Writer",
        goal=(
            "Add new content to the document at the correct position. "
            "Always check the document structure first before adding."
        ),
        backstory=(
            "You are the content creator. Before adding any line, you "
            "ALWAYS use search_document or vector_search to understand the "
            "paragraph and line numbering around where you're adding — you "
            "never try to view the whole document, since a 100-page report "
            "would never fit in context. You then add content at the right spot."
        ),
        tools=[
            search_document,
            vector_search,
            add_line,
            show_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- UPDATER AGENT ----
    updater = Agent(
        role="Document Updater",
        goal=(
            "Modify existing document content accurately. "
            "Find the exact line first, then update it."
        ),
        backstory=(
            "You are the editor. You never update blindly. Your workflow: "
            "1) Search for the text to change, 2) Note the exact P and L "
            "numbers, 3) Update with the new text, 4) Verify the change. "
            "For bulk changes across many lines, use replace_all."
        ),
        tools=[
            search_document,
            vector_search,
            update_line,
            replace_all,
            show_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- DELETER AGENT ----
    deleter = Agent(
        role="Document Deleter",
        goal=(
            "Remove specific content from the document safely. "
            "Always confirm the exact line before deleting."
        ),
        backstory=(
            "You are the cleanup specialist. Deletion is irreversible, "
            "so you are extra careful. Your workflow: 1) Search with "
            "search_document or vector_search to find the content, "
            "confirming it's the right line, 2) Delete using the exact "
            "P and L numbers. You never try to view the whole document."
        ),
        tools=[
            search_document,
            vector_search,
            delete_line,
            show_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- MANAGER AGENT ----
    manager = Agent(
        role="Document Operations Manager",
        goal=(
            "Understand the user's request and delegate to the right agent. "
            "READ/SEARCH/VIEW requests -> Reader Agent. "
            "ADD/INSERT/WRITE requests -> Writer Agent. "
            "UPDATE/EDIT/MODIFY/REPLACE requests -> Updater Agent. "
            "DELETE/REMOVE requests -> Deleter Agent. "
            "Review results before returning to the user."
        ),
        backstory=(
            "You are the team lead. You receive user requests and figure "
            "out which specialist should handle it. Sometimes a task needs "
            "multiple steps (e.g., 'find X and change it to Y' needs the "
            "Reader first, then the Updater). You coordinate the work and "
            "make sure the final result is accurate."
        ),
        tools=[
            search_document,
            vector_search,
            show_history,
            save_copy,
            reload_from_disk,
        ],
        allow_delegation=True,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    return {
        "reader": reader,
        "writer": writer,
        "updater": updater,
        "deleter": deleter,
        "manager": manager,
    }


# ================================================================
# SECTION 9: CREW BUILDER
# ================================================================

def build_crew(agents: dict, user_request: str):
    """
    Build a CrewAI Crew for one user request.

    - Process: HIERARCHICAL (manager routes to specialists)
    - Memory: DISABLED — CrewAI's built-in memory needs its own embedder,
      and without one explicitly configured it defaults to trying an
      OpenAI embedding call. Since this project has no OpenAI key (and
      is meant to run fully offline on the local sentence-transformers
      model — see vector_config.py), that default silently stalls/retries
      on every single kickoff, which is a big chunk of the "everything
      is slow" complaint. We already have our own persistent, local,
      offline RAG layer (Chroma) for file content, so CrewAI's separate
      conversational memory isn't needed on top of it.
    - Verbose: DISABLED at the crew level too — same reasoning as the
      agents: writing full reasoning traces to the console on every
      step adds real wall-clock time over a long multi-agent run.
    - A new Task is created from the user's input
    """
    task = Task(
        description=(
            f"USER REQUEST:\n"
            f"{user_request}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Determine what the user wants (read/search/add/update/delete)\n"
            f"2. Delegate to the right specialist agent\n"
            f"3. The specialist should use tools to complete the task\n"
            f"4. Return a clear, concise summary of what was done or found"
        ),
        expected_output=(
            "A clear summary of the action taken and its result. "
            "If content was found, show it. "
            "If content was changed, show what was changed."
        ),
    )

    crew = Crew(
        agents=[
            agents["reader"],
            agents["writer"],
            agents["updater"],
            agents["deleter"],
        ],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=agents["manager"],
        memory=False,
        verbose=True,
    )

    return crew


# ================================================================
# SECTION 10: MAIN SYSTEM (ties everything together)
# ================================================================

class DocsAgentSystem:
    """
    The main orchestrator.

    Takes a filepath, processes it, and runs a chatbot loop
    where CrewAI agents handle document operations.

    Usage:
        system = DocsAgentSystem("report.docx")
        system.chat()   # interactive loop
        # OR
        result = system.process_request("find all mentions of revenue")
    """

    def __init__(self, filepath: str, llm=None):
        """
        Initialize the system.

        Args:
            filepath: path to the .docx file
            llm: (optional) LangChain LLM instance for the agents.
                 Defaults to OpenAI gpt-4o-mini.
                 You can pass any LangChain-compatible LLM.
        """
        self.filepath = filepath

        # ---- Set up the LLM ----
        if llm:
            self.llm = llm
        else:
            # Default: load saved Gemini credentials and build a
            # CrewAI-native LLM (provider-prefixed model string).
            from doxl_ai_terminal.pipeline.config import get_crewai_llm
            from doxl_ai_terminal.pipeline.credential_store import load_credentials

            cred = load_credentials()
            if not cred:
                raise RuntimeError(
                    "No saved Gemini credentials found. Run onboarding first "
                    "(or pass an explicit llm= to DocsAgentSystem)."
                )
            print(f"Using CrewAI Gemini LLM with model: {cred['model']}")
            self.llm = get_crewai_llm(cred["api_key"], cred["model"])

        # ---- Step 1: Process the file ----
        print("=" * 55)
        print("  DOXL AI — Document Agent System")
        print("=" * 55)
        process_file(filepath)

        # ---- Step 2: Build agents ----
        self.agents = build_agents(self.llm)

        print("=" * 55)
        print("  Agents Ready:")
        print("    - Reader   (exact + RAG semantic search)")
        print("    - Writer   (add content)")
        print("    - Updater  (modify content)")
        print("    - Deleter  (remove content)")
        print("    - Manager  (routes your requests)")
        print("=" * 55)
        print("  Commands:")
        print("    Type your request in plain English (e.g. 'find mentions of")
        print("    revenue' or 'what does it say about deadlines') — full-")
        print("    document 'view' is disabled since reports can run 100+ pages.")
        print("    'history' — see all changes made")
        print("    'quit'    — exit the system")
        print("=" * 55)

    def process_request(self, user_input: str) -> str:
        """
        Send one request through CrewAI and get the result.

        The Manager agent reads the request, picks the right
        specialist, and returns the result. If anything in this
        request wrote to the document, the (expensive) vector index is
        rebuilt exactly once here — after the whole request finishes —
        rather than after every individual tool call.
        """
        crew = build_crew(self.agents, user_input)
        result = crew.kickoff()
        rebuild_doc_vector_index_if_dirty()
        return str(result)

    def chat(self):
        """
        Interactive chatbot loop.

        Keeps running until the user types 'quit'.
        Each request goes through CrewAI agents.
        Memory is preserved across requests in the same session.
        """
        while True:
            # ---- Get user input ----
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # ---- Check for exit ----
            if user_input.lower() in ("quit", "exit", "q", "bye"):
                print("Goodbye!")
                break

            # ---- Quick shortcuts (skip CrewAI for speed) ----
            if user_input.lower() == "view":
                print(
                    "\n  'view' is disabled — a document can run 100+ pages, "
                    "which won't fit in context.\n"
                    "  Try searching naturally instead, e.g. 'find the section "
                    "about pricing'."
                )
                continue

            if user_input.lower() == "history":
                print("\n" + show_history.invoke("all"))
                continue

            # ---- Process through CrewAI ----
            try:
                print("\n  Processing your request...\n")
                result = self.process_request(user_input)
                print(f"\nAgent: {result}")

            except Exception as e:
                print(f"\n  Error: {e}")
                print("  Try rephrasing your request.")


# ================================================================
# SECTION 11: ENTRY POINT
# ================================================================

def main():
    """CLI entry point: python docs_agent.py <filepath.docx>"""

    if len(sys.argv) < 2:
        print("Usage: python docs_agent.py <filepath.docx>")
        print("Example: python docs_agent.py report.docx")
        sys.exit(1)

    filepath = sys.argv[1]

    # ---- Configure LLM (change this to use a different model) ----
    #
    # Option 1: OpenAI (default)
    #   Set env: OPENAI_API_KEY=your-key
    #   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #
    # Option 2: Ollama (local)
    #   from langchain_ollama import ChatOllama
    #   llm = ChatOllama(model="llama3.1", temperature=0)
    #
    # Option 3: Groq (fast + free tier)
    #   from langchain_groq import ChatGroq
    #   llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    #
    llm = None  # None = uses default (OpenAI gpt-4o-mini)

    system = DocsAgentSystem(filepath, llm=llm)
    system.chat()


if __name__ == "__main__":
    main()