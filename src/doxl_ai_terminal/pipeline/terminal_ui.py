#terminal_ui.py
"""
Terminal UI helpers — colored output, banners, prompts.
"""


# --- ANSI Colors ---

class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def success(text: str):
    print(f"{Color.GREEN}✓ {text}{Color.RESET}")


def error(text: str):
    print(f"{Color.RED}✗ {text}{Color.RESET}")


def info(text: str):
    print(f"{Color.CYAN}→ {text}{Color.RESET}")


def warn(text: str):
    print(f"{Color.YELLOW}⚠ {text}{Color.RESET}")


def agent_say(agent_name: str, text: str):
    print(f"{Color.BOLD}{Color.CYAN}[{agent_name}]{Color.RESET} {text}")


def divider():
    print(f"{Color.DIM}{'─' * 50}{Color.RESET}")


def banner():
    print(f"""
{Color.BOLD}{Color.CYAN}╔══════════════════════════════════════════╗
║       LangGraph Multi-Agent System       ║
║              Terminal Agent               ║
╚══════════════════════════════════════════╝{Color.RESET}
""")


def prompt_input(label: str) -> str:
    return input(f"{Color.YELLOW}{label}{Color.RESET}").strip()


def interrupt_prompt(question: str) -> str:
    """Human-in-the-loop prompt"""
    divider()
    print(f"{Color.BOLD}{Color.YELLOW}🔔 INTERRUPT — Agent needs your input:{Color.RESET}")
    print(f"   {question}")
    answer = input(f"{Color.YELLOW}   Your answer: {Color.RESET}").strip()
    divider()
    return answer