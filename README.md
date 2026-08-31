# doxl-ai-terminal

AI-powered terminal for reading and writing Excel & Word documents.

## Installation

### From source (editable / development mode)

```bash
# Clone the repository
git clone https://github.com/monishkumaarm-5/Excel-docs-reader-writer-AI.git
cd Excel-docs-reader-writer-AI

# Install in editable mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### From built package

```bash
pip install .
```

## Usage

### CLI

After installation, the `doxl-ai` command is available:

```bash
# Show help
doxl-ai --help

# Show version
doxl-ai --version
```

### Python API

```python
import doxl_ai_terminal
print(doxl_ai_terminal.__version__)

# Import sub-modules
from doxl_ai_terminal.agents import excel_agent, docs_agent
from doxl_ai_terminal.data_handler import vector_config
from doxl_ai_terminal.pipeline import pipeliner
```

### As a module

```bash
python -m doxl_ai_terminal --help
```

## Project Structure

```
src/
└── doxl_ai_terminal/
    ├── __init__.py          # Package root & public API
    ├── __main__.py          # python -m support
    ├── cli.py               # CLI entry point
    ├── agents/              # AI agents for different file types
    │   ├── config_agent.py
    │   ├── docs_agent.py
    │   ├── excel_agent.py
    │   └── terminal_agent.py
    ├── data_handler/        # Vector DB & data operations
    │   ├── vector_config.py
    │   └── vector_db_operation.py
    └── pipeline/            # Processing pipelines
        └── pipeliner.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## License

MIT License — see [LICENSE](LICENSE) for details.
