# Repository Guidelines
    
## Project Structure & Module Organization

The project follows a standard Python package structure:

- `src/swarmx/` - Core package source code
    - `agent.py` - Agent class and orchestration logic
    - `cli.py` - Command-line interface implementation
    - `mcp_client.py` - Model Context Protocol client
    - `utils.py` - Utility functions and helpers
    - `types.py` - Type definitions and Pydantic models
    - `hook.py` - Hook system for custom extensions

- `tests/` - Test suite with comprehensive coverage
    - `test_agent.py` - Agent orchestration tests
    - `test_cli.py` - CLI command tests
    - `test_mcp_client.py` - MCP integration tests
    - `threads/` - Test fixtures and thread data

- `docs/` - Documentation and examples
- `examples/` - Usage examples and patterns

## Build, Test, and Development Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Managing Scripts with PEP 723 Inline Metadata

- Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
- You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
- Or using uv CLI:
    - `uv add package-name --script script.py`
    - `uv remove package-name --script script.py`

## Coding Style & Naming Conventions

- **Python 3.11+**: Uses modern Python features (async/await, type hints)
- **Ruff**: Primary linter and formatter (configured in pyproject.toml)
- **Type Hints**: Extensive use of type annotations throughout
- **Pydantic**: Data validation and serialization models
- **Naming**:
    - Classes: `PascalCase` (e.g., `Agent`, `Swarm`)
    - Functions/Variables: `snake_case` (e.g., `run_agent`, `message_slice`)
    - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MODEL`)

## Testing Guidelines

- **Framework**: pytest with pytest-cov for coverage
- **Coverage**: Aim for >90% test coverage
- **Test Structure**:
    - Tests mirror source structure (test_*.py files)
    - Use pytest fixtures for test setup
    - Async tests use pytest-asyncio
- **Naming**: Test functions start with `test_`
- **Running**: Use `pytest -xvs` for verbose test execution

## Commit & Pull Request Guidelines

**Commit Messages:**
- Follow KeepAChangelog guideline
- Use imperative mood ("Added", "Fixed", "Changed", "Removed")
- Keep first line under 50 characters
- Provide context in body if needed
- Reference issues when applicable

**Pull Requests:**
- Include clear description of changes
- Link related issues
- Ensure all tests pass
- Update documentation if needed
- Follow existing code style patterns

## Security & Configuration Tips

- Environment variables loaded from `.env` file
- API keys should never be committed
- Use `.env.example` as template for required variables
- MCP servers require proper authentication setup

## Agent-Specific Instructions

**Agent Design:**
- Keep agents focused and single-purpose
- Use hooks (`on_llm_start`, `on_handoff`) for custom behavior
- Leverage context variables (`background`, `message_slice`, `tools`)

**Workflow Patterns:**
- Use function-based edge transfers for agent routing
- Implement context compression with `message_slice`
- Support dynamic tool selection via `tools` context

**MCP Integration:**
- Configure MCP servers in environment
- Use `mcp_client.py` for protocol interactions
- Follow MCP specification for tool definitions
