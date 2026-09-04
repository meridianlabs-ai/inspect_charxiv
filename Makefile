# Local development tasks. CI runs the same commands.
#
#   make check      run lint, typecheck, and tests
#   make format     auto-fix lint issues and reformat

.PHONY: check lint format typecheck test help

check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest

help:
	@echo "make check      lint + typecheck + test"
	@echo "make lint       ruff check and format --check"
	@echo "make format     ruff --fix and format (modifies files)"
	@echo "make typecheck  mypy on src"
	@echo "make test       pytest"
