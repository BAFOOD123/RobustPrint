# RobustPrint: common tasks. Uses a local .venv (created with uv if available, else venv).
PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup setup-locked test lint fast full figures public-data app docs clean

setup:
	@if command -v uv >/dev/null 2>&1; then uv venv -p $(PY) $(VENV); \
	  uv pip install -p $(BIN)/python torch --index-url https://download.pytorch.org/whl/cpu; \
	  uv pip install -p $(BIN)/python -e ".[dev,app]"; \
	else $(PY) -m venv $(VENV); $(BIN)/pip install --upgrade pip; \
	  $(BIN)/pip install torch --index-url https://download.pytorch.org/whl/cpu; \
	  $(BIN)/pip install -e ".[dev,app]"; fi

setup-locked:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(BIN)/pip install -e . --no-deps

test:
	$(BIN)/pytest --cov=robustprint --cov-report=term --cov-fail-under=85

lint:
	$(BIN)/ruff check src tests app scripts
	$(BIN)/ruff format --check src tests app scripts
	$(BIN)/mypy

fast:
	$(BIN)/robustprint reproduce --fast

full:
	$(BIN)/robustprint reproduce --full

figures:
	$(BIN)/robustprint figures results/full

public-data:
	$(BIN)/robustprint public-data

app:
	$(BIN)/streamlit run app/streamlit_app.py

docs:
	@echo "Documentation is Markdown: README.md and docs/*.md (no build step)."

clean:
	rm -rf results/fast .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
