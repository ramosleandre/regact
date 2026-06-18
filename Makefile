# Prefer the project venv so gates use the pinned ruff/mypy/pytest, not PATH's.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.12)
PKG := regact

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── setup ────────────────────────────────────────────────────────────────────
.PHONY: install
install:  ## Editable install with dev extras (py3.12)
	$(PYTHON) -m pip install -e ".[dev]"

# ── quality gates ────────────────────────────────────────────────────────────
.PHONY: lint
lint:  ## Ruff lint
	$(PYTHON) -m ruff check src tests

.PHONY: format
format:  ## Ruff format + autofix
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

.PHONY: typecheck
typecheck:  ## mypy
	$(PYTHON) -m mypy src

.PHONY: test
test:  ## Unit tests (no LLM, no game)
	$(PYTHON) -m pytest -m "not integration and not slow" -q

.PHONY: test-int
test-int:  ## Integration tests (scripted agent + fake env, no LLM)
	$(PYTHON) -m pytest -m integration -q

.PHONY: test-all
test-all:  ## All tests
	$(PYTHON) -m pytest -q

.PHONY: check
check: lint typecheck test  ## The CI gate: lint + typecheck + unit tests

# ── run / inspect (light up as blocks land) ──────────────────────────────────
.PHONY: serve
serve:  ## Launch the env server alone (GAME=...)
	$(PYTHON) -m $(PKG).env.server --problem $(GAME)

.PHONY: play
play:  ## Human-play an env, no agent (GAME=...)
	$(PYTHON) -m $(PKG).play --problem $(GAME)

.PHONY: prompt
prompt:  ## Show the assembled prompt for a task (GAME=...)
	$(PYTHON) -m $(PKG).prompt show --problem $(GAME)

.PHONY: viz
viz:  ## Launch the experiment visualizer: make viz EXP=experiments/<run> [PORT=8030]
	PYTHONPATH=src $(PYTHON) -m $(PKG).viz.app --experiment $(EXP) --port $(or $(PORT),8030)

.PHONY: run-exp
run-exp:  ## Research run (Hydra overrides via ARGS=...)
	$(PYTHON) -m $(PKG).run_exp $(ARGS)

.PHONY: run-kaggle
run-kaggle:  ## Competition run (flags via ARGS=...)
	$(PYTHON) -m $(PKG).run_kaggle $(ARGS)

.PHONY: debug
debug:  ## Run a per-block debug smoke script: make debug D=block2_env [ARGS=...]
	PYTHONPATH=src $(PYTHON) debug/$(D).py $(ARGS)

# ── housekeeping ─────────────────────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
