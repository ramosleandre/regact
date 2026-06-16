PYTHON ?= python3.12
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
	ruff check src tests

.PHONY: format
format:  ## Ruff format + autofix
	ruff format src tests
	ruff check --fix src tests

.PHONY: typecheck
typecheck:  ## mypy
	mypy src

.PHONY: test
test:  ## Unit tests (no LLM, no game)
	pytest -m "not integration and not slow" -q

.PHONY: test-int
test-int:  ## Integration tests (scripted agent + fake env, no LLM)
	pytest -m integration -q

.PHONY: test-all
test-all:  ## All tests
	pytest -q

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
viz:  ## Launch the experiment visualizer
	$(PYTHON) -m $(PKG).viz

.PHONY: run-exp
run-exp:  ## Research run (Hydra overrides via ARGS=...)
	$(PYTHON) -m $(PKG).run_exp $(ARGS)

.PHONY: run-kaggle
run-kaggle:  ## Competition run (flags via ARGS=...)
	$(PYTHON) -m $(PKG).run_kaggle $(ARGS)

.PHONY: debug
debug:  ## Run a per-block debug smoke script: make debug D=block2_env
	PYTHONPATH=src $(PYTHON) debug/$(D).py

# ── housekeeping ─────────────────────────────────────────────────────────────
.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
