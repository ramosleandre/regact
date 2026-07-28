# Prefer the project venv so gates use the pinned ruff/mypy/pytest, not PATH's.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.12)
PKG := regact
# Run the package from ./src even when an editable install didn't take (a known macOS
# .pth quirk under ~/Desktop; harmless everywhere else). Tests inject src themselves.
RUN := PYTHONPATH=src $(PYTHON)

.DEFAULT_GOAL := help

# `make help` groups targets by the "## <category> ── <description>" tag on each rule.
.PHONY: help
help:  ## general ── Show this help, grouped by category
	@echo "regact — make targets"
	@awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / { \
		split($$2, p, " ── "); cat=p[1]; desc=p[2]; \
		if (cat != last) { printf "\n\033[1m%s\033[0m\n", cat; last=cat } \
		printf "  \033[36m%-12s\033[0m %s\n", $$1, desc \
	}' $(MAKEFILE_LIST)

# ── setup ─────────────────────────────────────────────────────────────────────
.PHONY: install
install:  ## setup ── Editable install with dev extras (py3.12)
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: doctor
doctor:  ## setup ── Check this machine can run regact (sandbox, agent CLIs, LLM endpoint)
	$(RUN) -m $(PKG).doctor

# ── quality gate ──────────────────────────────────────────────────────────────
.PHONY: lint
lint:  ## gate ── Ruff lint
	$(PYTHON) -m ruff check src tests

.PHONY: format
format:  ## gate ── Ruff format + autofix
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

.PHONY: typecheck
typecheck:  ## gate ── mypy
	$(PYTHON) -m mypy src

.PHONY: test
test:  ## gate ── Tests needing no real LLM or game (the exact selector CI runs)
	$(PYTHON) -m pytest -m "not live and not slow" -q

.PHONY: test-all
test-all:  ## gate ── Every test, including the live ones (needs alancode / a game)
	$(PYTHON) -m pytest -q

.PHONY: check
check: lint typecheck test  ## gate ── The CI gate: lint + typecheck + unit tests

# ── run & inspect ─────────────────────────────────────────────────────────────
.PHONY: run
run:  ## run ── Research run via Hydra (overrides via ARGS=..., e.g. ARGS="agent=codex problem=arc_agi")
	$(RUN) -m $(PKG).run_exp $(ARGS)

.PHONY: run-kaggle
run-kaggle:  ## run ── Competition run (flags via ARGS=..., e.g. ARGS="--games ls20")
	$(RUN) -m $(PKG).run_kaggle $(ARGS)

.PHONY: viz
viz:  ## run ── Launch the visualizer: make viz EXP=experiments/<name>/latest [PORT=8030]
	$(RUN) -m $(PKG).viz.app --experiment $(EXP) --port $(or $(PORT),8030)

# ── diagnostics (every one also runs as `python -m regact.<name>`, since an HPC
#    job has no make; add --json to capture a verdict in the job log) ───────────
.PHONY: probe
probe:  ## diagnose ── Does the sandbox honor the R1-R6 contract here?
	$(RUN) -m $(PKG).security.probe --sandbox

.PHONY: agentcheck
agentcheck:  ## diagnose ── Do the agent backends actually launch here? (ARGS="--verbose")
	$(RUN) -m $(PKG).agentcheck --all $(ARGS)

.PHONY: prompt
prompt:  ## diagnose ── Print the prompt a run would send (ARGS="--problem minigrid")
	$(RUN) -m $(PKG).prompt_preview $(ARGS)

# ── housekeeping ──────────────────────────────────────────────────────────────
.PHONY: clean
clean:  ## housekeeping ── Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
