# OpenRSI developer shortcuts. Everything goes through uv (https://docs.astral.sh/uv/).
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh

LOGGING_DIR ?= $(CURDIR)/.logs

.DEFAULT_GOAL := help

.PHONY: help setup test lock upgrade export clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv and install everything (fetches Python 3.12 if needed)
	uv sync --all-groups

test:  ## Run the ctf_gym suite
	@mkdir -p "$(LOGGING_DIR)"
	LOGGING_DIR="$(LOGGING_DIR)" uv run python -m unittest ctf_gym.tests.test_ctf_gym -v

lock:  ## Re-resolve uv.lock after editing a pyproject.toml
	uv lock

upgrade:  ## Re-resolve, allowing newer versions
	uv lock --upgrade

export:  ## Regenerate the pip-compatible requirements.txt from uv.lock
	uv export --no-hashes --no-emit-project --format requirements-txt -o requirements.txt

clean:  ## Remove the virtualenv and caches
	rm -rf .venv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path "./.git/*" -exec rm -rf {} +
