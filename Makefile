.DEFAULT_GOAL := help

.PHONY: help install test lint typecheck mock ingest up down clean setup dev run check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (dev + optional real)
	uv sync --extra dev --extra real

install-light: ## Install only core dependencies (no optional real deps)
	uv sync --extra dev

test: ## Run all tests
	uv run pytest -v

test-coverage: ## Run tests with coverage report
	uv run pytest -v --cov=app --cov-report=term

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check . --fix

typecheck: ## Run mypy strict type checking
	uv run mypy app

check: lint typecheck test ## Run all checks (lint + typecheck + test)

setup: install up ## Prepare environment: install deps + start services (Qdrant)

dev: setup mock ## Full dev cycle: setup + run demo in mock mode

run: mock ## Quick start — run demo in mock mode (assumes deps installed)

mock: ## Run Adaptive RAG demo in mock mode (no external services needed)
	@echo "Starting Adaptive RAG demo (mock mode)..."
	uv run python -m app.adaptive_rag.demo

ingest: ## Run ingestion pipeline (requires APP_MODE=real + credentials)
	@echo "Starting ingestion pipeline (real mode)..."
	uv run python -m app.adaptive_rag.ingest

up: ## Start docker services (Qdrant)
	docker compose up -d

down: ## Stop docker services
	docker compose down

clean: ## Clean caches and temp artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
