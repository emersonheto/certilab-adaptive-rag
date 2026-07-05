.PHONY: setup mock real ingest test lint clean help

help: ## Mostrar ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Levantar Qdrant + MySQL con datos seed
	docker compose up -d
	@echo ""
	@echo "✅ Servicios levantados:"
	@echo "   Qdrant: http://localhost:6333"
	@echo "   MySQL:  localhost:3306 (certilab_test)"
	@echo ""
	@echo "   Para indexar certificados: make ingest"
	@echo "   Para probar el demo:       make real"

mock: ## Demo en modo mock (sin servicios)
	uv run python -m app.adaptive_rag.demo "¿Qué procedimiento de calibración sigue INDECOPI?"

real: ## Demo en modo real (requiere make setup + make ingest)
	APP_MODE=real uv run python -m app.adaptive_rag.demo "¿Qué certificados tiene ALS PERU?"

ingest: ## Indexar certificados en Qdrant (requiere make setup + .env con OPENAI_API_KEY)
	APP_MODE=real DB_HOST=127.0.0.1 DB_USERNAME=certilab DB_PASSWORD=certilab \
	DB_DATABASE=certilab_test QDRANT_URL=http://localhost:6333 \
	uv run python -m app.adaptive_rag.ingest

notebook: ## Abrir el notebook
	uv run jupyter notebook notebooks/adaptive_rag_demo.ipynb

test: ## Ejecutar tests
	uv run pytest -q

lint: ## Linting + type check
	uv run ruff check .
	uv run mypy app/

clean: ## Detener y limpiar servicios
	docker compose down -v
	@echo "✅ Servicios detenidos y datos eliminados"
