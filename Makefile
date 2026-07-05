.PHONY: setup mock real ingest restore test lint clean help

help: ## Mostrar ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Levantar Qdrant + MySQL con datos seed
	docker compose up -d
	@echo ""
	@echo "✅ Servicios levantados:"
	@echo "   Qdrant: http://localhost:6333"
	@echo "   MySQL:  localhost:3306 (certilab_test / certilab)"
	@echo ""
	@echo "   Siguiente paso: make restore"

restore: ## Restaurar Qdrant con vectores pre-calculados (sin gastar OpenAI)
	@echo "📥 Descargando backup (28 MB)..."
	curl -sL https://github.com/emersonheto/certilab-adaptive-rag/releases/download/v1.0-data/certilab-rag-backup.jsonl.gz -o /tmp/qdrant-backup.jsonl.gz
	@echo "🔄 Restaurando 3,848 puntos en Qdrant..."
	uv run python scripts/restore_qdrant.py /tmp/qdrant-backup.jsonl.gz
	@rm -f /tmp/qdrant-backup.jsonl.gz

quickstart: setup restore ## Setup completo: servicios + datos vectoriales
	@echo ""
	@echo "✅ Todo listo. Probá: make real"

mock: ## Demo en modo mock (sin servicios, sin Docker)
	uv run python -m app.adaptive_rag.demo "¿Qué procedimiento de calibración sigue INDECOPI?"

real: ## Demo en modo real (requiere make quickstart)
	APP_MODE=real DB_HOST=127.0.0.1 DB_USERNAME=certilab DB_PASSWORD=certilab \
	DB_DATABASE=certilab_test QDRANT_URL=http://localhost:6333 \
	uv run python -m app.adaptive_rag.demo "¿Qué certificados tiene ALS PERU?"

ingest: ## Re-indexar desde cero con OpenAI (alternativa a make restore, gasta API)
	APP_MODE=real DB_HOST=127.0.0.1 DB_USERNAME=certilab DB_PASSWORD=certilab \
	DB_DATABASE=certilab_test QDRANT_URL=http://localhost:6333 \
	uv run python -m app.adaptive_rag.ingest

notebook: ## Abrir el notebook
	uv run jupyter notebook notebooks/adaptive_rag_demo.ipynb

test: ## Ejecutar tests (58)
	uv run pytest -q

lint: ## Linting + type check
	uv run ruff check .
	uv run mypy app/

clean: ## Detener y limpiar todo
	docker compose down -v
	@echo "✅ Servicios detenidos y datos eliminados"
