# Instrucciones para el profesor

## 🔗 Repositorio

**https://github.com/emersonheto/certilab-adaptive-rag**

## 🚀 Cómo ejecutar

### 1. Clonar e instalar

```bash
git clone https://github.com/emersonheto/certilab-adaptive-rag.git
cd certilab-adaptive-rag
uv sync
cp .env.example .env
```

### 2. Configurar API key

Editar `.env` y agregar:
```
OPENAI_API_KEY=sk-...
```

### 3. Ejecutar el demo (modo mock — sin base de datos)

```bash
uv run python -m app.adaptive_rag.demo "¿Qué procedimiento de calibración sigue INDECOPI?"
```

### 4. Ejecutar el notebook

```bash
uv run jupyter notebook notebooks/adaptive_rag_demo.ipynb
```

El notebook ejecuta el grafo con `graph.stream()`, mostrando cada nodo y los loops de auto-corrección.

### 5. Ejecutar pruebas

```bash
uv run pytest -q       # 58 tests
uv run ruff check .    # Linting
uv run mypy app/       # Type checking
```

## 📂 Qué evaluar

| Entregable | Ubicación |
|---|---|
| Sistema Agentic RAG | `app/adaptive_rag/` — 7 nodos, 2 loops de auto-corrección |
| Notebook | `notebooks/adaptive_rag_demo.ipynb` — 14 celdas con ejecución live |
| README | `README.md` — español, diagrama Mermaid, explicación completa |
| Buenas prácticas | Conventional commits, 58 tests, ruff + mypy clean, lazy imports, Protocols |
| Documentación | `docs/course-alignment.md` — alineación detallada con requisitos |

## 📊 Stack completo

- Python 3.11+, LangGraph, LangChain, OpenAI (GPT-4o-mini + text-embedding-3-small)
- Tavily Search API, Pydantic v2, Qdrant, PyMuPDF, Camelot, Unstructured
- Phoenix/OpenInference (observabilidad), MySQL + S3 (datos reales)

## 🗂️ Proyecto complementario

Este repo es la entrega académica del Adaptive RAG. El sistema productivo completo (API, Chainlit UI, tenant isolation, integración real con MySQL/S3) está en:

**https://github.com/emersonheto/certilab-agentic-rag**
