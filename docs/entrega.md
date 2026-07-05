# Instrucciones para el profesor

## 🔗 Repositorio

**https://github.com/emersonheto/certilab-adaptive-rag**

## 🚀 Cómo ejecutar

### Opción 1: Google Colab (sin instalación)

1. Abrir [Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub → `emersonheto/certilab-adaptive-rag`
3. Seleccionar `notebooks/adaptive_rag_demo.ipynb`
4. Ejecutar celdas en orden. La celda 1 clona el repo e instala dependencias.
5. La celda 2 pedirá la API key (Colab Secrets o input manual)

### Opción 2: Local

```bash
git clone https://github.com/emersonheto/certilab-adaptive-rag.git
cd certilab-adaptive-rag
uv sync
cp .env.example .env
# Editar .env con OPENAI_API_KEY
```

#### Demo CLI (modo mock — sin base de datos)

```bash
uv run python -m app.adaptive_rag.demo "¿Qué procedimiento de calibración sigue INDECOPI?"
```

#### Notebook

```bash
uv run jupyter notebook notebooks/adaptive_rag_demo.ipynb
```

### Opción 3: Modo real (con datos reales)

```bash
docker compose up -d qdrant
uv sync --extra real
APP_MODE=real uv run python -m app.adaptive_rag.ingest   # ingesta 154 certificados
APP_MODE=real uv run python -m app.adaptive_rag.demo "¿Qué certificados tiene ALS PERU?"
```

## 🧪 Verificar calidad del código

```bash
uv run pytest -q       # 58 tests
uv run ruff check .    # Linting
```

## 📂 Qué evaluar

| Entregable | Ubicación | Descripción |
|---|---|---|
| Sistema Agentic RAG | `app/adaptive_rag/` | 7 nodos, 2 loops de auto-corrección, Pydantic structured output |
| Notebook | `notebooks/adaptive_rag_demo.ipynb` | 20 celdas, ejecución live con `graph.stream()` |
| README | `README.md` | 227 líneas, español, diagrama Mermaid |
| Buenas prácticas | Git history | 24 conventional commits, 58 tests, ruff clean |
| Alineación | `docs/course-alignment.md` | Mapeo detallado con requisitos del enunciado |
| Instrucciones | `docs/entrega.md` | Este archivo |

## 📊 Stack completo

| Capa | Herramienta |
|---|---|
| Grafo RAG | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Vector store | Qdrant (tenant isolation por `customer_id`) |
| Extracción de texto | PyMuPDF (fitz) |
| Extracción de tablas | Camelot (99% accuracy) |
| Chunking semántico | Unstructured (`chunk_by_title`) |
| Web search | Tavily Search API |
| Schemas | Pydantic v2 (structured output) |
| Observabilidad | Phoenix / OpenInference |
| Datos | MySQL + Amazon S3 |

## 🗂️ Proyecto complementario

El sistema productivo completo (API FastAPI, Chainlit UI, integración Laravel/MySQL/S3) está en:

**https://github.com/emersonheto/certilab-agentic-rag**
