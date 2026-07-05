# Instrucciones para el profesor

## 🔗 Repositorio

**https://github.com/emersonheto/certilab-adaptive-rag**

---

## 🚀 Cómo ejecutar — 3 opciones

### Opción 1: Mock (sin Docker, sin base de datos) — la más rápida

```bash
git clone https://github.com/emersonheto/certilab-adaptive-rag.git
cd certilab-adaptive-rag
uv sync
cp .env.example .env
# Editar .env: OPENAI_API_KEY=sk-...
make mock
```

✅ Funciona con datos de prueba locales. No necesita Docker ni servicios externos.

---

### Opción 2: Real con datos pre-indexados (recomendada)

```bash
git clone https://github.com/emersonheto/certilab-adaptive-rag.git
cd certilab-adaptive-rag
uv sync
cp .env.example .env
# Editar .env: OPENAI_API_KEY=sk-...

make quickstart    # 1. Levanta Qdrant + MySQL (Docker)
                   # 2. Restaura 3,848 vectores desde GitHub Release
                   # 3. Carga 177 certificados en MySQL

make real          # Demo con datos reales: "¿Qué certificados tiene ALS PERU?"
```

✅ No gasta en OpenAI para embeddings (ya están calculados).
✅ Solo necesita OPENAI_API_KEY para las consultas (routing, grading, generation).
✅ No necesita acceso a S3 ni credenciales externas.

**Consultas de ejemplo:**

```bash
# Por cliente
APP_MODE=real DB_HOST=127.0.0.1 DB_USERNAME=certilab DB_PASSWORD=certilab \
  DB_DATABASE=certilab_test QDRANT_URL=http://localhost:6333 \
  uv run python -m app.adaptive_rag.demo "¿Qué certificados tiene ALS PERU?"

# Por procedimiento
APP_MODE=real DB_HOST=127.0.0.1 DB_USERNAME=certilab DB_PASSWORD=certilab \
  DB_DATABASE=certilab_test QDRANT_URL=http://localhost:6333 \
  uv run python -m app.adaptive_rag.demo "¿Qué procedimiento sigue la norma INDECOPI?"

# Por temperatura
APP_MODE=real DB_HOST=127.0.0.1 DB_USERNAME=certilab DB_PASSWORD=certilab \
  DB_DATABASE=certilab_test QDRANT_URL=http://localhost:6333 \
  uv run python -m app.adaptive_rag.demo "¿Cuál fue la temperatura máxima a 105°C?"
```

---

### Opción 3: Google Colab (sin instalación local)

1. Abrir [Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub → `emersonheto/certilab-adaptive-rag`
3. Seleccionar `notebooks/adaptive_rag_demo.ipynb`
4. Ejecutar celdas en orden:
   - Celda 1: clona el repo e instala dependencias
   - Celda 2: pide `OPENAI_API_KEY` (Colab Secrets o input manual)
   - Celdas 3+: construye el grafo y ejecuta 5 consultas de demo

---

## 📋 Comandos disponibles (Makefile)

```bash
make help        # ver todos los comandos
make mock        # demo sin Docker (datos locales)
make quickstart  # setup completo: Docker + datos vectoriales
make real        # demo con datos reales
make restore     # restaurar vectores de Qdrant (solo)
make ingest      # re-indexar desde cero (gasta OpenAI API)
make notebook    # abrir Jupyter
make test        # 58 tests
make lint        # ruff + mypy
make clean       # detener Docker y borrar datos
```

---

## 🧪 Verificar calidad del código

```bash
make test    # 58 tests
make lint    # ruff + mypy
```

---

## 📂 Qué evaluar

| Entregable | Ubicación | Descripción |
|---|---|---|
| Sistema Agentic RAG | `app/adaptive_rag/` | 7 nodos, 2 loops de auto-corrección |
| Notebook | `notebooks/adaptive_rag_demo.ipynb` | 20 celdas, `graph.stream()` |
| README | `README.md` | Español, Mermaid, stack completo |
| Alineación con enunciado | `docs/course-alignment.md` | Mapeo detallado requisito por requisito |
| Datos seed | `data/sql/seed.sql` | 12 clientes + 177 certificados |
| Backup de vectores | [Release v1.0-data](https://github.com/emersonheto/certilab-adaptive-rag/releases/tag/v1.0-data) | 3,848 puntos con embeddings |

---

## 📊 Stack completo

| Capa | Herramienta |
|---|---|
| Grafo RAG | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Vector store | Qdrant (tenant isolation) |
| Extracción de texto | PyMuPDF (fitz) |
| Extracción de tablas | Camelot (99% accuracy) |
| Chunking | Unstructured (`chunk_by_title`) |
| Web search | Tavily Search API |
| Schemas | Pydantic v2 (structured output) |
| Base de datos | MySQL 8.0 (Docker) |
| Observabilidad | Phoenix / OpenInference |

---

## 🗂️ Proyecto complementario

El sistema productivo completo (API FastAPI, Chainlit UI, integración Laravel/MySQL/S3) está en:

**https://github.com/emersonheto/certilab-agentic-rag**
