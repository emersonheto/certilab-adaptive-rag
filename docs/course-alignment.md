# Documento de Alineación — Curso AI Engineering

Este documento demuestra cómo el proyecto **certilab-adaptive-rag** satisface cada requisito del enunciado.

## 📋 Requisitos del enunciado

> *"Implementar un sistema Agentic RAG. Leer el artículo y llevar el notebook incluido a un repositorio de GitHub."*

> *"Un repositorio público aplicando buenas prácticas: estructura de carpetas, commits descriptivos, .gitignore."*

> *"Un archivo README.md claro y completo."*

---

## 1. Sistema Agentic RAG implementado ✅

### Grafo Adaptive RAG con LangGraph

El sistema implementa un **StateGraph de LangGraph con 7 nodos y 2 loops de auto-corrección**, siguiendo fielmente la topología del artículo de Abhinav Kumar:

| Nodo | Función | Tecnología |
|---|---|---|
| `route_question` | Clasifica vectorstore vs web search | Pydantic structured output + GPT-4o-mini |
| `retrieve` | Recupera documentos con tenant isolation | Qdrant (real) o InMemory (mock) |
| `grade_documents` | Evalúa relevancia de cada documento | GPT-4o-mini + GradeDocuments schema |
| `transform_query` | Reescribe la pregunta (loop 1) | GPT-4o-mini + StrOutputParser |
| `web_search` | Busca conocimiento externo | Tavily Search API |
| `generate` | Genera respuesta con contexto | GPT-4o-mini + RAG prompt |
| `hallucination_check` | Verifica groundedness + utilidad (loop 2) | GPT-4o-mini + GradeHallucinations + GradeAnswer |

### Loops de auto-corrección

1. **Rewrite loop**: docs irrelevantes → reescribe query → re-intenta (máx. 3)
2. **Regenerate loop**: respuesta alucina → regenera (máx. 2). Si no es útil → reescribe

### Pydantic Structured Output

5 schemas Pydantic v2 implementados:
- `RouteQuery(route: Literal["vectorstore", "web_search"])`
- `GradeDocuments(score: Literal["yes", "no"])`
- `GradeHallucinations(score: Literal["yes", "no"])`
- `GradeAnswer(score: Literal["yes", "no"])`
- `QuestionRewriter(question: str)`

---

## 2. Notebook incluido ✅

**Ubicación**: `notebooks/adaptive_rag_demo.ipynb`

14 celdas que ejecutan el grafo con `graph.stream()` en vivo:
- Celda 1: título + requisito de API key
- Celda 2: validación de OPENAI_API_KEY
- Celdas 3-4: imports + construcción del grafo
- Celdas 5-6: consulta de ejemplo (vectorstore route)
- Celdas 7-9: demo del rewrite loop
- Celdas 10-12: demo del hallucination check
- Celda 13: visualización del grafo
- Celda 14: resumen

---

## 3. Buenas prácticas ✅

| Práctica | Evidencia |
|---|---|
| **Estructura de carpetas** | `app/` (domain, adaptive_rag, ingestion, retrieval, tools, observability, security), `tests/`, `data/`, `notebooks/`, `docs/` |
| **Commits descriptivos** | 10 commits con conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`) |
| **.gitignore** | Cubre `__pycache__`, `.venv`, `.env`, `.DS_Store`, builds |
| **Testing** | 58 tests unitarios + integración, pytest |
| **Linting** | Ruff (all checks passed) |
| **Type checking** | Mypy strict mode |
| **Lazy imports** | Dependencias pesadas importadas dentro de funciones (certilab-rag-patterns) |
| **Protocol interfaces** | `VectorIndex`, `RagPipeline` definidos como `typing.Protocol` |

---

## 4. README claro y completo ✅

**Ubicación**: `README.md` (183 líneas en español)

Contenido:
- Descripción del proyecto + referencia al artículo
- Diagrama Mermaid del grafo Adaptive RAG
- Explicación nodo por nodo
- Tabla de modos (mock vs real)
- Pipeline de ingesta documentado
- Instalación y uso
- Estructura del proyecto
- Tecnologías utilizadas
- Referencias

---

## 5. Stack del curso aplicado ✅

| Tema del curso | Dónde se aplica |
|---|---|
| **LangGraph** | `app/adaptive_rag/graph.py` — StateGraph con nodos y edges condicionales |
| **LangChain** | `langchain-openai` para ChatOpenAI con structured output |
| **OpenAI API** | GPT-4o-mini para routing, grading, generation. text-embedding-3-small para embeddings |
| **Tavily** | `app/tools/web_search.py` — búsqueda web integrada |
| **Pydantic v2** | Schemas para routing, grading, y configuración via `BaseSettings` |
| **FastAPI** (bonus) | Pipeline productivo en repo hermano (`certilab-agentic-rag`) |
| **Chainlit** (bonus) | UI en repo hermano |
| **Observabilidad** | Phoenix/OpenInference con `trace_span` en todos los nodos |

---

## 6. Adicionales (fuera del alcance del enunciado)

| Componente | Descripción |
|---|---|
| **Pipeline de ingesta real** | MySQL → S3 → PyMuPDF → Camelot → Unstructured → Qdrant |
| **Datos reales** | 154 certificados de calibración con metadatos de clientes reales |
| **Tenant isolation** | Filtrado por `customer_id` en Qdrant para consultas multi-cliente |
| **Observabilidad** | Phoenix tracing en todos los nodos del grafo |
| **Dual mode** | Mock (sin dependencias) + Real (MySQL/S3/Qdrant) |

---

## 🔗 Enlaces

- **Repositorio**: https://github.com/emersonheto/certilab-adaptive-rag
- **Artículo de referencia**: https://levelup.gitconnected.com/building-an-adaptive-rag-system-with-langgraph-openai-and-tavily-c4ee39d2f021
- **Notebook**: `notebooks/adaptive_rag_demo.ipynb`
- **Proyecto productivo complementario**: https://github.com/emersonheto/certilab-agentic-rag
