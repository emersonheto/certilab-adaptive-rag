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

### Prompts optimizados

Todos los prompts del sistema aplican patrones de prompt engineering:
- Estructura `<role>`, `<task>`, `<constraints>`, `<few_shot_examples>`
- Few-shot con ejemplos del dominio (certificados, clientes, mediciones)
- Constraints explícitas para reducir alucinaciones y mejorar routing

---

## 2. Notebook incluido ✅

**Ubicación**: `notebooks/adaptive_rag_demo.ipynb` — 20 celdas, nbformat 4.5

Soporta ejecución en **Google Colab** y **local**:

| Celda | Contenido |
|---|---|
| 0 | Título + referencia al artículo |
| 1 | Setup Colab (clona repo + instala deps) / local (saltear) |
| 2 | Configuración de API keys (Colab Secrets / .env / input manual) |
| 3 | Imports |
| 4 | Construcción de componentes (Qdrant real o InMemory mock) |
| 5 | Helpers: `make_state()` + `run()` con output limpio |
| 6-7 | **Query A**: Procedimiento INDECOPI (vectorstore route) |
| 8-9 | **Query B**: Tenant isolation — ALS PERU |
| 10-11 | **Self-correction intro** + Loop 1: Reescritura (query ambigua) |
| 12-13 | **Query D**: Datos de tablas (medición 105°C) |
| 14-15 | **Query E**: Metadatos (fecha + tipo de documento) |
| 16-17 | Visualización del grafo (Mermaid) |
| 18-19 | Resumen + stack |

### Output limpio

Cada query muestra solo lo relevante:
```
Ruta: vectorstore
Reescrituras: 1
Regeneraciones: 0
Verificado: grounded | Util: useful

La norma INDECOPI sigue el procedimiento SNM PC-018 (2° ed. –2009)...
```

---

## 3. Buenas prácticas ✅

| Práctica | Evidencia |
|---|---|
| **Estructura de carpetas** | `app/` (adaptive_rag, domain, ingestion, retrieval, tools, observability, security), `tests/`, `data/`, `notebooks/`, `docs/` |
| **Commits descriptivos** | 24 commits con conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) |
| **.gitignore** | Cubre `__pycache__`, `.venv`, `.env`, `.DS_Store`, builds, `.codegraph/` |
| **Testing** | 58 tests (unitarios + integración), pytest, 6 tests para ingest helpers |
| **Linting** | Ruff — All checks passed (notebooks excluidos como presentation code) |
| **Type checking** | Mypy strict mode |
| **Lazy imports** | `ChatOpenAI`, `StrOutputParser`, `langgraph` importados dentro de funciones (certilab-rag-patterns rule 1) |
| **Protocol interfaces** | `VectorIndex`, `RagPipeline` definidos como `typing.Protocol` |
| **modern-python** | `from __future__ import annotations` en todos los archivos, `X \| None`, `match/case`, `@dataclass(frozen=True)`, `@functools.cache` |
| **VS Code config** | `.vscode/settings.json` para resolución de `.env` en notebooks |

---

## 4. README claro y completo ✅

**Ubicación**: `README.md` — 227 líneas en español

Contenido:
- Descripción del proyecto + referencia al artículo
- Tabla de modos (mock vs real)
- Diagrama Mermaid del grafo Adaptive RAG
- Explicación nodo por nodo con tabla de tecnologías
- Descripción de los 3 loops de auto-corrección
- Pipeline de ingesta documentado (MySQL → S3 → PyMuPDF → Camelot → Unstructured → Qdrant)
- Instalación (incluyendo `--extra real`)
- Uso: demo CLI, notebook, ingesta real
- Consultas de ejemplo
- Tabla completa de tecnologías (15 herramientas)
- Estructura del proyecto
- Pruebas
- Referencias (artículo, LangGraph, PyMuPDF, Camelot, Unstructured)

---

## 5. Stack del curso aplicado ✅

| Tema del curso | Dónde se aplica |
|---|---|
| **LangGraph** | `app/adaptive_rag/graph.py` — StateGraph con nodos, edges condicionales y 2 loops |
| **LangChain** | `langchain-openai` para ChatOpenAI con `.with_structured_output()` |
| **OpenAI API** | GPT-4o-mini (routing, grading, generation) + text-embedding-3-small (embeddings) |
| **Tavily** | `app/tools/web_search.py` — búsqueda web con gating opcional |
| **Pydantic v2** | 5 schemas para structured output + `BaseSettings` con `.env` loading |
| **Chunking** | `RecursiveCharacterTextSplitter` (patrón del profesor) + `chunk_by_title` de Unstructured |
| **Vector store** | Qdrant con payload filtering (tenant isolation por `customer_id`) |
| **Embeddings** | OpenAI text-embedding-3-small (1536-dim) con Phoenix tracing |
| **Observabilidad** | Phoenix/OpenInference con `trace_span` en todos los nodos del grafo |

---

## 6. Adicionales (más allá del enunciado)

| Componente | Descripción |
|---|---|
| **Pipeline de ingesta real** | MySQL → S3 → PyMuPDF (texto) → Camelot (tablas, 99% accuracy) → Unstructured (chunking semántico) → OpenAI embeddings → Qdrant |
| **Datos reales** | 154 certificados de calibración con 3,848 chunks indexados, metadatos de 12 clientes reales |
| **Tenant isolation** | Filtrado por `customer_id` en Qdrant (`allowed_customer_ids` en payload filter) |
| **Metadatos enriquecidos** | Cada chunk incluye `certificate_code`, `customer_name`, `issue_date` (en español: "mayo 2026"), `status` (Pendiente/Firmado), `document_type` (Acreditado/No acreditado) |
| **Table splitting** | Tablas time-series grandes (>30K chars) se splitean en sub-chunks de ~1.5K con header repetido |
| **Metadata-only chunks** | Certificados sin PDF también se indexan (status, fecha, cliente) |
| **Dual mode** | Mock (sin dependencias, datos locales) + Real (MySQL/S3/Qdrant) |
| **Detección de cliente** | El demo detecta automáticamente el cliente mencionado en la pregunta y filtra el scope |
| **Prompt engineering** | Todos los prompts aplican patrones profesionales: `<role>`, `<task>`, `<constraints>`, `<few_shot_examples>` |

---

## 7. Métricas finales

| Métrica | Valor |
|---|---|
| Archivos | 64 |
| Tests | 58 passed |
| Commits | 24 (conventional) |
| Puntos en Qdrant | 3,848 |
| Certificados indexados | 176 (154 con PDF + 22 metadata-only) |
| Celdas del notebook | 20 |
| Ruff | ✅ All checks passed |
| Pytest | ✅ 58 passed |
| README | 227 líneas, español |

---

## 🔗 Enlaces

- **Repositorio**: https://github.com/emersonheto/certilab-adaptive-rag
- **Artículo de referencia**: https://levelup.gitconnected.com/building-an-adaptive-rag-system-with-langgraph-openai-and-tavily-c4ee39d2f021
- **Notebook**: `notebooks/adaptive_rag_demo.ipynb`
- **Instrucciones para el profesor**: `docs/entrega.md`
- **Proyecto productivo complementario**: https://github.com/emersonheto/certilab-agentic-rag
