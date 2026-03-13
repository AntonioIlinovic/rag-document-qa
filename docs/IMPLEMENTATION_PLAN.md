# RAG Document QA — Implementation Plan

A comprehensive, phased plan for building a Python REST API that ingests documents (PDF/images), extracts text, and answers questions via a RAG pipeline — designed for extensibility and clean engineering practices.

---

## 1. Technology Choices

For each key technology, options are listed with a **recommended** pick.

### Web Framework
| Option | Pros | Cons |
|--------|------|------|
| **FastAPI** ✅ | Async, auto OpenAPI docs, Pydantic-native, dependency injection | Slightly more opinionated |
| Flask | Simple, huge ecosystem | No built-in async, no auto docs, manual validation |

**Recommendation:** FastAPI — auto-generated Swagger docs are great for a demo, native Pydantic integration reduces boilerplate, and dependency injection makes testing easy.

### Package Manager
| Option | Pros | Cons |
|--------|------|------|
| **uv** ✅ | Extremely fast, resolves + locks, replaces pip/venv/poetry | Newer, less community examples |
| Poetry | Mature, well-known | Slower resolution |
| pip + venv | Universal, zero learning curve | No lockfile by default, manual venv |

**Recommendation:** uv — fast, modern, good impression for a tech assignment. Falls back gracefully to standard pip if needed.

### PDF Text Extraction
| Option | Pros | Cons |
|--------|------|------|
| **PyMuPDF (fitz)** ✅ | Fast, reliable, extracts text + metadata, small footprint | No OCR for scanned PDFs |
| pdfplumber | Good table extraction | Slower |
| PyPDF2 | Lightweight | Less reliable text extraction |

**Recommendation:** PyMuPDF — fast and reliable for text-based PDFs. Pair with an OCR solution for scanned documents.

### OCR (Image / Scanned Document Extraction)
| Option | Pros | Cons |
|--------|------|------|
| **EasyOCR** ✅ | Multi-language, good accuracy, pure Python | Heavy (~1GB models), slow on CPU |
| Tesseract (pytesseract) | Lightweight, fast | Requires system-level install, lower accuracy on complex layouts |
| docTR | Modern, good accuracy | Fewer community examples |

**Recommendation:** EasyOCR — best accuracy out-of-the-box, assignment specifically mentions it. Accept the heavier Docker image.

### Embeddings
| Option | Pros | Cons |
|--------|------|------|
| **sentence-transformers (local)** ✅ | Free, no API key, fast after load, offline | ~400MB model, slower first load |
| OpenAI Embeddings API | High quality, tiny client | Costs money, requires API key, external dependency |

**Recommendation:** sentence-transformers with `all-MiniLM-L6-v2` — free, runs locally, good quality for this use case. Keeps the embedding step independent of the LLM choice.

### Vector Store
| Option | Pros | Cons |
|--------|------|------|
| **ChromaDB** ✅ | Built-in persistence, metadata filtering, simpler API | Extra dependency, server mode adds complexity |
| FAISS | Fast, battle-tested, no server needed | File-based persistence (pickle), no built-in metadata filtering |
| Qdrant | Production-grade, rich filtering | Overkill for this scope |

**Recommendation:** ChromaDB — built-in persistence, metadata filtering, simpler API. For MVP, file-based persistence is fine. Document in README that a production system would use a dedicated vector DB.

### QA / LLM
| Option | Pros | Cons |
|--------|------|------|
| **OpenAI API** ✅ (cloud) | Best quality, easy integration, well-documented | Requires API key, costs money |
| OpenRouter | Access to many models, pay-per-use | Extra abstraction layer, less predictable |
| **DistilBERT QA** ✅ (local) | Free, no API key, runs in Docker | Lower quality answers, extractive only (not generative) |
| Ollama + local LLM | Generative, free | Heavy resource requirements, complex Docker setup |

**Recommendation:** Implement **both** behind an abstract interface. Default to OpenAI API for quality; DistilBERT as a zero-cost fallback. Config-driven switching via environment variable.

### Text Chunking
| Option | Pros | Cons |
|--------|------|------|
| **LangChain RecursiveCharacterTextSplitter** ✅ | Smart splitting, widely used | Pulls in langchain-text-splitters dependency |
| Custom splitter | No dependencies | Must handle edge cases yourself |

**Recommendation:** LangChain's text splitter — well-tested, minimal dependency (`langchain-text-splitters` is a standalone lightweight package).

### Testing
| Option | Pros | Cons |
|--------|------|------|
| **pytest** ✅ | Standard, rich plugin ecosystem, fixtures | None significant |
| unittest | Built-in | More verbose |

**Recommendation:** pytest with `httpx` for async FastAPI testing (via `pytest-asyncio`).

---

## 2. Project Folder Structure

```
rag-document-qa/
├── .env.example                    # Example env vars (committed)
├── .gitignore
├── docker-compose.yml              # Orchestrates backend (+ optional frontend)
├── README.md                       # Setup, usage, approach description
├── dummy_docs/                     # Sample PDFs/images for testing & demo
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml              # uv-managed dependencies
│   ├── uv.lock
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app factory, lifespan, CORS
│   │   ├── config.py               # Pydantic BaseSettings (single source of truth)
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py           # Aggregates all route modules
│   │   │   ├── upload.py           # POST /upload endpoint
│   │   │   ├── ask.py              # POST /ask endpoint
│   │   │   └── deps.py             # FastAPI dependency injection (get_session, etc.)
│   │   │
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── upload.py           # UploadResponse, DocumentInfo
│   │   │   └── ask.py              # AskRequest, AskResponse
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── extraction/
│   │       │   ├── __init__.py
│   │       │   ├── base.py         # ABC: BaseExtractor
│   │       │   ├── pdf.py          # PyMuPDFExtractor
│   │       │   └── ocr.py          # EasyOCRExtractor
│   │       ├── rag/
│   │       │   ├── __init__.py
│   │       │   ├── chunker.py      # Text splitting logic
│   │       │   ├── embedder.py     # ABC + SentenceTransformerEmbedder
│   │       │   └── store.py        # ChromaDB vector store (save/load per session)
│   │       ├── qa/
│   │       │   ├── __init__.py
│   │       │   ├── base.py         # ABC: BaseQAEngine
│   │       │   ├── cloud.py        # OpenAI / API-based QA
│   │       │   └── local.py        # DistilBERT extractive QA
│   │       └── session.py          # Session lifecycle (create, get, delete)
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py             # Shared fixtures (test client, temp sessions)
│   │   ├── test_extraction.py
│   │   ├── test_rag.py
│   │   ├── test_qa.py
│   │   └── test_api.py             # Integration tests for endpoints
│   │
│   └── data/                       # Runtime storage: sessions, ChromaDB indexes (gitignored)
│
└── frontend/                       # OPTIONAL: Streamlit UI (separate service)
    ├── Dockerfile
    ├── pyproject.toml
    └── app.py
```

### Structure Rationale
- **`app/`** is the importable Python package; `main.py` is the entry point
- **`api/`** holds only HTTP concerns (routes, deps); no business logic
- **`services/`** holds all business logic, organized by domain (extraction, rag, qa)
- **`schemas/`** holds Pydantic models for request/response validation
- **Abstract base classes** in `base.py` files enable the Strategy pattern — swap implementations via config
- **`session.py`** is a single file for now; when user-based auth is added, it becomes a `session/` package
- **`tests/`** mirrors the service structure; `conftest.py` provides shared fixtures

---

## 3. Implementation Phases (Recommended Order)

### Phase 1: Project Scaffolding & Configuration
- Initialize `backend/` with `uv init`, set up `pyproject.toml`
- Create `app/config.py` with Pydantic `BaseSettings`:
  - `OPENAI_API_KEY` (optional — only required if using cloud QA)
  - `QA_ENGINE` (enum: `"cloud"` | `"local"`, default `"cloud"`)
  - `DATA_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`
- Create `.env.example`, `.gitignore`
- Create `app/main.py` with basic FastAPI app factory and health check (`GET /health`)
- **Verify:** `uv run uvicorn app.main:app --reload` starts and `/health` returns 200

### Phase 2: Document Extraction
- Implement `BaseExtractor` ABC with method `extract(file_bytes, filename) -> str`
- Implement `PyMuPDFExtractor` for `.pdf` files
- Implement `EasyOCRExtractor` for `.png`, `.jpg`, `.jpeg`, `.tiff`
- Create a factory function: `get_extractor(filename) -> BaseExtractor`
- Add dummy test documents to `dummy_docs/` (at least 1 PDF, 1 image)
- **Write tests:** Verify extraction produces non-empty text for each file type

### Phase 3: RAG Pipeline (Chunking + Embedding + Vector Store)
- Implement `chunker.py`: split text into overlapping chunks (configurable size/overlap)
- Implement `embedder.py`: ABC + `SentenceTransformerEmbedder` using `all-MiniLM-L6-v2`
- Implement `store.py`: ChromaDB vector store wrapper with:
  - `add_documents(chunks, embeddings)`
  - `search(query_embedding, top_k) -> list[str]`
  - `save(path)` / `load(path)` for persistence per session
- **Write tests:** Chunk a sample text, embed it, store it, query it, verify relevant chunks returned

### Phase 4: QA Engine
- Implement `BaseQAEngine` ABC with method `answer(question, context_chunks) -> str`
- Implement `CloudQAEngine` (OpenAI API): send question + context as prompt, return response
- Implement `LocalQAEngine` (DistilBERT): use `transformers` pipeline, return extracted answer
- Factory function: `get_qa_engine(config) -> BaseQAEngine`
- **Write tests:** Mock API calls for cloud; verify local model returns a string

### Phase 5: Session Management & API Endpoints
- Implement `session.py`:
  - `create_session() -> session_id` (UUID-based)
  - `get_session(session_id) -> SessionData` (extracted text, ChromaDB collection name, metadata)
  - Store session state on disk under `data/{session_id}/`
- Implement `POST /upload`:
  - Accept `multipart/form-data` with one or more files
  - Optional `session_id` param (create new if not provided)
  - Extract text → chunk → embed → store in ChromaDB
  - Return `session_id` + list of processed documents
- Implement `POST /ask`:
  - Accept JSON `{ "session_id": "...", "question": "..." }`
  - Load session's ChromaDB collection → search → pass top-k chunks to QA engine → return answer
  - Include source chunks in response for transparency
- Add proper error handling: 404 for invalid session, 422 for bad input, 500 with structured error
- **Write integration tests:** Upload a doc, ask a question, verify response structure

### Phase 6: Dockerization
- Create `backend/Dockerfile`:
  - Multi-stage build (builder with uv + deps, runtime with minimal image)
  - Install system deps for PyMuPDF and EasyOCR (`libgl1`, etc.)
  - Pre-download models in build stage for faster cold starts
- Create `docker-compose.yml`:
  - Backend service with volume mount for `data/`
  - Environment variables from `.env`
- **Verify:** `docker compose up --build` → Swagger UI works → upload + ask works

### Phase 7: Documentation (README.md)
- Project description and approach rationale
- Setup instructions (manual with uv + Docker)
- Example API requests/responses (curl + Swagger screenshots)
- Architecture overview (brief)
- Technology choices and why

---

## 4. Development Practices & Notes

### Code Quality
- **Type hints everywhere** — FastAPI and Pydantic rely on them; they also serve as documentation
- **Dependency injection** via FastAPI's `Depends()` — makes testing trivial (swap real services for mocks)
- **Abstract base classes** for all swappable components (extractors, QA engines, embedders) — Strategy pattern
- **Pydantic models** for all API input/output — automatic validation and documentation
- **No business logic in route handlers** — routes call services, services do the work

### Error Handling
- Use FastAPI's `HTTPException` with meaningful status codes and detail messages
- Create custom exception classes for domain errors (e.g., `SessionNotFoundError`, `ExtractionError`)
- Add a global exception handler that catches unhandled errors and returns structured JSON

### Configuration
- **Single source of truth:** `config.py` with Pydantic `BaseSettings`
- All secrets via environment variables, never hardcoded
- `.env.example` committed with placeholder values; `.env` gitignored
- Use `Field(default=...)` for sensible defaults so the app runs with minimal config

### Testing Strategy
- **Unit tests** for each service in isolation (mock external dependencies)
- **Integration tests** for API endpoints using FastAPI's `TestClient`
- Use pytest fixtures for common setup (temp directories, mock sessions)
- Aim for tests that verify behavior, not implementation details

### Docker Best Practices
- Multi-stage builds to keep image size down
- Pin base image versions (e.g., `python:3.12-slim`)
- Use `.dockerignore` to exclude tests, docs, `.git`
- Pre-download ML models during build to avoid runtime downloads

### Git Practices
- Meaningful commit messages per phase
- `.gitignore`: `.env`, `data/`, `__pycache__/`, `.venv/`, `*.pyc`, model caches

---

## 5. API Design

### `POST /upload`
```
Content-Type: multipart/form-data

Fields:
  - files: one or more files (PDF, PNG, JPG, TIFF)
  - session_id: (optional) string — existing session to add documents to

Response 200:
{
  "session_id": "a1b2c3d4-...",
  "documents": [
    {
      "filename": "contract.pdf",
      "pages": 12,
      "chunks": 47,
      "status": "processed"
    }
  ]
}
```

### `POST /ask`
```
Content-Type: application/json

Body:
{
  "session_id": "a1b2c3d4-...",
  "question": "What is the termination clause?"
}

Response 200:
{
  "answer": "The termination clause states that...",
  "sources": [
    {
      "chunk": "Section 8.1 — Either party may terminate...",
      "score": 0.87
    }
  ]
}
```

### `GET /health`
```
Response 200:
{
  "status": "ok"
}
```

---

## 6. Optional Enhancements (Post-MVP)

Each enhancement is assessed for **complexity**, **files affected**, and **what changes**.

### 6.1 Named Entity Recognition (NER)
- **Complexity:** Medium (1-2 hours)
- **Approach:** Add spaCy with a pre-trained model (`en_core_web_sm`) or a transformer-based NER
- **Changes needed:**
  - New service: `services/ner.py` with entity extraction logic
  - Update `schemas/ask.py`: add `entities` field to `AskResponse`
  - Update `api/ask.py`: call NER service on the answer text
  - Update `Dockerfile`: install spaCy model
- **Notes:** Can run NER on the answer text and/or the source chunks. spaCy is lightweight; transformer NER is more accurate but heavier.

### 6.2 Caching Layer (Redis)
- **Complexity:** Medium (2-3 hours)
- **Approach:** Cache document embeddings and optionally QA results
- **Changes needed:**
  - Add Redis to `docker-compose.yml`
  - New service: `services/cache.py` (or integrate into embedder/store)
  - Update `config.py`: Redis connection settings
  - Update `embedder.py`: check cache before computing embeddings
  - Optionally cache QA results keyed by `(session_id, question_hash)`
- **Notes:** Biggest win is caching embeddings — avoids recomputation on re-upload. Redis also enables TTL-based session expiry.

### 6.3 Streamlit UI
- **Complexity:** Low-Medium (2-3 hours)
- **Approach:** Streamlit app in `frontend/` that calls the backend API
- **Changes needed:**
  - Create `frontend/app.py`, `Dockerfile`, `pyproject.toml`
  - Add frontend service to `docker-compose.yml`
  - Add CORS middleware to backend (already in plan)
- **Notes:** Use `st.file_uploader` for documents, `st.chat_input` + `st.chat_message` for Q&A. Store `session_id` in `st.session_state`.

### 6.4 MLOps (CI/CD, Logging, Monitoring)
- **Complexity:** Medium-High (3-5 hours)
- **Approach:**
  - **CI/CD:** GitHub Actions workflow for linting (ruff), testing (pytest), Docker build
  - **Logging:** Replace `print` with `structlog` or Python's `logging` with JSON formatter
  - **Monitoring:** Add `/metrics` endpoint or integrate with Prometheus
- **Changes needed:**
  - New: `.github/workflows/ci.yml`
  - Update `config.py`: log level settings
  - Add structured logging throughout services
  - Optional: health check enhancements, Prometheus metrics
- **Notes:** Start with CI/CD + structured logging. Monitoring can be deferred.

### 6.5 Performance Optimization
- **Complexity:** Medium (2-4 hours)
- **Approach:** Profile inference latency, optimize embedding batch size, async processing
- **Changes needed:**
  - Add timing middleware to measure request latency
  - Batch embedding calls in `embedder.py`
  - Consider background task processing for `/upload` (FastAPI `BackgroundTasks`)
  - Profile with `cProfile` or `py-spy`
- **Notes:** The biggest bottleneck will be embedding generation and LLM calls. Background processing for upload is the highest-impact optimization.

### 6.6 Security (JWT Auth, Rate Limiting, Input Sanitization)
- **Complexity:** Medium-High (3-4 hours)
- **Approach:** JWT auth for user-based access, rate limiting via middleware, strict input validation
- **Changes needed:**
  - New: `services/auth.py` (JWT creation/validation)
  - New: `api/deps.py` additions for `get_current_user` dependency
  - Add `slowapi` or custom middleware for rate limiting
  - Update `config.py`: JWT secret, token expiry, rate limits
  - Input sanitization: already using Pydantic (validates types); add file size limits, allowed MIME types
- **Notes:** This is the path to upgrade from session-based to user-based. Design session management with a clear interface so swapping in user-scoped sessions is straightforward.

### 6.7 User-Based Document Management (upgrade from session-based)
- **Complexity:** Medium (2-3 hours, depends on 6.6)
- **Approach:** Add user registration/login, scope documents to users instead of sessions
- **Changes needed:**
  - Depends on Security (6.6) for JWT auth
  - New: user model + simple storage (SQLite or JSON file for MVP)
  - Update `session.py` → `user_session.py`: scope ChromaDB collections by user ID
  - Update API deps: inject `current_user` instead of `session_id`
- **Notes:** The abstract session interface makes this a clean swap. Main decision is user storage backend (SQLite vs. file-based vs. full DB).

---

## 7. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| EasyOCR OOM in Docker | Increase Docker memory limit (4GB+); fall back to PyMuPDF-only for demo |
| OpenAI API rate limits / costs | Implement LocalQAEngine as fallback; add mock engine for testing |
| Large file uploads | Set file size limits in FastAPI (`UploadFile` + config); validate MIME types |
| Model download on first run | Pre-download models in Dockerfile build stage |
