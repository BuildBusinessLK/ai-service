# BuildBusinessLK AI Service - Documentation

## 1. Overview
A Retrieval-Augmented Generation (RAG) based AI service designed to provide domain-specific knowledge about the Sri Lankan agriculture industry (Coconut, Kithul, Palmyra) and market prices.

## 2. Architecture & Project Structure

The system consists of a **Spring Boot** gateway and a **Python FastAPI** AI engine.

```text
BuildBusinessLK/
├── ai-service/ (Python - FastAPI)
│   ├── data/                 # Knowledge Base (.txt files + optional _scraped/)
│   ├── scripts/              # Optional fetch helpers for public portals
│   ├── rag/                  # RAG Implementation
│   │   ├── ingest.py         # Converts text to Vector Embeddings
│   │   ├── query.py          # Handles Retrieval & LLM Generation
│   │   └── vectorstore/      # Local FAISS Vector Database
│   └── app.py                # FastAPI endpoints
└── backend/ (Java - Spring Boot)
    └── src/main/java/com/backend/
        ├── controller/       # AiController (External API gateway)
        ├── service/          # AiService (Communicates with AI Engine)
        └── dto/              # Structured JSON Request/Response DTOs
```

## 3. Libraries & Tools
- **Frameworks**: FastAPI (Python), Spring Boot (Java)
- **AI Orchestration**: LangChain
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM**: Ollama (`llama3`)
- **Communication**: RestTemplate (Java to Python)

## 4. Data conversion & access
The AI uses flat-file retrieval from UTF-8 `.txt` files under `data/` (including subfolders such as `data/_scraped/`).

1. **Categorization**: Domain files like `coconut.txt`, `kithul.txt`, `official_sources_lk_institutions.txt`, etc.
2. **Standardization**: Structured UTF-8 for LangChain `TextLoader`.
3. **Vectorization**: `ingest.py` splits into ~800-character chunks with ~120 overlap.
4. **Rebuild after edits**: run `python rag/ingest.py` so FAISS matches the new text.

Optional **live web snippets** for keywords like `latest`, `price`, `export` (see `rag/query.py`): set `ENABLE_WEB_SEARCH=true` in `.env` (uses the `ddgs` package).

Optional **HTML fetch** for official sites (EDB, CDA, PDB, KDB):

```bash
pip install -r requirements.txt
python scripts/fetch_lk_institution_pages.py
python rag/ingest.py
```

Respect each site's terms and robots rules; treat this as an occasional refresh, not a hammer.

## 5. SME advisor behaviour
The prompt in `rag/query.py` tells the model to synthesize answers from the knowledge base (and optional web context), avoid homework-style commands (“do market research”) without concrete sector findings, present options with **Pros/Cons** when useful, and ask short clarifying questions when user inputs are vague.

## 6. System workflow
1. **User request**: JSON to Spring Boot `/ask` (question + `conversationId`). Chat history is stored in the DB and replayed to Python.
2. **Inter-service call**: Spring posts to FastAPI `POST /ask`.
3. **Retrieval**: FAISS MMR over local embeddings (see `k` / `fetch_k` in `rag/query.py`).
4. **Generation**: **Ollama `llama3`** (or `OLLAMA_MODEL` override) with the SME system prompt.
5. **UI**: Assistant replies are rendered with paragraphs/lists/bold via `frontend/src/utils/assistantTextFormat.jsx`.

## 7. Next steps for improvement
* Dataset expansion (PDFs, internal DBs, verified price feeds).
* Hybrid search (BM25 + vectors) for HS codes and product names.
* Fine-tuning or LoRA on Sri Lankan SME dialogues (beyond RAG + prompt).
* Agent orchestration for authenticated APIs instead of generic web snippets.
