# BuildBusinessLK AI Service - Documentation

## 1. Overview
A Retrieval-Augmented Generation (RAG) based AI service designed to provide domain-specific knowledge about the Sri Lankan agriculture industry (Coconut, Kithul, Palmyra) and market prices.

## 2. Architecture & Project Structure

The system consists of a **Spring Boot** gateway and a **Python FastAPI** AI engine.

```text
BuildBusinessLK/
├── ai-service/ (Python - FastAPI)
│   ├── data/                 # Knowledge Base (.txt files)
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

## 4. Data Conversion & Access
The AI uses a "Flat File Retrieval" strategy:
1. **Categorization**: Data is grouped into domain-specific text files (`coconut.txt`, `prices.txt`, etc.).
2. **Standardization**: Raw info is converted to structured UTF-8 `.txt` files to ensure character compatibility and clean parsing by LangChain's `TextLoader`.
3. **Vectorization**: `ingest.py` splits these files into 1000-character chunks with a 200-character overlap, allowing the AI to find relevant context without losing meaning.

## 5. System Workflow
1. **User Request**: User sends a JSON request to the Spring Boot `/ask` endpoint.
2. **Inter-Service Call**: Spring Boot forwards the request to the FastAPI engine.
3. **Retrieval**: LangChain searches the FAISS index to find the top 3 most relevant context chunks from the `.txt` data.
4. **Generation**: The relevant context + user question is sent to the local **Llama3** model.
5. **Structured Response**: The final answer is wrapped in a JSON body (Success flag + Code + Answer) and returned to the user.

## 6. Next Steps for Improvement
*   **Data Expansion**: Integrate PDF loaders and direct Database connectors for real-time market data.
*   **Hybrid Search**: Combine Vector Search with Keyword (BM25) search to improve accuracy for specific terminology.
*   **Agentic Orchestration**: Use LangGraph or LangChain Agents to allow the AI to decide when to call external Price APIs vs. searching the static knowledge base.
*   **Query Refinement**: Implement a "re-ranking" step to ensure the most relevant documents are prioritized before passing to the LLM.
