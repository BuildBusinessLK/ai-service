import os
import re
import importlib

# Keep HuggingFace in offline mode once ingested — no live model downloads during inference.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional model adapters — import dynamically so the service can run without all
# adapters installed. We prefer Ollama as the default (existing behaviour).
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

ChatOpenAI = None
try:
    ChatOpenAI = importlib.import_module("langchain.chat_models").ChatOpenAI
except Exception:
    ChatOpenAI = None

# Google / Gemini adapter (may not be present in every environment). We try
# a few known module locations so the factory below can pick the best option.
ChatGoogleGemini = None
for candidate in ("langchain.chat_models.google", "langchain.chat_models.vertex_ai"):
    try:
        mod = importlib.import_module(candidate)
        # Some langchain versions expose a Gemini/Vertex wrapper with different names
        for attr in ("ChatGoogleGenerativeAI", "ChatVertexAI", "ChatGoogleGemini"):
            if hasattr(mod, attr):
                ChatGoogleGemini = getattr(mod, attr)
                break
        if ChatGoogleGemini:
            break
    except Exception:
        continue

DB_PATH = os.getenv("VECTORSTORE_PATH", "rag/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OLLAMA").upper()


def _get_llm(temperature: float | None = None):
    """Factory to return an LLM/chat model instance based on env vars.

    Supports:
    - OLLAMA: existing Ollama adapter (`langchain_ollama.ChatOllama`).
    - OPENAI: LangChain `ChatOpenAI` wrapper (useful if you route Gemini-like
      models through an OpenAI-compatible API).
    - GEMINI / GOOGLE: LangChain Google/Vertex adapter if available.
    """
    temp = float(temperature) if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.3"))
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).upper()

    if provider == "OLLAMA" and ChatOllama is not None:
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL), temperature=temp)

    if provider in ("OPENAI",) and ChatOpenAI is not None:
        model_name = os.getenv("OPENAI_MODEL", os.getenv("GEMINI_MODEL", "gpt-4o-mini"))
        return ChatOpenAI(model_name=model_name, temperature=temp)

    if provider in ("GEMINI", "GOOGLE") and ChatGoogleGemini is not None:
        model_name = os.getenv("GEMINI_MODEL", "gemini")
        try:
            # different adapters expect different param names; try common ones
            return ChatGoogleGemini(model=model_name, temperature=temp)
        except TypeError:
            return ChatGoogleGemini(model_name=model_name, temperature=temp)

    # Fallbacks
    if ChatOllama is not None:
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", OLLAMA_MODEL), temperature=temp)
    if ChatOpenAI is not None:
        return ChatOpenAI(model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temp)

    raise RuntimeError("No LLM adapter is installed. Install one of: langchain-ollama, langchain (OpenAI), or the Google/Vertex adapters.")

# ──────────────────────────────────────────────
# Domain guard
# ──────────────────────────────────────────────

SUPPORTED_SECTORS = {
    "coconut": ("coconut", "pol", "coco", "copra", "coir"),
    "palmyrah": ("palmyrah", "thal", "palmyra"),
    "kithul": ("kithul", "kitul"),
}

KNOWN_UNSUPPORTED_TERMS = (
    "rubber", "tea", "coffee", "cinnamon", "pepper",
    "rice", "paddy", "spice", "spices",
)


def _get_supported_sectors(question: str) -> list[str]:
    normalized = question.lower()
    return [
        sector
        for sector, keywords in SUPPORTED_SECTORS.items()
        if any(kw in normalized for kw in keywords)
    ]


def _get_unsupported_terms(question: str) -> list[str]:
    normalized = question.lower()
    return [
        term
        for term in KNOWN_UNSUPPORTED_TERMS
        if re.search(rf"\b{re.escape(term)}\b", normalized)
    ]


def _unsupported_message(terms: list[str] | None = None) -> str:
    topic = ", ".join(terms) if terms else "that sector"
    return (
        "I’m sorry, I only have good, verified data for coconut (pol), palmyrah/thal, "
        "and kithul right now. I can’t answer in depth about "
        f"{topic} yet, but I’m happy to help with coconut, palmyrah, or kithul questions."
    )


def _domain_notice(unsupported_terms: list[str]) -> str:
    if not unsupported_terms:
        return "The user asked only about supported sectors."
    return (
        f"The user also asked about unsupported sectors: {', '.join(unsupported_terms)}. "
        "Do NOT provide recommendations, prices, market claims, or guesses for those sectors. "
        "Briefly acknowledge the gap and redirect. Answer only the supported part."
    )


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────

def _format_documents(documents) -> str:
    if not documents:
        return "No matching local documents were found."
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown source")
        parts.append(f"Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_chat_history(chat_history) -> str:
    if not chat_history:
        return "No previous messages in this conversation."
    lines = []
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "No previous messages in this conversation."


# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are BuildBusinessLK — a practical AI business advisor for Sri Lankan SMEs in the
coconut (pol), palmyrah/thal, and kithul value chains.

Registered user & business context (personalise your tone using this; never invent facts):
{user_context}

Your communication style:
- Answer in plain, clear English. Use Sri Lankan product names naturally (pol = coconut, thal = palmyrah, kitul = kithul).
- Do NOT use Markdown headings (#). Use **bold** for key terms sparingly.
- Be conversational and warm — you are talking to a small business owner, not an academic.
- Keep answers focused and actionable. Avoid generic business-school advice.

Your job:
- YOU synthesise insight from the local knowledge base and the user/business context below.
- If the user context contains an ML recommendation summary, treat it as a trusted guidance signal and use it to shape the answer.
- Do NOT tell the owner to "conduct research" as a standalone task.
  Instead, share what you already know from the knowledge base, then give concrete next steps.
- If the question is vague, ask 1–3 short clarifying questions BEFORE giving generic advice.
  Example questions: product form, district, target channel (retail vs export), monthly volume, budget.

Answer structure (adapt as needed):
1. Start with a 2–3 sentence direct answer or situation summary.
2. Then provide either:
   (A) 2–3 strategic options — each with Pros, Cons, and who it fits.
   (B) A short numbered action list — each step specific to this sector and Sri Lankan SME reality.
3. Flag risks or unknowns honestly. If a price, law, or certification is uncertain, say so and suggest
   where to verify (e.g. contact CDA, PDB, EDB) — but do not make this the user's entire homework.
4. For unsupported sectors: follow the domain guard below.

Formatting:
- Use short plain-text section titles (no # symbols).
- Use "1. ", "2. " for numbered lists; "- " for bullet points.
- Leave a blank line before lists.

Domain guard:
{domain_notice}

Local knowledge base:
{context}

Conversation so far:
{chat_history}

Use the conversation history to avoid repeating yourself and to build on what the user already shared.
"""


# ──────────────────────────────────────────────
# Chain
# ──────────────────────────────────────────────

class SMEAdvisorChain:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs: dict) -> dict:
        question = inputs["input"]
        chat_history = _format_chat_history(inputs.get("chat_history", []))
        supported_sectors = _get_supported_sectors(question)
        unsupported_terms = _get_unsupported_terms(question)

        # Hard guard — no supported sector mentioned at all
        if not supported_sectors:
            return {
                "answer": _unsupported_message(unsupported_terms),
                "context": [],
            }

        user_context = str(inputs.get("user_context") or "").strip()
        if not user_context:
            user_context = "No registered user or business profile was provided."

        documents = self.retriever.invoke(question)
        context = _format_documents(documents)

        answer = self.chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "domain_notice": _domain_notice(unsupported_terms),
            "user_context": user_context,
            "input": question,
        })

        return {
            "answer": answer,
            "context": documents,
        }


def get_qa_chain() -> SMEAdvisorChain:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # MMR retrieval — fetch_k=16 candidates, return top 6 diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8},
    )

    llm = _get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    return SMEAdvisorChain(retriever, llm, prompt)