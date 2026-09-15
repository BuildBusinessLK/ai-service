import os
import re

# Set writable cache directory for serverless environments (Vercel has read-only root)
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

from langchain_community.vectorstores import FAISS

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
    USE_FASTEMBED = True
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings
    USE_FASTEMBED = False
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_factory import get_llm

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "vectorstore")
DB_PATH = os.getenv("VECTORSTORE_PATH", DEFAULT_DB_PATH)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Decide offline mode: only force offline mode if HF_FORCE_OFFLINE is explicitly set.
force_offline = os.getenv("HF_FORCE_OFFLINE", "").lower() in ("1", "true", "yes")
if force_offline:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    LOCAL_FILES_ONLY = True
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    LOCAL_FILES_ONLY = False

# ──────────────────────────────────────────────
# Domain guard
# ──────────────────────────────────────────────

SUPPORTED_SECTORS = {
    "coconut": ("coconut", "pol", "coco", "copra", "coir"),
    "palmyrah": ("palmyrah", "thal", "palmyra"),
    "kithul": ("kithul", "kitul"),
}

KNOWN_UNSUPPORTED_TERMS = (
    "rubber", "tea", "coffee", "cinnamon", "pepper", "clove", "cardamom",
    "rice", "paddy", "spice", "spices", "garment", "garments", "textile",
    "textiles", "gem", "gems", "handicraft", "handicrafts", "fishery", "fish",
    "retail", "tourism", "apparel"
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
    topic = ", ".join(terms) if terms else "that domain"
    return (
        "BuildBusinessLK exclusively supports Sri Lanka's **Coconut (Pol)**, **Kithul**, "
        "and **Palmyrah (Thal)** indigenous value chains. We do not provide advisory or recommendations for "
        f"**{topic}**. Please select or ask about opportunities in Coconut, Kithul, or Palmyrah (Thal)."
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

BuildBusinessLK Built-in Features (let the user know and encourage them to use these features when relevant):
- Website Builder: Users can generate and host their own modern SME business website in one click under Marketing > Website (or by clicking the 'Create website' button).
- Ad & Social Media Generator: Generate Facebook, Instagram, and WhatsApp ads with instant copy and visuals under Marketing > Social.
- Email Campaigns: Generate professional email newsletters and send targeted campaigns to customer groups under Marketing > Email.
- ML Business & Product Recommendation: AI/ML tool to find which product or business suits the user based on budget, yield, and workforce.
- Business Profile: Store registration, contact, and product catalog details under Business Profile.

When a user asks how to create a website, create ads, or send marketing emails, provide helpful steps and explicitly guide them to use BuildBusinessLK's built-in feature, mentioning the action button available below your message!

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

def _extract_sector_from_context(user_context: str) -> str | None:
    if not user_context:
        return None
    normalized = user_context.lower()
    for sector, keywords in SUPPORTED_SECTORS.items():
        if any(kw in normalized for kw in keywords):
            return sector
    return None


class SMEAdvisorChain:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs: dict) -> dict:
        question = inputs["input"]
        chat_history = _format_chat_history(inputs.get("chat_history", []))
        user_context = str(inputs.get("user_context") or "").strip()
        if not user_context:
            user_context = "No registered user or business profile was provided."

        supported_sectors = _get_supported_sectors(question)
        unsupported_terms = _get_unsupported_terms(question)

        # Hard guard only when user explicitly asks about known unsupported agricultural sectors
        if unsupported_terms and not supported_sectors:
            return {
                "answer": _unsupported_message(unsupported_terms),
                "context": [],
            }

        history_sector = None
        if not supported_sectors and chat_history:
            history_sector = _extract_sector_from_context(chat_history)

        context_sector = _extract_sector_from_context(user_context)
        effective_sector = supported_sectors[0] if supported_sectors else (history_sector or context_sector or "coconut")

        retrieval_query = question if supported_sectors else f"{effective_sector} {question}"
        documents = self.retriever.invoke(retrieval_query)
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
    if USE_FASTEMBED:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_dir="/tmp")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": LOCAL_FILES_ONLY},
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

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    return SMEAdvisorChain(retriever, llm, prompt)