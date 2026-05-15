import importlib
import os
import re

# The API should run fully free/local after `rag/ingest.py` has downloaded
# the embedding model and rebuilt the FAISS index.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_PATH = os.getenv("VECTORSTORE_PATH", "rag/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
WEB_SEARCH_LIMIT = int(os.getenv("WEB_SEARCH_LIMIT", "3"))

SUPPORTED_SECTORS = {
    "coconut": ("coconut", "pol", "coco", "copra", "coir"),
    "palmyrah": ("palmyrah", "thal", "palmyra"),
    "kithul": ("kithul", "kitul"),
}

KNOWN_UNSUPPORTED_TERMS = (
    "rubber",
    "tea",
    "coffee",
    "cinnamon",
    "pepper",
    "rice",
    "paddy",
    "spice",
    "spices",
)


def _get_supported_sectors(question: str) -> list[str]:
    normalized_question = question.lower()
    return [
        sector
        for sector, keywords in SUPPORTED_SECTORS.items()
        if any(keyword in normalized_question for keyword in keywords)
    ]


def _get_unsupported_terms(question: str) -> list[str]:
    normalized_question = question.lower()
    return [
        term
        for term in KNOWN_UNSUPPORTED_TERMS
        if re.search(rf"\b{re.escape(term)}\b", normalized_question)
    ]


def _unsupported_dataset_message(unsupported_terms: list[str] | None = None) -> str:
    topic = ", ".join(unsupported_terms) if unsupported_terms else "that sector"
    return (
        "Sorry, at the moment BuildBusinessLK only has verified data for coconut "
        "(pol), thal/palmyrah, and kithul. We do not have enough dataset coverage "
        f"to answer about {topic} yet. We will request this dataset from admins "
        "and support these answers very soon. Right now, I can help you with "
        "coconut, thal/palmyrah, or kithul business questions."
    )


def _domain_notice(unsupported_terms: list[str]) -> str:
    if not unsupported_terms:
        return "The user asked only about supported sectors."

    return (
        "The user also asked about unsupported sectors: "
        f"{', '.join(unsupported_terms)}. Do not provide recommendations, product "
        "lists, pricing, market claims, or web-based guesses for those sectors. "
        "Briefly say BuildBusinessLK currently has verified data only for coconut "
        "(pol), thal/palmyrah, and kithul, and that the missing dataset will be "
        "requested from admins soon. Answer only the supported part of the question."
    )


def _question_needs_web_search(question: str) -> bool:
    keywords = (
        "latest",
        "current",
        "today",
        "recent",
        "trend",
        "price",
        "export",
        "buyer",
        "competitor",
    )
    normalized_question = question.lower()
    has_supported_sector = bool(_get_supported_sectors(question))
    has_unsupported_sector = bool(_get_unsupported_terms(question))
    return (
        ENABLE_WEB_SEARCH
        and has_supported_sector
        and not has_unsupported_sector
        and any(keyword in normalized_question for keyword in keywords)
    )


def _search_web(question: str) -> str:
    if not _question_needs_web_search(question):
        return "No live web search was used for this answer."

    try:
        search_module = importlib.import_module("ddgs")
    except ImportError:
        return (
            "Live web search is available only after installing the free "
            "`ddgs` package from requirements.txt."
        )

    DDGS = search_module.DDGS

    search_query = (
        f"{question} Sri Lanka SME coconut palmyrah kithul business "
        "marketing export agriculture"
    )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=WEB_SEARCH_LIMIT))
    except Exception as exc:
        return f"Live web search failed, so only local knowledge was used. Error: {exc}"

    if not results:
        return "Live web search returned no useful results."

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled result")
        body = result.get("body", "No summary available")
        href = result.get("href", "No URL available")
        formatted_results.append(f"{index}. {title}\nSummary: {body}\nURL: {href}")

    return "\n\n".join(formatted_results)


def _format_documents(documents) -> str:
    if not documents:
        return "No matching local documents were found."

    formatted_docs = []
    for document in documents:
        source = document.metadata.get("source", "unknown source")
        formatted_docs.append(f"Source: {source}\n{document.page_content}")

    return "\n\n---\n\n".join(formatted_docs)


def _format_chat_history(chat_history) -> str:
    if not chat_history:
        return "No previous messages in this conversation."

    formatted_messages = []
    for message in chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        if content:
            formatted_messages.append(f"{role}: {content}")

    return "\n".join(formatted_messages) or "No previous messages in this conversation."


class SMEAdvisorChain:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs):
        question = inputs["input"]
        chat_history = _format_chat_history(inputs.get("chat_history", []))
        supported_sectors = _get_supported_sectors(question)
        unsupported_terms = _get_unsupported_terms(question)

        if not supported_sectors:
            return {
                "answer": _unsupported_dataset_message(unsupported_terms),
                "context": [],
                "web_context": "No live web search was used for this answer.",
            }

        documents = self.retriever.invoke(question)
        context = _format_documents(documents)
        web_context = _search_web(question)
        answer = self.chain.invoke(
            {
                "context": context,
                "web_context": web_context,
                "chat_history": chat_history,
                "domain_notice": _domain_notice(unsupported_terms),
                "input": question,
            }
        )
        return {
            "answer": answer,
            "context": documents,
            "web_context": web_context,
        }


def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 14},
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.35,
    )

    system_prompt = """
You are BuildBusinessLK: an AI business growth assistant for Sri Lankan SMEs in coconut (pol),
palmyrah/thal, and kithul value chains.

Language & names:
- Answer in clear English. Local names: pol=coconut, thal=palmyrah, kithul=kitul.
- Do not output Markdown headings with # symbols. You may use **bold** and *emphasis* sparingly for key terms only.

Your job (not the user's homework):
- YOU synthesize market and sector insight from the Local knowledge and Live web context sections below.
- Do NOT tell the owner to "conduct market research", "analyze competitors", or "study demand" as a standalone
  to-do unless you immediately pair it with concrete findings or comparisons drawn FROM the provided context.
  Instead, phrase it as guidance based on retrieved information, e.g. "In Sri Lanka's coconut sector, export-oriented
  SME products often compete on X; typical constraints include Y" — then give practical next steps.

If the question is thin on detail:
- Offer the best provisional guidance you can from context, then ask 1–4 short clarifying questions
  (product form, scale, district, target channel retail vs export, monthly volume, equipment budget).

How to structure answers:
1. Start with 2–4 sentences: direct answer or situation summary for this SME in Sri Lanka.
2. Then give either:
   (A) Two or three strategic OPTIONS (e.g. "Focus on retail", "Pilot export niche", "Stabilize supply first").
      For EACH option include Pros, Cons, and who it fits (budget, risk tolerance, time horizon).
   OR (B) A numbered list of concrete next actions the owner can start this week — each step must be specific
      to coconut/palmyrah/kithul and SME reality (not generic business textbook steps).

3. Include marketing that is realistic: low-cost social posts, local fairs, wholesale shops, cooperatives,
   EDB-style programmes where mentioned in context—but do not invent programme names or guarantees.

4. Risks & unknowns: state what cannot be known without more data; avoid inventing exact prices, laws,
   certifications, grants, or export rules. If unsure, say so briefly and suggest verification paths
   (e.g. contact CDA/PDB/EDB) without treating that as "the user must do research alone".

5. Unsupported sectors (tea, rubber, etc.): obey Domain guard below; do not fabricate sector facts.

Formatting:
- Use short section titles as plain text lines (no #).
- Prefer numbered lists like "1. " "2. " with a blank line before the list when helpful.

Domain guard:
{domain_notice}

Local knowledge:
{context}

Live web context:
{web_context}

Conversation so far:
{chat_history}

Use the conversation to remember what the user already said. Keep tone practical, respectful, and concise.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    return SMEAdvisorChain(retriever, llm, prompt)