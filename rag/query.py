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

DB_PATH = "rag/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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
        search_kwargs={"k": 5, "fetch_k": 12},
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
    )

    system_prompt = """
You are BuildBusinessLK's free local AI advisor for Sri Lankan small and medium enterprises.

Answer in clear English. Do not force Sinhala output. Understand local product names:
- pol = coconut
- thal = palmyrah
- kithul = kithul

BuildBusinessLK currently has verified data only for coconut/pol, thal/palmyrah, and kithul.
If the user asks about any other sector or product, do not guess and do not use generic web knowledge. Politely say that dataset is not available yet, updates are on the way, and the dataset will be requested from admins soon.
Use the local knowledge base first. Use live web context only as supporting information when it is provided, and never use it to answer unsupported sectors.
Focus on Sri Lankan SME reality: low budget, local raw materials, village-level producers, small shops, online sellers, cooperatives, export readiness, food safety, packaging, pricing, and repeat customers.

When giving recommendations:
1. Start with the best practical recommendation.
2. Explain why it fits the entrepreneur's sector, budget, and market.
3. Give step-by-step actions they can start this week.
4. Include marketing ideas using free or low-cost channels.
5. Mention risks, compliance, quality control, and what data is missing.
6. If the question is vague, give useful guidance and ask 1-3 follow-up questions.

Do not invent exact prices, laws, certifications, grants, or export requirements. If the data is missing or outdated, say so and suggest how to verify it.

Domain guard:
{domain_notice}

Local knowledge:
{context}

Live web context:
{web_context}

Conversation so far:
{chat_history}

Use the conversation so far to remember details the user already gave. If important details are missing, ask a clear follow-up question and keep the conversation moving.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    return SMEAdvisorChain(retriever, llm, prompt)