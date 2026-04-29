import importlib
import os

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
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
WEB_SEARCH_LIMIT = int(os.getenv("WEB_SEARCH_LIMIT", "3"))


def _question_needs_web_search(question: str) -> bool:
    keywords = (
        "latest",
        "current",
        "today",
        "recent",
        "trend",
        "market",
        "price",
        "export",
        "buyer",
        "competitor",
        "marketing",
        "social media",
        "online",
        "web",
    )
    normalized_question = question.lower()
    return any(keyword in normalized_question for keyword in keywords)


def _search_web(question: str) -> str:
    if not ENABLE_WEB_SEARCH or not _question_needs_web_search(question):
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


class SMEAdvisorChain:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs):
        question = inputs["input"]
        documents = self.retriever.invoke(question)
        context = _format_documents(documents)
        web_context = _search_web(question)
        answer = self.chain.invoke(
            {
                "context": context,
                "web_context": web_context,
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

Use the local knowledge base first. Use live web context only as supporting information when it is provided.
Focus on Sri Lankan SME reality: low budget, local raw materials, village-level producers, small shops, online sellers, cooperatives, export readiness, food safety, packaging, pricing, and repeat customers.

When giving recommendations:
1. Start with the best practical recommendation.
2. Explain why it fits the entrepreneur's sector, budget, and market.
3. Give step-by-step actions they can start this week.
4. Include marketing ideas using free or low-cost channels.
5. Mention risks, compliance, quality control, and what data is missing.
6. If the question is vague, give useful guidance and ask 1-3 follow-up questions.

Do not invent exact prices, laws, certifications, grants, or export requirements. If the data is missing or outdated, say so and suggest how to verify it.

Local knowledge:
{context}

Live web context:
{web_context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    return SMEAdvisorChain(retriever, llm, prompt)