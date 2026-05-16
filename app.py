import json
import os
import re
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from rag.query import get_qa_chain

load_dotenv()

app = FastAPI(title="BuildBusinessLK AI Service", version="3.0")

# Allow Spring Boot backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8083", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialise RAG chain at startup (loads FAISS index + embeddings once)
qa_chain = get_qa_chain()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    question: str
    chat_history: List[ChatMessage] = Field(default_factory=list)
    userProfile: Optional[dict] = None
    businessProfile: Optional[dict] = None


class WebsiteCopyBody(BaseModel):
    businessProfile: Optional[dict] = None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _message_to_dict(msg: ChatMessage) -> dict:
    return msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()


def _format_profiles(user_profile: Optional[dict], business_profile: Optional[dict]) -> str:
    parts: list[str] = []
    if user_profile:
        parts.append("User profile (JSON):\n" + json.dumps(user_profile, ensure_ascii=False, indent=2))
    else:
        parts.append("User profile: not provided.")
    if business_profile:
        parts.append("Business profile (JSON):\n" + json.dumps(business_profile, ensure_ascii=False, indent=2))
    else:
        parts.append("Business profile: not provided.")
    return "\n\n".join(parts)


def _parse_json_object(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return json.loads(s)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL}


@app.post("/chat")
def chat(body: ChatBody):
    """
    Primary chat endpoint called by the Spring Boot backend.
    Accepts the user question + chat history + user/business profile context.
    Returns the AI answer as {"message": "..."}
    """
    user_context = _format_profiles(body.userProfile, body.businessProfile)
    result = qa_chain.invoke({
        "input": body.question,
        "chat_history": [_message_to_dict(m) for m in body.chat_history],
        "user_context": user_context,
    })
    return {"message": result["answer"]}


@app.post("/website-copy")
def website_copy(body: WebsiteCopyBody):
    """
    Generates short marketing copy (hero, about, marketing text) for an SME website
    based on the stored business profile.
    """
    bp = body.businessProfile or {}
    raw = json.dumps(bp, ensure_ascii=False, indent=2)

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.35)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You write website copy for Sri Lankan SMEs in coconut, kithul, and palmyrah value chains. "
            "Return ONLY a single JSON object with keys heroText, aboutText, marketingText. "
            "No markdown fences. Use British or Sri Lankan English. "
            "Do not invent certifications, awards, or guarantees.",
        ),
        (
            "human",
            "Business context (JSON):\n{business}\n\nWrite short, convincing copy for a one-page website.",
        ),
    ])

    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"business": raw})

    try:
        data = _parse_json_object(out)
    except Exception:
        name = (bp.get("businessName") or "Our business").strip()
        data = {
            "heroText": f"Welcome to {name}",
            "aboutText": "Quality Sri Lankan products, crafted with care and delivered with integrity.",
            "marketingText": "Explore our range and connect with us today.",
        }

    return {
        "heroText": str(data.get("heroText", "")).strip(),
        "aboutText": str(data.get("aboutText", "")).strip(),
        "marketingText": str(data.get("marketingText", "")).strip(),
    }
