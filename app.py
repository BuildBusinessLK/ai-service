import json
import os
import re
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from rag.query import get_qa_chain

load_dotenv()

app = FastAPI(title="BuildBusinessLK AI", version="2.0")

qa_chain = get_qa_chain()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    question: str
    chat_history: List[ChatMessage] = Field(default_factory=list)
    userProfile: Optional[dict] = None
    businessProfile: Optional[dict] = None


class LegacyQuery(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    chat_history: List[ChatMessage] = Field(default_factory=list)
    user_context: Optional[str] = None


class WebsiteCopyBody(BaseModel):
    businessProfile: Optional[dict] = None


def _message_to_dict(message: ChatMessage) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


def _format_profiles(user_profile: Optional[dict], business_profile: Optional[dict]) -> str:
    parts: List[str] = []
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


@app.post("/chat")
def chat(body: ChatBody):
    user_context = _format_profiles(body.userProfile, body.businessProfile)
    result = qa_chain.invoke(
        {
            "input": body.question,
            "chat_history": [_message_to_dict(m) for m in body.chat_history],
            "user_context": user_context,
        }
    )
    return {"message": result["answer"]}


@app.post("/ask")
def ask_legacy(query: LegacyQuery):
    """Backward-compatible endpoint; prefer POST /chat."""
    result = qa_chain.invoke(
        {
            "input": query.question,
            "chat_history": [_message_to_dict(m) for m in query.chat_history],
            "user_context": (query.user_context or "").strip(),
        }
    )
    msg = result["answer"]
    return {"answer": msg, "message": msg}


@app.post("/website-copy")
def website_copy(body: WebsiteCopyBody):
    bp = body.businessProfile or {}
    raw = json.dumps(bp, ensure_ascii=False, indent=2)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.35)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write website copy for Sri Lankan SMEs in coconut, kithul, and palmyrah value chains. "
                "Return ONLY a single JSON object with keys heroText, aboutText, marketingText. "
                "No markdown fences. British or Sri Lankan English. No invented certifications or guarantees.",
            ),
            (
                "human",
                "Business context (JSON):\n{business}\n\nProduce short, convincing copy suitable for a one-page site.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"business": raw})
    try:
        data = _parse_json_object(out)
    except Exception:
        name = (bp.get("businessName") or "Our business").strip()
        data = {
            "heroText": f"Welcome to {name}",
            "aboutText": "We bring quality Sri Lankan products to customers who care about authenticity.",
            "marketingText": "Explore our range and connect with us today.",
        }
    return {
        "heroText": str(data.get("heroText", "")).strip(),
        "aboutText": str(data.get("aboutText", "")).strip(),
        "marketingText": str(data.get("marketingText", "")).strip(),
    }
