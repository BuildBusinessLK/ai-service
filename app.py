import json
import os
import re
from typing import Any, List, Optional

from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_ollama import ChatOllama  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from rag.query import get_qa_chain

app = FastAPI(title="BuildBusinessLK AI Service", version="3.0")


class ServiceState:
    def __init__(self) -> None:
        self.qa_chain = None
        self.initialization_error: Optional[str] = None


service_state = ServiceState()

# Allow Spring Boot backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8083", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialise RAG chain at startup (loads FAISS index + embeddings once)
try:
    service_state.qa_chain = get_qa_chain()
except Exception as exc:  # pragma: no cover - defensive startup fallback
    service_state.initialization_error = str(exc)

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


class AdGenerationBody(BaseModel):
    prompt: str
    businessProfile: Optional[dict] = None
    userProfile: Optional[dict] = None


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
    if service_state.qa_chain is None:
        fallback = (
            "The AI service is currently running in fallback mode because the local knowledge model could not be loaded. "
            "Please try again shortly or check the local AI service logs."
        )
        return {"message": fallback}

    result = service_state.qa_chain.invoke({
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

    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.35)
    except Exception:
        llm = None

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

    if llm is None:
        return {
            "heroText": f"Welcome to {bp.get('businessName') or 'our business'}",
            "aboutText": "Quality Sri Lankan products, crafted with care and delivered with integrity.",
            "marketingText": "Explore our range and connect with us today.",
        }

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


# @app.post("/ad-generate")
def ad_generate(body: AdGenerationBody):
    """
    Generates structured advertisements for multiple platforms.
    """

    prompt = body.prompt or "Write a marketing advertisement."

    business_profile = body.businessProfile or {}
    user_profile = body.userProfile or {}

    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.4
        )
    except Exception:
        llm = None

    system_prompt = """
You are an expert AI Marketing Strategist.

Your job is to create high-quality marketing advertisements.

Rules:

- Never invent information.
- Only use the supplied business profile.
- Make the advertisements persuasive.
- Use professional English.
- Return ONLY valid JSON.
- Never return markdown.
- Never return explanations.
"""

    if llm is None:

        return {
            "headline": "Advertisement",
            "facebook": f"Discover {business_profile.get('businessName','our business')} today.",
            "instagram": "Visit us today.",
            "google": "Quality Products",
            "headlines": [
                "Best Quality",
                "Shop Today",
                "Special Offer",
                "Trusted Business",
                "Contact Us"
            ],
            "hashtags": [
                "#Business",
                "#SriLanka",
                "#Quality",
                "#SupportLocal",
                "#ShopNow"
            ],
            "marketingTips": "Promote this advertisement using Facebook and Instagram."
        }

    prompt_template = ChatPromptTemplate.from_messages([

        ("system", system_prompt),

        ("human", """
Business Profile

{business_profile_json}

User Profile

{user_profile_json}

Marketing Request

{marketing_prompt}

Return ONLY this JSON format.

{
    "headline":"",

    "facebook":"",

    "instagram":"",

    "google":"",

    "headlines":[
        "",
        "",
        "",
        "",
        ""
    ],

    "hashtags":[
        "",
        "",
        "",
        "",
        ""
    ],

    "marketingTips":""
}

Do NOT include markdown.

Do NOT include explanations.

Return JSON only.
""")
    ])

    chain = prompt_template | llm | StrOutputParser()

    try:

        generated = chain.invoke({

            "business_profile_json": json.dumps(
                business_profile,
                indent=2,
                ensure_ascii=False
            ),

            "user_profile_json": json.dumps(
                user_profile,
                indent=2,
                ensure_ascii=False
            ),

            "marketing_prompt": prompt

        })

        ads = _parse_json_object(generated)

    except Exception:

        ads = {

            "headline": "Advertisement",

            "facebook": generated if 'generated' in locals() else "",

            "instagram": "",

            "google": "",

            "headlines": [],

            "hashtags": [],

            "marketingTips": ""

        }

    return ads