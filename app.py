import json
import os
import re
from types import SimpleNamespace
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import traceback
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from rag.query import get_qa_chain

load_dotenv()

app = FastAPI(title="BuildBusinessLK AI Service", version="3.0")
service_state = SimpleNamespace(qa_chain=None, initialization_error=None)

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
    idea: Optional[str] = None
    tone: Optional[str] = None
    platform: Optional[str] = None
    website: Optional[str] = None
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


def _build_business_advisor_prompt(recommended_business: str, sector: str) -> str:
    return (
        f"Recommended business: {recommended_business}. "
        f"Sector: {sector}. "
        "Explain how to start and grow this business in Sri Lanka with practical steps for a small SME owner. "
        "Keep the guidance grounded in coconut, palmyrah, or kithul local value chains and mention actionable next steps."
    )


def _build_local_ad_fallback(body: AdGenerationBody) -> str:
    business_name = (body.businessProfile or {}).get("businessName") or "your business"
    sector = (body.businessProfile or {}).get("sector") or "your industry"
    user_name = (body.userProfile or {}).get("fullName") or "our team"
    
    idea_str = body.idea or ""
    if "Current ad copy:" in idea_str and "Edit request:" in idea_str:
        try:
            parts = idea_str.split("Current ad copy:")
            after_ad = parts[1].split("Edit request:")
            current_ad = after_ad[0].strip()
            edit_request = after_ad[1].strip()
            
            # Match change X to Y or replace X with Y
            match = re.search(r'(?:change|replace)\s+(.+?)\s+(?:to|with)\s+(.+)', edit_request, re.IGNORECASE)
            if match:
                old_val = match.group(1).strip()
                new_val = match.group(2).strip()
                # Run case-insensitive replace on the current ad text
                pattern = re.compile(re.escape(old_val), re.IGNORECASE)
                updated_ad = pattern.sub(new_val, current_ad)
                return updated_ad
            return current_ad
        except Exception:
            pass

    request = idea_str.strip() or "our latest offer"

    return f"""Facebook Ad
--------------------
{business_name} is excited to introduce {request} for customers who value quality and trust.
Discover more today and experience the difference.

Instagram Ad
--------------------
{business_name} brings {request} to life with care, quality, and a personal touch.
Follow us and stay connected for the latest updates.

WhatsApp Advertisement
--------------------
Hello! We are {business_name}, and we are proud to share {request} with you.
Reach out today to learn more about our offer.

Short Headline
--------------------
Fresh solutions from {business_name}

Call to Action
--------------------
Contact us today or visit our website to learn more.

Hashtags
--------------------
#{business_name.replace(' ', '')} #BusinessGrowth #SME #DigitalMarketing #{sector.replace(' ', '')}

Suggested Tone
--------------------
Friendly and professional for {user_name}.
"""


_PROMPT_ECHO_MARKERS = (
    "you are an expert",
    "business information",
    "==============================",
    "instructions",
)


def _looks_like_echoed_prompt(text: str) -> bool:
    """Heuristic guard: if the model's output contains the scaffolding of our
    own prompt (headers, meta-instructions) instead of actual ad copy, treat
    it as a bad generation so we fall back to a clean template instead of
    showing the user a garbled response."""
    if not text:
        return True
    lowered = text.lower()
    hits = sum(1 for marker in _PROMPT_ECHO_MARKERS if marker in lowered)
    return hits >= 2


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

@app.post("/ad-generate")
def generate_ad(body: AdGenerationBody):
    try:

        business = json.dumps(
            body.businessProfile or {},
            ensure_ascii=False,
            indent=2
        )

        user = json.dumps(
            body.userProfile or {},
            ensure_ascii=False,
            indent=2
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an expert Sri Lankan SME marketing assistant and advertising copywriter.

You will be given a business profile, a user profile, and a campaign brief
describing what the advertisement should be about. Write real, ready-to-post
advertisement copy that personalizes the message using the business and user
details provided.

Rules:
- Output ONLY the advertisement content in the format below. Never repeat,
  quote, or summarize these instructions or the brief itself in your answer.
- Do not include section separators like "====" or restate field labels
  such as "Business Information" or "Instructions".
- Keep each section short and platform-appropriate.

Return exactly these sections, in this order:

Facebook Ad
--------------------
<ad text>

Instagram Ad
--------------------
<ad text>

WhatsApp Advertisement
--------------------
<ad text>

Short Headline
--------------------
<one headline>

Call to Action
--------------------
<one call to action>

Hashtags
--------------------
<3-5 hashtags>
"""
            ),
            (
                "human",
                """
Business profile (JSON):
{business}

User profile (JSON):
{user}

Campaign brief:
{prompt}
"""
            )
        ])

        try:
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.5
            )
        except Exception as e:
            traceback.print_exc()
            return {
                "generatedAds": _build_local_ad_fallback(body)
            }

        chain = prompt | llm | StrOutputParser()

        print("========================")
        print(body.model_dump())
        print("========================")

        try:
            if not body.prompt:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt cannot be empty."
                )
            result = chain.invoke({
                "business": business,
                "user": user,
                "prompt": body.prompt
            })
            result_text = str(result).strip() if result else ""
            if not result_text or _looks_like_echoed_prompt(result_text):
                # The model failed to follow instructions and echoed the
                # prompt/brief back instead of writing ad copy — use the
                # clean template fallback rather than show garbage.
                return {
                    "generatedAds": _build_local_ad_fallback(body)
                }
        except Exception:
            traceback.print_exc()
            return {
                "generatedAds": _build_local_ad_fallback(body)
            }

        return {
            "generatedAds": result
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }