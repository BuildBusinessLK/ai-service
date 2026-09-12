import json
import os
import re
from types import SimpleNamespace
from typing import Any, List, Optional

os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import traceback
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_factory import get_llm
from pydantic import BaseModel, Field

from rag.query import get_qa_chain

load_dotenv()

load_dotenv()

app = FastAPI(title="BuildBusinessLK AI Service", version="3.0")
service_state = SimpleNamespace(qa_chain=None, initialization_error=None)

# Allow Spring Boot backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse

qa_chain = None

def _get_chain():
    global qa_chain
    if qa_chain is None:
        qa_chain = get_qa_chain()
    return qa_chain

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


def _extract_number(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1).replace(",", "")
    if value.isdigit():
        return int(value)
    return None


def _parse_experience(text: str) -> Optional[str]:
    if re.search(r"\b(beginner|new|no experience|fresh|first time)\b", text):
        return "beginner"
    if re.search(r"\b(advanced|experienced|expert|professional|many years|5\+|6\+|7\+|8\+|9\+)\b", text):
        return "advanced"
    if re.search(r"\b(intermediate|some experience|a few years|couple years|2 years|3 years|4 years)\b", text):
        return "intermediate"
    year_match = re.search(r"(\d+)\s*(?:years|yrs?)\b", text)
    if year_match:
        years = int(year_match.group(1))
        if years <= 1:
            return "beginner"
        if years <= 4:
            return "intermediate"
        return "advanced"
    return None


def _parse_business_profile(question: str) -> dict[str, Any]:
    text = question.lower()
    sector = None
    for candidate in ["palmyrah", "kithul", "coconut"]:
        if candidate in text:
            sector = candidate
            break

    budget = _extract_number(text, r"(?:budget|investment|invest|capital|lkr|rs|rupees)[^\d]{0,20}([0-9][0-9,]*)")
    monthly_yield = _extract_number(text, r"(?:yield|production|output|sap|liters|kgs|kg)[^\d]{0,20}([0-9][0-9,]*)")
    employees = _extract_number(text, r"(?:employee|staff|worker|team)[^\d]{0,20}([0-9][0-9,]*)")
    experience = _parse_experience(text)

    # Fallback if yield is described using only a number and budget is present
    if monthly_yield is None:
        numbers = re.findall(r"([0-9][0-9,]*)", text)
        if sector and budget is not None and len(numbers) >= 2:
            budget_text = str(budget)
            if numbers[0].replace(",", "") == budget_text:
                monthly_yield = int(numbers[1].replace(",", ""))

    if employees is None:
        employees = 1

    profile: dict[str, Any] = {}
    if sector:
        profile["sector"] = sector
    if budget is not None:
        profile["budget"] = budget
    if monthly_yield is not None:
        profile["monthly_yield"] = monthly_yield
    if employees is not None:
        profile["employees"] = employees
    if experience:
        profile["experience"] = experience
    else:
        profile["experience"] = "intermediate"
        profile["experience_assumed"] = "intermediate"

    return profile


def _build_ml_context(business_profile: dict[str, Any]) -> str:
    if not business_profile:
        return ""
    values = []
    for key in ["sector", "budget", "monthly_yield", "employees", "experience", "recommendedBusiness"]:
        if key in business_profile:
            values.append(f"{key}: {business_profile[key]}")
    if "experience_assumed" in business_profile:
        values.append("experience_assumed: assumed intermediate because it was not provided")
    summary = "Business profile summary: " + "; ".join(values) if values else ""
    if "recommendedBusiness" in business_profile:
        summary += (
            "\nUse the ML recommendation above as a guiding suggestion when answering the user. "
            "If the user is asking for the best product or business option, mention the recommended business first."
        )
    return summary


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

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    provider = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama")
    model = os.getenv("GROQ_MODEL" if provider == "groq" else "OLLAMA_MODEL", "groq/compound-mini" if provider == "groq" else "llama3")
    return {
        "status": "ok",
        "service": "BuildBusinessLK AI Service",
        "provider": provider,
        "model": model,
        "vectorstore": os.path.exists(os.getenv("VECTORSTORE_PATH", "rag/vectorstore")),
    }


@app.post("/chat")
def chat(body: ChatBody):
    """
    Primary chat endpoint called by the Spring Boot backend.
    Accepts the user question + chat history + user/business profile context.
    Returns the AI answer as {"message": "..."}
    """
    try:
        chain = _get_chain()

        parsed_ml = _parse_business_profile(body.question)
        ml_profile = dict(body.businessProfile or {})
        ml_profile.update(parsed_ml)
        if ml_profile.get("sector") and ml_profile.get("budget") and ml_profile.get("monthly_yield"):
            try:
                from ml.predict import recommend_business
                rec = recommend_business(
                    sector=str(ml_profile["sector"]),
                    budget=int(ml_profile["budget"]),
                    monthly_yield=int(ml_profile["monthly_yield"]),
                    employees=int(ml_profile.get("employees", 1)),
                    experience=str(ml_profile.get("experience", "intermediate")),
                )
                ml_profile["recommendedBusiness"] = rec
            except Exception:
                pass

        ml_context = _build_ml_context(ml_profile)
        user_context = _format_profiles(body.userProfile, body.businessProfile)
        if ml_context:
            user_context += "\n\n" + ml_context

        result = chain.invoke({
            "input": body.question,
            "chat_history": [_message_to_dict(m) for m in body.chat_history],
            "user_context": user_context,
        })
        return {"message": result["answer"]}
    except Exception as e:
        import logging, traceback
        logging.error(f"Chat error: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"AI service error: {str(e)}"})


@app.post("/website-copy")
def website_copy(body: WebsiteCopyBody):
    """
    Generates short marketing copy (hero, about, marketing text) for an SME website
    based on the stored business profile.
    """
    bp = body.businessProfile or {}
    raw = json.dumps(bp, ensure_ascii=False, indent=2)

    llm = get_llm(temperature=0.35)
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


@app.post("/business-advisor")
def business_advisor(body: BusinessAdvisorBody):
    missing_fields = []
    business_profile = body.businessProfile or {}

    for field_name in ["sector", "budget", "monthly_yield", "employees", "experience"]:
        value = business_profile.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field_name)

    if missing_fields:
        return {
            "message": (
                "Please complete your business profile with the following fields: "
                f"{', '.join(missing_fields)}."
            )
        }

    try:
        recommendation = recommend_business(
            sector=business_profile["sector"],
            budget=int(business_profile["budget"]),
            monthly_yield=int(business_profile["monthly_yield"]),
            employees=int(business_profile["employees"]),
            experience=business_profile["experience"],
        )
    except ValueError as exc:
        return {"message": str(exc)}
    except Exception:
        return {
            "message": (
                "Unable to generate a recommendation with the provided profile. "
                "Please check the values and try again."
            )
        }

    user_context = _format_profiles(body.userProfile, business_profile)
    prompt = _build_business_advisor_prompt(recommendation, str(business_profile.get("sector", "")))
    result = qa_chain.invoke({
        "input": prompt,
        "chat_history": [],
        "user_context": user_context,
    })

    return {
        "recommendedBusiness": recommendation,
        "guidance": result["answer"],
    }


@app.post("/recommend-business")
def recommend(body: RecommendationBody):

    recommendation = recommend_business(
        sector=body.sector,
        budget=body.budget,
        monthly_yield=body.monthly_yield,
        employees=body.employees,
        experience=body.experience
    )

    return {
        "recommendation": recommendation
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
            llm = get_llm(temperature=0.5)
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