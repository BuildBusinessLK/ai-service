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
try:
    service_state.qa_chain = get_qa_chain()
except Exception as exc:  # pragma: no cover - defensive startup fallback
    service_state.initialization_error = str(exc)

MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "OLLAMA")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")


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


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "provider": MODEL_PROVIDER, "model": MODEL_NAME}


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
        "user_context": combined_context,
    })
    return {"message": result["answer"], "recommendedBusiness": recommended}


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