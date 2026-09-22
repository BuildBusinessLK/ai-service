from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_factory import get_llm
from ml.predict import MODEL_VERSION, load_model, recommend_business, recommend_products
from rag.query import get_qa_chain

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")


# ──────────────────────────────────────────────
# Lifespan Management
# ──────────────────────────────────────────────

qa_chain = None


def _get_chain():
    global qa_chain
    if qa_chain is None:
        qa_chain = get_qa_chain()
    return qa_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing BuildBusinessLK AI Service...")
    load_model()
    try:
        _get_chain()
    except Exception as e:
        logger.warning(f"RAG initialization warning: {e}")
    yield
    logger.info("Shutting down BuildBusinessLK AI Service.")


app = FastAPI(
    title="BuildBusinessLK AI Service",
    version="3.0",
    lifespan=lifespan,
)

# Allow Spring Boot backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    question: str
    chat_history: List[ChatMessage] = Field(default_factory=list)
    userProfile: Optional[dict] = None
    businessProfile: Optional[dict] = None


class BusinessProfile(BaseModel):
    sector: Optional[str] = None
    budget_lkr: Optional[float] = Field(default=250000, gt=0)
    monthly_yield_kg: Optional[float] = Field(default=None, ge=0)
    employees: Optional[int] = Field(default=None, ge=1)
    experience_years: Optional[float] = Field(default=None, ge=0)


class RecommendationDto(BaseModel):
    rank: int
    product: str
    confidence: float


class FeasibilityDto(BaseModel):
    capitalFit: int
    yieldFit: int
    staffingFit: int


class BusinessRecommendationResponse(BaseModel):
    modelVersion: str = MODEL_VERSION
    recommendedBusiness: Optional[str] = None
    recommendations: List[RecommendationDto] = Field(default_factory=list)
    feasibility: Optional[FeasibilityDto] = None
    guidance: Optional[str] = None
    actions: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    sessionId: Optional[int] = None


class BusinessAdvisorBody(BaseModel):
    sessionId: Optional[int] = None
    userProfile: Optional[dict] = None
    businessProfile: Optional[dict] = None


class RecommendationBody(BaseModel):
    sector: Optional[str] = "coconut"
    budget: Optional[float] = 250000
    monthly_yield: Optional[float] = 1000
    employees: Optional[int] = 2
    experience: Optional[Any] = 1.0


class RecommendationRequest(BaseModel):
    business: Optional[BusinessProfile] = None
    sector: Optional[str] = None
    budget: Optional[float] = None
    monthly_yield: Optional[float] = None
    employees: Optional[int] = None
    experience: Optional[Any] = None
    top_k: int = Field(default=3, ge=1, le=5)


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


class EmailGenerationRequest(BaseModel):
    goal: Optional[str] = "GENERAL_ANNOUNCEMENT"
    productName: Optional[str] = None
    sector: Optional[str] = None
    targetAudience: Optional[str] = None
    keyOffer: Optional[str] = None
    tone: Optional[str] = "professional"
    companyName: Optional[str] = None
    userName: Optional[str] = None
    contactPhone: Optional[str] = None
    contactEmail: Optional[str] = None
    userProfile: Optional[dict] = None
    businessProfile: Optional[dict] = None
    idea: Optional[str] = None


class EmailGenerationResponse(BaseModel):
    subject: str
    body: str
    suggestedCallToAction: Optional[str] = None
    targetAudienceNotes: Optional[str] = None



# ──────────────────────────────────────────────
# NLP Helpers & Intent Routing
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


def _extract_amount(text: str) -> Optional[int]:
    """Extracts money amounts handling 'k', 'lakhs', commas, etc."""
    lowered = text.lower()
    # e.g., 400k, 500 k
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", lowered)
    if k_match:
        return int(float(k_match.group(1)) * 1000)

    # e.g., 4 lakh, 5 lakhs, 2.5 lakhs
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?)\b", lowered)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)

    # Word numbers for lakhs e.g. "four lakhs"
    word_lakhs = {"one": 100000, "two": 200000, "three": 300000, "four": 400000, "five": 500000, "ten": 1000000}
    for word, amt in word_lakhs.items():
        if f"{word} lakh" in lowered:
            return amt

    # Explicit numbers near currency cues
    curr_match = re.search(r"(?:budget|investment|invest|capital|lkr|rs\.?|rupees)[^\d]{0,15}([0-9][0-9,]*)", lowered)
    if curr_match:
        val = curr_match.group(1).replace(",", "")
        if val.isdigit() and int(val) > 1000:
            return int(val)

    # General large number if greater than 10,000
    for num_str in re.findall(r"\b([0-9][0-9,]{3,})\b", text):
        cleaned = num_str.replace(",", "")
        if cleaned.isdigit() and int(cleaned) >= 20000:
            return int(cleaned)

    return None


def _parse_experience(text: str) -> float:
    lowered = text.lower()
    if re.search(r"\b(beginner|new|no experience|fresh|first time)\b", lowered):
        return 0.5
    if re.search(r"\b(advanced|expert|professional|many years|5\+|6\+|7\+|8\+)\b", lowered):
        return 5.0
    if re.search(r"\b(intermediate|some experience|a few years|couple years|2 years|3 years)\b", lowered):
        return 2.5
    year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years|yrs?)\b", lowered)
    if year_match:
        return float(year_match.group(1))
    return 2.0


def _parse_yield(text: str) -> Optional[int]:
    lowered = text.lower()
    match = re.search(r"(?:yield|production|output|sap|raw material|liters?|kgs?|kg)[^\d]{0,15}([0-9][0-9,]*)", lowered)
    if match:
        val = match.group(1).replace(",", "")
        if val.isdigit():
            return int(val)
    return None


def _detect_recommendation_intent(question: str) -> bool:
    lowered = question.lower()
    strong_cues = [
        "what should i start",
        "what can i start",
        "which product",
        "what product",
        "what business",
        "best business",
        "recommend",
        "match product",
        "product fit",
        "which is better",
        "feasibility",
        "what to produce",
        "suggest a business",
        "what to do with",
        "which suits",
        "profitable",
    ]
    if any(sc in lowered for sc in strong_cues):
        return True

    # Action word + Sector
    has_action = any(re.search(rf"\b{cue}\b", lowered) for cue in ["start", "launch", "begin", "make", "produce", "setup", "set up", "invest", "build"])
    has_sector = any(sec in lowered for sec in ["coconut", "kithul", "palmyra", "palmyrah"])
    return bool(has_action and has_sector)


def _detect_email_campaign_intent(question: str) -> bool:
    lowered = question.lower()
    email_cues = [
        "draft email",
        "write email",
        "send email",
        "email campaign",
        "email exporters",
        "email to exporters",
        "email to coconut",
        "email to kithul",
        "email to palmyrah",
        "email pitch",
        "outreach email",
        "email marketing",
        "cold email",
        "wholesale email",
        "sample email",
        "mail campaign",
        "compose email",
        "marketing email",
    ]
    return any(cue in lowered for cue in email_cues)


def generate_email_core(req: EmailGenerationRequest) -> Dict[str, Any]:
    bp = req.businessProfile or {}
    up = req.userProfile or {}

    company_name = req.companyName or bp.get("businessName") or "Lanka Value Agribusiness"
    user_name = req.userName or (up.get("fullName") if up else "Founder & Managing Director")
    sector = (req.sector or bp.get("sector") or "Coconut").capitalize()
    product = req.productName or (f"Certified {sector} Value-Added Products")
    audience = req.targetAudience or "EDB Registered Sri Lankan Exporters & Wholesale Buyers"
    goal = req.goal or "GENERAL_ANNOUNCEMENT"
    tone = req.tone or "Professional B2B"
    key_offer = req.keyOffer or req.idea or f"Introduce {company_name}'s premium {sector} products, certified farmgate sourcing, and wholesale packaging availability."
    phone = req.contactPhone or "+94 77 123 4567"
    email = req.contactEmail or "inquiries@buildbusinesslk.com"

    # Sector-specific origin and credential grounding
    origin_map = {
        "Coconut": "Sri Lanka Coconut Triangle (Kurunegala, Puttalam & Gampaha estates)",
        "Kithul": "Central Highlands & Sabaragamuwa Rainforest Buffer Zones (Ratnapura/Kegalle)",
        "Palmyrah": "Northern & Eastern Province (Jaffna Peninsula & Mannar)",
    }
    origin_region = origin_map.get(sector, "Authentic Sri Lankan Agro-Processing Estates")

    prompt_text = f"""
You are an expert commercial business email copywriter and export market consultant for Sri Lankan MSMEs.
Write a concise, high-converting commercial business email tailored to the following scenario:

Campaign Goal: {goal}
Company Name: {company_name}
Sender Name: {user_name}
Sector: {sector}
Product / Focus: {product}
Target Audience: {audience}
Tone: {tone}
Key Offer / Brief: {key_offer}
Sender Contact Phone: {phone}
Sender Contact Email: {email}

Strict Domain & Tone Guidelines:
1. Subject line must be punchy, commercial, professional (no spam words, no ALL-CAPS, under 65 chars).
2. GEOGRAPHIC ACCURACY (CRITICAL):
   - Sector {sector}: Sourcing origin MUST strictly be {origin_region}.
   - NEVER attribute Coconut products to Palmyrah estates or vice versa.
3. CAMPAIGN GOAL CONSISTENCY:
   - If Goal is WHOLESALE_PITCH or EXPORTER_SAMPLE_OFFER: Target B2B export houses and distributors. Focus on export container loads / MOQ, commercial grading (SLS / ISO / organic), packaging (bulk drums, HDPE, private label), and offering a sample dispatch pack. DO NOT refer to retail discounts.
   - If Goal is RETAIL_DISCOUNT: Target retail buyers / consumers. Focus on consumer packaging, introductory discounts, and ordering directly.
4. Structure the body into 3-4 cleanly spaced paragraphs with bullet points for key commercial specs (Purity, Origin: {origin_region}, Certification, Packaging, MOQ/Terms).
5. Include a professional, non-pushy Call to Action.
6. Professional signature block with sender name, title, company, phone, and email.

Return ONLY a JSON object with this exact schema:
{{
  "subject": "...",
  "body": "...",
  "suggestedCallToAction": "...",
  "targetAudienceNotes": "..."
}}
"""

    llm = get_llm(temperature=0.3)
    if llm is not None:
        try:
            chain = ChatPromptTemplate.from_messages([
                ("system", "You are an expert Sri Lankan agribusiness export copywriter. Output strictly valid JSON without markdown fences. Ensure geographic origins strictly match Sri Lankan agricultural regions."),
                ("human", "{prompt}"),
            ]) | llm | StrOutputParser()
            raw_out = chain.invoke({"prompt": prompt_text})
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_out.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            parsed = json.loads(cleaned)
            if "subject" in parsed and "body" in parsed:
                return {
                    "subject": str(parsed["subject"]).strip(),
                    "body": str(parsed["body"]).strip(),
                    "suggestedCallToAction": str(parsed.get("suggestedCallToAction", "Request wholesale pricing & sample pack")).strip(),
                    "targetAudienceNotes": str(parsed.get("targetAudienceNotes", f"Targeting {audience}")).strip(),
                }
        except Exception as e:
            logger.warning(f"LLM email generation failed: {e}. Using deterministic fallback.")

    # High quality deterministic fallback
    subject_map = {
        "WHOLESALE_PITCH": f"Wholesale Supply Inquiry: Export-Grade {product} from {company_name}",
        "EXPORTER_SAMPLE_OFFER": f"Export-Grade {product} Samples Available – {company_name}",
        "RETAIL_DISCOUNT": f"Special Commercial Discount on Fresh Batch of {product}",
        "HARVEST_ANNOUNCEMENT": f"New Harvest Ready: Certified Sri Lankan {sector} Supply",
    }
    subj = subject_map.get(goal, f"Commercial Partnership Inquiry – {product} by {company_name}")

    body_text = f"""Dear Commercial Partner,

I hope this email finds you well.

I am writing to you on behalf of {company_name}, a verified Sri Lankan producer operating in the {sector} value chain. We are currently offering our latest harvest batch of {product}, processed to meet stringent export and commercial grading criteria.

Key Commercial Specifications:
• Purity: 100% pure, single-origin {sector.lower()} agro-processing
• Origin: {origin_region}
• Certification: SLS (Sri Lanka Standards) & Food Safety compliant
• Packaging: Commercial bulk containers and food-grade export packaging
• Commercial Terms: {key_offer}

We would welcome the opportunity to courier a complimentary evaluation sample pack and our wholesale specification sheet to your procurement team.

Please reply to this email or contact me directly at {phone} to request sample shipments or container pricing.

Warm regards,

{user_name}
{company_name}
Phone: {phone}
Email: {email}
"""
    return {
        "subject": subj,
        "body": body_text,
        "suggestedCallToAction": "Request complimentary evaluation sample pack",
        "targetAudienceNotes": f"Tailored for {audience} with verified {origin_region} origin.",
    }




def _extract_slots(question: str, bp: Optional[dict] = None) -> Dict[str, Any]:
    lowered = question.lower()
    bp = bp or {}

    # Sector - strictly restricted to Coconut, Kithul, Palmyrah (Thal)
    sector = None
    if any(k in lowered for k in ["palmyrah", "palmyra", "thal"]):
        sector = "palmyrah"
    elif any(k in lowered for k in ["kithul", "kitul"]):
        sector = "kithul"
    elif any(k in lowered for k in ["coconut", "coco", "pol", "copra", "coir"]):
        sector = "coconut"

    if not sector and bp.get("sector"):
        bp_s = str(bp["sector"]).lower()
        if "palmyr" in bp_s or "thal" in bp_s:
            sector = "palmyrah"
        elif "kithul" in bp_s or "kitul" in bp_s:
            sector = "kithul"
        elif "coconut" in bp_s or "coco" in bp_s or "pol" in bp_s:
            sector = "coconut"

    # Budget
    budget = _extract_amount(question)
    if budget is None and bp.get("budget"):
        budget = int(bp["budget"])

    # Yield
    yield_val = _parse_yield(question)
    if yield_val is None and bp.get("monthly_yield"):
        yield_val = int(bp["monthly_yield"])

    # Employees
    emp_match = re.search(r"(\d+)\s*(?:employee|staff|worker|people|team)\b", lowered)
    employees = int(emp_match.group(1)) if emp_match else (int(bp.get("employees")) if bp.get("employees") else 2)

    # Experience
    experience = _parse_experience(question)
    if bp.get("experience") and experience == 2.0:
        experience = _parse_experience(str(bp["experience"]))

    return {
        "sector": sector,
        "budget_lkr": budget,
        "monthly_yield_kg": yield_val,
        "employees": employees,
        "experience_years": experience,
    }


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

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "status": "ok",
        "service": "BuildBusinessLK AI Service",
        "health": "/health",
        "docs": "/docs",
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    provider = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama")
    model = os.getenv("GROQ_MODEL" if provider == "groq" else "OLLAMA_MODEL", "groq/compound-mini" if provider == "groq" else "llama3")
    ml_loaded = load_model() is not None
    return {
        "status": "ok",
        "service": "BuildBusinessLK AI Service",
        "provider": provider,
        "model": model,
        "mlModelLoaded": ml_loaded,
        "modelVersion": MODEL_VERSION,
        "vectorstore": os.path.exists(os.getenv("VECTORSTORE_PATH", "rag/vectorstore")),
    }


@app.get("/model-info")
def model_info():
    metrics_file = Path(__file__).resolve().parent / "ml" / "models" / "metrics_v1.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_version": MODEL_VERSION,
        "status": "active",
        "features": ["sector", "budget_lkr", "monthly_yield_kg", "employees", "experience_years"],
    }


@app.post("/chat")
def chat(body: ChatBody):
    """
    Primary chat endpoint.
    Performs Intent Classification and Slot Extraction:
    - If user asks for a business recommendation or product match, executes the ML model
      and returns structured recommendations, feasibility metrics, and RAG explanation.
    - Otherwise, routes to standard RAG domain consultation.
    """
    try:
        chain = _get_chain()
        is_rec_intent = _detect_recommendation_intent(body.question)
        is_email_intent = _detect_email_campaign_intent(body.question)
        slots = _extract_slots(body.question, body.businessProfile)

        # 1. Email Campaign Intent Routing
        if is_email_intent:
            sector = slots.get("sector") or (body.businessProfile.get("sector") if body.businessProfile else "Coconut")
            lowered_q = body.question.lower()
            if "sample" in lowered_q:
                goal = "EXPORTER_SAMPLE_OFFER"
            elif "wholesale" in lowered_q or "pitch" in lowered_q or "bulk" in lowered_q:
                goal = "WHOLESALE_PITCH"
            elif "discount" in lowered_q or "sale" in lowered_q or "promo" in lowered_q:
                goal = "RETAIL_DISCOUNT"
            elif "harvest" in lowered_q or "fresh" in lowered_q or "season" in lowered_q:
                goal = "HARVEST_ANNOUNCEMENT"
            else:
                goal = "GENERAL_ANNOUNCEMENT"

            email_req = EmailGenerationRequest(
                goal=goal,
                sector=str(sector),
                keyOffer=body.question,
                targetAudience="EDB Registered Exporters & Wholesale Distributors",
                businessProfile=body.businessProfile,
                userProfile=body.userProfile,
            )
            email_res = generate_email_core(email_req)
            return {
                "type": "EMAIL_CAMPAIGN",
                "message": f"I've drafted a targeted B2B outreach email for your **{str(sector).capitalize()}** campaign. You can preview, edit, or launch the campaign simulation in our Email Campaign Studio.",
                "emailCampaign": {
                    "goal": goal,
                    "sector": str(sector).upper(),
                    "subject": email_res["subject"],
                    "body": email_res["body"],
                    "targetAudience": email_res["targetAudienceNotes"],
                    "suggestedCallToAction": email_res["suggestedCallToAction"],
                },
                "actions": [
                    "Open in Email Campaign Studio",
                    "Regenerate with formal export tone",
                    "Target EDB Exporters Directory",
                ],
            }

        # 2. Recommendation Intent Routing
        if is_rec_intent or (slots.get("sector") and slots.get("budget_lkr")):
            sector = slots.get("sector")
            budget = slots.get("budget_lkr")

            # Missing slot handling
            if not sector:
                return {
                    "type": "QUESTION",
                    "message": "Which agricultural value chain are you exploring? We currently specialize in **Coconut**, **Kithul**, and **Palmyrah**.",
                }

            if not budget:
                return {
                    "type": "QUESTION",
                    "message": f"To find the best {sector.capitalize()} product match, what is your approximate initial investment budget in LKR?",
                }

            # If yield is not specified, assign a realistic baseline for the sector
            yield_val = slots.get("monthly_yield_kg")
            assumed_yield = False
            if not yield_val:
                yield_val = 1500 if sector == "coconut" else 500
                slots["monthly_yield_kg"] = yield_val
                assumed_yield = True

            # Run ML model
            try:
                rec_result = recommend_products(slots, top_k=3)
                top_product = rec_result["recommendations"][0]["product"]
                top_conf = rec_result["recommendations"][0]["confidence"]

                # Generate domain explanation via RAG
                user_context = _format_profiles(body.userProfile, body.businessProfile)
                rec_prompt = (
                    f"The user has LKR {budget:,} capital in the {sector} sector "
                    f"with ~{yield_val} kg/L monthly raw material access and {slots['employees']} workers. "
                    f"The ML recommendation model matched them with '{top_product}' ({top_conf}% Match). "
                    f"Briefly explain why this product fits their capital and resource scale, and what first step they should take in Sri Lanka."
                )
                try:
                    rag_result = chain.invoke({
                        "input": rec_prompt,
                        "chat_history": [_message_to_dict(m) for m in body.chat_history[-4:]],
                        "user_context": user_context,
                    })
                    explanation = rag_result["answer"]
                except Exception as rag_err:
                    logger.warning(f"RAG guidance failed, using localized guidance: {rag_err}")
                    explanation = f"{top_product} offers strong local and export value addition for your LKR {budget:,} capital in the {sector} value chain. Prioritize basic processing equipment, hygienic bottling/packaging, and target local retail or regional collection centers."

                assumed_note = f" *(calculated assuming ~{yield_val:,} kg monthly raw material availability and {slots['employees']} workers)*" if assumed_yield else ""
                message_text = f"Based on your budget of **LKR {budget:,}** in the **{sector.capitalize()}** sector{assumed_note}, **{top_product}** is your strongest match."

                return {
                    "type": "RECOMMENDATION",
                    "message": message_text,
                    "recommendation": {
                        "modelVersion": rec_result["modelVersion"],
                        "recommendedBusiness": top_product,
                        "recommendations": rec_result["recommendations"],
                        "feasibility": rec_result["feasibility"],
                        "guidance": explanation,
                        "actions": [
                            f"Show required machinery for {top_product}",
                            f"Draft a business plan for {top_product}",
                            f"Calculate profit margins for {top_product}",
                            f"Export requirements for {top_product}",
                        ],
                    },
                }
            except Exception as ml_err:
                logger.warning(f"ML recommendation fallback to RAG: {ml_err}")

        # Standard RAG Chat
        user_context = _format_profiles(body.userProfile, body.businessProfile)
        try:
            result = chain.invoke({
                "input": body.question,
                "chat_history": [_message_to_dict(m) for m in body.chat_history],
                "user_context": user_context,
            })
            return {
                "type": "TEXT",
                "message": result["answer"],
            }
        except Exception as llm_err:
            logger.warning(f"LLM call failed: {llm_err}")
            return {
                "type": "TEXT",
                "message": "I am having temporary trouble reaching the external AI service. You can use the '✨ Match Products' button above for direct ML product recommendations without waiting for the LLM.",
            }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"AI service error: {str(e)}"})


@app.post("/recommend-business")
def recommend_endpoint(body: RecommendationRequest):
    """
    Direct ML endpoint returning Top-3 products and feasibility metrics.
    """
    try:
        profile_data = {}
        if body.business:
            profile_data = body.business.model_dump()
        else:
            profile_data = {
                "sector": body.sector,
                "budget_lkr": body.budget,
                "monthly_yield_kg": body.monthly_yield,
                "employees": body.employees,
                "experience_years": body.experience,
            }
        result = recommend_products(profile_data, top_k=body.top_k)
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        logger.error(f"Recommendation endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business-advisor")
def business_advisor(body: BusinessAdvisorBody):
    """
    Called by Spring Boot Gateway for the Business Recommendation Dialog.
    Runs ML Top-3 recommendation + Feasibility + RAG domain guidance.
    """
    bp = body.businessProfile or {}
    sector = bp.get("sector") or "coconut"
    budget = bp.get("budget_lkr") or bp.get("budget") or 250000
    monthly_yield = bp.get("monthly_yield_kg") or bp.get("monthly_yield") or 1000
    employees = bp.get("employees") or 2
    experience = bp.get("experience_years") or bp.get("experience") or 1.0

    features = {
        "sector": sector,
        "budget_lkr": budget,
        "monthly_yield_kg": monthly_yield,
        "employees": employees,
        "experience_years": experience,
    }

    try:
        rec_res = recommend_products(features, top_k=3)
    except Exception as e:
        logger.error(f"ML Recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ML service unavailable: {e}")

    top_product = rec_res["recommendations"][0]["product"]
    user_context = _format_profiles(body.userProfile, bp)
    prompt = _build_business_advisor_prompt(top_product, str(sector))

    chain = _get_chain()
    guidance = ""
    try:
        result = chain.invoke({
            "input": prompt,
            "chat_history": [],
            "user_context": user_context,
        })
        guidance = result.get("answer", "")
    except Exception as e:
        logger.warning(f"RAG guidance error in business-advisor: {e}")
        guidance = f"Focus on setting up hygienic production standards for {top_product}. Target local retail and export quality compliance."

    return {
        "modelVersion": rec_res["modelVersion"],
        "recommendedBusiness": top_product,
        "recommendations": rec_res["recommendations"],
        "feasibility": rec_res["feasibility"],
        "guidance": guidance,
        "actions": [
            f"Show required machinery for {top_product}",
            f"Draft a business plan for {top_product}",
            f"Calculate profit margins for {top_product}",
            f"Export requirements for {top_product}",
        ],
        "sessionId": body.sessionId,
    }


@app.post("/website-copy")
def website_copy(body: WebsiteCopyBody):
    """
    Generates short marketing copy for an SME website.
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
        data = json.loads(re.sub(r"^```[a-zA-Z]*\n", "", out.strip()).rstrip("```").strip())
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
    """
    Generates multi-platform SME marketing copy.
    """
    try:
        business = json.dumps(body.businessProfile or {}, ensure_ascii=False, indent=2)
        user = json.dumps(body.userProfile or {}, ensure_ascii=False, indent=2)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert Sri Lankan SME marketing assistant and advertising copywriter.
Write ready-to-post advertisement copy for Facebook, Instagram, and WhatsApp tailored to the Sri Lankan market.""",
            ),
            (
                "human",
                """Business profile (JSON):\n{business}\n\nUser profile (JSON):\n{user}\n\nCampaign brief:\n{prompt}""",
            ),
        ])

        llm = get_llm(temperature=0.5)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"business": business, "user": user, "prompt": body.prompt})
        return {"generatedAds": result}
    except Exception as e:
        logger.error(f"Ad generation error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/email-generate", response_model=EmailGenerationResponse)
def generate_email_endpoint(body: EmailGenerationRequest):
    """
    Generates high-converting commercial business emails tailored to Sri Lankan MSMEs
    in Coconut, Kithul, and Palmyrah sectors for B2B export outreach or retail promotions.
    """
    try:
        res = generate_email_core(body)
        return EmailGenerationResponse(
            subject=res["subject"],
            body=res["body"],
            suggestedCallToAction=res.get("suggestedCallToAction"),
            targetAudienceNotes=res.get("targetAudienceNotes"),
        )
    except Exception as e:
        logger.error(f"Email generation endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)