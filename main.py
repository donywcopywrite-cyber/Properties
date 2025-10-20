import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load local .env when running on your machine. On Render, env vars come from the dashboard.
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="ListingMatcher API", version="1.0.0")


class Criteria(BaseModel):
    location: str = Field(..., description="City/Area. e.g., 'Laval, QC'")
    budget: Optional[str] = Field(None, description="Price filter, e.g., '<= 450000 CAD'")
    beds: Optional[int] = Field(None, description="Minimum bedrooms")
    baths: Optional[int] = Field(None, description="Minimum bathrooms")
    keywords: Optional[str] = Field(None, description="Comma-separated tags like 'near metro, new construction'")
    limit: int = Field(10, ge=1, le=20, description="Max properties to return (5–12 recommended)")
    language: str = Field("fr", description="Response language preference: 'fr' or 'en'")

class Listing(BaseModel):
    mls: Optional[str] = None
    url: Optional[str] = None
    address: Optional[str] = None
    price_cad: Optional[int] = None
    beds: Optional[int] = None
    baths: Optional[float] = None
    type: Optional[str] = None
    note: Optional[str] = None

class MatchResponse(BaseModel):
    reply: str
    properties: List[Listing] = []


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/listingmatcher/run", response_model=MatchResponse)
def run_listingmatcher(payload: Criteria = Body(...)):
    """
    Call OpenAI Responses API with guardrails & structured output.
    Designed for Quebec listings; bilingual response (FR first by default).
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: OPENAI_API_KEY missing")

    # Build system instructions & a compact user instruction block
    lang_pref = "fr" if (payload.language or "").lower().startswith("fr") else "en"
    language_line = (
        "Réponds en français d’abord, puis fournis une courte version en anglais."
        if lang_pref == "fr" else
        "Answer in English first, then provide a short French version."
    )

    system_instructions = (
        "You are ListingMatcher, a real-estate assistant for Québec.\n"
        "- Return only currently-listed properties in Québec (public sites such as Centris, Realtor.ca, Royal LePage, RE/MAX Québec, DuProprio may be referenced, but if you are not sure, say so).\n"
        "- Output 5–12 results (use 'limit' if provided), deduplicate by MLS when present.\n"
        "- For each property include: MLS, URL, address, price CAD, beds, baths, property type, and a one-line note.\n"
        "- Be cautious about legal/contractual statements; if uncertain, state uncertainty and suggest a next step.\n"
        "- ALWAYS return a machine-readable JSON block named 'properties' following the JSON schema provided.\n"
        f"- {language_line}"
    )

    user_criteria = (
        f"Location: {payload.location}\n"
        f"Budget: {payload.budget or 'N/A'}\n"
        f"Beds: {payload.beds or 'N/A'}\n"
        f"Baths: {payload.baths or 'N/A'}\n"
        f"Keywords: {payload.keywords or 'N/A'}\n"
        f"Limit: {payload.limit}\n"
        "Return a concise human-friendly summary AND a JSON array 'properties' that matches the schema."
    )

    # Ask the model for structured output: we'll validate into MatchResponse
    # Responses API call: https documentation shows client.responses.create(model=..., input=...) pattern
    # We'll request JSON using a schema in the text output.
    try:
        resp = client.responses.create(
            model="gpt-4.1",  # strong instruction-following & tool support; Responses API recommended path
            input=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_criteria},
            ],
            # Ask it to produce both: natural text + a JSON block. For strictness you could use JSON Schema tools.
        )  # See Responses API quickstart. 
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    text = getattr(resp, "output_text", None) or ""

    # --- Minimal JSON extraction ---
    # We expect the model to include a JSON array named "properties" somewhere in the output.
    # For reliability in production, switch to Responses API JSON Schema (tools/text.format) or a parser like 'instructor'.
    properties: List[Dict[str, Any]] = []
    import json, re
    match = re.search(r'"properties"\s*:\s*(\[[\s\S]*?\])', text)
    if match:
        try:
            properties = json.loads(match.group(1))  # parse the array only
        except Exception:
            properties = []

    # Normalize & cast fields
    normalized: List[Listing] = []
    for p in properties[: payload.limit]:
        try:
            normalized.append(Listing(**p))
        except Exception:
            # best-effort coercion
            normalized.append(Listing(**{k: p.get(k) for k in ["mls","url","address","price_cad","beds","baths","type","note"] if k in p}))

    # Build bilingual reply header if absent
    reply = text.strip() or "Aucune réponse générée / No response generated."

    return MatchResponse(reply=reply, properties=normalized)
