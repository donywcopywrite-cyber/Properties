import os, json, re, math
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# ---------- Setup ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="ListingMatcher API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class Criteria(BaseModel):
    location: str
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    beds_min: Optional[int] = None
    baths_min: Optional[float] = None
    property_types: Optional[List[str]] = None
    keywords: Optional[str] = None

class RequestBody(BaseModel):
    limit: Optional[int] = 6
    language: Optional[str] = "fr"
    criteria: Criteria

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

# ---------- Helpers ----------
async def web_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    """Search listings using Serper.dev"""
    if not SERPER_API_KEY:
        return [{"title": "No SERPER_API_KEY configured", "url": "", "snippet": ""}]

    async with httpx.AsyncClient(timeout=20) as s:
        resp = await s.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num}
        )
    data = resp.json()
    return [
        {"title": o.get("title"), "url": o.get("link"), "snippet": o.get("snippet")}
        for o in (data.get("organic") or [])
    ][:num]


def extract_json(text: str) -> List[Listing]:
    """Extract valid property JSON list from model output."""
    try:
        match = re.search(r'"properties"\s*:\s*(\[[\s\S]*?\])', text)
        if not match:
            return []
        props = json.loads(match.group(1))
        return [Listing(**p) for p in props]
    except Exception:
        return []

# ---------- Route ----------
@app.post("/listingmatcher/run", response_model=MatchResponse)
async def listingmatcher_run(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payload = RequestBody(**body)
    c = payload.criteria
    lang = payload.language.lower()

    # Step 1: Run a real search
    search_query = (
        f"site:centris.ca OR site:realtor.ca OR site:remax-quebec.com OR site:royallepage.ca OR site:duproprio.com "
        f"{c.location} {c.keywords or ''} "
        f'{c.min_price or ""} to {c.max_price or ""} {" ".join(c.property_types or [])}'
    )
    search_results = await web_search(search_query, num=8)
    if not search_results:
        raise HTTPException(status_code=502, detail="No search results returned")

    # Step 2: Feed those results into GPT for filtering
    search_text = "\n\n".join(
        [f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in search_results]
    )

    system_prompt = f"""
You are ListingMatcher, a bilingual (French first, then English) assistant for Québec real estate.

Your task:
- Analyze the provided web search results.
- Identify only **current residential property listings**.
- Only keep entries that are **actively listed for sale** in Québec.
- Never invent any MLS numbers. If not visible, set `"mls": null`.
- Deduplicate results (same MLS or same address).
- Include 5–12 listings maximum.
- Output JSON under key "properties" with these fields:
  mls, url, address, price_cad, beds, baths, type, note.
"""

    user_prompt = f"""
Recherche effectuée sur Google avec les critères:
- Lieu: {c.location}
- Budget: {c.min_price or '?'} à {c.max_price or '?'} CAD
- Chambres: minimum {c.beds_min or '?'}
- Salles de bain: minimum {c.baths_min or '?'}
- Type: {", ".join(c.property_types or []) or 'Indifférent'}
- Mots-clés: {c.keywords or 'Aucun'}

Résultats trouvés:
{search_text}

Donne-moi un résumé concis et le JSON demandé.
Réponds en français d'abord, puis ajoute une courte version anglaise.
"""

    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

    reply_text = getattr(response, "output_text", "")
    properties = extract_json(reply_text)

    return MatchResponse(reply=reply_text, properties=properties)
