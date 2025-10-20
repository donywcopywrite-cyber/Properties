import os, json, re, math
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="ListingMatcher API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Models ----------
class FlatCriteria(BaseModel):
    location: str
    budget: Optional[str] = None
    beds: Optional[int] = None
    baths: Optional[float] = None
    keywords: Optional[str] = None
    limit: int = 10
    language: str = "fr"
    allow_web: bool = False

class BubbleCriteria(BaseModel):
    location: str
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    beds_min: Optional[int] = None
    baths_min: Optional[float] = None
    property_types: Optional[List[str]] = None
    keywords: Optional[str] = None

class BubblePayload(BaseModel):
    conversation_id: Optional[str] = None
    limit: Optional[int] = 10
    language: Optional[str] = "fr"
    allow_web: Optional[bool] = False
    criteria: BubbleCriteria

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

# ---------- Helpers ----------
def normalize_to_flat(payload: dict) -> FlatCriteria:
    if "criteria" in payload and isinstance(payload["criteria"], dict):
        c = payload["criteria"]
        # Budget text
        if c.get("min_price") is not None and c.get("max_price") is not None:
            budget_text = f"{c['min_price']}-{c['max_price']} CAD"
        elif c.get("min_price") is not None:
            budget_text = f">= {c['min_price']} CAD"
        elif c.get("max_price") is not None:
            budget_text = f"<= {c['max_price']} CAD"
        else:
            budget_text = None
        # Keywords + property types
        kw_extra = ""
        if c.get("property_types"):
            kw_extra = " | types: " + ", ".join(c["property_types"])
        keywords = ((c.get("keywords") or "") + kw_extra).strip(" |")
        return FlatCriteria(
            location=c.get("location", ""),
            budget=budget_text,
            beds=c.get("beds_min"),
            baths=c.get("baths_min"),
            keywords=keywords or None,
            limit=payload.get("limit", 10),
            language=payload.get("language", "fr"),
            allow_web=payload.get("allow_web", False),
        )
    return FlatCriteria(**payload)

# Tool: web search (serper.dev if key is present; simple fallback otherwise)
async def tool_web_search(query: str, num: int = 5) -> Dict[str, Any]:
    serper = os.getenv("SERPER_API_KEY")
    if serper:
        try:
            async with httpx.AsyncClient(timeout=20) as s:
                r = await s.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper, "Content-Type": "application/json"},
                    json={"q": query, "num": num}
                )
            data = r.json()
            items = []
            for obj in (data.get("organic", []) or [])[:num]:
                items.append({"title": obj.get("title"), "url": obj.get("link"), "snippet": obj.get("snippet")})
            return {"engine": "serper", "query": query, "results": items}
        except Exception as e:
            return {"engine": "serper", "query": query, "results": [], "error": str(e)}
    # Fallback: DuckDuckGo lite JSON via html scraping is unreliable; return a hint for the model
    return {
        "engine": "fallback",
        "query": query,
        "results": [],
        "note": "No SERPER_API_KEY set; provide explicit URLs to fetch with http_get."
    }

# Tool: fetch a page
async def tool_http_get(url: str, max_chars: int = 20000) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "Mozilla/5.0 LM Agents"}) as s:
            r = await s.get(url, follow_redirects=True)
        text = r.text
        if len(text) > max_chars:
            # keep start + end to preserve metadata/footer where MLS may appear
            head = text[: math.floor(max_chars * 0.7)]
            tail = text[-math.floor(max_chars * 0.3):]
            text = head + "\n...TRUNCATED...\n" + tail
        return {"status": r.status_code, "url": str(r.url), "content": text}
    except Exception as e:
        return {"status": 0, "url": url, "error": str(e)}

# ---------- Main endpoint with tools ----------
@app.post("/listingmatcher/run", response_model=MatchResponse)
async def run_listingmatcher(request: Request):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: OPENAI_API_KEY missing")

    try:
        incoming = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    flat = normalize_to_flat(incoming)

    lang_pref = "fr" if (flat.language or "").lower().startswith("fr") else "en"
    language_line = (
        "Réponds en français d’abord, puis fournis une courte version en anglais."
        if lang_pref == "fr" else
        "Answer in English first, then provide a short French version."
    )

    # MLS patterns guidance for the model
    mls_hint = (
        "For MLS extraction:\n"
        "- Centris QC: often 7 digits, e.g., 1234567 (use \\b\\d{7}\\b).\n"
        "- REALTOR.ca: can be alphanumeric/hyphen, use \\b[A-Z0-9-]{6,12}\\b (filters out false positives with context like 'MLS').\n"
        "If MLS not on page, set mls:null (do not guess)."
    )

    system_instructions = (
        "You are ListingMatcher, a real-estate assistant for Québec.\n"
        "- Return only CURRENT listings in Québec; if uncertain, say so.\n"
        "- Output 5–12 results, deduplicate by MLS.\n"
        "- Fields: mls, url, address, price_cad, beds, baths, type, note (one-line).\n"
        f"- {mls_hint}\n"
        "- When tools are available, you may search and open pages to verify MLS and details.\n"
        "- ALWAYS return a machine-readable JSON array named 'properties' at the end."
    )

    user_block = (
        f"Location: {flat.location}\n"
        f"Budget: {flat.budget or 'N/A'}\n"
        f"Beds: {flat.beds or 'N/A'}\n"
        f"Baths: {flat.baths or 'N/A'}\n"
        f"Keywords: {flat.keywords or 'N/A'}\n"
        f"Limit: {flat.limit}\n"
        f"{language_line}\n"
        "If you use tools, search with site filters like 'site:centris.ca' or 'site:realtor.ca', then open a few promising URLs and extract MLS/address/price from page content. "
        "Return a concise human summary PLUS a JSON array named 'properties'."
    )

    # Tool specs exposed to the model
    tool_defs = []
    if flat.allow_web:
        tool_defs = [
            {
                "type": "function",
                "name": "web_search",
                "description": "Web search for property listings. Prefer centris.ca, realtor.ca, remax-quebec.com, royallepage.ca, duproprio.com",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num": {"type": "integer", "minimum": 1, "maximum": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "http_get",
                "description": "Fetch the HTML content of a given URL for MLS extraction.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            }
        ]

    # Tool loop
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_block},
    ]

    def run_once(msgs):
        return client.responses.create(
            model="gpt-4.1",
            input=msgs,
            tools=tool_defs if flat.allow_web else None,
        )

    # up to 4 tool iterations
    for _ in range(4):
        resp = run_once(messages)
        # collect tool calls (function calls)
        tool_calls = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "tool_call":
                tool_calls.append(item)
        if not tool_calls:
            # no tool calls; take text
            text = getattr(resp, "output_text", "") or ""
            break

        # execute calls
        tool_outputs: List[Dict[str, Any]] = []
        for call in tool_calls:
            name = call.tool_name
            args = call.arguments or {}
            if name == "web_search":
                out = await tool_web_search(args.get("query", ""), int(args.get("num", 5) or 5))
                tool_outputs.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
            elif name == "http_get":
                out = await tool_http_get(args.get("url", ""))
                tool_outputs.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
            else:
                tool_outputs.append({"role": "tool", "tool_name": name, "content": json.dumps({"error":"unknown tool"})})

        # feed results back
        messages.append({
            "role": "assistant",
            "content": [{"type":"tool_result", "tool_results":[{"call_id": tc.id, "output": to["content"]} for tc,to in zip(tool_calls, tool_outputs)]}]
        })

    else:
        text = getattr(resp, "output_text", "") or ""

    # Extract JSON array "properties"
    props = []
    try:
        m = re.search(r'"properties"\s*:\s*(\[[\s\S]*?\])', text)
        if m:
            props = json.loads(m.group(1))
    except Exception:
        props = []

    # Normalize list
    out: List[Listing] = []
    for p in props[: flat.limit]:
        try:
            out.append(Listing(**p))
        except Exception:
            out.append(Listing(**{k: p.get(k) for k in ["mls","url","address","price_cad","beds","baths","type","note"] if k in p}))

    reply = text.strip() or ("Aucune réponse / No response.")
    return MatchResponse(reply=reply, properties=out)
