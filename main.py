import os, json, re, math
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# ---------- Env & Client ----------
load_dotenv()

# (Optional) neutralize proxies that sometimes break httpx/OpenAI on hosts
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.setdefault("NO_PROXY", "api.openai.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- FastAPI App ----------
app = FastAPI(title="ListingMatcher API", version="1.2.2")

# CORS (handy for Bubble/GHL previews or browser calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Models ----------
class FlatCriteria(BaseModel):
    location: str
    budget: Optional[str] = None           # e.g., "300000-650000 CAD" or "<= 450000 CAD"
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

# ---------- Utilities ----------
def normalize_to_flat(payload: dict) -> FlatCriteria:
    """
    Accepts either:
      - flat {location,budget,beds,baths,keywords,limit,language,allow_web}
      - nested {limit,language,allow_web, criteria:{location,min_price,max_price,beds_min,baths_min,property_types,keywords}}
    Returns FlatCriteria.
    """
    if "criteria" in payload and isinstance(payload["criteria"], dict):
        c = payload["criteria"]
        # budget text
        if c.get("min_price") is not None and c.get("max_price") is not None:
            budget_text = f"{c['min_price']}-{c['max_price']} CAD"
        elif c.get("min_price") is not None:
            budget_text = f">= {c['min_price']} CAD"
        elif c.get("max_price") is not None:
            budget_text = f"<= {c['max_price']} CAD"
        else:
            budget_text = None
        # keywords + property types
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

async def tool_web_search(query: str, num: int = 5) -> Dict[str, Any]:
    """
    Searches with serper.dev if SERPER_API_KEY is set. Otherwise returns a hint.
    """
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
    return {"engine": "fallback", "query": query, "results": [], "note": "No SERPER_API_KEY set; provide explicit URLs to fetch with http_get."}

async def tool_http_get(url: str, max_chars: int = 20000) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "Mozilla/5.0 LM Agents"}) as s:
            r = await s.get(url, follow_redirects=True)
        text = r.text
        if len(text) > max_chars:
            head = text[: math.floor(max_chars * 0.7)]
            tail = text[-math.floor(max_chars * 0.3):]
            text = head + "\n...TRUNCATED...\n" + tail
        return {"status": r.status_code, "url": str(r.url), "content": text}
    except Exception as e:
        return {"status": 0, "url": url, "error": str(e)}

def extract_text(api_resp) -> str:
    """Robustly gather assistant text from Responses API output."""
    try:
        if getattr(api_resp, "output_text", None):
            return api_resp.output_text
        txt = []
        for item in (getattr(api_resp, "output", None) or []):
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t == "message":
                content = getattr(item, "content", None) or (isinstance(item, dict) and item.get("content") or [])
                for block in content or []:
                    if isinstance(block, dict):
                        if block.get("type") in ("output_text", "text") and block.get("text"):
                            txt.append(block["text"])
                        elif block.get("type") == "input_text" and block.get("text"):
                            txt.append(block["text"])
            elif t == "output_text":
                text_val = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text"))
                if text_val:
                    txt.append(text_val)
        return "\n".join(txt).strip()
    except Exception:
        return ""

def extract_tool_calls(api_resp):
    """Find tool/function calls in Responses output."""
    calls = []
    for item in (getattr(api_resp, "output", None) or []):
        obj = item if isinstance(item, dict) else item.__dict__
        if obj.get("type") in ("tool_call", "function_call"):
            calls.append(obj)
    return calls

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

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

    mls_hint = (
        "For MLS extraction:\n"
        "- Centris QC: often 7 digits, e.g., 1234567 (use \\b\\d{7}\\b).\n"
        "- REALTOR.ca: can be alphanumeric/hyphen, use \\b[A-Z0-9-]{6,12}\\b.\n"
        "If MLS not on page, set mls:null (do not guess)."
    )

    system_instructions = (
        "You are ListingMatcher, a real-estate assistant for Québec.\n"
        "- Return only CURRENT listings in Québec; if uncertain, say so.\n"
        "- Output 5–12 results, deduplicate by MLS.\n"
        "- Fields: mls, url, address, price_cad, beds, baths, type, note (one-line).\n"
        f"- {mls_hint}\n"
        "- When tools are available, you may search and open pages to verify MLS and details.\n"
        "- ALWAYS end with a machine-readable JSON array named 'properties'."
    )

    user_block = (
        f"Location: {flat.location}\n"
        f"Budget: {flat.budget or 'N/A'}\n"
        f"Beds: {flat.beds or 'N/A'}\n"
        f"Baths: {flat.baths or 'N/A'}\n"
        f"Keywords: {flat.keywords or 'N/A'}\n"
        f"Limit: {flat.limit}\n"
        f"{language_line}\n"
        "If you use tools, search with site filters like 'site:centris.ca' or 'site:realtor.ca', then open promising URLs and extract MLS/address/price from page content. "
        "Return a concise human summary PLUS a JSON array named 'properties'."
    )

    # tools offered to the model (only if allowed)
    tool_defs = []
    if flat.allow_web:
        tool_defs = [
            {
                "type": "function",
                "name": "web_search",
                "description": "Web search for listings (centris, realtor.ca, remax-quebec, royallepage, duproprio).",
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
                "description": "Fetch a page's HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            }
        ]

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_block},
    ]

    def run_once(msgs):
        return client.responses.create(
            model="gpt-4.1",
            input=msgs,
            tools=tool_defs or None
        )

    text = ""
    last_raw = None

    # ---- Tool loop (max 3 rounds) ----
    for _ in range(3):
        try:
            api_resp = run_once(messages)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

        last_raw = api_resp
        tool_calls = extract_tool_calls(api_resp)

        # If no tool calls, collect any text and break
        if not tool_calls:
            text = extract_text(api_resp) or text
            break

        # Execute each tool call and append a separate role="tool" message with tool_call_id
        for i, call in enumerate(tool_calls):
            call_id = call.get("id") or call.get("tool_call_id") or str(i)
            name = call.get("tool_name") or call.get("name")
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}

            if name == "web_search":
                out = await tool_web_search(args.get("query", ""), int(args.get("num", 5) or 5))
                output_text = json.dumps(out)
            elif name == "http_get":
                out = await tool_http_get(args.get("url", ""))
                output_text = json.dumps(out)
            else:
                output_text = json.dumps({"error": f"unknown tool: {name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": str(call_id),
                "content": [
                    {"type": "output_text", "text": output_text}
                ]
            })

    if not text and last_raw:
        text = extract_text(last_raw)

    # ---- Extract JSON array "properties" from the model's text ----
    props = []
    try:
        m = re.search(r'"properties"\s*:\s*(\[[\s\S]*?\])', text or "")
        if m:
            props = json.loads(m.group(1))
    except Exception:
        props = []

    out: List[Listing] = []
    for p in props[: flat.limit]:
        try:
            out.append(Listing(**p))
        except Exception:
            out.append(Listing(**{k: p.get(k) for k in ["mls","url","address","price_cad","beds","baths","type","note"] if k in p}))

    if not text and not out:
        raise HTTPException(status_code=502, detail="Model returned no text. Try allow_web=false to isolate tools, or set SERPER_API_KEY.")

    return MatchResponse(reply=text or "—", properties=out)
