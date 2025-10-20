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
        "If MLS not on page, set mls:null."
    )

    system_instructions = (
        "You are ListingMatcher, a real-estate assistant for Québec.\n"
        "- Return only CURRENT listings in Québec; if uncertain, say so.\n"
        "- Output 5–12 results, deduplicate by MLS.\n"
        "- Fields: mls, url, address, price_cad, beds, baths, type, note (one-line).\n"
        f"- {mls_hint}\n"
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
        "If tools are available, you may search with site filters like 'site:centris.ca' or 'site:realtor.ca', then open promising URLs and extract MLS/address/price from page content. "
        "Return a concise human summary PLUS a JSON array named 'properties'."
    )

    # tools schema only if allowed
    tool_defs = []
    if flat.allow_web:
        tool_defs = [
            {
                "type": "function",
                "name": "web_search",
                "description": "Web search for listings (centris, realtor.ca, remax-quebec, royallepage, duproprio).",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "num": {"type": "integer"}},
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

    # helper: extract assistant text from Responses API output
    def extract_text(api_resp) -> str:
      try:
        if getattr(api_resp, "output_text", None):
            return api_resp.output_text
        txt = []
        for item in (getattr(api_resp, "output", None) or []):
            # item may be dict-like in new SDKs
            t = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if t == "message":
                content = getattr(item, "content", None) or (isinstance(item, dict) and item.get("content") or [])
                # content is a list of blocks
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") in ("output_text", "text") and block.get("text"):
                            txt.append(block["text"])
                        elif block.get("type") == "input_text" and block.get("text"):  # fallback
                            txt.append(block["text"])
            elif t == "output_text":
                # some SDKs flatten this
                text_val = getattr(item, "text", None) or (isinstance(item, dict) and item.get("text"))
                if text_val:
                    txt.append(text_val)
        return "\n".join(txt).strip()
      except Exception:
        return ""

    # helper: extract tool calls (robust to SDK shape)
    def extract_tool_calls(api_resp):
        calls = []
        for item in (getattr(api_resp, "output", None) or []):
            obj = item if isinstance(item, dict) else item.__dict__
            if obj.get("type") in ("tool_call", "function_call"):
                calls.append(obj)
        return calls

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_block},
    ]

    # run + tool loop
    text = ""
    last_raw = None
    for _ in range(3):  # up to 3 tool iterations
        try:
            api_resp = client.responses.create(
                model="gpt-4.1",
                input=messages,
                tools=tool_defs or None
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

        last_raw = api_resp
        tool_calls = extract_tool_calls(api_resp)

        if not tool_calls:
            text = extract_text(api_resp) or text
            break

        # execute tools and feed back
        tool_results_blocks = []
        for call in tool_calls:
            name = call.get("tool_name") or call.get("name")
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {}
            if name == "web_search":
                out = await tool_web_search(args.get("query",""), int(args.get("num",5) or 5))
                tool_results_blocks.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
            elif name == "http_get":
                out = await tool_http_get(args.get("url",""))
                tool_results_blocks.append({"role": "tool", "tool_name": name, "content": json.dumps(out)})
            else:
                tool_results_blocks.append({"role":"tool","tool_name": name or "unknown","content": json.dumps({"error":"unknown tool"})})

        # IMPORTANT: feed results as a normal assistant turn with a special tool_result block
        messages.append({
            "role": "assistant",
            "content": [{"type":"tool_result","tool_results":[{"call_id": str(i), "output": tr["content"]} for i,tr in enumerate(tool_results_blocks) ]}]
        })

    # final text fallback
    if not text and last_raw:
        text = extract_text(last_raw)

    # extract JSON array "properties"
    props = []
    try:
        m = re.search(r'"properties"\s*:\s*(\[[\s\S]*?\])', text)
        if m:
            props = json.loads(m.group(1))
    except Exception:
        props = []

    out = []
    for p in props[: flat.limit]:
        try: out.append(Listing(**p))
        except: out.append(Listing(**{k: p.get(k) for k in ["mls","url","address","price_cad","beds","baths","type","note"] if k in p}))

    if not text and not out:
        # helpful debug for you in Bubble
        raise HTTPException(status_code=502, detail="Model returned no text. Try allow_web=false to isolate tools, or ensure SERPER_API_KEY is set.")

    return MatchResponse(reply=text or "—", properties=out)
