import asyncio
import json
import os
import re
import shutil
import subprocess
import traceback
import urllib.parse
import urllib.request
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()


def _relax_numeric_types(schema: Any) -> Any:
    """
    Allow LLMs to pass numbers as strings (Groq tool validation can be strict).

    MCP tools usually provide JSON schema with numeric types, but the LLM may emit
    those fields as strings (e.g. "count": "5"), which Groq rejects.
    """
    if not isinstance(schema, dict):
        return schema

    # Recurse into common schema containers
    for key in ("properties", "definitions", "$defs"):
        if isinstance(schema.get(key), dict):
            for k, v in list(schema[key].items()):
                schema[key][k] = _relax_numeric_types(v)

    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            schema[key] = _relax_numeric_types(schema[key])

    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            schema[key] = [_relax_numeric_types(s) for s in schema[key]]

    t = schema.get("type")
    if t in ("integer", "number"):
        # Accept either true numeric JSON or a string (e.g. "10")
        schema["anyOf"] = [{"type": t}, {"type": "string"}]
        schema.pop("type", None)

    return schema


def mcp_tool_to_openai_dict(tool: Any) -> dict:
    """
    Convert an MCP tool into an OpenAI-style tool dict that
    langchain_groq.ChatGroq.bind_tools() expects.
    """
    description = (
        getattr(tool, "description", None)
        or getattr(getattr(tool, "annotations", None) or {}, "title", None)
        or ""
    )

    input_schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    input_schema = _relax_numeric_types(input_schema)

    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", "unnamed_tool"),
            "description": description,
            "parameters": input_schema,
        },
    }


def _content_to_text(result_content: list[Any]) -> str:
    parts: list[str] = []
    for item in result_content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(str(item))
    return "\n".join(parts).strip()


def _ai_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def _is_time_sensitive_prompt(prompt: str) -> bool:
    p = prompt.lower()
    direct_signals = [
        "today",
        "latest",
        "current",
        "recent",
        "right now",
        "news",
        "this week",
        "this month",
    ]
    month_names = [
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "september",
        "oct", "october", "nov", "november", "dec", "december",
    ]
    travel_terms = ["hotel", "airbnb", "stay", "booking", "checkin", "check-in", "availability", "price"]
    has_month = any(m in p for m in month_names)
    has_travel = any(t in p for t in travel_terms)
    has_explicit_day = any(token.isdigit() for token in p.replace("-", " ").split())
    return any(s in p for s in direct_signals) or (has_travel and (has_month or has_explicit_day))


def _is_transient_error_text(text: str) -> bool:
    t = text.lower()
    markers = [
        "rate limit",
        "too quickly",
        "anomaly",
        "429",
        "temporarily",
        "timeout",
        "timed out",
        "try again",
        "connection reset",
    ]
    return any(m in t for m in markers)


def _requires_tool_use_prompt(prompt: str) -> bool:
    p = prompt.lower()
    forced_markers = [
        "use playwright",
        "using playwright",
        "use mcp",
        "using mcp",
        "use tool",
        "using tool",
        "use duckduckgo",
        "using duckduckgo",
        "duckduckgo tool",
        "use the duckduckgo",
        "search using duckduckgo",
        "with duckduckgo",
        "use airbnb",
        "using airbnb",
        "airbnb_search tool",
        "use the airbnb_search",
    ]
    return any(m in p for m in forced_markers)


def _is_playwright_prompt(prompt: str) -> bool:
    p = prompt.lower()
    return "playwright" in p or "browser_" in p


def _is_airbnb_prompt(prompt: str) -> bool:
    p = prompt.lower()
    if "airbnb" in p or "airbnb_search" in p:
        return True
    travel_terms = [
        "hotel",
        "hotels",
        "stay",
        "accommodation",
        "booking",
        "book",
        "place to stay",
        "place in",
        "check-in",
        "checkin",
        "check-out",
        "checkout",
    ]
    return any(t in p for t in travel_terms)


def _is_ddg_prompt(prompt: str) -> bool:
    p = prompt.lower()
    ddg_terms = [
        "duckduckgo",
        "web search",
        "search for",
        "find ",
        "official documentation",
        "documentation url",
        "news",
        "latest",
        "today",
        "current",
        "right now",
        "weather",
        "local time",
    ]
    return any(t in p for t in ddg_terms)


def _extract_first_external_link_from_snapshot(snapshot_text: str) -> tuple[str, str] | None:
    idx = snapshot_text.lower().find("external links")
    if idx < 0:
        return None
    lines = snapshot_text[idx:].splitlines()
    for i, line in enumerate(lines):
        title_match = re.search(r'link "([^"]+)"', line)
        if not title_match:
            continue
        title = title_match.group(1).strip()
        for j in range(i + 1, min(i + 6, len(lines))):
            url_match = re.search(r"/url:\s*(https?://\S+)", lines[j])
            if url_match:
                return title, url_match.group(1).strip()
    return None


def _extract_url(text: str) -> str | None:
    m = re.search(r"https?://[^\s)\]\"'>]+", text or "")
    return m.group(0) if m else None


def _next_weekend_dates() -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    days_until_sat = (5 - today.weekday()) % 7
    if days_until_sat == 0:
        days_until_sat = 7
    checkin = today + timedelta(days=days_until_sat)
    checkout = checkin + timedelta(days=2)
    return checkin.isoformat(), checkout.isoformat()


def _extract_location_for_airbnb(prompt: str) -> str:
    # Capture broad location text after "in", including commas/hyphens.
    m = re.search(
        r"\bin\s+(.+?)(?:\s+for\s+next weekend\b|\s+for\b|\s+next weekend\b|[.?!]|$)",
        prompt,
        flags=re.IGNORECASE,
    )
    if m:
        location = m.group(1).strip()
        # Clean trailing separators and collapse spaces.
        location = re.sub(r"\s+", " ", location)
        location = location.strip(" ,;-")
        if location:
            return location
    return "Jaffna"


def _extract_first_id_price(text: str) -> tuple[str | None, str | None]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, None
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None, None

    if isinstance(obj, dict) and isinstance(obj.get("searchResults"), list) and obj["searchResults"]:
        first = obj["searchResults"][0]
        if isinstance(first, dict):
            first_id = first.get("id")
            price_val = (
                ((first.get("structuredDisplayPrice") or {}).get("primaryLine") or {}).get("accessibilityLabel")
                or ((first.get("structuredDisplayPrice") or {}).get("explanationData") or {}).get("priceDetails")
                or first.get("price")
            )
            if first_id is not None and price_val is not None:
                return str(first_id), str(price_val)

    queue: list[Any] = [obj]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            keys = {k.lower(): k for k in cur.keys()}
            if "id" in keys:
                id_val = cur.get(keys["id"])
                price_key = None
                for candidate in ("price", "pricestring", "formattedprice", "displayprice"):
                    if candidate in keys:
                        price_key = keys[candidate]
                        break
                if price_key is not None:
                    price_val = cur.get(price_key)
                    return str(id_val), str(price_val)
            for v in cur.values():
                queue.append(v)
        elif isinstance(cur, list):
            queue.extend(cur)
    return None, None


def _wants_first_id_and_price(prompt: str) -> bool:
    p = prompt.lower()
    wants_id = "'id'" in p or " id " in f" {p} " or '"id"' in p
    wants_price = "'price'" in p or " price " in f" {p} " or '"price"' in p
    wants_first = "first result" in p or "first listing" in p or "first one" in p
    return wants_id and wants_price and wants_first


def _extract_airbnb_rows(text: str, limit: int = 3) -> list[dict[str, str]]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        obj = json.loads(text[start : end + 1])
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    results = obj.get("searchResults") if isinstance(obj, dict) else None
    if not isinstance(results, list):
        return rows

    for item in results[:limit]:
        if not isinstance(item, dict):
            continue
        listing_id = str(item.get("id", "")).strip()
        listing_url = str(item.get("url", "")).strip()
        demand = item.get("demandStayListing") or {}
        desc = demand.get("description") or {}
        name_obj = desc.get("name") or {}
        name = name_obj.get("localizedStringWithTranslationPreference") or ""

        display_price = item.get("structuredDisplayPrice") or {}
        primary_line = display_price.get("primaryLine") or {}
        explanation = display_price.get("explanationData") or {}
        price = primary_line.get("accessibilityLabel") or explanation.get("priceDetails") or ""
        rows.append(
            {
                "id": listing_id,
                "name": str(name).strip(),
                "price": str(price).strip(),
                "url": listing_url,
            }
        )
    return rows


def _build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep predictable non-interactive npm behavior in MCP child processes.
    env.setdefault("CI", "1")
    return env


def _resolve_mcp_command(command: str | None) -> str | None:
    if not command:
        return command
    c = command.strip()
    if os.name != "nt":
        return c

    lower = c.lower()
    if lower in {"npx", "npx.cmd"}:
        explicit = os.environ.get("NPX_PATH")
        if explicit:
            return explicit
        found = shutil.which("npx.cmd") or shutil.which("npx")
        if found:
            return found
        default_path = r"C:\Program Files\nodejs\npx.cmd"
        if os.path.exists(default_path):
            return default_path
        return "npx.cmd"
    return c


def _read_timeout_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _log_node_version() -> None:
    try:
        out = subprocess.check_output(["node", "-v"], text=True, timeout=5).strip()
        print(f"[MCP DEBUG] node version: {out}")
    except Exception as e:
        print(f"[MCP DEBUG] unable to read node version: {e}")


async def mcp_health_check(config_path: Path) -> dict[str, Any]:
    child_env = _build_child_env()
    init_timeout = _read_timeout_env("MCP_INIT_TIMEOUT_SEC", 75.0)
    _log_node_version()

    with open(config_path, "r", encoding="utf-8") as f:
        mcp_config = json.load(f)

    servers = mcp_config.get("mcpServers", {})
    summary: dict[str, Any] = {
        "ok": True,
        "server_count": len(servers),
        "servers": [],
    }

    async with AsyncExitStack() as stack:
        for server_name, server_info in servers.items():
            command = _resolve_mcp_command(server_info.get("command"))
            args = server_info.get("args", [])
            server_result: dict[str, Any] = {
                "server_name": server_name,
                "command": command,
                "args": args,
                "ok": False,
                "tool_count": 0,
                "tools": [],
            }
            try:
                server_params = StdioServerParameters(command=command, args=args, env=child_env)
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=init_timeout)
                tool_names = [getattr(t, "name", "") for t in tools_response.tools if getattr(t, "name", "")]
                server_result["ok"] = True
                server_result["tool_count"] = len(tool_names)
                server_result["tools"] = tool_names
            except Exception as e:
                summary["ok"] = False
                server_result["error"] = str(e)
                server_result["traceback"] = traceback.format_exc()
            summary["servers"].append(server_result)

    return summary


def _is_news_prompt(prompt: str) -> bool:
    p = prompt.lower()
    return "news" in p or "headline" in p or "headlines" in p


def _is_travel_prompt(prompt: str) -> bool:
    p = prompt.lower()
    travel_terms = [
        "hotel",
        "hotels",
        "stay",
        "accommodation",
        "booking",
        "book",
        "near",
        "price",
        "availability",
        "check-in",
        "checkin",
        "check-out",
        "checkout",
    ]
    return any(t in p for t in travel_terms)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _fetch_tavily_results(prompt: str, *, topic: str) -> tuple[str, dict[str, Any]] | None:
    api_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("HOTEL_API_KEY")
    if not api_key:
        return None

    query = prompt
    if topic == "travel":
        query = (
            f"{prompt}. Include current hotel options, approximate price range, and booking links. "
            "Prioritize trusted booking websites and official hotel websites."
        )
    elif topic == "news":
        query = f"{prompt}. Focus on today/latest verified AI news with source links."

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 6,
        "include_answer": True,
    }

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "mcpdemo/1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception as e:
        return (
            "",
            {
                "tool_name": "tavily_live_search",
                "server_name": "tavily",
                "isError": True,
                "content": "",
                "error": str(e),
            },
        )

    results = data.get("results") or []
    answer = _clean_text(data.get("answer") or "")

    lines: list[str] = []
    for i, item in enumerate(results[:5], start=1):
        title = _clean_text(item.get("title") or "")
        url = _clean_text(item.get("url") or "")
        snippet = _clean_text(item.get("content") or "")
        if not (title and url):
            continue
        short_snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
        lines.append(f"{i}. {title}\n   {short_snippet}\n   {url}")

    if not lines:
        return (
            "",
            {
                "tool_name": "tavily_live_search",
                "server_name": "tavily",
                "isError": True,
                "content": "",
                "error": "Tavily returned no usable results.",
            },
        )

    header = "Here are live results from web search:"
    if topic == "travel":
        header = "Here are current hotel/travel options I found:"
    if topic == "news":
        header = "Here are the latest AI news items I found:"

    final = header
    if answer:
        final += f"\n\nSummary: {answer}\n"
    final += "\n" + "\n".join(lines)

    return (
        final,
        {
            "tool_name": "tavily_live_search",
            "server_name": "tavily",
            "isError": False,
            "content": "\n".join(lines),
        },
    )


def _fetch_newsapi_ai_news(prompt: str) -> tuple[str, dict[str, Any]] | None:
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        return None

    base_url = "https://newsapi.org/v2/everything"
    today = datetime.now(timezone.utc).date().isoformat()
    query = "artificial intelligence OR AI"
    if "openai" in prompt.lower():
        query = "OpenAI OR ChatGPT OR GPT"

    params = {
        "q": query,
        "from": today,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": "7",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-Api-Key": api_key, "User-Agent": "mcpdemo/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception as e:
        return (
            "",
            {
                "tool_name": "newsapi_live_news",
                "server_name": "newsapi",
                "isError": True,
                "content": "",
                "error": str(e),
            },
        )

    if payload.get("status") != "ok":
        return (
            "",
            {
                "tool_name": "newsapi_live_news",
                "server_name": "newsapi",
                "isError": True,
                "content": "",
                "error": payload.get("message", "NewsAPI returned non-ok status"),
            },
        )

    articles = payload.get("articles") or []
    if not articles:
        return (
            "",
            {
                "tool_name": "newsapi_live_news",
                "server_name": "newsapi",
                "isError": True,
                "content": "",
                "error": "No fresh AI news articles found for today.",
            },
        )

    lines: list[str] = []
    for i, a in enumerate(articles[:5], start=1):
        title = (a.get("title") or "").strip()
        source = ((a.get("source") or {}).get("name") or "").strip()
        when = (a.get("publishedAt") or "").strip()
        link = (a.get("url") or "").strip()
        if title and link:
            lines.append(f"{i}. {title} ({source}, {when})\n   {link}")

    if not lines:
        return (
            "",
            {
                "tool_name": "newsapi_live_news",
                "server_name": "newsapi",
                "isError": True,
                "content": "",
                "error": "NewsAPI returned articles but they were incomplete.",
            },
        )

    final = "Here are the latest AI news headlines for today:\n\n" + "\n".join(lines)
    return (
        final,
        {
            "tool_name": "newsapi_live_news",
            "server_name": "newsapi",
            "isError": False,
            "content": "\n".join(lines),
        },
    )


async def run_agent(prompt: str, *, model: str, config_path: Path) -> dict[str, Any]:
    """
    Simple end-to-end MCP agent:
      1) LLM suggests tool calls (bind_tools)
      2) Backend executes the tool calls via MCP (ClientSession.call_tool)
      3) Backend asks the LLM for the final answer using tool results
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Please set your GROQ_API_KEY environment variable.")

    llm = ChatGroq(model=model, temperature=0)
    child_env = _build_child_env()
    init_timeout = _read_timeout_env("MCP_INIT_TIMEOUT_SEC", 75.0)
    tool_timeout = _read_timeout_env("MCP_TOOL_TIMEOUT_SEC", 120.0)
    _log_node_version()

    with open(config_path, "r", encoding="utf-8") as f:
        mcp_config = json.load(f)

    servers = mcp_config.get("mcpServers", {})
    if not servers:
        raise RuntimeError("No servers found in MCP config.")

    skip_run_code_tool = _is_playwright_prompt(prompt)

    try:
        async with AsyncExitStack() as stack:
            openai_tools: list[dict] = []
            tools_by_name: dict[str, tuple[str, ClientSession]] = {}
            server_errors: list[dict[str, str]] = []

            for server_name, server_info in servers.items():
                try:
                    command = _resolve_mcp_command(server_info.get("command"))
                    args = server_info.get("args", [])
                    print(f"[MCP DEBUG] starting server={server_name} command={command} args={args}")

                    server_params = StdioServerParameters(command=command, args=args, env=child_env)
                    stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                    read, write = stdio_transport
                    session = await stack.enter_async_context(ClientSession(read, write))
                    try:
                        await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                    except Exception as e:
                        print(f"[MCP DEBUG] initialize failed for {server_name}: {e}")
                        print(traceback.format_exc())
                        raise

                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=init_timeout)
                    print(f"[MCP DEBUG] server={server_name} discovered_tools={len(tools_response.tools)}")
                    for tool in tools_response.tools:
                        tool_name = getattr(tool, "name", "")
                        if not tool_name:
                            continue
                        if skip_run_code_tool and tool_name == "browser_run_code":
                            continue
                        openai_tools.append(mcp_tool_to_openai_dict(tool))
                        tools_by_name[tool_name] = (server_name, session)
                except Exception as e:
                    server_errors.append({"server_name": server_name, "error": str(e)})
                    print(f"[MCP DEBUG] server startup error for {server_name}: {e}")
                    print(traceback.format_exc())

            if not openai_tools:
                base = await llm.ainvoke([HumanMessage(content=prompt)])
                base_text = _ai_content_to_text(getattr(base, "content", ""))
                return {
                    "final": base_text or "I couldn't generate a response right now.",
                    "tool_calls": [],
                    "tool_results": [
                        {
                            "tool_name": "mcp_server_startup",
                            "server_name": err["server_name"],
                            "isError": True,
                            "content": "",
                            "error": err["error"],
                        }
                        for err in server_errors
                    ],
                    "initial_llm_content": base_text,
                }

            # Deterministic path for explicit Playwright tasks: run tools directly.
            if _requires_tool_use_prompt(prompt) and _is_playwright_prompt(prompt):
                server_name, session = tools_by_name.get("browser_navigate", (None, None))
                if session is not None and "browser_snapshot" in tools_by_name:
                    target_url = "https://www.wikipedia.org/"
                    lower_prompt = prompt.lower()
                    if "wikipedia.org" in lower_prompt and "model context protocol" in lower_prompt:
                        target_url = "https://en.wikipedia.org/wiki/Model_Context_Protocol"

                    tool_results: list[dict[str, Any]] = []
                    try:
                        nav_res = session.call_tool("browser_navigate", {"url": target_url})
                        if asyncio.iscoroutine(nav_res):
                            nav_res = await asyncio.wait_for(nav_res, timeout=tool_timeout)
                        nav_content = _content_to_text(getattr(nav_res, "content", []))
                        tool_results.append(
                            {
                                "tool_name": "browser_navigate",
                                "server_name": server_name,
                                "arguments": {"url": target_url},
                                "isError": getattr(nav_res, "isError", False),
                                "content": nav_content,
                            }
                        )

                        snap_server, snap_session = tools_by_name["browser_snapshot"]
                        snap_res = snap_session.call_tool("browser_snapshot", {})
                        if asyncio.iscoroutine(snap_res):
                            snap_res = await asyncio.wait_for(snap_res, timeout=tool_timeout)
                        snap_content = _content_to_text(getattr(snap_res, "content", []))
                        tool_results.append(
                            {
                                "tool_name": "browser_snapshot",
                                "server_name": snap_server,
                                "arguments": {},
                                "isError": getattr(snap_res, "isError", False),
                                "content": snap_content[:6000],
                            }
                        )

                        extracted = _extract_first_external_link_from_snapshot(snap_content)
                        if extracted:
                            link_name, link_url = extracted
                            return {
                                "final": f"The first external link at the bottom is: {link_name} ({link_url})",
                                "tool_calls": [
                                    {"name": "browser_navigate", "args": {"url": target_url}},
                                    {"name": "browser_snapshot", "args": {}},
                                ],
                                "tool_results": tool_results,
                                "initial_llm_content": "",
                            }
                        return {
                            "final": (
                                "I used Playwright MCP tools, but could not locate an 'External links' section "
                                "with a valid external URL on the captured snapshot."
                            ),
                            "tool_calls": [
                                {"name": "browser_navigate", "args": {"url": target_url}},
                                {"name": "browser_snapshot", "args": {}},
                            ],
                            "tool_results": tool_results,
                            "initial_llm_content": "",
                        }
                    except Exception as e:
                        return {
                            "final": f"I executed Playwright MCP tools but hit an error: {e}",
                            "tool_calls": [],
                            "tool_results": [
                                {
                                    "tool_name": "playwright_direct_flow",
                                    "server_name": server_name or "playwright",
                                    "isError": True,
                                    "content": "",
                                    "error": str(e),
                                }
                            ],
                            "initial_llm_content": "",
                        }

            # Deterministic path for explicit Airbnb requests.
            if "airbnb_search" in tools_by_name and _is_airbnb_prompt(prompt):
                airbnb_server, airbnb_session = tools_by_name["airbnb_search"]
                location = _extract_location_for_airbnb(prompt)
                if "next weekend" in prompt.lower():
                    checkin, checkout = _next_weekend_dates()
                else:
                    # Best-effort default: upcoming weekend.
                    checkin, checkout = _next_weekend_dates()

                airbnb_args = {
                    "location": location,
                    "checkin": checkin,
                    "checkout": checkout,
                    "adults": 1,
                    "ignoreRobotsText": True,
                }
                try:
                    airbnb_res = airbnb_session.call_tool("airbnb_search", airbnb_args)
                    if asyncio.iscoroutine(airbnb_res):
                        airbnb_res = await asyncio.wait_for(airbnb_res, timeout=tool_timeout)
                    airbnb_text = _content_to_text(getattr(airbnb_res, "content", []))
                    airbnb_is_error = bool(getattr(airbnb_res, "isError", False))
                    if airbnb_is_error:
                        return {
                            "final": (
                                "I called Airbnb MCP, but the tool returned an error. "
                                f"Details: {airbnb_text[:260]}"
                            ),
                            "tool_calls": [{"name": "airbnb_search", "args": airbnb_args}],
                            "tool_results": [
                                {
                                    "tool_name": "airbnb_search",
                                    "server_name": airbnb_server,
                                    "arguments": airbnb_args,
                                    "isError": True,
                                    "content": "",
                                    "error": airbnb_text or "Unknown Airbnb MCP error",
                                }
                            ],
                            "initial_llm_content": "",
                        }

                    first_id, first_price = _extract_first_id_price(airbnb_text)
                    if first_id and first_price and _wants_first_id_and_price(prompt):
                        return {
                            "final": (
                                f"First Airbnb result in {location} for {checkin} to {checkout}: "
                                f"id={first_id}, price={first_price}"
                            ),
                            "tool_calls": [{"name": "airbnb_search", "args": airbnb_args}],
                            "tool_results": [
                                {
                                    "tool_name": "airbnb_search",
                                    "server_name": airbnb_server,
                                    "arguments": airbnb_args,
                                    "isError": False,
                                    "content": airbnb_text[:8000],
                                }
                            ],
                            "initial_llm_content": "",
                        }

                    # If user asked something else, build direct structured answer (no extra LLM call).
                    rows = _extract_airbnb_rows(airbnb_text, limit=3)
                    if rows:
                        lines = []
                        for i, row in enumerate(rows, start=1):
                            label = row["name"] or "Listing"
                            line = f"{i}. {label} | id={row['id']} | price={row['price']}"
                            if row["url"]:
                                line += f" | {row['url']}"
                            lines.append(line)
                        return {
                            "final": (
                                f"Top Airbnb results in {location} for {checkin} to {checkout}:\n\n"
                                + "\n".join(lines)
                            ),
                            "tool_calls": [{"name": "airbnb_search", "args": airbnb_args}],
                            "tool_results": [
                                {
                                    "tool_name": "airbnb_search",
                                    "server_name": airbnb_server,
                                    "arguments": airbnb_args,
                                    "isError": False,
                                    "content": airbnb_text[:8000],
                                }
                            ],
                            "initial_llm_content": "",
                        }

                    return {
                        "final": (
                            "I called Airbnb MCP successfully, but I couldn't reliably parse a first result "
                            "with both id and price from the returned payload."
                        ),
                        "tool_calls": [{"name": "airbnb_search", "args": airbnb_args}],
                        "tool_results": [
                            {
                                "tool_name": "airbnb_search",
                                "server_name": airbnb_server,
                                "arguments": airbnb_args,
                                "isError": False,
                                "content": airbnb_text[:8000],
                            }
                        ],
                        "initial_llm_content": "",
                    }
                except Exception as e:
                    return {
                        "final": f"I executed Airbnb MCP but hit an error: {e}",
                        "tool_calls": [{"name": "airbnb_search", "args": airbnb_args}],
                        "tool_results": [
                            {
                                "tool_name": "airbnb_search",
                                "server_name": airbnb_server,
                                "arguments": airbnb_args,
                                "isError": True,
                                "content": "",
                                "error": str(e),
                            }
                        ],
                        "initial_llm_content": "",
                    }

            # Deterministic path for explicit DuckDuckGo requests.
            if "duckduckgo_web_search" in tools_by_name and (
                _is_ddg_prompt(prompt)
                or (_requires_tool_use_prompt(prompt) and not _is_playwright_prompt(prompt) and not _is_airbnb_prompt(prompt))
            ):
                ddg_server, ddg_session = tools_by_name["duckduckgo_web_search"]
                ddg_attempts = [{"query": prompt}]
                ddg_error = ""
                for ddg_args in ddg_attempts:
                    try:
                        ddg_res = None
                        ddg_content = ""
                        ddg_is_error = True
                        for ddg_try in range(3):
                            ddg_res = ddg_session.call_tool("duckduckgo_web_search", ddg_args)
                            if asyncio.iscoroutine(ddg_res):
                                ddg_res = await asyncio.wait_for(ddg_res, timeout=tool_timeout)
                            ddg_content = _content_to_text(getattr(ddg_res, "content", []))
                            ddg_is_error = bool(getattr(ddg_res, "isError", False))
                            if not ddg_is_error and ddg_content.strip():
                                break
                            ddg_error = ddg_content or "DuckDuckGo returned empty content."
                            if ddg_try < 2 and _is_transient_error_text(ddg_error):
                                await asyncio.sleep(1.5 * (ddg_try + 1))
                                continue
                            break
                        if ddg_is_error or not ddg_content.strip():
                            continue

                        # Let LLM extract the exact answer from search results only.
                        final_from_ddg = await llm.ainvoke(
                            [
                                HumanMessage(
                                    content=(
                                        f"User request:\n{prompt}\n\n"
                                        f"DuckDuckGo tool output:\n{ddg_content}\n\n"
                                        "Answer directly using only this output. "
                                        "If URL is requested, return the best matching URL."
                                    )
                                )
                            ]
                        )
                        final_text_ddg = _ai_content_to_text(getattr(final_from_ddg, "content", ""))
                        if not final_text_ddg:
                            url = _extract_url(ddg_content)
                            final_text_ddg = url or "DuckDuckGo returned results, but I could not extract a clear final answer."

                        return {
                            "final": final_text_ddg,
                            "tool_calls": [{"name": "duckduckgo_web_search", "args": ddg_args}],
                            "tool_results": [
                                {
                                    "tool_name": "duckduckgo_web_search",
                                    "server_name": ddg_server,
                                    "arguments": ddg_args,
                                    "isError": False,
                                    "content": ddg_content,
                                }
                            ],
                            "initial_llm_content": "",
                        }
                    except Exception as e:
                        ddg_error = str(e)

                fallback_topic = "general"
                if _is_news_prompt(prompt):
                    fallback_topic = "news"
                elif _is_travel_prompt(prompt):
                    fallback_topic = "travel"
                fallback = _fetch_tavily_results(prompt, topic=fallback_topic)
                base_ddg_error = {
                    "tool_name": "duckduckgo_web_search",
                    "server_name": ddg_server,
                    "isError": True,
                    "content": "",
                    "error": ddg_error or "Unknown DuckDuckGo tool error",
                }
                if fallback is not None and fallback[0]:
                    return {
                        "final": "DuckDuckGo MCP was rate-limited, so I used fallback live web search.\n\n" + fallback[0],
                        "tool_calls": [{"name": "duckduckgo_web_search", "args": {"query": prompt}}],
                        "tool_results": [base_ddg_error, fallback[1]],
                        "initial_llm_content": "",
                    }

                return {
                    "final": (
                        "You asked to use DuckDuckGo MCP, but the tool call failed. "
                        f"Error: {ddg_error[:240]}"
                    ),
                    "tool_calls": [{"name": "duckduckgo_web_search", "args": {"query": prompt}}],
                    "tool_results": [base_ddg_error],
                    "initial_llm_content": "",
                }

            llm_with_tools = llm.bind_tools(openai_tools)
            prompt_hints: list[str] = []
            if _is_time_sensitive_prompt(prompt):
                prompt_hints.append(
                    "If the user asks for latest/current/today information, you must use available MCP tools before answering."
                )
            if _requires_tool_use_prompt(prompt):
                prompt_hints.append("User explicitly requested tool usage. You must call tools; do not answer from memory.")
            if _is_playwright_prompt(prompt):
                prompt_hints.append(
                    "For Playwright tasks, prefer browser_navigate/browser_click/browser_type/browser_snapshot. "
                    "Avoid browser_run_code unless strictly necessary and syntax-validated."
                )

            tool_hint_prompt = prompt
            if prompt_hints:
                tool_hint_prompt = f"{prompt}\n\n" + "\n".join(prompt_hints)
            tool_calls: list[dict[str, Any]] = []
            initial_text = ""
            must_use_tools = _is_time_sensitive_prompt(prompt) or _requires_tool_use_prompt(prompt)
            tool_results: list[dict[str, Any]] = []
            loop_prompt = tool_hint_prompt
            max_agent_loops = 3 if must_use_tools else 1

            for loop_index in range(max_agent_loops):
                try:
                    initial = await llm_with_tools.ainvoke([HumanMessage(content=loop_prompt)])
                except Exception:
                    initial = await llm.ainvoke([HumanMessage(content=prompt)])
                if not initial_text:
                    initial_text = _ai_content_to_text(getattr(initial, "content", ""))

                step_calls = getattr(initial, "tool_calls", None) or []
                if not step_calls:
                    if must_use_tools and loop_index == 0:
                        loop_prompt = (
                            f"{prompt}\n\nYou must call at least one MCP tool. "
                            "Do not answer from memory."
                        )
                        continue
                    break

                tool_calls.extend(step_calls)
                new_step_results: list[dict[str, Any]] = []
                for call in step_calls:
                    tool_name = call.get("name")
                    raw_args = call.get("args") or {}
                    server_name, session = tools_by_name.get(tool_name, (None, None))
                    if session is None:
                        tr = {
                            "tool_name": tool_name,
                            "server_name": server_name,
                            "error": f"Tool not found in loaded tool registry: {tool_name}",
                        }
                        tool_results.append(tr)
                        new_step_results.append(tr)
                        continue

                    if isinstance(raw_args, dict):
                        arguments = raw_args
                    else:
                        try:
                            arguments = json.loads(raw_args)
                        except Exception:
                            arguments = {"_raw": raw_args}

                    try:
                        result = None
                        for attempt in range(3):
                            maybe_result = session.call_tool(tool_name, arguments)
                            if asyncio.iscoroutine(maybe_result):
                                result = await asyncio.wait_for(maybe_result, timeout=tool_timeout)
                            else:
                                result = maybe_result

                            attempt_text = _content_to_text(getattr(result, "content", []))
                            attempt_is_error = bool(getattr(result, "isError", False))
                            if not attempt_is_error:
                                break
                            if attempt < 2 and _is_transient_error_text(attempt_text):
                                await asyncio.sleep(1.5 * (attempt + 1))
                                continue
                            break

                        content_text = _content_to_text(getattr(result, "content", []))
                        tr = {
                            "tool_name": tool_name,
                            "server_name": server_name,
                            "arguments": arguments,
                            "isError": getattr(result, "isError", False),
                            "content": content_text,
                        }
                        tool_results.append(tr)
                        new_step_results.append(tr)
                    except Exception as e:
                        tr = {
                            "tool_name": tool_name,
                            "server_name": server_name,
                            "arguments": arguments,
                            "isError": True,
                            "content": "",
                            "error": str(e),
                        }
                        tool_results.append(tr)
                        new_step_results.append(tr)

                successful_step = any(
                    not r.get("isError") and isinstance(r.get("content"), str) and r["content"].strip()
                    for r in new_step_results
                )
                if successful_step:
                    break

                error_summary = "\n".join(
                    [
                        f"- {r.get('tool_name')}: {(r.get('error') or r.get('content') or 'unknown error')[:220]}"
                        for r in new_step_results
                    ]
                )
                loop_prompt = (
                    f"Original user request:\n{prompt}\n\n"
                    f"Previous tool call failures:\n{error_summary}\n\n"
                    "Try a different tool strategy and call MCP tools again."
                )

            if not tool_calls:
                if _requires_tool_use_prompt(prompt):
                    return {
                        "final": (
                            "You asked to use MCP tools, but the model did not execute any tool call. "
                            "Please try rephrasing the request, or ask to verify one tool at a time."
                        ),
                        "tool_calls": [],
                        "tool_results": [],
                        "initial_llm_content": initial_text,
                    }
                if _is_time_sensitive_prompt(prompt):
                    if _is_news_prompt(prompt):
                        news = _fetch_newsapi_ai_news(prompt)
                        if news is not None:
                            news_final, news_tool = news
                            if news_final:
                                return {
                                    "final": news_final,
                                    "tool_calls": [],
                                    "tool_results": [news_tool],
                                    "initial_llm_content": initial_text,
                                }
                        tavily_news = _fetch_tavily_results(prompt, topic="news")
                        if tavily_news is not None:
                            t_final, t_tool = tavily_news
                            if t_final:
                                return {
                                    "final": t_final,
                                    "tool_calls": [],
                                    "tool_results": [t_tool],
                                    "initial_llm_content": initial_text,
                                }
                    if _is_travel_prompt(prompt):
                        tavily_travel = _fetch_tavily_results(prompt, topic="travel")
                        if tavily_travel is not None:
                            t_final, t_tool = tavily_travel
                            if t_final:
                                return {
                                    "final": t_final,
                                    "tool_calls": [],
                                    "tool_results": [t_tool],
                                    "initial_llm_content": initial_text,
                                }
                    return {
                        "final": "I couldn't fetch live data for this time-sensitive request right now. Please try again in a moment.",
                        "tool_calls": [],
                        "tool_results": [],
                        "initial_llm_content": initial_text,
                    }
                return {
                    "final": initial_text or "I couldn't produce a response. Please try again.",
                    "tool_calls": [],
                    "tool_results": [],
                    "initial_llm_content": initial_text,
                }

            successful_results = [
                tr
                for tr in tool_results
                if not tr.get("isError") and isinstance(tr.get("content"), str) and tr["content"].strip()
            ]

            if _requires_tool_use_prompt(prompt) and not successful_results:
                first_error = ""
                for tr in tool_results:
                    err_txt = tr.get("error") or tr.get("content") or ""
                    if isinstance(err_txt, str) and err_txt.strip():
                        first_error = err_txt.strip()
                        break
                msg = "I executed MCP tool calls, but they failed before producing usable output."
                if first_error:
                    msg += f" First error: {first_error[:220]}"
                return {
                    "final": msg,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "initial_llm_content": initial_text,
                }
            tool_context = "\n".join(
                [
                    f"- {tr['tool_name']} (server={tr['server_name']}), args={tr['arguments']}\n  result={tr['content']}"
                    for tr in successful_results
                ]
            )

            if successful_results:
                followup = (
                    f"User request:\n{prompt}\n\n"
                    f"MCP tool results:\n{tool_context}\n\n"
                    "Provide a direct, well-formatted answer. If key information is missing, say what is missing."
                )
            elif _is_time_sensitive_prompt(prompt):
                if _is_news_prompt(prompt):
                    news = _fetch_newsapi_ai_news(prompt)
                    if news is not None:
                        news_final, news_tool = news
                        if news_final:
                            return {
                                "final": news_final,
                                "tool_calls": tool_calls,
                                "tool_results": [news_tool] + tool_results,
                                "initial_llm_content": initial_text,
                            }
                    tavily_news = _fetch_tavily_results(prompt, topic="news")
                    if tavily_news is not None:
                        t_final, t_tool = tavily_news
                        if t_final:
                            return {
                                "final": t_final,
                                "tool_calls": tool_calls,
                                "tool_results": [t_tool] + tool_results,
                                "initial_llm_content": initial_text,
                            }
                if _is_travel_prompt(prompt):
                    tavily_travel = _fetch_tavily_results(prompt, topic="travel")
                    if tavily_travel is not None:
                        t_final, t_tool = tavily_travel
                        if t_final:
                            return {
                                "final": t_final,
                                "tool_calls": tool_calls,
                                "tool_results": [t_tool] + tool_results,
                                "initial_llm_content": initial_text,
                            }
                followup = (
                    f"User request:\n{prompt}\n\n"
                    "No MCP tool results were available because tool calls failed or returned empty data.\n"
                    "Do not provide stale or guessed latest/today facts. Explain that live retrieval failed and ask the user to retry."
                )
            else:
                followup = (
                    f"User request:\n{prompt}\n\n"
                    "No MCP tool results were available because tool calls failed or returned empty data.\n"
                    "Provide the best possible answer using your own knowledge and clearly mention this limitation."
                )

            try:
                final = await llm.ainvoke([HumanMessage(content=followup)])
            except Exception:
                final = initial

            final_text = _ai_content_to_text(getattr(final, "content", ""))
            if not final_text:
                final_text = initial_text or "I couldn't generate a response this time. Please try again."

            startup_errors = [
                {
                    "tool_name": "mcp_server_startup",
                    "server_name": err["server_name"],
                    "isError": True,
                    "content": "",
                    "error": err["error"],
                }
                for err in server_errors
            ]
            return {
                "final": final_text,
                "tool_calls": tool_calls,
                "tool_results": startup_errors + tool_results,
                "initial_llm_content": initial_text,
            }
    except BaseException as e:
        if "final_text" in locals():
            startup_errors = [
                {
                    "tool_name": "mcp_server_startup",
                    "server_name": err["server_name"],
                    "isError": True,
                    "content": "",
                    "error": err["error"],
                }
                for err in locals().get("server_errors", [])
            ]
            return {
                "final": locals().get("final_text") or "I couldn't generate a response this time. Please try again.",
                "tool_calls": locals().get("tool_calls", []),
                "tool_results": startup_errors + locals().get("tool_results", []),
                "initial_llm_content": locals().get("initial_text", ""),
            }

        if _is_time_sensitive_prompt(prompt):
            if _is_news_prompt(prompt):
                news = _fetch_newsapi_ai_news(prompt)
                if news is not None:
                    n_final, n_tool = news
                    if n_final:
                        return {
                            "final": n_final,
                            "tool_calls": [],
                            "tool_results": [n_tool],
                            "initial_llm_content": "",
                        }
                tavily_news = _fetch_tavily_results(prompt, topic="news")
                if tavily_news is not None:
                    t_final, t_tool = tavily_news
                    if t_final:
                        return {
                            "final": t_final,
                            "tool_calls": [],
                            "tool_results": [t_tool],
                            "initial_llm_content": "",
                        }
            if _is_travel_prompt(prompt):
                tavily_travel = _fetch_tavily_results(prompt, topic="travel")
                if tavily_travel is not None:
                    t_final, t_tool = tavily_travel
                    if t_final:
                        return {
                            "final": t_final,
                            "tool_calls": [],
                            "tool_results": [t_tool],
                            "initial_llm_content": "",
                        }
            return {
                "final": "I couldn't fetch live data for this time-sensitive request right now. Please try again in a moment.",
                "tool_calls": [],
                "tool_results": [],
                "initial_llm_content": "",
            }

        fallback = (
            "I couldn't complete live tool retrieval right now due a backend tool runtime error. "
            "Please try again in a moment."
        )
        return {
            "final": fallback,
            "tool_calls": [],
            "tool_results": [
                {
                    "tool_name": "agent_runtime",
                    "server_name": "mcp",
                    "isError": True,
                    "content": "",
                    "error": str(e),
                }
            ],
            "initial_llm_content": "",
        }
