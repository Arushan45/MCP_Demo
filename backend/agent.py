import asyncio
import json
import os
from contextlib import AsyncExitStack
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

    with open(config_path, "r", encoding="utf-8") as f:
        mcp_config = json.load(f)

    servers = mcp_config.get("mcpServers", {})
    if not servers:
        raise RuntimeError("No servers found in MCP config.")

    async with AsyncExitStack() as stack:
        openai_tools: list[dict] = []
        tools_by_name: dict[str, tuple[str, ClientSession]] = {}

        for server_name, server_info in servers.items():
            command = server_info.get("command")
            args = server_info.get("args", [])

            if os.name == "nt" and command == "npx":
                command = "npx.cmd"

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None,
            )

            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                tool_name = getattr(tool, "name", "")
                if not tool_name:
                    continue
                openai_tools.append(mcp_tool_to_openai_dict(tool))
                tools_by_name[tool_name] = (server_name, session)

        if not openai_tools:
            raise RuntimeError("No MCP tools loaded.")

        llm_with_tools = llm.bind_tools(openai_tools)

        initial = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])
        tool_calls = getattr(initial, "tool_calls", None) or []

        tool_results: list[dict[str, Any]] = []
        for call in tool_calls:
            # `call` is usually a dict like: {"name": "...", "args": {...}}
            tool_name = call.get("name")
            raw_args = call.get("args") or {}

            server_name, session = tools_by_name.get(tool_name, (None, None))
            if session is None:
                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "server_name": server_name,
                        "error": f"Tool not found in loaded tool registry: {tool_name}",
                    }
                )
                continue

            # Ensure tool args is a dict
            arguments: dict[str, Any]
            if isinstance(raw_args, dict):
                arguments = raw_args
            else:
                # Some models may pass strings; best-effort parse.
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {"_raw": raw_args}

            result = session.call_tool(tool_name, arguments)
            # call_tool is sync method but returns awaitable? In our mcp client it's async-capable.
            if asyncio.iscoroutine(result):
                result = await result

            content_text = _content_to_text(getattr(result, "content", []))
            tool_results.append(
                {
                    "tool_name": tool_name,
                    "server_name": server_name,
                    "arguments": arguments,
                    "isError": getattr(result, "isError", False),
                    "content": content_text,
                }
            )

        # If the model didn't call any tool, return its initial response.
        if not tool_calls:
            return {
                "final": getattr(initial, "content", "") or "",
                "tool_calls": [],
                "tool_results": [],
            }

        # Ask for final answer with tool outputs as context.
        tool_context = "\n".join(
            [
                f"- {tr['tool_name']} (server={tr['server_name']}), args={tr['arguments']}\n  result={tr['content']}"
                for tr in tool_results
            ]
        )
        followup = (
            f"User request:\n{prompt}\n\n"
            f"MCP tool results (use only these):\n{tool_context}\n\n"
            f"Provide the best final answer to the user."
        )

        final = await llm.ainvoke([HumanMessage(content=followup)])
        return {
            "final": getattr(final, "content", "") or "",
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "initial_llm_content": getattr(initial, "content", "") or "",
        }

