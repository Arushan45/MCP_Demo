import asyncio
import json
import os
from contextlib import AsyncExitStack
from dotenv import load_dotenv

load_dotenv()
# Standard MCP SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# LangChain imports - Swapped to Groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def _relax_numeric_types(schema):
    """Allow LLMs to pass numbers as strings (Groq tool validation is strict)."""
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


def mcp_tool_to_openai_dict(tool) -> dict:
    """
    Convert an MCP tool object into an OpenAI-style tool dict
    that langchain_groq.ChatGroq.bind_tools() can understand.
    """
    # Fallbacks for missing description/schema
    description = getattr(tool, "description", None) or getattr(
        getattr(tool, "annotations", None) or {}, "title", None
    ) or ""

    input_schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    input_schema = _relax_numeric_types(input_schema)

    # Wrap as an OpenAI "tool" with a "function" definition
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", "unnamed_tool"),
            "description": description,
            "parameters": input_schema,
        },
    }


async def main():
    # 1. Ensure you have your Groq API key set
    if not os.environ.get("GROQ_API_KEY"):
        print("Please set your GROQ_API_KEY environment variable.")
        return

    # Initialize the Groq LLM
    # You can change the model to mixtral-8x7b-32768 or others supported by Groq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # 2. Load your MCP server configuration
    config_path = "browser_mcp.json"
    try:
        with open(config_path, "r") as f:
            mcp_config = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {config_path}. Make sure it is in the same directory.")
        return

    servers = mcp_config.get("mcpServers", {})
    if not servers:
        print("No servers found in the configuration.")
        return

    # 3. Connect to the servers and gather tools
    async with AsyncExitStack() as stack:
        all_tools = []
        
        print("Connecting to MCP servers...")
        for server_name, server_info in servers.items():
            command = server_info.get("command")
            args = server_info.get("args", [])
            
            # Windows tip: npx sometimes needs to be explicitly called as npx.cmd 
            if os.name == 'nt' and command == 'npx':
                command = 'npx.cmd'

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None
            )
            
            try:
                # Connect to the stdio stream
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                
                # Initialize the session
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                # Fetch available tools from this specific server
                tools_response = await session.list_tools()
                print(f"[OK] Connected to {server_name}. Found {len(tools_response.tools)} tools.")
                
                # Add them to our master list
                all_tools.extend(tools_response.tools)
                
            except Exception as e:
                print(f"[ERR] Failed to connect to {server_name}: {e}")

        if not all_tools:
            print("No tools could be loaded. Exiting.")
            return

        print(f"\nTotal tools loaded: {len(all_tools)}")
        print("Binding tools to the Groq LLM...")

        # 4. Convert MCP tools into OpenAI-style tool dicts, then bind
        openai_tools = [mcp_tool_to_openai_dict(t) for t in all_tools]
        llm_with_tools = llm.bind_tools(openai_tools)

        # 5. Test the Agent
        test_prompt = "Search DuckDuckGo for the top 3 things to do in Tokyo, and check Airbnb for places to stay."
        print(f"\nSending test prompt to LLM: '{test_prompt}'")
        
        response = await llm_with_tools.ainvoke([HumanMessage(content=test_prompt)])
        
        print("\n=== LLM Response ===")
        print(response.content)
        if response.tool_calls:
            print("\nThe LLM decided to use the following tools:")
            for tool_call in response.tool_calls:
                print(f"- {tool_call['name']} with arguments: {tool_call['args']}")

if __name__ == "__main__":
    asyncio.run(main())