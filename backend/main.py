from pathlib import Path
from typing import Any, Optional

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import mcp_health_check, run_agent


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "browser_mcp.json"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="MCP Agent Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3005",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, description="User prompt for the MCP agent")
    model: Optional[str] = Field(default=None, description="Override Groq model name")


class AgentResponse(BaseModel):
    final: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    initial_llm_content: str = ""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/mcp-health")
async def mcp_health() -> dict[str, Any]:
    return await mcp_health_check(DEFAULT_MCP_CONFIG_PATH)


@app.post("/api/agent", response_model=AgentResponse)
async def agent(req: AgentRequest) -> Any:
    model = req.model or DEFAULT_MODEL
    try:
        result = await run_agent(
            req.prompt,
            model=model,
            config_path=DEFAULT_MCP_CONFIG_PATH,
        )
        return result
    except Exception as e:
        return {
            "final": "I hit a backend error while processing your request. Please try again.",
            "tool_calls": [],
            "tool_results": [
                {
                    "tool_name": "backend_exception",
                    "server_name": "api",
                    "isError": True,
                    "content": "",
                    "error": str(e),
                }
            ],
            "initial_llm_content": "",
        }

