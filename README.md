# MCP Demo

This project demonstrates how to connect multiple MCP servers and bind their tools to a Groq LLM using Python.

## Features

- Loads environment variables from `.env`
- Connects to MCP servers from `browser_mcp.json`
- Collects available tools from each server
- Converts MCP tools into OpenAI-style function tools for Groq
- Sends a test prompt and prints model/tool-call output

## Requirements

- Python 3.12+
- A valid `GROQ_API_KEY`
- MCP servers defined in `browser_mcp.json`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Arushan45/MCP_Demo.git
   cd MCP_Demo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -U pip
   pip install langchain-groq langchain-core python-dotenv mcp
   ```

4. Create a `.env` file:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## Run

```bash
python app.py
```

## Notes

- `.env`, `.venv`, and local MCP cache folders are excluded via `.gitignore`.
- If a model name is deprecated on Groq, replace it in `app.py` with a currently supported model.
