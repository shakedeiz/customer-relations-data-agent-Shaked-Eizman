# Bitext Customer Service Data Analyst Agent

**Shaked Eizman**

## Project Overview

This project is a **Customer Service Data Analyst Agent** built with [LangGraph](https://github.com/langchain-ai/langgraph) using a **ReAct (Reason + Act)** architecture.

The agent answers questions about the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — a synthetic dataset of ~27 000 instruction/response pairs across 11 categories and ~77 intents.

Key features:
- **ReAct loop**: the agent reasons about what data it needs, calls tools to retrieve it, and synthesises a final answer.
- **Query router**: a lightweight LLM classifies every incoming message as `structured`, `unstructured`, or `out_of_scope` before the agent sees it.
- **Three data tools**: `get_samples`, `get_aggregate`, `get_linguistic_profile`.
- **Persistent memory**: conversation history is stored in a local SQLite database and survives restarts.
- **MCP server**: the same three tools are exposed via FastMCP so any MCP-compatible client can call them independently of the chat interface.
- **Streamlit UI**: a notebook-style web interface for interactive sessions.

## Architecture & Model Selection

Both models are served via the [Nebius Token Factory](https://nebius.com/studio).

| Role | Model | Why |
|------|-------|-----|
| Primary agent | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Flagship instruct model fine-tuned for tool use; avoids the repeated-tool-call bug seen in Llama-3.3-70B |
| Query router | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 3B active-parameter MoE; ~70 tok/s at negligible cost; classifies queries before the heavy model is invoked |

### Model Selection Details

For this project, I utilize a two-tier model approach, leveraging the Nebius Token Factory:

* **Primary Agent (`Qwen/Qwen3-235B-A22B-Instruct-2507`):** Drives the ReAct loop. I initially used `meta-llama/Llama-3.3-70B-Instruct`, but it exhibited a systematic tool-calling loop — re-issuing identical tool calls after already receiving valid results. I switched to Qwen3-235B-A22B-Instruct, Nebius's flagship instruct model explicitly fine-tuned for tool use. As an additional safeguard, `agent_node` dynamically appends a synthesis reminder to the system prompt whenever a `ToolMessage` is present in state, nudging the model to answer rather than call again.

* **Query Router (`Qwen3-30B-A3B-Instruct-2507`):** I implemented a lightweight routing node to act as a gatekeeper for all incoming user queries. By utilizing this highly optimized Instruct model (with a highly efficient 3B active parameter MoE architecture), I minimize latency (achieving ~70 Tok/s) and keep classification costs negligible. This ensures that the heavier, more expensive 70B+ model is only engaged for complex data synthesis queries it is specifically needed to resolve.

**Justification Summary:** This two-tier strategy optimizes for both cost and efficiency. By offloading classification to a specialized router, I improve the overall responsiveness of the product, while reserving the 70B model's superior reasoning for the complex logic required during data analysis.

### Router Logic

- The router classifies each turn as `structured`, `unstructured`, or `out_of_scope`.
- Classification uses both the latest user query and recent conversation context (not just the last message).
- Recent context window is configurable via `ROUTER_HISTORY_USER_TURNS` (default: `3` user turns).
- Follow-up messages are context-aware:
  - Refinement of summaries/explanations is routed as `unstructured`.
  - Requests for counts/rows/examples are routed as `structured`.
  - Requests like "give concrete examples for this summary" are routed as `structured`.
- Deterministic retry guard: short retry prompts (`"try again"`, `"retry"`, including common typo `"try agian"`) inherit the latest in-scope intent to avoid false `out_of_scope`.
- The router stores a rolling `intent_history` and consults at least the last two previous intents for robust follow-up handling.

## Setup Instructions

**1. Clone the repository**

```bash
git clone <repo-url>
cd customer-relations-data-agent-Shaked-Eizman
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Create a `.env` file in the project root**

```env
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1/
HF_TOKEN=your_huggingface_token_here
```

- `NEBIUS_API_KEY`: obtain from the [Nebius Studio console](https://studio.nebius.ai/).
- `HF_TOKEN`: obtain from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Required to download the Bitext dataset on first run.
- The dataset is cached locally in `.hf_cache/` after the first download.

**Optional environment variables** (add to `.env` to override defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Model used by the main agent |
| `ROUTER_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Model used by the query router |
| `ROUTER_HISTORY_USER_TURNS` | `3` | Number of prior user turns the router sees for follow-up context |
| `PROFILE_HISTORY_USER_TURNS` | `5` | Number of prior user turns the memory node uses for profile extraction |

## Running the Agent

### CLI

```bash
.\venv\Scripts\python.exe main.py   # Windows
python main.py                       # Mac/Linux
```

On startup the agent will prompt for a session ID:

```
Enter a session ID (or press Enter to start a new session):
```

- **Type a name** (e.g. `user_1`) to resume a previous conversation — the full message history is loaded from `agent_memory.sqlite`.
- **Press Enter** to generate a random 8-character ID and start a fresh session.

**Example — testing persistence:**
1. Run → enter `user_1` → ask *"Show me 3 examples from the REFUND category"* → type `quit`
2. Run again → enter `user_1` → ask *"Show me 3 more"*
3. The agent loads the prior context from SQLite and returns examples 4–6.

### Streamlit UI

```bash
.\venv\Scripts\streamlit run streamlit_app.py   # Windows
streamlit run streamlit_app.py                   # Mac/Linux
```

Open `http://localhost:8501` in your browser. The UI provides a notebook-style interface with numbered turns, a session sidebar for resuming past conversations, and a **Show Profile** button to inspect the current user profile.

## Starting the MCP Server

`mcp_server.py` exposes `get_samples`, `get_aggregate`, and `get_linguistic_profile` as MCP tools via [FastMCP](https://github.com/jlowin/fastmcp).

**Start the server:**

```bash
.\venv\Scripts\python.exe mcp_server.py   # Windows
python mcp_server.py                       # Mac / Linux
```

Expected output:
```
INFO  Starting MCP server 'BitextCustomerService_DataAnalysis_Shaked' with transport 'stdio'
```

### Connecting a client

Any MCP client that supports **stdio transport** can connect. Point it at the venv Python executable and the path to `mcp_server.py`. For clients that use a JSON config (e.g. Claude Desktop), add this entry:

```json
{
  "mcpServers": {
    "BitextCustomerService_DataAnalysis_Shaked": {
      "command": "C:\\Users\\Shakede\\Downloads\\SelfDev\\AI Performance Engineer\\From AI model to AI product-1\\Assignment 3\\customer-relations-data-agent-Shaked-Eizman\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\Shakede\\Downloads\\SelfDev\\AI Performance Engineer\\From AI model to AI product-1\\Assignment 3\\customer-relations-data-agent-Shaked-Eizman\\mcp_server.py"
      ]
    }
  }
}
```

> **Note:** Update `command` and the path in `args` to match your local installation before using.

Save the config and restart the client. The three tools will appear and can be called directly.

**To test without a full client** — use [MCP Inspector](https://github.com/modelcontextprotocol/inspector) (requires Node.js):

```bash
npx @modelcontextprotocol/inspector "...\venv\Scripts\python.exe" "...\mcp_server.py"
```

Open `http://localhost:5173`, click **Connect**, select `get_aggregate`, and call it with:

```json
{ "aggregation_type": "count", "category": "REFUND" }
```

The server returns the row count for that category.

## Defined Tools

### `get_samples`
Returns up to `n` formatted instruction/response pairs from the dataset, optionally filtered by `category` and/or `intent`. Supports sequential pagination via `offset` and random sampling via `randomize`. Use this for requests like "show me examples" or "give me 5 rows from REFUND".

### `get_aggregate`
Returns a numeric summary of the dataset filtered by `category` and/or `intent`. Supports two modes: `count` (total matching rows) and `distribution` (breakdown by intent within a category, or by category across the full dataset). Use this for "how many", "total", or "what is the distribution of" questions.

### `get_linguistic_profile`
Returns a percentage breakdown of all 12 linguistic variation flags across the filtered subset of the dataset. Runs over the entire filtered subset for statistical accuracy (no sampling). Filters accept the same optional `category` and `intent` as the other tools.

The 12 flags are:

| Flag | Meaning |
|------|---------|
| B | Basic structure |
| I | Interrogative |
| C | Coordinated structure |
| N | Negation |
| P | Polite register |
| Q | Colloquial language |
| W | Offensive language |
| K | Keyword / shorthand |
| M | Morphological variant |
| L | Semantic variant |
| E | Abbreviations |
| Z | Errors and typos |

Use this for questions like "how polite is the REFUND category?", "what linguistic patterns appear in SHIPPING?", or "what percentage of ACCOUNT rows contain typos?".
