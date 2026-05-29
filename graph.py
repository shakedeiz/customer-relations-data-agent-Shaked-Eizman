from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent import agent_node, memory_node
from router import router_node
from state import AgentState
from tools import tools

# ---- Graph Definition ----

# Node: Refusal (handles out-of-scope queries)
def refusal_node(state: AgentState):
    return {
        "messages": [
            AIMessage(content="Sorry, this request is out of scope for this assistant.")
        ]
    }


def route_after_router(state: AgentState):
    intent = state["classification"]
    if intent == "out_of_scope":
        return "refusal"
    return "agent"


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# Exporting uncompiled builder for main.py (checkpointer can be attached there).
builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("refusal", refusal_node)
builder.add_node("memory", memory_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_after_router, {"refusal": "refusal", "agent": "agent"})
builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: "memory"})
builder.add_edge("tools", "agent")
builder.add_edge("refusal", "memory")
builder.add_edge("memory", END)

# Alias kept for compatibility with existing imports that expect `graph`.
graph = builder