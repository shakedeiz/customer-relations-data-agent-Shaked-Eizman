import os
import sqlite3
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph

# Ensure project root is importable when running this file directly from venv/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from graph import builder

MAX_ITERATIONS = 12
GRAPH_IMAGE_FILENAME = "graph_visualization.png"


def visualize_graph(graph: StateGraph, filename: str = "graph.png"):
    """
    Visualizes the given StateGraph and saves it as an image file.
    
    Args:
        graph (StateGraph): The graph to visualize.
        filename (str): The name of the output image file (default is "graph.png").
    """
    graph_png = graph.get_graph().draw_mermaid_png()
    with open(filename, "wb") as f:
        f.write(graph_png)


load_dotenv()


def _print_event(event: dict):
    """Pretty-print one streamed event from LangGraph."""
    for node_name, payload in event.items():
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not messages:
            continue

        last = messages[-1]
        print(f"\n[{node_name.upper()}]")

        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            print("Tool calls:")
            for call in tool_calls:
                name = call.get("name", "unknown_tool")
                args = call.get("args", {})
                print(f"- {name} args={args}")

        content = getattr(last, "content", "")
        if content:
            print(content)


def main():
    conn = sqlite3.connect(str(PROJECT_ROOT / "agent_memory.sqlite"), check_same_thread=False)
    memory = SqliteSaver(conn)

    app = builder.compile(checkpointer=memory)
    visualize_graph(app, filename=GRAPH_IMAGE_FILENAME)

    session_id = input("Enter a session ID (or press Enter to start a new session): ").strip()
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        print(f"New session started: {session_id}")
    else:
        print(f"Resuming session: {session_id}")

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_ITERATIONS,
    }

    print("Customer Relations Agent is ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("You have successfully exited the Customer Relations Agent. See you next time!")
            break
        if not user_input:
            continue

        print("\nAssistant:")
        try:
            stream = app.stream(
                {"messages": [("user", user_input)]},
                config=config,
            )

            for event in stream:
                _print_event(event)

        except GraphRecursionError:
            print(
                f"I could not finish this request within the step limit of {MAX_ITERATIONS} steps. "
                "Please rephrase or narrow the request and try again."
            )


if __name__ == "__main__":
    main()