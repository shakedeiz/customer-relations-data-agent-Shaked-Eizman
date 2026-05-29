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
    # Assuming 'app' is your compiled graph
    graph_png = graph.get_graph().draw_mermaid_png()
    with open(filename, "wb") as f:
        f.write(graph_png)


# ==========================================
# main.py: Entry Point & CLI Interface
# ==========================================

# TODO: Import os and dotenv (load API keys)
# TODO: Import sqlite3 and SqliteSaver from langgraph.checkpoint.sqlite
# TODO: Import compiled graph (or builder) from graph.py

# TODO: Load .env variables
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
    # TODO: Initialize SqliteSaver checkpointer for persistent memory (Task 2a)
    conn = sqlite3.connect(str(PROJECT_ROOT / "agent_memory.sqlite"), check_same_thread=False)
    memory = SqliteSaver(conn)

    # TODO: Compile graph with checkpointer: app = builder.compile(checkpointer=memory)
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
    # TODO: Start a while True loop
    while True:
        # TODO: Get user input via input() and strip whitespace
        user_input = input("\nYou: ").strip()

        # TODO: Break loop if user types "exit" or "quit"
        if user_input.lower() in {"exit", "quit"}:
            print("You have successfully exited the Customer Relations Agent. See you next time!")
            break
        if not user_input:
            continue

        print("\nAssistant:")
        try:
            # TODO: Call app.stream() with user input and config
            stream = app.stream(
                {"messages": [("user", user_input)]},
                config=config,
            )

            # TODO: Iterate through stream events
            # TODO: Print agent reasoning (tool calls/results) clearly (Task 1d)
            for event in stream:
                _print_event(event)

        except GraphRecursionError:
            # TODO: Print final answer
            print(
                f"I could not finish this request within the step limit of {MAX_ITERATIONS} steps. "
                "Please rephrase or narrow the request and try again."
            )


# TODO: Add standard if __name__ == "__main__": main()
if __name__ == "__main__":
    main()

# TODO: Define main() function
    # TODO: Initialize SqliteSaver checkpointer for persistent memory (Task 2a)
    # TODO: Compile graph with checkpointer: app = builder.compile(checkpointer=memory)
    # TODO: Set up config with a thread_id (e.g., {"configurable": {"thread_id": "1"}})
    
    # --- Interactive CLI Loop ---
    # TODO: Start a while True loop

    # TODO: Get user input via input()
    # TODO: Break loop if user types "exit" or "quit"
    # TODO: Call app.stream() with user input and config
    # TODO: Iterate through stream events
        # TODO: Print agent reasoning (tool calls/results) clearly (Task 1d)
        # TODO: Print final answer
        
# TODO: Add standard if __name__ == "__main__": main()
visualize_graph(app, filename=GRAPH_IMAGE_FILENAME)