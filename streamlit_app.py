import sqlite3
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph import builder

MAX_ITERATIONS = 12
DB_PATH = PROJECT_ROOT / "agent_memory.sqlite"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Relations Agent",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate Teal Notebook CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #F0F4F8; }

/* Notebook cell wrapper */
.nb-cell {
    background: #FFFFFF;
    border-radius: 8px;
    border-left: 5px solid #0694A2;
    padding: 18px 22px 14px 22px;
    margin-bottom: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.nb-turn-label {
    font-size: 10px;
    font-weight: 700;
    color: #0694A2;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.nb-question {
    font-size: 16px;
    font-weight: 600;
    color: #1E3A5F;
    margin: 0 0 10px 0;
}
.nb-divider {
    border: none;
    border-top: 1px solid #E2EAF0;
    margin: 10px 0;
}
.nb-answer {
    color: #1A1A2E;
    line-height: 1.7;
    font-size: 15px;
}
.nb-reasoning-step {
    background: #F7FAFC;
    border-left: 3px solid #CBD5E0;
    padding: 6px 10px;
    margin: 4px 0;
    font-size: 13px;
    color: #6B7280;
    border-radius: 0 4px 4px 0;
    font-family: monospace;
}
.nb-empty {
    color: #9CA3AF;
    font-style: italic;
    font-size: 14px;
    padding: 20px 0;
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E3A5F !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #D1E3F0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] input {
    background-color: #2A4A70 !important;
    color: #FFFFFF !important;
    border: 1px solid #0694A2 !important;
}
section[data-testid="stSidebar"] input::placeholder {
    color: #A0BCD4 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: #0694A2 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #047A8A !important;
}
section[data-testid="stSidebar"] hr { border-color: #2A4A70; }

/* Profile JSON viewer in sidebar — light background, dark text */
section[data-testid="stSidebar"] .stJson,
section[data-testid="stSidebar"] .stJson * {
    color: #1A1A2E !important;
    background-color: #EEF4FA !important;
}

/* Page title */
h1 { color: #1E3A5F !important; border-bottom: 3px solid #0694A2; padding-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Compiled graph (cached across reruns) ─────────────────────────────────────
@st.cache_resource
def get_app():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


app = get_app()


# ── Helper: rebuild display history from checkpointer ─────────────────────────
def _msgs_to_display(messages: list) -> list:
    display = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content, "steps": []})
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            display.append({"role": "assistant", "content": msg.content, "steps": []})
    return display


# ── Helper: render one notebook cell ──────────────────────────────────────────
def render_cell(turn: int, question: str, steps: list[str], answer: str, live: bool = False):
    st.markdown(f"""
    <div class="nb-cell">
        <div class="nb-turn-label">Turn {turn}</div>
        <div class="nb-question">Q: {question}</div>
        <hr class="nb-divider">
    </div>
    """, unsafe_allow_html=True)

    if steps:
        with st.expander("Reasoning steps", expanded=live):
            for step in steps:
                st.markdown(
                    f'<div class="nb-reasoning-step">{step}</div>',
                    unsafe_allow_html=True,
                )

    if answer:
        st.markdown(
            f'<div style="background:#FFFFFF; border-radius:6px; padding:14px 20px; '
            f'margin-top:8px; border-left:5px solid #0694A2; box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
            f'<span style="font-size:10px;font-weight:700;color:#0694A2;'
            f'text-transform:uppercase;letter-spacing:0.12em;">Answer</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(answer)


# ── Session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "steps"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Session")
    st.markdown(f"**Active:** `{st.session_state.session_id}`")
    st.divider()

    raw_sid = st.text_input("Session ID", placeholder="Leave blank for new session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Resume", use_container_width=True):
            sid = raw_sid.strip() or str(uuid.uuid4())[:8]
            st.session_state.session_id = sid
            config = {"configurable": {"thread_id": sid}, "recursion_limit": MAX_ITERATIONS}
            try:
                state = app.get_state(config)
                prior = state.values.get("messages", []) if state.values else []
            except Exception:
                prior = []
            st.session_state.chat_history = _msgs_to_display(prior)
            st.rerun()
    with col2:
        if st.button("New", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("**Tips**")
    st.markdown("- Ask for counts, distributions, or examples")
    st.markdown("- Type a session ID and click **Resume** to reload a past conversation")
    st.markdown("- Each turn is numbered so you can reference earlier results")

    st.divider()
    st.markdown("**User Profile**")
    if st.button("Show Profile", use_container_width=True):
        try:
            _config = {"configurable": {"thread_id": st.session_state.session_id}, "recursion_limit": MAX_ITERATIONS}
            _state = app.get_state(_config)
            profile = (_state.values or {}).get("user_profile") or {}
        except Exception:
            profile = {}
        if profile and any(profile.values()):
            st.json(profile)
        else:
            st.caption("No profile data yet — ask a few questions first.")


# ── Page title ────────────────────────────────────────────────────────────────
st.title("Customer Relations Data Agent")
st.caption(f"Session `{st.session_state.session_id}` · Bitext Customer Service Dataset")

# ── Render existing notebook cells ───────────────────────────────────────────
history = st.session_state.chat_history
turns = []
i = 0
while i < len(history):
    if history[i]["role"] == "user":
        question = history[i]["content"]
        answer = ""
        steps = []
        if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
            answer = history[i + 1]["content"]
            steps = history[i + 1].get("steps", [])
            i += 2
        else:
            i += 1
        turns.append((question, steps, answer))
    else:
        i += 1

if not turns:
    st.markdown('<div class="nb-empty">No turns yet — ask a question below.</div>', unsafe_allow_html=True)

for idx, (question, steps, answer) in enumerate(turns, start=1):
    render_cell(idx, question, steps, answer, live=False)


# ── Config for this session ───────────────────────────────────────────────────
config = {
    "configurable": {"thread_id": st.session_state.session_id},
    "recursion_limit": MAX_ITERATIONS,
}

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about the Bitext Customer Service dataset…")

if user_input:
    turn_number = len(turns) + 1
    st.session_state.chat_history.append({"role": "user", "content": user_input, "steps": []})

    steps: list[str] = []
    final_content = ""

    # Render live cell header
    st.markdown(f"""
    <div class="nb-cell">
        <div class="nb-turn-label">Turn {turn_number}</div>
        <div class="nb-question">Q: {user_input}</div>
        <hr class="nb-divider">
    </div>
    """, unsafe_allow_html=True)

    with st.status("Thinking…", expanded=True) as status:
        try:
            for event in app.stream(
                {"messages": [("user", user_input)]},
                config=config,
            ):
                for node_name, payload in event.items():
                    if not isinstance(payload, dict):
                        continue

                    if node_name == "router":
                        route = payload.get("classification", "")
                        if route:
                            line = f"Router → <code>{route}</code>"
                            st.markdown(
                                f'<div class="nb-reasoning-step">{line}</div>',
                                unsafe_allow_html=True,
                            )
                            steps.append(f"Router → `{route}`")
                        continue

                    msgs = payload.get("messages", [])
                    if not msgs:
                        continue
                    last = msgs[-1]

                    if node_name == "agent":
                        tool_calls = getattr(last, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                line = f"<b>{tc['name']}</b> — <code>{tc['args']}</code>"
                                st.markdown(
                                    f'<div class="nb-reasoning-step">{line}</div>',
                                    unsafe_allow_html=True,
                                )
                                steps.append(f"`{tc['name']}` args=`{tc['args']}`")
                        elif getattr(last, "content", ""):
                            final_content = last.content

                    elif node_name == "tools":
                        content = getattr(last, "content", "")
                        if content:
                            st.markdown(f"**Result:**")
                            st.markdown(content)
                            steps.append(f"Result: {content}")

                    elif node_name == "refusal":
                        final_content = getattr(last, "content", "")

            status.update(label="Done ✓", state="complete", expanded=False)

        except GraphRecursionError:
            final_content = (
                f"Could not complete within the {MAX_ITERATIONS}-step limit. "
                "Please rephrase or narrow your request."
            )
            status.update(label="Step limit reached", state="error", expanded=False)

    # Answer block
    st.markdown(
        '<div style="background:#FFFFFF; border-radius:6px; padding:14px 20px; '
        'margin-top:8px; border-left:5px solid #0694A2; box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
        '<span style="font-size:10px;font-weight:700;color:#0694A2;'
        'text-transform:uppercase;letter-spacing:0.12em;">Answer</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(final_content)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": final_content,
        "steps": steps,
    })
