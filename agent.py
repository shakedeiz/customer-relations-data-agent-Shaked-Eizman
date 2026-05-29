import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from state import AgentState, UserProfile
from tools import tools

load_dotenv()

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
AGENT_MODEL = os.getenv("AGENT_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
PROFILE_HISTORY_USER_TURNS = int(os.getenv("PROFILE_HISTORY_USER_TURNS", "5"))

SYSTEM_PROMPT = (
    "You are a customer-relations data assistant with access to tools that query "
    "the Bitext Customer Service dataset.\n\n"
    "USER MEMORY:\n"
    "You maintain a profile of the user that is updated after every turn. "
    "When a USER PROFILE section appears below, use it to personalise your responses and "
    "to answer questions like 'what do you remember about me?' or 'what do I usually ask about?'. "
    "When no USER PROFILE section is present, it means not enough conversation history exists yet "
    "to build one — tell the user this honestly rather than claiming you have no memory system.\n\n"
    "TOOL SELECTION:\n"
    "- Use 'get_samples' for requests about examples, sample rows, or dataset evidence.\n"
    "- Use 'get_aggregate' for counts, totals, distributions, or breakdowns.\n"
    "- Use 'get_linguistic_profile' for questions about language style, tone, politeness, "
    "typos, colloquial language, or any breakdown of the 12 linguistic variation flags "
    "(e.g. 'how polite is the REFUND category?', 'what linguistic patterns appear in SHIPPING?').\n\n"
    "STOPPING RULE:\n"
    "- Do NOT call a tool with the exact same arguments more than once. "
    "If you have the data, proceed immediately to synthesising your final answer.\n"
    "- After calling a tool, check the conversation history. If a ToolMessage already "
    "contains the data you need, write your final answer in plain text immediately.\n"
    "- You may call different tools in sequence if the user's question genuinely requires "
    "both (e.g. a count AND examples), but never repeat a call you have already made."
)

PROFILE_EXTRACTION_PROMPT = (
    "Review the following conversation history. Extract any recurring topics, specific intents, "
    "or preferences the user has shown, and output them matching the provided schema.\n\n"
    "RULES:\n"
    "- Use only evidence from the provided history.\n"
    "- Keep values concise and factual.\n"
    "- If a value is unknown, keep it empty (string fields) or [] (list fields).\n"
    "- Preserve existing profile signal when the current turn does not add new evidence."
)

def _format_profile_for_prompt(profile: dict) -> str:
    """Format the stored user profile as a system-prompt section for the agent."""
    if not profile:
        return ""
    parts = []
    if profile.get("frequent_intents"):
        parts.append(f"  - Frequent topics: {', '.join(profile['frequent_intents'])}")
    if profile.get("product_area_focus"):
        parts.append(f"  - Product areas of interest: {', '.join(profile['product_area_focus'])}")
    if profile.get("communication_style"):
        parts.append(f"  - Communication style: {profile['communication_style']}")
    if profile.get("preferred_response_length"):
        parts.append(f"  - Preferred response length: {profile['preferred_response_length']}")
    if profile.get("technical_level"):
        parts.append(f"  - Technical level: {profile['technical_level']}")
    if profile.get("recent_queries"):
        parts.append(f"  - Recent queries: {'; '.join(profile['recent_queries'][-3:])}")
    if not parts:
        return ""
    return "\n\nUSER PROFILE (observed from conversation history):\n" + "\n".join(parts)


# Initialized once and reused across turns.
llm = ChatOpenAI(
    model=AGENT_MODEL,
    base_url=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)
profile_llm = llm.with_structured_output(UserProfile)


def _format_conversation_for_profile(state: AgentState) -> str:
    """Create a compact transcript from the most recent user turns."""
    transcript_messages = []
    for msg in state.get("messages", []):
        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", "")
        if not content:
            continue
        if msg_type == "human":
            transcript_messages.append(("User", content))
        elif msg_type == "ai":
            transcript_messages.append(("Assistant", content))

    if not transcript_messages:
        return ""

    user_indexes = [
        idx for idx, (speaker, _) in enumerate(transcript_messages) if speaker == "User"
    ]
    if not user_indexes:
        return ""

    if PROFILE_HISTORY_USER_TURNS <= 0:
        start_index = user_indexes[-1]
    elif len(user_indexes) <= PROFILE_HISTORY_USER_TURNS:
        start_index = 0
    else:
        start_index = user_indexes[-PROFILE_HISTORY_USER_TURNS]

    recent_messages = transcript_messages[start_index:]
    return "\n".join(f"{speaker}: {content}" for speaker, content in recent_messages)


def agent_node(state: AgentState):
    """Main reasoning node that can call bound tools."""
    from langchain_core.messages import ToolMessage

    last = state["messages"][-1] if state["messages"] else None
    has_tool_results = isinstance(last, ToolMessage)

    profile_section = _format_profile_for_prompt(state.get("user_profile") or {})
    system_content = SYSTEM_PROMPT + profile_section

    if has_tool_results:
        # Tool results are already in the conversation. Extend the system prompt
        # to remind the model to synthesise rather than re-call, but keep tools
        # available so it can still call a *different* tool if genuinely needed.
        synthesis_reminder = (
            "\n\nIMPORTANT: Tool results are now in the conversation above. "
            "First decide: do you have everything you need to answer the user? "
            "If yes, write your final answer in plain text NOW — do not call any tool. "
            "Only call another tool if the user's question requires data you have NOT yet retrieved."
        )
        system = SystemMessage(content=system_content + synthesis_reminder)
    else:
        system = SystemMessage(content=system_content)

    msg = llm_with_tools.invoke([system, *state["messages"]])
    return {"messages": [msg]}


def memory_node(state: AgentState):
    """Extract and persist a structured user profile before graph termination."""
    existing_profile = state.get("user_profile") or UserProfile().model_dump()
    transcript = _format_conversation_for_profile(state)

    if not transcript.strip():
        return {"user_profile": existing_profile}

    extraction_input = (
        f"Current profile (JSON):\n{existing_profile}\n\n"
        f"Conversation history:\n{transcript}"
    )

    try:
        updated_profile = profile_llm.invoke(
            [
                SystemMessage(content=PROFILE_EXTRACTION_PROMPT),
                HumanMessage(content=extraction_input),
            ]
        )
        return {"user_profile": updated_profile.model_dump()}
    except Exception:
        # Fail-safe: do not block the turn if extraction fails.
        return {"user_profile": existing_profile}
