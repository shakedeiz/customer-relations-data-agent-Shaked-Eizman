import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from state import AgentState

load_dotenv()

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
ROUTER_HISTORY_USER_TURNS = int(os.getenv("ROUTER_HISTORY_USER_TURNS", "3"))

class RouteQuery(BaseModel):
    """Analyze the user's input and classify it accurately based on the Bitext Customer Service dataset context."""
    
    justification: str = Field(
        description="A step-by-step internal thought process justifying why this query belongs to the chosen intent type, referencing specific dataset features or boundaries."
    )
    intent: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="The final classification label: 'structured' for rigorous data lookups/metrics, 'unstructured' for open-ended summaries/text patterns, or 'out_of_scope' for irrelevant queries."
    )


# Initialize once and reuse across turns.
router_llm = ChatOpenAI(
    model=ROUTER_MODEL,
    base_url=NEBIUS_BASE_URL,
    api_key=NEBIUS_API_KEY,
    temperature=0,
)
structured_llm = router_llm.with_structured_output(RouteQuery)

RETRY_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(try\s+ag(?:a|ai)n|try\s+again|retry|again|once\s+more|one\s+more\s+time)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

PROFILE_QUERY_PATTERN = re.compile(
    r"(what do you (remember|know|recall)|remember about me|learned about me"
    r"|what do i (usually|typically|often|normally|tend to)"
    r"|what (have i|did i).{0,30}(ask|talk|discuss)"
    r"|my (profile|preferences|history|style|habits)"
    r"|what are my (interests|preferences|habits)"
    r"|show.{0,10}my profile)",
    re.IGNORECASE,
)


def _format_router_context(state: AgentState) -> str:
    """Build a compact recent transcript so the router can resolve follow-up queries."""
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

    if ROUTER_HISTORY_USER_TURNS <= 0:
        start_index = user_indexes[-1]
    elif len(user_indexes) <= ROUTER_HISTORY_USER_TURNS:
        start_index = 0
    else:
        start_index = user_indexes[-ROUTER_HISTORY_USER_TURNS]

    recent_messages = transcript_messages[start_index:]
    return "\n".join(f"{speaker}: {content}" for speaker, content in recent_messages)


def _is_retry_followup(message: str) -> bool:
    """Return True for short retry/follow-up phrases like 'try again'."""
    return bool(RETRY_FOLLOWUP_PATTERN.match(message or ""))


def _last_two_intents(state: AgentState) -> list[str]:
    """Return up to the two most recent prior router intents."""
    history = state.get("intent_history") or []
    valid = [intent for intent in history if intent in {"structured", "unstructured", "out_of_scope"}]
    return valid[-2:]


def router_node(state: AgentState):
    # Router runs on a lighter model than the main agent.
    
    # 1. Define the descriptive system prompt with dataset boundaries and few-shot examples
    router_system_prompt = SystemMessage(content="""You are an expert query router for a Customer Service Data Analyst Agent. Your single task is to classify incoming user queries based on their relationship to the 'Bitext Customer Service Dataset'.

        ABOUT THE DATASET:
        The dataset contains synthetic customer support text rows categorized by high-level 'categories' (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION) and granular 'intents' (e.g., cancel_order, track_refund, create_account). It contains text entries consisting of an 'instruction' (user request), a 'response' (example assistant answer), entity tags, and linguistic phenomenon flags (such as typos, politeness, or colloquial variations).

        CLASSIFICATION CRITERIA & RULES:
        1. 'structured': Choose this if the query asks for concrete numbers, row counts, categorical listings, metric distributions, specific subsets of examples, or precise data aggregations that require executing filtering functions on rows.
        2. 'unstructured': Choose this if the query asks for open-ended thematic analysis, conversational summaries, text pattern syntheses, or stylistic/linguistic overviews of how support rows are written.
        3. 'out_of_scope': Choose this if the query asks about real-world events, general knowledge, custom creative generation tasks (like writing poems), or any company data outside the defined customer support domains of this generic dataset.
        4. 'unstructured': Also choose this for meta-questions about the conversation itself or what the agent remembers about the user (e.g., "what do you remember about me?", "what do I usually ask about?", "show me my profile"). These are in-scope agent-awareness queries, not dataset queries.

        FOLLOW-UP RULE:
        Treat short follow-up requests (e.g., "give me 5 more", "more", "another 3", "same for REFUND", "be more specific", "give me a more detailed answer") as in-scope when recent conversation context is in-scope.
        Do NOT default follow-ups to 'structured'. Infer from context:
        - If the follow-up asks to continue/refine a thematic summary or narrative explanation, classify as 'unstructured'.
        - If the follow-up asks for counts, listings, rows, or concrete sampled records, classify as 'structured'.
        - If the follow-up asks for "concrete examples" after an unstructured summary, classify as 'structured' (it requests dataset evidence).

        TOOL SELECTION HINTS:
        - Questions about how many, total, count, breakdown, distribution, or split by category/intent should be treated as structured and are good candidates for get_aggregate.
        - Questions asking for examples, sample rows, or "show me" style evidence should be treated as structured and are good candidates for get_samples.

        FEW-SHOT EXAMPLES:

        User Input: "What categories exist in the dataset?"
        Intent: structured

        User Input: "How many refund requests did we get?"
        Intent: structured

        User Input: "What is the intent distribution for ACCOUNT?"
        Intent: structured

        User Input: "How many SHIPPING_ADDRESS rows are there?"
        Intent: structured

        User Input: "Show me 3 examples from the SHIPPING intent."
        Intent: structured

        User Input: "What is the distribution of intents in the ACCOUNT category?"
        Intent: structured

        User Input: "Summarize the FEEDBACK category."
        Intent: unstructured

        User Input: "How do customer service representatives typically respond to cancellation requests?"
        Intent: unstructured

        User Input: "Who won the 2024 Champions League?"
        Intent: out_of_scope

        User Input: "Write me a poem about customer service."
        Intent: out_of_scope

        User Input: "What do you remember about me?"
        Intent: unstructured

        User Input: "What do I usually ask about?"
        Intent: unstructured

        User Input: "Do you know anything about my preferences?"
        Intent: unstructured

        User Input (after assistant showed SHIPPING samples): "give me 5 more"
        Intent: structured

        User Input (after assistant gave a thematic FEEDBACK summary): "please be more specific"
        Intent: unstructured

        User Input (after assistant gave a thematic FEEDBACK summary): "give me a more detailed answer"
        Intent: unstructured

        User Input (after assistant gave a thematic FEEDBACK summary): "give me concrete examples for this summary"
        Intent: structured

        INSTRUCTIONS:
        First provide a clear, objective 'justification' statement outlining your analytical reasoning, then select the final matching 'intent' label.""")

    # 2. Build recent context so the router can interpret follow-up ellipsis.
    user_message = state["messages"][-1].content
    last_two_intents = _last_two_intents(state)

    # Deterministic guard: profile/memory questions are always in-scope.
    if PROFILE_QUERY_PATTERN.search(user_message or ""):
        updated_history = (state.get("intent_history") or []) + ["unstructured"]
        print("\n[Router Heuristic]: Profile/memory query detected -> routing as unstructured.")
        print("[Router Final Route]: unstructured\n")
        return {
            "classification": "unstructured",
            "intent_history": updated_history[-20:],
        }

    # Deterministic guard: retry-only follow-ups should inherit the previous
    # in-scope intent instead of being misclassified as out_of_scope.
    previous_in_scope = [i for i in reversed(last_two_intents) if i in {"structured", "unstructured"}]
    inherited_intent = previous_in_scope[0] if previous_in_scope else None
    if _is_retry_followup(user_message) and inherited_intent:
        updated_history = (state.get("intent_history") or []) + [inherited_intent]
        print("\n[Router Heuristic]: Retry follow-up detected -> inheriting previous intent.")
        print(f"[Router Final Route]: {inherited_intent}\n")
        return {
            "classification": inherited_intent,
            "intent_history": updated_history[-20:],
        }

    recent_context = _format_router_context(state)
    router_input = (
        f"Latest user query:\n{user_message}\n\n"
        f"Last two previous intents: {last_two_intents}\n\n"
        f"Recent conversation context:\n{recent_context}"
    )
    
    # 3. Combine prompt messages and invoke the structured LLM
    messages = [router_system_prompt, HumanMessage(content=router_input)]
    classification = structured_llm.invoke(messages)
    
    # Optional: print the justification to the terminal trace for easier grading evaluation
    print(f"\n[Router Justification]: {classification.justification}")
    print(f"[Router Final Route]: {classification.intent}\n")
    updated_history = (state.get("intent_history") or []) + [classification.intent]
    
    # 4. Return the classification value to populate your state dictionary
    return {
        "classification": classification.intent,
        "intent_history": updated_history[-20:],
    }