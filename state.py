from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Persistent profile built up from the user's conversation history."""
    frequent_intents: List[str] = Field(
        default_factory=list,
        description="Recurring goals or intents observed across the user's queries.",
    )
    recent_queries: List[str] = Field(
        default_factory=list,
        description="The most recent raw queries submitted by the user.",
    )
    communication_style: str = Field(
        default="",
        description="Inferred tone/style of the user (e.g. 'formal', 'casual', 'technical').",
    )
    preferred_response_length: str = Field(
        default="",
        description="Preferred response length (e.g. 'short' or 'detailed').",
    )
    technical_level: str = Field(
        default="",
        description="Estimated technical proficiency (e.g. 'beginner', 'intermediate', 'advanced').",
    )
    product_area_focus: List[str] = Field(
        default_factory=list,
        description="Product areas/features the user asks about most often.",
    )


class AgentState(TypedDict):
    # This gives me the flexibility to add more fields to the state, unlike MessagesState
    messages: Annotated[List[BaseMessage], add_messages]
    
    # I add this field to store my router's classification.
    # This allows downstream nodes (like the agent or a refusal node)
    # to know if the query was deemed 'out_of_scope' by the router.
    classification: str

    # Rolling history of router decisions across turns.
    intent_history: List[str]

    # Persistent user profile populated by the profiling node.
    user_profile: dict  # serialised representation of a UserProfile instance
