from collections import Counter
from langchain_core.tools import tool
import os
import pandas as pd
from pathlib import Path
from difflib import get_close_matches
from typing import Optional, Literal
from pydantic import BaseModel, Field
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Loading Dataset (HF-only, pinned + cached)
HF_DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
HF_REVISION = "main"  
HF_CACHE_DIR = ".hf_cache"
HF_TOKEN = os.getenv("HF_TOKEN")

def _load_df() -> pd.DataFrame:
    required = ["category", "intent", "instruction", "response", "flags"]
    try:
        ds = load_dataset(
            HF_DATASET_ID,
            revision=HF_REVISION,
            cache_dir=HF_CACHE_DIR,
            token=HF_TOKEN,
        )
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split_name].to_pandas()

        # Keep required columns only; add missing ones as empty strings
        for col in required:
            if col not in df.columns:
                df[col] = ""
        return df[required]
    except Exception:
        return pd.DataFrame(columns=required)

DF = _load_df()

# --- Data Analysis Tools (Task 1c) ---

# Tool 1: get_samples
# This tool allows the agent to retrieve example instruction/response pairs from the dataset, filtered by category and/or intent. 
# This is crucial for grounding the agent's responses in actual data patterns and providing concrete examples when users ask for them.

"""
Description:
Returns up to n formatted instruction / response pairs from the dataset, optionally filtered by category and/or intent. 
If no filters are provided, samples from the full dataset. 
If an invalid category or intent is supplied, returns an informative error string listing the valid options - never raises an exception.
"""

class GetSamplesInput(BaseModel):
    # Optional filters for category and intent, plus a required 'n' for how many examples to return.
    category: Optional[str] = Field(
        None,
        description="High-level category (case-insensitive). Leave None for all categories.",
    )
    intent: Optional[str] = Field(
        None,
        description="Specific intent (case-insensitive). Leave None for all intents.",
    )
    n: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of samples to return.",
    )
    offset: int = Field(
        0,
        ge=0,
        description=(
            "Number of matching rows to skip before returning results. "
            "Use this for pagination: if you already showed n rows, "
            "set offset=n to get the next batch. Ignored when randomize=True."
        ),
    )
    randomize: bool = Field(
        False,
        description=(
            "If True, return a random selection of rows instead of sequential ones. "
            "Use this when the user asks for varied, diverse, or random examples. "
            "When True, offset is ignored."
        ),
    )


def _suggest(term: str, options: list[str]) -> str:
    # Helper function to suggest close matches for invalid category/intent inputs
    m = get_close_matches(term, options, n=1, cutoff=0.6)
    return f"\nDid you mean '{m[0]}'?" if m else ""


@tool("get_samples", args_schema=GetSamplesInput)
def get_samples(category: Optional[str] = None, intent: Optional[str] = None, n: int = 5, offset: int = 0, randomize: bool = False) -> str:
    """Return up to n instruction/response examples, optionally filtered by category and/or intent."""
    # Normalize LLM-supplied "null"/"none" strings to Python None
    if category and category.strip().lower() in ("null", "none", ""):
        category = None
    if intent and intent.strip().lower() in ("null", "none", ""):
        intent = None

    if DF.empty:
        return (
            "Error: Dataset is empty or failed to load from Hugging Face. "
            "Check HF_DATASET_ID, HF_REVISION, and your internet connection."
        )

    required = {"category", "intent", "instruction", "response", "flags"}
    missing = [c for c in required if c not in DF.columns]
    if missing:
        return f"Error: Dataset is missing required columns: {missing}"

    # Normalize with read-only series to avoid mutating/copying large DF
    cat_norm = DF["category"].astype(str).str.strip().str.upper()
    intent_norm = DF["intent"].astype(str).str.strip().str.lower()
    mask = pd.Series(True, index=DF.index)

    # Category validation/filter
    if category:
        category_in = category.strip().upper()
        valid_categories = sorted(cat_norm.dropna().unique().tolist())
        if category_in not in valid_categories:
            return (
                f"Error: '{category}' is not a valid category.\n"
                f"Valid categories are: {valid_categories}."
                f"{_suggest(category_in, valid_categories)}"
            )
        mask &= cat_norm == category_in

    # Intent validation/filter (within current filtered set)
    if intent:
        intent_in = intent.strip().lower()
        valid_intents = sorted(intent_norm[mask].dropna().unique().tolist())
        if intent_in not in valid_intents:
            return (
                f"Error: '{intent}' is not a valid intent.\n"
                f"Valid intents are: {valid_intents}."
                f"{_suggest(intent_in, valid_intents)}"
            )
        mask &= intent_norm == intent_in
    # Here I use the final boolean mask to filter the original DF and select only the instruction and response columns for output
    df = DF.loc[mask, ["instruction", "response"]]

    if df.empty:
        return "No samples found for the given filters."

    if randomize:
        rows = df.sample(min(n, len(df))).fillna("")
        label_start = 1
    else:
        rows = df.iloc[offset: offset + n].fillna("")
        label_start = offset + 1

    if rows.empty:
        return f"No more samples available after offset {offset}."
    lines = []
    for i, row in enumerate(rows.itertuples(index=False), start=label_start):
        lines.append(
            f"[{i}] INSTRUCTION: {row.instruction}\n"
            f"    RESPONSE: {row.response}"
        )
    return "\n\n".join(lines)

# Tool 2: get_aggregate
# This tool allows the agent to retrieve numeric summaries from the dataset, filtered by category and/or intent. 
# This is crucial for grounding the agent's responses in actual data patterns and providing concrete metrics when users ask for them.

"""
Description:
Returns a pre-computed numeric summary of the dataset, optionally filtered by category and/or intent. 
When aggregation_type='count', returns a single integer. When aggregation_type='distribution', returns a {intent: count} dict if a category is given, or a {category: count} dict if neither filter is given. 
Invalid parameters return an informative error string.
"""

class GetAggregateInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description="High-level category to filter on. Case-insensitive.",
    )
    intent: Optional[str] = Field(
        None,
        description="Specific intent to filter on. Case-insensitive.",
    )
    aggregation_type: Literal["count", "distribution"] = Field(
        ...,
        description=(
            "Use 'count' for total row counts. Use 'distribution' for a breakdown: "
            "returns {intent: count} if category is provided, or {category: count} "
            "across the full dataset if neither filter is given."
        ),
    )


@tool("get_aggregate", args_schema=GetAggregateInput)
def get_aggregate(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    aggregation_type: Literal["count", "distribution"] = "count",
) -> str:
    """Return a numeric summary of the dataset, optionally filtered by category and/or intent."""
    # Normalize LLM-supplied "null"/"none" strings to Python None
    if category and category.strip().lower() in ("null", "none", ""):
        category = None
    if intent and intent.strip().lower() in ("null", "none", ""):
        intent = None

    if DF.empty:
        return (
            "Error: Dataset is empty or failed to load from Hugging Face. "
            "Check HF_DATASET_ID, HF_REVISION, and your internet connection."
        )

    required = {"category", "intent"}
    missing = [c for c in required if c not in DF.columns]
    if missing:
        return f"Error: Dataset is missing required columns: {missing}"

    cat_norm = DF["category"].astype(str).str.strip().str.upper()
    intent_norm = DF["intent"].astype(str).str.strip().str.lower()
    mask = pd.Series(True, index=DF.index)

    if category:
        category_in = category.strip().upper()
        valid_categories = sorted(cat_norm.dropna().unique().tolist())
        if category_in not in valid_categories:
            return (
                f"Error: '{category}' is not a valid category.\n"
                f"Valid categories are: {valid_categories}."
                f"{_suggest(category_in, valid_categories)}"
            )
        mask &= cat_norm == category_in

    if intent:
        intent_in = intent.strip().lower()
        valid_intents = sorted(intent_norm[mask].dropna().unique().tolist())
        if intent_in not in valid_intents:
            return (
                f"Error: '{intent}' is not a valid intent.\n"
                f"Valid intents are: {valid_intents}."
                f"{_suggest(intent_in, valid_intents)}"
            )
        mask &= intent_norm == intent_in

    filtered = DF.loc[mask]

    if aggregation_type == "count":
        if category and intent:
            return f"Total records matching [category={category}, intent={intent}]: {len(filtered)}"
        if category:
            return f"Total records matching [category={category}]: {len(filtered)}"
        if intent:
            return f"Total records matching [intent={intent}]: {len(filtered)}"
        return f"Total records in dataset: {len(filtered)}"

    if intent and not category:
        return "Error: aggregation_type='distribution' requires a category when intent is provided."

    if category:
        distribution = filtered["intent"].astype(str).str.strip().str.lower().value_counts().sort_index().to_dict()
        return f"Intent distribution for category '{category}':\n{distribution}"

    distribution = filtered["category"].astype(str).str.strip().str.upper().value_counts().sort_index().to_dict()
    return f"Category distribution across the full dataset:\n{distribution}"

# Tool 3: Get Linguistic Profile
# This tool allows the agent to retrieve a linguistic profile of the dataset based on the 12 variation flags, optionally filtered by category and/or intent. 
# This is crucial for grounding the agent's responses in actual data patterns and providing insights into language
"""
Description:
Reads the flags column for a filtered subset of the dataset and returns a pre-computed percentage breakdown of all 12 language variation tags 
(e.g., Polite, Colloquial, Offensive, Typos). Takes the same optional category and intent filters as the other tools, 
and returns the same informative error string on invalid input. No sampling- it runs over the entire filtered subset for statistical accuracy.
"""
FLAG_LABELS = {
    'B': 'Basic structure',       'I': 'Interrogative',
    'C': 'Coordinated structure', 'N': 'Negation',
    'P': 'Polite register',       'Q': 'Colloquial language',
    'W': 'Offensive language',    'K': 'Keyword / shorthand',
    'M': 'Morphological variant', 'L': 'Semantic variant',
    'E': 'Abbreviations',         'Z': 'Errors and typos',
}


class GetLinguisticProfileInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description="High-level category to profile (e.g. 'REFUND'). "
                    "Leave None to profile the entire dataset."
    )
    intent: Optional[str] = Field(
        None,
        description="Specific intent to profile (e.g. 'track_refund'). "
                    "Leave None to profile all intents within the given category."
    )


@tool("get_linguistic_profile", args_schema=GetLinguisticProfileInput)
def get_linguistic_profile(category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Return a percentage breakdown of the 12 linguistic variation flags for a filtered subset of the dataset."""
    if category and category.strip().lower() in ("null", "none", ""):
        category = None
    if intent and intent.strip().lower() in ("null", "none", ""):
        intent = None

    if DF.empty:
        return (
            "Error: Dataset is empty or failed to load from Hugging Face. "
            "Check HF_DATASET_ID, HF_REVISION, and your internet connection."
        )

    if "flags" not in DF.columns:
        return "Error: Dataset is missing the 'flags' column required for linguistic profiling."

    cat_norm = DF["category"].astype(str).str.strip().str.upper()
    intent_norm = DF["intent"].astype(str).str.strip().str.lower()
    mask = pd.Series(True, index=DF.index)

    if category:
        category_in = category.strip().upper()
        valid_categories = sorted(cat_norm.dropna().unique().tolist())
        if category_in not in valid_categories:
            return (
                f"Error: '{category}' is not a valid category.\n"
                f"Valid categories are: {valid_categories}."
                f"{_suggest(category_in, valid_categories)}"
            )
        mask &= cat_norm == category_in

    if intent:
        intent_in = intent.strip().lower()
        valid_intents = sorted(intent_norm[mask].dropna().unique().tolist())
        if intent_in not in valid_intents:
            return (
                f"Error: '{intent}' is not a valid intent.\n"
                f"Valid intents are: {valid_intents}."
                f"{_suggest(intent_in, valid_intents)}"
            )
        mask &= intent_norm == intent_in

    subset = DF.loc[mask]
    counts = Counter("".join(subset["flags"].dropna().astype(str)))
    total = sum(counts.values())

    if total == 0:
        return "No linguistic flag data found for the given filters."

    lines = [
        f"{FLAG_LABELS[k]}: {v} ({v / total * 100:.1f}%)"
        for k, v in counts.most_common()
        if k in FLAG_LABELS
    ]
    header = f"Linguistic profile — {category or 'ALL'} / {intent or 'ALL INTENTS'} ({len(subset)} rows)"
    return header + "\n" + "\n".join(lines)


tools = [get_samples, get_aggregate, get_linguistic_profile]
# ...existing code...
# TODO: Define tools using @tool 

# --- Profiling Tools (Task 2b) ---
# TODO: Define @tool for update_user_profile (input: dict of interests/preferences)
# TODO: Define @tool for get_user_profile ()

# TODO: Create a list variable 'tools' containing all defined @tool functions