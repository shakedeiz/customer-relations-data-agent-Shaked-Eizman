# =============================================================================
# mcp_server.py  —  FastMCP server for the Bitext Customer Service dataset
#
# Exposes three tools to any MCP client:
#   1. get_samples           – retrieve example instruction/response pairs
#   2. get_aggregate         – count or distribution summaries
#   3. get_linguistic_profile – breakdown of linguistic variation flags
#
# Dataset and helpers are imported from tools.py (single source of truth).
# Run standalone:  python mcp_server.py
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────

from collections import Counter
from typing import Literal, Optional, Union

import pandas as pd
from fastmcp import FastMCP

# Shared dataset (pre-loaded DataFrame) and helpers from tools.py.
# We call plain Python logic here — NOT the LangGraph @tool wrappers.
from tools import DF, FLAG_LABELS, _suggest

# ── Server instance ───────────────────────────────────────────────────────────

mcp = FastMCP("BitextCustomerService_DataAnalysis_Shaked")

# ── Shared helpers ────────────────────────────────────────────────────────────

_NULL_STRINGS = {"null", "none", ""}

#
def _clean(value: Optional[str]) -> Optional[str]:
    """Normalise a user-supplied string; return None for empty/null sentinels."""
    return None if (value is None or value.strip().lower() in _NULL_STRINGS) else value


def _build_mask(
    category: Optional[str],
    intent: Optional[str],
) -> Union[tuple[Optional[str], Optional[str], "pd.Series[bool]"], str]:
    """Validate and build a boolean row-mask for the given category/intent filters.

    Returns either:
      (category_in, intent_in, mask)  — on success
      str                             — an error message on invalid input
    """
    cat_norm    = DF["category"].astype(str).str.strip().str.upper()
    intent_norm = DF["intent"].astype(str).str.strip().str.lower()
    mask = pd.Series(True, index=DF.index)

    category_in = intent_in = None

    if category:
        category_in = category.strip().upper()
        valid = sorted(cat_norm.dropna().unique().tolist())
        if category_in not in valid:
            return (
                f"Error: '{category}' is not a valid category.\n"
                f"Valid categories: {valid}.{_suggest(category_in, valid)}"
            )
        mask &= cat_norm == category_in

    if intent:
        intent_in = intent.strip().lower()
        valid = sorted(intent_norm[mask].dropna().unique().tolist())
        if intent_in not in valid:
            return (
                f"Error: '{intent}' is not a valid intent.\n"
                f"Valid intents: {valid}.{_suggest(intent_in, valid)}"
            )
        mask &= intent_norm == intent_in

    return category_in, intent_in, mask


# ── Tool 1: get_samples ───────────────────────────────────────────────────────

@mcp.tool()
def get_samples(
    category:  Optional[str] = None,
    intent:    Optional[str] = None,
    n:         int  = 5,
    offset:    int  = 0,
    randomize: bool = False,
) -> str:
    """Return formatted instruction/response example pairs from the dataset.

    Args:
        category:  High-level category (e.g. 'REFUND', 'SHIPPING'). Case-insensitive.
        intent:    Granular intent (e.g. 'track_refund'). Case-insensitive.
        n:         Number of examples to return (default 5).
        offset:    Rows to skip — use for pagination. Ignored when randomize=True.
        randomize: Return a random selection instead of sequential rows.

    Returns:
        Numbered INSTRUCTION / RESPONSE pairs, or an error string on bad input.
    """
    category, intent = _clean(category), _clean(intent)

    if DF.empty:
        return "Error: Dataset failed to load. Check your HF_TOKEN and internet connection."

    result = _build_mask(category, intent)
    if isinstance(result, str):
        return result
    _, _, mask = result

    df = DF.loc[mask, ["instruction", "response"]]
    if df.empty:
        return "No samples found for the given filters."

    if randomize:
        rows, label_start = df.sample(min(n, len(df))).fillna(""), 1
    else:
        rows, label_start = df.iloc[offset: offset + n].fillna(""), offset + 1

    if rows.empty:
        return f"No more samples available after offset {offset}."

    lines = [
        f"[{i}] INSTRUCTION: {row.instruction}\n    RESPONSE: {row.response}"
        for i, row in enumerate(rows.itertuples(index=False), start=label_start)
    ]
    return "\n\n".join(lines)


# ── Tool 2: get_aggregate ─────────────────────────────────────────────────────

@mcp.tool()
def get_aggregate(
    aggregation_type: Literal["count", "distribution"],
    category: Optional[str] = None,
    intent:   Optional[str] = None,
) -> str:
    """Return a numeric summary of the dataset.

    Args:
        aggregation_type:
            'count'        — total rows matching the filters.
            'distribution' — {intent: count} when category is set;
                             {category: count} across the full dataset otherwise.
        category: High-level category (e.g. 'ACCOUNT', 'PAYMENT'). Case-insensitive.
        intent:   Granular intent (e.g. 'create_account'). Case-insensitive.

    Returns:
        A plain-text count or distribution, or an error string on bad input.
    """
    category, intent = _clean(category), _clean(intent)

    if DF.empty:
        return "Error: Dataset failed to load. Check your HF_TOKEN and internet connection."

    result = _build_mask(category, intent)
    if isinstance(result, str):
        return result
    category_in, intent_in, mask = result

    filtered = DF.loc[mask]

    if aggregation_type == "count":
        if category_in and intent_in:
            label = f"category={category_in}, intent={intent_in}"
        elif category_in:
            label = f"category={category_in}"
        elif intent_in:
            label = f"intent={intent_in}"
        else:
            return f"Total records in dataset: {len(filtered)}"
        return f"Total records matching [{label}]: {len(filtered)}"

    # aggregation_type == "distribution"
    if intent_in and not category_in:
        return "Error: aggregation_type='distribution' requires a category when intent is provided."
    if category_in:
        dist = filtered["intent"].astype(str).str.strip().str.lower().value_counts().sort_index().to_dict()
        return f"Intent distribution for category '{category_in}':\n{dist}"
    dist = filtered["category"].astype(str).str.strip().str.upper().value_counts().sort_index().to_dict()
    return f"Category distribution across the full dataset:\n{dist}"


# ── Tool 3: get_linguistic_profile ────────────────────────────────────────────

@mcp.tool()
def get_linguistic_profile(
    category: Optional[str] = None,
    intent:   Optional[str] = None,
) -> str:
    """Return a percentage breakdown of the 12 linguistic variation flags.

    Runs over the full filtered subset (no sampling) for statistical accuracy.
    Flag codes: B=Basic, I=Interrogative, C=Coordinated, N=Negation,
    P=Polite, Q=Colloquial, W=Offensive, K=Keyword, M=Morphological,
    L=Semantic, E=Abbreviations, Z=Typos.

    Args:
        category: High-level category to profile. Leave as None for all categories.
        intent:   Granular intent to profile. Leave as None for all intents.

    Returns:
        Flag labels with counts and percentages (most to least frequent),
        or an error string on bad input.
    """
    category, intent = _clean(category), _clean(intent)

    if DF.empty:
        return "Error: Dataset failed to load. Check your HF_TOKEN and internet connection."
    if "flags" not in DF.columns:
        return "Error: Dataset is missing the 'flags' column required for linguistic profiling."

    result = _build_mask(category, intent)
    if isinstance(result, str):
        return result
    category_in, intent_in, mask = result

    subset = DF.loc[mask]
    counts = Counter("".join(subset["flags"].dropna().astype(str)))
    total  = sum(counts.values())

    if total == 0:
        return "No linguistic flag data found for the given filters."

    lines = [
        f"{FLAG_LABELS[k]}: {v} ({v / total * 100:.1f}%)"
        for k, v in counts.most_common()
        if k in FLAG_LABELS
    ]
    header = (
        f"Linguistic profile — {category_in or 'ALL'} / {intent_in or 'ALL INTENTS'}"
        f"  ({len(subset)} rows)"
    )
    return header + "\n" + "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
