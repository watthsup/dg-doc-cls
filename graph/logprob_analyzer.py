"""Logprob analysis — the Reliability Engine.

Extracts natural probability distributions from LLM response metadata
and calculates the Margin score between Top 1 and Top 2 predictions.

The Margin Formula:
    Margin = ln(P_Top1) - ln(P_Top2)

Decision Thresholds:
    - Margin >= 1.5: Safe path (auto-proceed)
    - Margin <  1.5: Ambiguous path (trigger HITL interrupt)
"""

from __future__ import annotations

import math

from schemas.models import LogprobAnalysis


def analyze_logprobs(
    response_metadata: dict,
    valid_tokens: list[str],
) -> LogprobAnalysis:
    """Extract margin score from natural logprob distribution.

    Filters logprobs to only consider valid classification tokens,
    then calculates the separation between Top 1 and Top 2.

    Args:
        response_metadata: Raw response_metadata from LangChain AIMessage.
        valid_tokens: List of valid 3-letter codes to consider (e.g. ["MED", "NON"]).

    Returns:
        LogprobAnalysis with margin score and confidence percentage.
    """
    logprobs_data = response_metadata.get("logprobs", {})
    content_logprobs = logprobs_data.get("content", [])

    if not content_logprobs:
        # Fallback: no logprobs available — return zero-confidence
        return LogprobAnalysis(
            top1_token="UNK",
            top1_logprob=-100.0,
            top2_token="UNK",
            top2_logprob=-100.0,
            margin_score=0.0,
            confidence_pct=0.0,
        )

    # Find the first token that actually contains text (skip leading spaces/newlines)
    first_token_data = None
    for token_data in content_logprobs:
        if token_data.get("token", "").strip():
            first_token_data = token_data
            break
            
    if not first_token_data:
        first_token_data = content_logprobs[0]  # Fallback
        
    top_logprobs = first_token_data.get("top_logprobs", [])

    # Filter to only valid classification tokens
    valid_entries = [
        entry for entry in top_logprobs
        if entry.get("token", "").strip().upper() in {t.upper() for t in valid_tokens}
    ]

    # Also consider the generated token itself
    generated_token = first_token_data.get("token", "").strip().upper()
    generated_logprob = first_token_data.get("logprob", -100.0)

    if generated_token in {t.upper() for t in valid_tokens}:
        # Check if already in valid_entries to avoid duplicates
        if not any(e.get("token", "").strip().upper() == generated_token for e in valid_entries):
            valid_entries.insert(0, {
                "token": generated_token,
                "logprob": generated_logprob,
            })

    # Sort by logprob descending (highest probability first)
    valid_entries.sort(key=lambda x: x.get("logprob", -100.0), reverse=True)

    if not valid_entries:
        return LogprobAnalysis(
            top1_token=generated_token or "UNK",
            top1_logprob=generated_logprob,
            top2_token="UNK",
            top2_logprob=-100.0,
            margin_score=abs(generated_logprob) if generated_logprob != -100.0 else 0.0,
            confidence_pct=min(100.0, math.exp(generated_logprob) * 100) if generated_logprob > -50 else 0.0,
            candidates=top_logprobs,
        )

    top1 = valid_entries[0]
    top1_token = top1.get("token", "UNK").strip().upper()
    top1_logprob = top1.get("logprob", -100.0)

    if len(valid_entries) >= 2:
        top2 = valid_entries[1]
        top2_token = top2.get("token", "UNK").strip().upper()
        top2_logprob = top2.get("logprob", -100.0)
    else:
        # Only one valid token — maximum confidence
        top2_token = "N/A"
        top2_logprob = -100.0

    margin = top1_logprob - top2_logprob
    confidence_pct = min(100.0, math.exp(top1_logprob) * 100)

    return LogprobAnalysis(
        top1_token=top1_token,
        top1_logprob=top1_logprob,
        top2_token=top2_token,
        top2_logprob=top2_logprob,
        margin_score=margin,
        confidence_pct=confidence_pct,
        candidates=top_logprobs,
    )
