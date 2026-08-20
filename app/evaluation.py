"""
Response Evaluation Module: heuristic quality checks for LLM-generated replies.

IMPORTANT — Methodology Disclosure:
    These are deterministic heuristic checks, NOT a scientifically validated
    hallucination detector or a semantic similarity model.

    What we DO:
      - Check response length (completeness proxy)
      - Check term overlap between ticket and response (relevance proxy)
      - Check for known unsafe patterns (safety heuristic)
      - Estimate groundedness using keyword overlap with retrieved KB chunks
      - Detect missing sign-off and missing next-step signals

    What we DON'T claim:
      - These checks do NOT guarantee factual accuracy.
      - Term overlap is a lexical proxy for relevance, not semantic understanding.
      - Groundedness score measures keyword co-occurrence, not true grounding.
      - Safety check uses regex patterns and cannot detect all unsafe content.

    Grounding score methodology:
      We compute the fraction of top-K RAG chunk keywords (content words > 4 chars)
      that appear in the LLM response. A high score means the response uses
      vocabulary from the retrieved context. A low score means the response may
      be using information not in the retrieved context (higher hallucination risk).
      This is a proxy metric — treat it as a signal, not a guarantee.

Design:
    All checks are deterministic and fast (<1ms). They do not call external
    APIs. The evaluation runs on every generated response before it is returned
    to the client.
"""

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration thresholds (tunable without code changes)
# ---------------------------------------------------------------------------

MIN_RESPONSE_LENGTH = 80          # characters — shorter replies are flagged incomplete
MAX_RESPONSE_LENGTH = 2000        # characters — extremely long replies are flagged
MIN_WORD_COUNT = 15               # words — below this is too short to be useful
RELEVANCE_OVERLAP_THRESHOLD = 0.05  # fraction of ticket terms in response
GROUNDING_OVERLAP_THRESHOLD = 0.08  # fraction of KB chunk terms in response
QUALITY_SCORE_THRESHOLD = 0.55    # below this -> needs_regeneration = True

# Safety patterns: strings that should NOT appear in a customer support reply
UNSAFE_PATTERNS = [
    r"\b\d{10,}\b",                  # Very long number sequences (fake phone/account numbers)
    r"http[s]?://(?!platform\.com|docs\.platform\.com|status\.platform\.com)",  # External links
    r"\b(?:your account number is|your password is)\b",  # Credential disclosure
    r"\bI cannot help with that\b",  # Unhelpful non-answer without escalation
]

# Signals that indicate a proper next step is present
NEXT_STEP_SIGNALS = [
    "please", "contact", "visit", "go to", "check", "try", "follow",
    "click", "navigate", "email", "reach", "submit", "update", "let us",
    "our team", "we will", "we can", "escalate", "investigate",
]

# Sign-off signals
SIGNOFF_SIGNALS = ["support team", "regards", "sincerely", "best,", "warm regards"]

# Common stop words for TF-IDF-style filtering (we want content words)
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that",
    "these", "those", "it", "its", "we", "our", "you", "your", "they",
    "their", "i", "my", "me", "us", "not", "can", "if", "as", "so",
    "about", "up", "out", "what", "which", "who", "when", "where", "how",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Heuristic evaluation result for a single LLM response.

    Fields:
        quality_score:        Weighted composite score [0.0, 1.0].
                              Higher is better. Not a scientifically validated metric.
        relevant:             True if ticket terms appear in response (lexical overlap heuristic).
        grounded:             True if response uses KB chunk vocabulary (groundedness heuristic).
        safe:                 True if no unsafe output patterns detected.
        complete:             True if response meets minimum length and includes a next step.
        has_signoff:          True if response ends with a recognizable sign-off.
        needs_regeneration:   True if quality_score < threshold.
        failure_reason:       Human-readable explanation if needs_regeneration is True.
        response_length:      Character count of the response.
        word_count:           Word count of the response.
        relevance_score:      Raw term-overlap ratio (0.0-1.0). Lexical proxy only.
        groundedness_score:   Raw KB-term overlap ratio (0.0-1.0). Heuristic proxy only.
        rag_chunks_available: Number of KB chunks available for grounding check.
    """
    quality_score: float = 0.0
    relevant: bool = False
    grounded: bool = False
    safe: bool = True
    complete: bool = False
    has_signoff: bool = False
    needs_regeneration: bool = True
    failure_reason: Optional[str] = None
    response_length: int = 0
    word_count: int = 0
    relevance_score: float = 0.0
    groundedness_score: float = 0.0
    rag_chunks_available: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_content_words(text: str) -> set[str]:
    """Extract lowercase content words (length >= 4, not stop words)."""
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _check_relevance(ticket_text: str, response: str) -> tuple[bool, float]:
    """
    Lexical relevance check: fraction of ticket content words appearing in response.

    This is a surface-form lexical check, not semantic similarity.
    A high score means the response uses similar vocabulary to the ticket.
    """
    ticket_words = _extract_content_words(ticket_text)
    if not ticket_words:
        return True, 1.0  # Cannot compute — assume relevant

    response_lower = response.lower()
    matched = sum(1 for w in ticket_words if w in response_lower)
    score = matched / len(ticket_words)
    return score >= RELEVANCE_OVERLAP_THRESHOLD, round(score, 4)


def _check_groundedness(
    response: str,
    kb_chunk_contents: list[str],
) -> tuple[bool, float]:
    """
    Heuristic groundedness check: fraction of KB content words in response.

    Methodology:
        1. Extract content words (len >= 4, non-stop-word) from all KB chunks.
        2. Count how many appear in the response.
        3. Score = matched / total_kb_content_words.

    This measures vocabulary overlap, not factual correctness.
    A score of 0.0 with no KB chunks means groundedness cannot be assessed.

    Limitations:
        - Does not detect semantic paraphrasing.
        - A response can score high by using common support vocabulary
          that happens to appear in KB chunks.
        - Does not verify that facts (numbers, dates, policies) are correct.
    """
    if not kb_chunk_contents:
        # No context available — cannot assess groundedness
        return True, 0.0  # Pass by default, score indicates no KB

    all_kb_words = set()
    for chunk_content in kb_chunk_contents:
        all_kb_words.update(_extract_content_words(chunk_content))

    if not all_kb_words:
        return True, 0.0

    response_lower = response.lower()
    matched = sum(1 for w in all_kb_words if w in response_lower)
    score = matched / len(all_kb_words)
    return score >= GROUNDING_OVERLAP_THRESHOLD, round(score, 4)


def _check_safety(response: str) -> tuple[bool, Optional[str]]:
    """
    Safety heuristic: check for known unsafe output patterns.

    Returns (is_safe, failure_reason).

    Limitations:
        This is a simple regex-based check. It catches known unsafe
        patterns but cannot detect all harmful content. It should not
        be treated as a comprehensive safety filter.
    """
    response_lower = response.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, response_lower):
            return False, f"Unsafe pattern detected: {pattern}"
    return True, None


def _check_completeness(response: str) -> tuple[bool, Optional[str]]:
    """
    Completeness check: length, word count, and next-step presence.

    Returns (is_complete, failure_reason).
    """
    length = len(response)
    words = len(response.split())

    if length < MIN_RESPONSE_LENGTH:
        return False, f"Response too short ({length} chars, minimum {MIN_RESPONSE_LENGTH})"

    if words < MIN_WORD_COUNT:
        return False, f"Response too brief ({words} words, minimum {MIN_WORD_COUNT})"

    if length > MAX_RESPONSE_LENGTH:
        return False, f"Response too long ({length} chars, maximum {MAX_RESPONSE_LENGTH})"

    # Check for actionable next step
    response_lower = response.lower()
    has_next_step = any(signal in response_lower for signal in NEXT_STEP_SIGNALS)
    if not has_next_step:
        return False, "Response does not include an actionable next step"

    return True, None


def _check_signoff(response: str) -> bool:
    """Check that the response ends with a recognizable sign-off."""
    response_lower = response.lower()
    return any(signal in response_lower for signal in SIGNOFF_SIGNALS)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_response(
    response: str,
    ticket_text: str,
    retrieved_chunk_contents: Optional[list[str]] = None,
) -> EvaluationResult:
    """
    Evaluate an LLM-generated response using heuristic checks.

    Args:
        response: The LLM-generated reply text.
        ticket_text: The original customer ticket text.
        retrieved_chunk_contents: List of KB chunk content strings from RAG retrieval.
                                   Pass empty list if no chunks were retrieved.
                                   Pass None to skip groundedness check.

    Returns:
        EvaluationResult with all check outcomes and a composite quality score.

    Quality score computation (heuristic weights — not scientifically validated):
        - Completeness:  35% weight
        - Relevance:     25% weight
        - Safety:        25% weight
        - Groundedness:  15% weight

    The weights are designed to prioritize completeness and relevance
    over groundedness, since groundedness check degrades gracefully
    when no KB context is available.
    """
    result = EvaluationResult(
        response_length=len(response),
        word_count=len(response.split()),
        rag_chunks_available=len(retrieved_chunk_contents) if retrieved_chunk_contents is not None else 0,
    )

    failure_reasons = []

    # --- Completeness check ---
    result.complete, completeness_reason = _check_completeness(response)
    if not result.complete and completeness_reason:
        failure_reasons.append(completeness_reason)

    # --- Relevance check ---
    result.relevant, result.relevance_score = _check_relevance(ticket_text, response)
    if not result.relevant:
        failure_reasons.append(
            f"Low relevance: only {result.relevance_score:.0%} of ticket terms in response"
        )

    # --- Safety check ---
    result.safe, safety_reason = _check_safety(response)
    if not result.safe and safety_reason:
        failure_reasons.append(safety_reason)

    # --- Groundedness check ---
    if retrieved_chunk_contents is not None:
        result.grounded, result.groundedness_score = _check_groundedness(
            response, retrieved_chunk_contents
        )
    else:
        result.grounded = True  # Cannot assess — pass by default
        result.groundedness_score = 0.0

    # --- Sign-off check ---
    result.has_signoff = _check_signoff(response)

    # --- Quality score (weighted composite, heuristic) ---
    completeness_weight = 0.35
    relevance_weight = 0.25
    safety_weight = 0.25
    groundedness_weight = 0.15

    completeness_score = 1.0 if result.complete else 0.0
    safety_score = 1.0 if result.safe else 0.0

    # Groundedness score: use raw float if KB chunks available, else neutral 0.5
    if retrieved_chunk_contents:
        g_score = min(1.0, result.groundedness_score * 8.0)  # scale up from ~0.1 range
    else:
        g_score = 0.5  # neutral when no context

    result.quality_score = round(
        completeness_score * completeness_weight
        + result.relevance_score * relevance_weight
        + safety_score * safety_weight
        + g_score * groundedness_weight,
        4,
    )

    # --- Regeneration decision ---
    result.needs_regeneration = result.quality_score < QUALITY_SCORE_THRESHOLD or not result.safe
    result.failure_reason = "; ".join(failure_reasons) if failure_reasons else None

    logger.debug(
        "Response evaluation: quality=%.3f relevant=%s grounded=%s safe=%s complete=%s",
        result.quality_score, result.relevant, result.grounded, result.safe, result.complete,
    )

    return result
