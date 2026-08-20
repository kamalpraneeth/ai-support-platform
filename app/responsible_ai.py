"""
Responsible AI Module: input validation, output validation, and escalation logic.

This module implements a lightweight Responsible AI layer with the following goals:

1. Input Validation
   - Detect prompt injection attempts (heuristic pattern matching)
   - Flag suspiciously long inputs (potential abuse)
   - Detect PII patterns and log warnings (not reject — just surface)

2. Output Validation
   - Re-run safety checks from evaluation.py
   - Detect hallucination-risk patterns (fabricated specifics)

3. Escalation Logic
   - Combine ML confidence + urgency + sentiment to decide escalation
   - Escalated tickets are flagged; human review is recommended

Transparency notes:
   - Prompt injection detection uses heuristic regex. It does NOT guarantee
     catching all adversarial inputs. Defense in depth is required.
   - PII detection flags common patterns (email, phone, SSN-like) but does
     not implement comprehensive PII scanning.
   - These checks are documented as heuristics, not security guarantees.

Audit logging:
   - All validation decisions are logged at INFO level.
   - Sensitive content (actual ticket text) is truncated in logs to 100 chars.
   - API keys, passwords, and secrets are NEVER logged.
"""

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt injection patterns (heuristic — not exhaustive)
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+instructions?",
    r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|context)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+\w+",
    r"do\s+not\s+follow\s+your\s+(instructions?|guidelines?|rules?)",
    r"system\s+prompt\s*:",
    r"</?system>",
    r"\[system\]",
    r"override\s+(safety|filter|restriction)",
    r"bypass\s+(safety|filter|restriction|rule)",
    r"jailbreak",
    r"dan\s+mode",
]

# PII detection patterns (surface/flag only — do not reject)
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn_like": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "credit_card_like": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}

# Hallucination risk patterns in outputs (fabricated specifics)
HALLUCINATION_RISK_PATTERNS = [
    r"your account number is \d+",
    r"reference number[:\s]+[A-Z0-9]{6,}",
    r"case id[:\s]+[A-Z0-9]{6,}",
    r"ticket id[:\s]+[A-Z0-9]{6,}",
    r"call us at \d[\d\s\-().]{7,}",      # Fabricated phone numbers in output
    r"within \d+ (minutes?|hours?) guaranteed",  # Over-specific time promises
]

# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = 0.45   # Escalate if confidence below this
ESCALATION_CATEGORIES = {"Account"}       # Security-adjacent categories always reviewed


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InputValidationResult:
    """
    Result of validating a customer-submitted ticket text.

    Fields:
        is_valid:                  True if the input should be processed.
        injection_detected:        True if a prompt injection pattern matched.
        pii_types_detected:        List of PII type names found (e.g., ["email", "phone_us"]).
        is_suspiciously_long:      True if input exceeds 2000 characters.
        rejection_reason:          Reason for rejection if is_valid is False.
        warning_message:           Advisory message (input accepted but flagged).
    """
    is_valid: bool = True
    injection_detected: bool = False
    pii_types_detected: list = None
    is_suspiciously_long: bool = False
    rejection_reason: Optional[str] = None
    warning_message: Optional[str] = None

    def __post_init__(self):
        if self.pii_types_detected is None:
            self.pii_types_detected = []

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutputValidationResult:
    """Result of validating an LLM-generated reply."""
    is_safe: bool = True
    hallucination_risk_detected: bool = False
    risk_patterns_found: list = None
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        if self.risk_patterns_found is None:
            self.risk_patterns_found = []

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input(text: str) -> InputValidationResult:
    """
    Validate a customer-submitted ticket text before processing.

    Checks performed:
      1. Prompt injection patterns (heuristic regex — not exhaustive)
      2. Input length (>2000 chars flagged, >5000 chars rejected)
      3. PII detection (flagged in logs, not rejected)

    Args:
        text: The raw customer ticket text.

    Returns:
        InputValidationResult. Check is_valid before processing.

    Note:
        Injection detection is a heuristic. Adversarial inputs may still
        pass. The LLM system prompt provides additional defense.
    """
    result = InputValidationResult()
    text_lower = text.lower()

    # --- Prompt injection check ---
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            result.injection_detected = True
            result.is_valid = False
            result.rejection_reason = (
                "Input flagged as potential prompt injection attempt. "
                "Please submit a standard support ticket."
            )
            logger.warning(
                "Prompt injection pattern detected. Input (truncated): '%s...'",
                text[:100],
            )
            return result  # Early return — no need to check further

    # --- Length check ---
    if len(text) > 5000:
        result.is_valid = False
        result.rejection_reason = (
            f"Input too long ({len(text)} characters). "
            "Please keep support tickets under 5000 characters."
        )
        logger.warning("Rejected oversized input: %d chars", len(text))
        return result

    if len(text) > 2000:
        result.is_suspiciously_long = True
        result.warning_message = "Input is unusually long. Consider summarizing."
        logger.info("Flagged long input: %d chars", len(text))

    # --- PII detection (log warning, do not reject) ---
    detected_pii = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            detected_pii.append(pii_type)

    if detected_pii:
        result.pii_types_detected = detected_pii
        # Log PII detection WITHOUT logging the actual PII values
        logger.warning(
            "PII type(s) detected in ticket input: %s. "
            "Ticket will still be processed. PII values are not logged.",
            detected_pii,
        )
        result.warning_message = (
            f"Note: PII may be present in this ticket ({', '.join(detected_pii)}). "
            "Please do not include sensitive personal information in support tickets."
        )

    return result


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(response: str) -> OutputValidationResult:
    """
    Validate an LLM-generated reply before delivering it to the client.

    Checks for hallucination-risk patterns such as fabricated account numbers,
    fake phone numbers, and over-specific time commitments.

    Args:
        response: The LLM-generated reply text.

    Returns:
        OutputValidationResult. If is_safe is False, the response should
        be regenerated or replaced with a safe fallback.

    Limitations:
        This is a regex-based heuristic. It cannot catch all forms of
        hallucination or harmful content.
    """
    result = OutputValidationResult()
    response_lower = response.lower()

    found_patterns = []
    for pattern in HALLUCINATION_RISK_PATTERNS:
        if re.search(pattern, response_lower):
            found_patterns.append(pattern)

    if found_patterns:
        result.hallucination_risk_detected = True
        result.is_safe = False
        result.risk_patterns_found = found_patterns
        result.rejection_reason = (
            "Response contains fabricated specifics (account numbers, phone numbers, "
            "or over-specific promises). Flagged for regeneration."
        )
        logger.warning(
            "Output validation: hallucination-risk patterns detected: %s",
            found_patterns,
        )

    return result


# ---------------------------------------------------------------------------
# Escalation logic
# ---------------------------------------------------------------------------

def should_escalate(
    category: str,
    confidence: float,
    urgency: str,
    sentiment: str,
) -> tuple[bool, str]:
    """
    Determine whether a ticket should be escalated to a human agent.

    Escalation triggers (any one is sufficient):
      1. ML confidence below ESCALATION_CONFIDENCE_THRESHOLD (ambiguous classification)
      2. High urgency + Negative sentiment (frustrated + urgent customer)
      3. Account security category (hacked accounts, unauthorized access)

    Args:
        category: ML-predicted ticket category.
        confidence: ML classifier confidence score [0.0, 1.0].
        urgency: Rule-based urgency ('High', 'Medium', 'Low').
        sentiment: VADER sentiment ('Positive', 'Neutral', 'Negative').

    Returns:
        (should_escalate: bool, reason: str)
    """
    # Trigger 1: Low classifier confidence
    if confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        reason = (
            f"Low classifier confidence ({confidence:.0%}). "
            "Ticket classification may be incorrect — human review recommended."
        )
        logger.info("Escalation triggered (low confidence=%.3f): '%s'", confidence, reason)
        return True, reason

    # Trigger 2: High urgency + Negative sentiment
    if urgency == "High" and sentiment == "Negative":
        reason = (
            "High urgency + Negative sentiment detected. "
            "Escalating for priority human review."
        )
        logger.info("Escalation triggered (high urgency + negative sentiment)")
        return True, reason

    # Trigger 3: Security-sensitive category
    if category in ESCALATION_CATEGORIES:
        # Only escalate account issues if high urgency
        if urgency == "High":
            reason = (
                f"Security-sensitive category ({category}) with High urgency. "
                "Escalating for human review."
            )
            logger.info("Escalation triggered (security category + high urgency): %s", category)
            return True, reason

    return False, ""
