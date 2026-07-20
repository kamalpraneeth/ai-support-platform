"""
Urgency Scorer: Rule-based keyword matching for ticket urgency classification.

Design: Rule-based is intentionally chosen here over another ML model because:
- The training dataset is too small to train a reliable urgency model separately.
- Keywords like "urgent", "immediately", "cannot access" are strong direct signals.
- Fully transparent: every decision is traceable to a keyword list (good for auditing).
- Easy to extend: add/remove keywords without retraining.
"""

import re

# High urgency signals: time-sensitive, blocking, security-related
HIGH_KEYWORDS = [
    r"\burgent\b", r"\bimmediately\b", r"\basap\b", r"\bcrash(ed|ing)?\b",
    r"\bcannot (login|access|connect|use)\b", r"\bbroken\b", r"\bdown\b",
    r"\bhack(ed)?\b", r"\bsuspended\b", r"\bunauthorized\b", r"\bfraud\b",
    r"\bcharged (twice|incorrectly|wrong)\b", r"\bdata loss\b",
    r"\bnot working\b", r"\b500 error\b", r"\blocke?d out\b",
]

# Medium urgency signals: inconvenience but not blocking
MEDIUM_KEYWORDS = [
    r"\bslow\b", r"\bbuffering\b", r"\bnot syncing\b", r"\bno notification\b",
    r"\bwrong (amount|plan|charge)\b", r"\bovercbarged?\b",
    r"\bcannot (export|upload|download|change)\b",
    r"\berror\b", r"\bissue\b", r"\bproblem\b", r"\bnot (loading|showing|working)\b",
    r"\brefund\b", r"\bdispute\b",
]


def score_urgency(text: str) -> str:
    """
    Score ticket urgency based on keyword matching.

    Returns:
        'High'   — matches one or more HIGH_KEYWORDS
        'Medium' — matches one or more MEDIUM_KEYWORDS (but no High match)
        'Low'    — no urgency signals found (informational / general inquiry)
    """
    lower = text.lower()

    for pattern in HIGH_KEYWORDS:
        if re.search(pattern, lower):
            return "High"

    for pattern in MEDIUM_KEYWORDS:
        if re.search(pattern, lower):
            return "Medium"

    return "Low"
