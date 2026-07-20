"""
Sentiment Analyzer: VADER + support-ticket-aware pre-screening.

Why VADER over a transformer model?
- Designed for short, informal text (support tickets, reviews, tweets).
- No model download required — pure Python lexicon, works fully offline.
- Fast: <1ms per prediction, no GPU needed.
- Interpretable: compound score maps directly to Positive/Neutral/Negative.

Known VADER limitation (and how we fix it):
  VADER is trained on general social-media text. It scores "help" as positive
  and "cannot" as only mildly negative, so factual support complaints like
  "I cannot login and need help urgently" come out positive even though the
  customer is clearly frustrated.

  Fix: we pre-screen for support-ticket-specific negative signal phrases
  (functional complaint language that VADER's lexicon underweights) and apply
  a compound-score penalty before the final threshold decision.
  Explicit emotional language ("frustrated", "angry", "terrible") still flows
  through VADER normally — the two layers are additive.
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create analyzer once at module load (avoids repeated lexicon loading)
_analyzer = SentimentIntensityAnalyzer()

# Phrases that strongly signal a negative support ticket even when written
# in neutral / factual language that VADER's lexicon underscores.
_TICKET_NEGATIVE_PATTERNS = [
    r"\bcannot (login|access|connect|use|open|load|view|change|update|export|download|upload)\b",
    r"\bcan't (login|access|connect|use|open|load|view|change|update)\b",
    r"\bnot (working|loading|syncing|showing|receiving|sending|responding)\b",
    r"\bcharged (twice|incorrectly|wrong|extra|again)\b",
    r"\bunauthorized charge\b",
    r"\bwas (hacked|suspended|locked|charged|billed|blocked)\b",
    r"\b(app|site|page|service|platform) (crash|crashes|crashed|crashing|down|broken|unavailable)\b",
    r"\bno (access|response|reply|notification|email)\b",
    r"\bnobody (is helping|helped|responded|replied)\b",
    r"\b(lost|missing|deleted|corrupted) (data|files?|account|records?)\b",
    r"\bneed (urgent|immediate|help urgently)\b",
    r"\burgent(ly)?\b",
    r"\bnot received\b",
    r"\bdouble.?charged?\b",
    r"\brefund\b",
    r"\bdispute\b",
    r"\b(locked|suspended|banned) (out|account)\b",
]

# Each matched pattern subtracts this from VADER's compound score.
# Set low enough to tip borderline positives into Negative territory,
# but not so aggressive it overrides genuinely positive text.
_PENALTY_PER_MATCH = 0.25
_MAX_PENALTY = 0.60  # cap so stacking many patterns doesn't over-penalise


def _ticket_negativity_penalty(text: str) -> float:
    """Return a score penalty [0.0 .. MAX_PENALTY] for implicit complaint signals."""
    lower = text.lower()
    hits = sum(1 for p in _TICKET_NEGATIVE_PATTERNS if re.search(p, lower))
    return min(hits * _PENALTY_PER_MATCH, _MAX_PENALTY)


def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of ticket text using VADER + ticket pre-screening.

    Args:
        text: Raw support ticket text.

    Returns:
        One of: 'Positive', 'Neutral', 'Negative'
    """
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Apply ticket-specific penalty to raw VADER compound score
    penalty = _ticket_negativity_penalty(text)
    adjusted = compound - penalty

    if adjusted >= 0.05:
        return "Positive"
    elif adjusted <= -0.05:
        return "Negative"
    else:
        return "Neutral"

