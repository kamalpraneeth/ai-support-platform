"""
Sentiment Analyzer: VADER (Valence Aware Dictionary and sEntiment Reasoner).

Why VADER over a transformer model?
- Designed specifically for short, informal text (support tickets, reviews, tweets).
- No model download required — pure Python lexicon, works fully offline.
- Fast: <1ms per prediction, no GPU needed.
- Interpretable: compound score maps directly to Positive/Neutral/Negative.

VADER compound score thresholds (from the original paper):
  >= 0.05  → Positive
  <= -0.05 → Negative
  else     → Neutral
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create analyzer once at module load (avoids repeated lexicon loading)
_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment of ticket text using VADER.

    Args:
        text: Raw support ticket text.

    Returns:
        One of: 'Positive', 'Neutral', 'Negative'
    """
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"
