"""
Tests for the ML layer: classifier, urgency scorer, and sentiment analyzer.

These tests validate the ML logic independently of the web server,
so they run fast and without any HTTP overhead.
"""

import pytest
from app.ml.urgency import score_urgency
from app.ml.sentiment import analyze_sentiment


# ── Urgency Scorer Tests ─────────────────────────────────────────────────────

class TestUrgencyScorer:
    def test_high_urgency_crash(self):
        """Crashing app is a high-urgency blocking issue."""
        assert score_urgency("The application keeps crashing every time I open it") == "High"

    def test_high_urgency_cannot_login(self):
        """Cannot login is a High urgency blocking issue."""
        assert score_urgency("I cannot login to my account") == "High"

    def test_high_urgency_unauthorized_charge(self):
        """Security/fraud keywords trigger High urgency."""
        assert score_urgency("There is an unauthorized charge on my account") == "High"

    def test_medium_urgency_slow(self):
        """Slow performance is inconvenient but not blocking — Medium."""
        assert score_urgency("The platform is very slow to load") == "Medium"

    def test_medium_urgency_error(self):
        """Generic 'error' keyword maps to Medium urgency."""
        assert score_urgency("I am seeing an error in the dashboard") == "Medium"

    def test_low_urgency_general_question(self):
        """General informational questions have Low urgency."""
        assert score_urgency("What are your pricing plans?") == "Low"

    def test_low_urgency_feature_request(self):
        """Feature requests are Low urgency."""
        assert score_urgency("I would love a dark mode option") == "Low"


# ── Sentiment Analyzer Tests ─────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_negative_sentiment_angry(self):
        """Frustrated, angry ticket should be Negative."""
        result = analyze_sentiment("I am extremely frustrated with the terrible service")
        assert result == "Negative"

    def test_positive_sentiment_happy(self):
        """Satisfied customer should be Positive."""
        result = analyze_sentiment("Thank you so much, the support was excellent and really helpful!")
        assert result == "Positive"

    def test_neutral_sentiment_question(self):
        """Factual question with no emotional valence should be Neutral."""
        result = analyze_sentiment("Where can I find the API documentation?")
        assert result == "Neutral"

    def test_returns_valid_label(self):
        """Sentiment always returns one of three valid labels."""
        result = analyze_sentiment("Something happened with my account")
        assert result in ("Positive", "Neutral", "Negative")


# ── Classifier Tests (with trained model) ────────────────────────────────────

class TestClassifier:
    @pytest.fixture(scope="class")
    def model(self):
        """Load the trained model once for all tests in this class."""
        from app.ml.classifier import load_model
        return load_model()

    def test_billing_classification(self, model):
        """Clear billing issue should be classified as Billing."""
        from app.ml.classifier import predict_category
        result = predict_category("I was charged twice for my subscription", model=model)
        assert result == "Billing"

    def test_technical_classification(self, model):
        """App crash should be classified as Technical."""
        from app.ml.classifier import predict_category
        result = predict_category("The application crashes every time I open it", model=model)
        assert result == "Technical"

    def test_account_classification(self, model):
        """Login issue should be classified as Account."""
        from app.ml.classifier import predict_category
        result = predict_category("I cannot log into my account and reset email is not arriving", model=model)
        assert result == "Account"

    def test_general_classification(self, model):
        """Pricing question should be classified as General."""
        from app.ml.classifier import predict_category
        result = predict_category("Do you offer a free trial for the premium plan?", model=model)
        assert result == "General"

    def test_returns_valid_category(self, model):
        """Prediction always returns one of the four valid categories."""
        from app.ml.classifier import predict_category, CATEGORIES
        result = predict_category("Some random support text", model=model)
        assert result in CATEGORIES
