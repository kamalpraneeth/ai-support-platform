# AI Support Platform — ML Pipeline

The platform uses a traditional Machine Learning pipeline to perform high-speed, robust ticket classification and scoring prior to LLM intervention.

## 1. Data Engineering Pipeline

Data quality is the foundation of the ML system. The `app/data_pipeline.py` script manages the data lifecycle.

**Pipeline Steps:**
1. **Validation**: Rejects rows missing text, missing categories, or having invalid categories. Enforces a minimum character length.
2. **Normalization**: Strips excess whitespace and collapses internal spaces.
3. **Deduplication**: Removes identical ticket texts to prevent model bias.
4. **Reporting**: Generates a `data_quality_report.json` summarizing invalid rows removed, duplicates removed, and final class distribution.

## 2. Text Classification Model

The core classifier (`app/ml/classifier.py`) is responsible for categorizing tickets into one of four categories:
- `Billing`
- `Technical`
- `Account`
- `General`

**Architecture:**
- **Vectorization**: `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) converts raw text into numerical features, capturing the importance of words relative to the corpus.
- **Classifier**: `LogisticRegression` provides fast, calibrated multi-class probability outputs.
- **Output**: The model predicts the category and returns a **Confidence Score** (probability).

**Why Logistic Regression?**
- Extremely fast inference (milliseconds), suitable for API request paths.
- Provides reliable probability estimates (confidence), which are critical for the Orchestrator's escalation logic.
- Requires very little memory compared to deep learning models.

## 3. Heuristic Scoring (Urgency & Sentiment)

Instead of relying on heavy ML models for simple tasks, the pipeline uses fast, deterministic heuristics:

- **Urgency Scoring**: Uses a keyword-based rules engine. Words like "crash", "hacked", or "urgent" elevate the urgency to `High`. Words like "slow" or "error" set it to `Medium`.
- **Sentiment Analysis**: Uses the `VADER` (Valence Aware Dictionary and sEntiment Reasoner) library. VADER is highly optimized for short, social-media-style texts and customer support queries, instantly classifying text as `Positive`, `Neutral`, or `Negative` based on compound polarity scores.

## 4. Training & Evaluation (`app/ml/train.py` & `app/ml/evaluate.py`)

The training script builds the TF-IDF + Logistic Regression pipeline and evaluates it robustly to prevent overfitting.

**Evaluation Strategy:**
- **Train/Test Split**: Uses a 75/25 stratified split to ensure class balance in both sets.
- **Cross-Validation**: Performs 5-fold cross-validation on the training set to estimate model stability and variance (`cv_mean`, `cv_std`).
- **Metrics Computed**:
  - Accuracy
  - Macro F1-Score
  - Weighted F1-Score
  - Per-class Precision, Recall, and F1
  - Confusion Matrix
- **Artifacts**: Generates `evaluation_report.json` and serializes the trained model pipeline to `classifier_pipeline.pkl`.

## 5. Inference Lifecycle

The model (`classifier_pipeline.pkl`) is loaded into memory exactly once during the FastAPI application lifespan (startup hook). All subsequent API calls share this in-memory instance, allowing classification to happen in less than 5 milliseconds per request.
