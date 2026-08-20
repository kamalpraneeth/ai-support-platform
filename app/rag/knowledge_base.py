"""
Knowledge Base Loader: reads and validates JSON knowledge-base documents.

Knowledge base files live in data/knowledge_base/*.json.
Each file is a JSON array of document objects with the schema:

  {
    "id":       string  (unique document ID, e.g. "billing_001")
    "title":    string  (short human-readable title)
    "content":  string  (full article content — this is what gets retrieved)
    "category": string  (one of: Billing, Technical, Account, General)
    "tags":     list[str] (keywords for supplementary matching)
  }

Documents with missing required fields are skipped with a warning.
The loader is intentionally format-agnostic — adding a new topic is as
simple as creating a new JSON file in the knowledge_base directory.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
KB_DIR = ROOT / "data" / "knowledge_base"

REQUIRED_FIELDS = {"id", "title", "content", "category"}
VALID_CATEGORIES = {"Billing", "Technical", "Account", "General"}


@dataclass
class KBDocument:
    """A single knowledge-base document after loading and validation."""
    doc_id: str
    title: str
    content: str
    category: str
    tags: list[str]

    @property
    def full_text(self) -> str:
        """Combined text used for TF-IDF indexing: title + content + tags."""
        tag_text = " ".join(self.tags) if self.tags else ""
        return f"{self.title}. {self.content} {tag_text}".strip()

    def to_dict(self) -> dict:
        return {
            "id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
        }


def _validate_document(raw: dict, source_file: str,
                       idx: int) -> Optional[KBDocument]:
    """
    Validate and parse a single raw document dict.

    Returns KBDocument on success, None if the document should be skipped.
    """
    missing = REQUIRED_FIELDS - raw.keys()
    if missing:
        logger.warning(
            "Skipping document #%d in %s: missing fields %s",
            idx, source_file, missing,
        )
        return None

    doc_id = str(raw.get("id", "")).strip()
    title = str(raw.get("title", "")).strip()
    content = str(raw.get("content", "")).strip()
    category = str(raw.get("category", "")).strip()
    tags = raw.get("tags", [])

    if not doc_id or not title or not content:
        logger.warning(
            "Skipping document #%d in %s: empty required field",
            idx,
            source_file)
        return None

    if category not in VALID_CATEGORIES:
        logger.warning(
            "Skipping document '%s' in %s: invalid category '%s'",
            doc_id, source_file, category,
        )
        return None

    if not isinstance(tags, list):
        tags = []

    return KBDocument(
        doc_id=doc_id,
        title=title,
        content=content,
        category=category,
        tags=[str(t) for t in tags],
    )


def load_knowledge_base(kb_dir: Path = KB_DIR) -> list[KBDocument]:
    """
    Load all JSON knowledge base documents from the knowledge_base directory.

    Args:
        kb_dir: Path to the directory containing *.json knowledge base files.

    Returns:
        List of validated KBDocument objects. Empty list if directory not found.

    Raises:
        ValueError: If any JSON file contains malformed JSON (syntax error).
    """
    if not kb_dir.exists():
        logger.warning("Knowledge base directory not found: %s", kb_dir)
        return []

    json_files = sorted(kb_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", kb_dir)
        return []

    documents: list[KBDocument] = []
    seen_ids: set[str] = set()

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                raw_docs = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in knowledge base file {json_file}: {exc}") from exc

        if not isinstance(raw_docs, list):
            logger.warning(
                "Skipping %s: expected a JSON array, got %s",
                json_file.name,
                type(raw_docs))
            continue

        file_count = 0
        for i, raw in enumerate(raw_docs):
            doc = _validate_document(raw, json_file.name, i)
            if doc is None:
                continue

            if doc.doc_id in seen_ids:
                logger.warning(
                    "Duplicate document ID '%s' in %s — skipping",
                    doc.doc_id,
                    json_file.name)
                continue

            seen_ids.add(doc.doc_id)
            documents.append(doc)
            file_count += 1

        logger.debug("Loaded %d documents from %s", file_count, json_file.name)

    logger.info(
        "Knowledge base loaded: %d documents from %d files in %s",
        len(documents), len(json_files), kb_dir,
    )
    return documents
