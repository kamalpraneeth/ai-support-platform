"""
Prompt Builder: assembles structured prompts from templates and runtime context.

The builder separates prompt construction from LLM calling code.
It takes ticket metadata + RAG results and produces a PromptPayload that
is ready to pass directly to the Groq API.

This keeps all prompt-construction logic in one place — changes to prompt
structure require editing only this module.
"""

import logging
from dataclasses import dataclass

from app.prompts.templates import (
    PromptTemplate,
    SUPPORT_REPLY_V1,
    SUPPORT_REPLY_FALLBACK_V1,
)
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class PromptPayload:
    """
    Ready-to-send prompt for the LLM.

    Contains the system message and user message as separate fields,
    matching the OpenAI / Groq messages format.
    """
    system_message: str
    user_message: str
    template_name: str
    template_version: str
    rag_chunks_used: int

    def to_messages(self) -> list[dict]:
        """Return the prompt in Groq/OpenAI chat messages format."""
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        ]


def build_support_prompt(
    ticket_text: str,
    category: str,
    urgency: str,
    sentiment: str,
    retrieved_chunks: list[RetrievedChunk],
    cv_objects: list[dict] | None = None,
    template: PromptTemplate = SUPPORT_REPLY_V1,
) -> PromptPayload:
    """
    Build a complete, structured support-reply prompt.

    Args:
        ticket_text: The raw customer ticket text.
        category: Predicted category from ML classifier.
        urgency: Rule-based urgency (High/Medium/Low).
        sentiment: VADER sentiment (Positive/Neutral/Negative).
        retrieved_chunks: RAG-retrieved knowledge base chunks.
        cv_objects: Optional list of objects detected by Computer Vision.
        template: The PromptTemplate to use (default: SUPPORT_REPLY_V1).

    Returns:
        PromptPayload ready to pass to the LLM.

    User message structure:
        [TICKET CONTEXT]
        [RELEVANT KNOWLEDGE]
        [CUSTOMER QUERY]
        [YOUR RESPONSE]
    """
    # --- Build context section ---
    context_lines = [
        "[TICKET CONTEXT]",
        f"Category: {category}",
        f"Urgency: {urgency}",
        f"Sentiment: {sentiment}",
    ]
    if urgency == "High":
        context_lines.append("Note: This is a HIGH URGENCY ticket. Prioritize acknowledgment and clear next steps.")
    elif sentiment == "Negative":
        context_lines.append("Note: The customer appears frustrated. Be especially empathetic in your response.")

    if cv_objects:
        labels = [obj["label"] for obj in cv_objects]
        context_lines.append(f"Image Upload: The user attached an image. Detected objects: {', '.join(labels)}.")
        context_lines.append("Note: Acknowledge these items in your response if relevant to their query.")

    context_block = "\n".join(context_lines)

    # --- Build knowledge section ---
    rag_chunks_used = len(retrieved_chunks)
    if retrieved_chunks:
        knowledge_lines = ["[RELEVANT KNOWLEDGE]"]
        for i, chunk in enumerate(retrieved_chunks, start=1):
            knowledge_lines.append(
                f"Article {i}: {chunk.title}\n"
                f"{chunk.content}"
            )
        knowledge_block = "\n\n".join(knowledge_lines)
        active_template = template  # Use RAG template
    else:
        knowledge_block = (
            "[RELEVANT KNOWLEDGE]\n"
            "No specific knowledge base articles were retrieved for this ticket. "
            "Respond based on general support best practices and offer to escalate if needed."
        )
        active_template = SUPPORT_REPLY_FALLBACK_V1  # Use fallback template

    # --- Assemble user message ---
    user_message_parts = [
        context_block,
        "",  # blank line separator
        knowledge_block,
        "",
        "[CUSTOMER QUERY]",
        ticket_text,
        "",
        "[YOUR RESPONSE]",
    ]
    user_message = "\n".join(user_message_parts)

    logger.debug(
        "Prompt built: template=%s/%s, rag_chunks=%d, urgency=%s",
        active_template.name, active_template.version, rag_chunks_used, urgency,
    )

    return PromptPayload(
        system_message=active_template.system_prompt,
        user_message=user_message,
        template_name=active_template.name,
        template_version=active_template.version,
        rag_chunks_used=rag_chunks_used,
    )
