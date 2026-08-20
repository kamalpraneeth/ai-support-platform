"""
Prompt Templates: versioned, structured system prompts for LLM calls.

Each template is a dataclass that can be serialized, logged, and compared.
Templates are versioned so prompt changes are traceable.

Template structure (SUPPORT_REPLY_V1):
  [SYSTEM INSTRUCTIONS] — role, tone, constraints
  [CONTEXT]             — ticket metadata (category, urgency, sentiment)
  [RELEVANT KNOWLEDGE]  — RAG-retrieved KB chunks
  [CUSTOMER QUERY]      — the raw ticket text
  [RESPONSE REQUIREMENTS] — format, length, sign-off
  [SAFETY RULES]        — grounding, escalation, no fabrication

Design principle:
  All prompts live here, not scattered in calling code.
  To change prompt behavior, update this file — the change propagates
  everywhere via the builder module.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """
    A versioned prompt template.

    Fields:
        version:        Identifier like "v1", "v2". Logged with every LLM call.
        system_prompt:  The system-role message sent to the LLM.
        user_prefix:    Prefix text prepended to the user message.
        name:           Human-readable template name.
    """
    version: str
    name: str
    system_prompt: str
    user_prefix: str


# ---------------------------------------------------------------------------
# SUPPORT_REPLY_V1 — primary template for RAG-augmented reply generation
# ---------------------------------------------------------------------------

SUPPORT_REPLY_V1 = PromptTemplate(
    version="v1",
    name="rag_support_reply",
    system_prompt=(
        "You are a professional and empathetic customer support agent.\n\n"
        "Your role is to write a helpful, accurate reply to a customer's support ticket.\n\n"
        "[RESPONSE REQUIREMENTS]\n"
        "- Write 3 to 5 sentences.\n"
        "- Be professional, clear, and empathetic.\n"
        "- Acknowledge the customer's issue.\n"
        "- Provide a concrete next step based on the RELEVANT KNOWLEDGE provided.\n"
        "- If the knowledge base does not contain enough information to answer fully, "
        "say so clearly and offer to escalate.\n"
        "- Sign off as 'Support Team'.\n\n"
        "[SAFETY RULES]\n"
        "- Only use information from RELEVANT KNOWLEDGE or verifiable general knowledge.\n"
        "- Do NOT invent specific details such as account numbers, dates, "
        "phone numbers, names, or reference numbers.\n"
        "- Do NOT promise specific timeframes unless stated in the knowledge base.\n"
        "- Do NOT claim capabilities or policies not mentioned in the knowledge base.\n"
        "- If you are unsure, say you will investigate and follow up.\n"
        "- For security or fraud issues, always recommend the customer "
        "contact the security team directly.\n"
        "- For billing disputes, acknowledge the issue and provide the "
        "dispute process from the knowledge base if available."
    ),
    user_prefix="",  # User message is fully built by the builder
)


# ---------------------------------------------------------------------------
# SUPPORT_REPLY_FALLBACK_V1 — template used when no RAG context is available
# ---------------------------------------------------------------------------

SUPPORT_REPLY_FALLBACK_V1 = PromptTemplate(
    version="v1",
    name="no_context_support_reply",
    system_prompt=(
        "You are a professional and empathetic customer support agent.\n\n"
        "You have received a customer support ticket but do not have specific "
        "knowledge base articles available for this query.\n\n"
        "[RESPONSE REQUIREMENTS]\n"
        "- Write 3 to 5 sentences.\n"
        "- Be professional, clear, and empathetic.\n"
        "- Acknowledge the customer's issue.\n"
        "- Tell the customer that your team will investigate and follow up.\n"
        "- Provide the expected response timeframe if applicable.\n"
        "- Sign off as 'Support Team'.\n\n"
        "[SAFETY RULES]\n"
        "- Do NOT invent specific details, policies, or timelines.\n"
        "- Do NOT claim to have information you do not have.\n"
        "- Keep the response honest and professional."
    ),
    user_prefix="",
)
