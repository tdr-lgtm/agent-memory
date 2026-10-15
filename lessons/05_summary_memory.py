
"""
Topic 5: Summary Memory (Abstractive + Progressive)

Example:
    You missed a 2-hour meeting.
    A friend gives you a short summary of what happened.
    Important information stays, but unnecessary details are removed.

Summary memory:
    Instead of sending all old messages, send a short summary
    of the conversation.

Abstractive:
    Writes the summary in new words instead of copying
    the original sentences.

Progressive:
    After each turn, update the old summary using the
    latest conversation.
    No need to summarize the entire chat from the beginning.

Compared to sliding window:
    Sliding window -> keeps only recent messages.
    Summary memory -> keeps important information from older messages.

Good:
    Uses fewer tokens.
    Can preserve important facts from older conversations.

Bad:
    Needs extra model calls to update the summary, which can
    increase cost.
    Small details and exact wording may be lost.
    If an important fact is removed from the summary, the
    agent may forget it.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from agent_memory.config import get_model

USER_TURNS = [
    "Hi, my name is Drv and I live in India.",
    "I'm learning about agent memory.",
    "Explain what a token is in two sentences.",
    "Give me one tip for writing clean Python.",
    "Suggest one name for a snack shop.",
    "What's my name and where do I live?",
]

SUMMARY_PROMPT = """Update the conversation summary.

Current summary:
{summary}

New exchange:
User: {user}
Assistant: {assistant}

Write the new summary in under 60 words.
Always keep facts about the user (name, location, goals).
Shorten long explanations to one line."""

def update_summary(model, summary: str, user: str, assistant: str):
    """
    Update the conversation summary using the latest exchange.

    Args:
        model: The chat model used to generate the summary.
        summary: The previous conversation summary.
        user: The latest user message.
        assistant: The latest assistant reply.

    Returns:
        A tuple containing the updated summary and the model reply.
    """
    prompt = SUMMARY_PROMPT.format(
        summary=summary or "(empty)",
        user=user,
        assistant=assistant,
    )
    reply = model.invoke(prompt)
    return reply.content, reply

def input_tokens_of(reply) -> int:
    """
    Get the number of input tokens used in a model reply.

    Args:
        reply: The response returned by the chat model.

    Returns:
        The number of input tokens, or 0 if usage data is unavailable.
    """
    usage = reply.usage_metadata or {}
    return usage.get("input_tokens", 0)

def main() -> None:
    model = get_model()
    summary = ""
    running_total = 0

    print(f"{'turn':<6}{'chat tokens':<13}{'summary tokens':<16}{'running total'}")
    print("-" * 50)

    for turn, text in enumerate(USER_TURNS, start=1):
        # send: summary (if any) + the new question. No old messages.
        to_send = []
        if summary:
            to_send.append(SystemMessage(f"Summary of the conversation so far:\n{summary}"))
        to_send.append(HumanMessage(text))

        reply = model.invoke(to_send)

        # extra call: fold this exchange into the summary
        summary, summary_reply = update_summary(model, summary, text, reply.content)

        chat_tokens = input_tokens_of(reply)
        summary_tokens = input_tokens_of(summary_reply)
        running_total += chat_tokens + summary_tokens

        print(f"{turn:<6}{chat_tokens:<13}{summary_tokens:<16}{running_total}")

    print("\nFinal summary:", summary)
    print("\nLast answer:", reply.content)


if __name__ == "__main__":
    main()