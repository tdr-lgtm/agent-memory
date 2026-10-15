
"""
Topic 6: Summary Buffer Memory

Example:
    You tell a friend:
    "Earlier, we decided to go to Goa."          <- Summary
    "Book the 6:40 train, coach B2, seat 31."    <- Exact recent message

Summary buffer memory:
    Combines sliding window memory (Topic 4) and summary memory (Topic 5).

    Recent messages -> Keep them exactly as they are.
    Older messages  -> Store them as a short summary.

    When a message leaves the window, add its important
    information to the summary.

    The summary is updated only when old messages leave
    the window, not after every turn.

Compared to:
    Sliding window -> Old messages are lost.
    Summary memory -> Keeps a summary but loses exact details.
    Summary buffer -> Keeps old information in a summary
                      and recent messages exactly.

Good:
    Remembers important old information.
    Keeps recent messages word for word.

Bad:
    Uses more tokens than summary memory because recent
    messages are sent with every request.
    Long messages can increase token usage.
    Exact details may be lost when messages leave the window
    and are summarized.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agent_memory.config import get_model

WINDOW = 4  # last 4 raw messages = last 2 back-and-forths

USER_TURNS = [
    "Hi, my name is Drv and I live in India.",
    "I'm learning about agent memory.",
    "Explain what a token is in two sentences.",
    "Give me one tip for writing clean Python.",
    "Suggest one name for a snack shop.",
    "What's my name, where do I live, and what was your Python tip, word for word?",
]

SUMMARY_PROMPT = """Update the conversation summary.

Current summary:
{summary}

Exchange to add:
User: {user}
Assistant: {assistant}

Write the new summary in under 60 words.
Always keep facts about the user (name, location, goals).
Shorten long explanations to one line."""


def fold_into_summary(model, summary: str, user_msg, ai_msg):
    # Fold one old exchange into the summary. Returns (new_summary, reply).
    prompt = SUMMARY_PROMPT.format(
        summary=summary or "(empty)",
        user=user_msg.content,
        assistant=ai_msg.content,
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
    recent = []  # raw messages inside the window
    running_total = 0

    print(f"{'turn':<6}{'raw msgs':<10}{'chat tokens':<13}{'summary tokens':<16}{'running total'}")
    print("-" * 60)

    for turn, text in enumerate(USER_TURNS, start=1):
        # send: summary (if any) + raw recent messages + new question
        to_send = []
        if summary:
            to_send.append(SystemMessage(f"Summary of the earlier conversation:\n{summary}"))
        to_send.extend(recent)
        to_send.append(HumanMessage(text))

        reply = model.invoke(to_send)
        chat_tokens = input_tokens_of(reply)
        raw_count = len(recent)

        recent.append(HumanMessage(text))
        recent.append(reply)

        # window overflowed -> fold the OLDEST exchange into the summary
        summary_tokens = 0
        while len(recent) > WINDOW:
            old_user, old_ai = recent.pop(0), recent.pop(0)
            summary, summary_reply = fold_into_summary(model, summary, old_user, old_ai)
            summary_tokens += input_tokens_of(summary_reply)

        running_total += chat_tokens + summary_tokens
        print(f"{turn:<6}{raw_count:<10}{chat_tokens:<13}{summary_tokens:<16}{running_total}")

    print("\nFinal summary:", summary)
    print("\nLast answer:", reply.content)


if __name__ == "__main__":
    main()