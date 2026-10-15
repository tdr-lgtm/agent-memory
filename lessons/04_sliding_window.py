"""
Topic 4: Sliding Window Memory

Example:
    A person who remembers only the most recent part of a conversation.
    When new messages come in, the oldest messages are removed.

Sliding window:
    Stores the full conversation but sends only the last N messages
    to the model.

    WINDOW = 4 means the model receives the last 4 messages,
    usually 2 user messages and 2 AI replies.

Compared to buffer memory:
    Buffer         -> sends all messages; token usage keeps growing.
    Sliding window -> sends only the last N messages; token usage stays limited.

Good:
    Uses fewer tokens and keeps costs under control.
    Works well for long conversations.

Bad:
    Cannot remember older messages outside the window.
    May forget important information, such as the user's name.

"""

from langchain_core.messages import HumanMessage
from agent_memory.config import get_model

WINDOW = 4  # last 4 messages = last 2 back-and-forths

USER_TURNS = [
    "Hi, my name is Drv and I live in India.",
    "I'm learning about agent memory.",
    "Explain what a token is in two sentences.",
    "Give me one tip for writing clean Python.",
    "Suggest one name for a snack shop.",
    "What's my name and where do I live?",
]

def main() -> None:
    model = get_model()
    # full history is stored, but not all of it is sent
    history = []
    running_total = 0

    print(f"{'turn':<6}{'msgs sent':<11}{'input tokens':<14}{'running total'}")
    print("-" * 45)

    for turn, text in enumerate(USER_TURNS, start=1):
        # only the last WINDOW messages + the new question
        recent = history[-WINDOW:]
        to_send = recent + [HumanMessage(text)]

        reply = model.invoke(to_send)

        # store both in full history
        history.append(HumanMessage(text))
        history.append(reply)

        usage = reply.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        running_total += input_tokens

        print(f"{turn:<6}{len(to_send):<11}{input_tokens:<14}{running_total}")

    print("\nLast answer:", reply.content)


if __name__ == "__main__":
    main()