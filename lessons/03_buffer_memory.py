
"""
Topic 3: Buffer Memory and Token Bloat

Example:
    A friend who reads the whole chat before every reply.
    They remember everything, but each reply takes more time and money.

Buffer memory:
    Stores all messages in a list and sends them to the model
    every time it replies.

Tokens:
    Small pieces of text that models process.
    API usage is often charged based on tokens.

Context window:
    The maximum amount of text and other information a model
    can process in one request.

Token bloat:
    Every new request includes old messages again.
    As the conversation grows, more tokens are used.
    This increases cost and can make responses slower.

Good:
    Remembers the full conversation.

Bad:
    Uses more tokens, costs more, and may exceed the context window.
"""

from langchain_core.messages import HumanMessage
from agent_memory.config import get_model

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
    # the whole conversation, kept forever
    buffer = []
    running_total = 0

    print(f"{'turn':<6}{'msgs sent':<11}{'input tokens':<14}{'running total'}")
    print("-" * 45)

    for turn, text in enumerate(USER_TURNS, start=1):
        buffer.append(HumanMessage(text))
        msgs_sent = len(buffer)

        # send EVERYTHING
        reply = model.invoke(buffer)
        # remember the reply too
        buffer.append(reply)

        usage = reply.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        running_total += input_tokens

        print(f"{turn:<6}{msgs_sent:<11}{input_tokens:<14}{running_total}")

    print("\nLast answer:", reply.content)


if __name__ == "__main__":
    main()