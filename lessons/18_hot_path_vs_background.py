"""
Topic 18: When to Write Memory (Hot Path vs Background)

Easy example:
    You tell a waiter: "I'm allergic to peanuts."

    Waiter A: writes it down BEFORE finishing your order.
             Safe, but takes more time. (hot path)

    Waiter B: finishes your order FIRST and writes it down afterward.
             Faster, but the kitchen may not know in time. (background)

Hot path:
    Save memory BEFORE sending the reply.
    The user waits for both the answer and the memory save.
    The new memory is ready before the next message is processed.

Background:
    Send the reply FIRST and save memory separately.
    The user waits only for the answer.
    But the next message may arrive before the memory is saved.

Jargon:
    Hot path              -> work the user must wait for.
    Background            -> work done separately without delaying the reply.
    Race condition        -> the result depends on which operation finishes first.
    Eventually consistent -> memory becomes up to date after some time,
                             but it may be temporarily outdated.

Example:
    Message 1: "I'm Drv and I just moved to Mumbai."
    Message 2: "Which city do I live in?"

    If message 2 arrives before the background save finishes,
    the agent may not know the answer because no chat history is provided.

Good (background):
    Replies are faster because memory saving does not delay the answer.

Bad (background):
    The next message may arrive before the latest memory is saved.

Best practice:
    Save important facts immediately using the hot path.
    Save less important information in the background.
    This balances fast replies with reliable memory.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_memory.config import get_model


class Profile(BaseModel):
    name: str | None = Field(None, description="The user's name")
    city: str | None = Field(None, description="City where the user currently lives")

EXTRACT_PROMPT = """Extract facts the user states about THEMSELVES.
Only fill a field if this message clearly mentions it.

Message: {message}"""

MESSAGES = [
    "Hi, I'm Drv and I just moved to Mumbai.",
    "Which city do I live in? Answer in one line.",
]

def chat(model, memory: dict, text: str) -> str:
    # Answer using ONLY the memory (no chat history).
    card = "\n".join(f"- {k}: {v}" for k, v in memory.items()) or "(nothing yet)"
    reply = model.invoke([
        SystemMessage(f"What you know about the user:\n{card}\nReply in one short line."),
        HumanMessage(text),
    ])
    return reply.content

def extract_and_save(extractor, memory: dict, lock: threading.Lock, text: str) -> None:
    # The memory write: extract facts, merge into memory.
    update = extractor.invoke(EXTRACT_PROMPT.format(message=text))
    filled = {k: v for k, v in update.model_dump().items() if v}
    with lock:  # only one thread changes memory at a time
        memory.update(filled)

def run_hot_path(model, extractor) -> None:
    print("HOT PATH: save memory BEFORE replying")
    memory, lock = {}, threading.Lock()

    for text in MESSAGES:
        seen = dict(memory)
        start = time.perf_counter()

        reply = chat(model, memory, text)
        extract_and_save(extractor, memory, lock, text)  # user waits for this too

        waited = time.perf_counter() - start
        print(f"  You: {text}")
        print(f"  memory seen: {seen}")
        print(f"  AI:  {reply}")
        print(f"  user waited: {waited:.1f}s\n")

def run_background(model, extractor) -> None:
    print("BACKGROUND: reply first, save memory in parallel")
    memory, lock = {}, threading.Lock()
    executor = ThreadPoolExecutor(max_workers=2)

    for text in MESSAGES:
        seen = dict(memory)
        start = time.perf_counter()

        reply = chat(model, memory, text)
        waited = time.perf_counter() - start  # user has the reply NOW

        executor.submit(extract_and_save, extractor, memory, lock, text)  # runs alongside

        print(f"  You: {text}")
        print(f"  memory seen: {seen}")
        print(f"  AI:  {reply}")
        print(f"  user waited: {waited:.1f}s\n")

    executor.shutdown(wait=True)  # let background saves finish
    print(f"  memory after background finished: {memory}\n")

def main() -> None:
    model = get_model()
    extractor = model.with_structured_output(Profile, method="function_calling")

    run_hot_path(model, extractor)
    run_background(model, extractor)

if __name__ == "__main__":
    main()