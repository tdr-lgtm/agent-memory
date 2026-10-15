"""
Topic 19: Memory Routing (pick the right memory for each question)

Example (receptionist):
    "What's my address on file?"     -> customer details cabinet
    "What happened at my last visit?" -> visit history cabinet
    "What's the capital of France?"   -> no cabinet, just answer
    Listen to the question FIRST, then open only the right cabinet.

Our memories:
    profile  -> facts true NOW            ("lives in Mumbai")
    episodes -> what HAPPENED, with dates ("Sep 15: finished tracker")
    facts    -> lasting habits, patterns  ("often stuck on async")
    none     -> general question, no memory needed

Why route?
    Every memory in every prompt -> costs more AND confuses the model
    (Topic 10: chai leaked into the dinner answer).

Flow:
    1. small model call reads the question -> picks ONE memory
    2. load only that memory
    3. answer

Jargon:
    Routing -> choosing where to send something
    Intent  -> what the user is really asking for

Good:
    Smaller prompts, less confusion.
Bad:
    Extra model call per question.
    Wrong route -> right answer is never loaded.
    Some questions need TWO memories (this version picks one).
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_memory.config import get_model

# The three memories (as if loaded from the store)
MEMORIES = {
    "profile": [
        "name: Drv",
        "city: Mumbai",
        "current project: chatbot",
    ],
    "episodes": [
        "2026-08-20: started an expense tracker with FastAPI",
        "2026-09-15: finished and deployed the expense tracker",
        "2026-10-05: moved from Delhi to Mumbai",
        "2026-10-10: started a chatbot, stuck on streaming",
    ],
    "facts": [
        "is vegetarian",
        "prefers short answers",
        "often gets stuck on async code",
        "usually chats late at night",
    ],
}


class Route(BaseModel):
    memory: Literal["profile", "episodes", "facts", "none"] = Field(
        description="Which memory to look in"
    )
    reason: str = Field(description="One short sentence: why this memory")

ROUTER_PROMPT = """Pick the ONE memory that best helps answer this question.

profile  -> current facts about the user: name, city, current project
episodes -> past events with dates: what the user did, finished, or when
facts    -> lasting habits and patterns: diet, preferences, common struggles
none     -> general question that needs nothing about the user

Suggestions and recommendations (food, plans, what to learn) are personal -> use facts.
If unsure, pick a memory instead of none: loading extra is safer than missing something.

Question: {question}"""

QUESTIONS = [
    "Where do I live?",
    "What did I finish last month?",
    "What do I usually struggle with when coding?",
    "Suggest a dinner for me tonight.",
    "What is a Python list?",
]

def main() -> None:
    model = get_model()
    router = model.with_structured_output(Route, method="function_calling")

    for question in QUESTIONS:
        # 1. pick a memory
        route = router.invoke(ROUTER_PROMPT.format(question=question))

        # 2. load only that memory
        loaded = MEMORIES.get(route.memory, [])
        memory_text = "\n".join(f"- {m}" for m in loaded) or "(none)"

        # 3. answer
        reply = model.invoke([
            SystemMessage(f"What you know about the user:\n{memory_text}\n"
                          "Answer in one line."),
            HumanMessage(question),
        ])

        print(f"Q: {question}")
        print(f"   route: {route.memory}  ({route.reason})")
        print(f"   loaded {len(loaded)} items")
        print(f"A: {reply.content}\n")

if __name__ == "__main__":
    main()