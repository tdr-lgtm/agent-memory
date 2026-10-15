"""
Topic 15: Semantic Memory (facts and repeated patterns)

Easy example:
    You read 6 diary entries about a topic.
    Instead of remembering every entry, you keep a few useful facts:
        - prefers short explanations
        - is vegetarian
        - often has difficulty with async code

    No full stories or event timeline -- just useful facts and patterns.

Semantic memory = a collection of facts and patterns learned from past events.

Episodic (Topic 14) vs Semantic:
    Episodic -> remembers events and their dates.
                Example: "Asked about async code on October 10."
    Semantic -> remembers facts and repeated patterns.
                Example: "Often struggles with async code."

    Episodic memory grows as new events happen.
    Semantic memory keeps a smaller collection of useful knowledge.
    Semantic facts do not automatically become less important just because
    they are old. However, facts can become outdated.

Jargon:
    Consolidation -> turning many diary entries into a few useful facts.
                     Usually happens in the background after sessions.
    Pattern       -> something that happens repeatedly in 2 or more entries.
    Evidence      -> the dates of the diary entries that support a fact.
                     Helps check whether the fact is correct.

Flow:
    Diary entries -> model extracts lasting facts and repeated patterns
                  -> attach evidence (source dates)
                  -> save facts in the store under ("users", id, "semantic")
                  -> answer questions using the facts instead of the full diary

Good:
    Stores useful knowledge in a compact form.
    Saves tokens because the model does not need the whole diary.
    Finds patterns that may not be obvious from a single event.

Bad:
    The model may call something a pattern after too little evidence.
    Old facts can become incorrect when circumstances change.
    Evidence dates may be wrong, so they must be checked.
    Existing facts need updating when new information changes them (Topic 20).
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from agent_memory.config import get_model

NAMESPACE = ("users", "drv", "semantic")

# Diary entries, as if loaded from Topic 14
DIARY = [
    {"date": "2026-08-03", "summary": "Asked for a FastAPI intro. Said long explanations confuse them and asked for short answers.", "outcome": "set up first API"},
    {"date": "2026-08-20", "summary": "Started an expense tracker with FastAPI. Chatted around 1 AM.", "outcome": "started"},
    {"date": "2026-08-28", "summary": "Got stuck on async database calls in the expense tracker.", "outcome": "stuck on async"},
    {"date": "2026-09-15", "summary": "Finished and deployed the expense tracker. Asked to keep the deploy steps short.", "outcome": "finished"},
    {"date": "2026-09-22", "summary": "Asked about protein sources and mentioned they don't eat meat.", "outcome": "got a list"},
    {"date": "2026-10-05", "summary": "Moved to Mumbai this week.", "outcome": "settling in"},
    {"date": "2026-10-10", "summary": "Started a chatbot in Python. Stuck on async streaming. Chatted around midnight.", "outcome": "stuck on async streaming"},
    {"date": "2026-10-12", "summary": "Asked for a dinner recipe and reminded the AI they are vegetarian.", "outcome": "got a recipe"},
]

class Fact(BaseModel):
    statement: str = Field(description="One lasting fact or pattern about the user, no dates")
    kind: Literal["fact", "pattern"] = Field(description="'pattern' only if it repeats in 2+ entries")
    evidence: list[str] = Field(description="Dates of the diary entries this comes from")

class Facts(BaseModel):
    facts: list[Fact]

EXTRACT_PROMPT = """Read these diary entries about one user.
Extract LASTING facts and REPEATED patterns about the user.

Rules:
- A pattern must appear in at least 2 entries.
- Skip one-time events (e.g. "got a recipe").
- Write each item without dates, as something true about the user.
- List the dates each item comes from as evidence.

Diary:
{diary}"""

def diary_text() -> str:
    return "\n".join(f"- {e['date']}: {e['summary']} Outcome: {e['outcome']}" for e in DIARY)

def consolidate(model, store: InMemoryStore) -> list[Fact]:
    # Many diary entries -> a few lasting facts, saved to the store.
    extractor = model.with_structured_output(Facts, method="function_calling")
    result = extractor.invoke(EXTRACT_PROMPT.format(diary=diary_text()))
    for i, fact in enumerate(result.facts):
        store.put(NAMESPACE, f"fact-{i}", fact.model_dump())
    return result.facts

def main() -> None:
    model = get_model()
    store = InMemoryStore()

    print(f"Consolidating {len(DIARY)} diary entries...\n")
    facts = consolidate(model, store)

    for f in facts:
        print(f"  [{f.kind:<7}] {f.statement}")
        print(f"            evidence: {', '.join(f.evidence)}")
    print()

    # Answer using FACTS only, not the whole diary
    facts_text = "\n".join(f"- {f.statement}" for f in facts)
    question = "I have a free weekend. What should I learn or practise? Answer in two lines."
    reply = model.invoke([
        SystemMessage(f"What you know about the user:\n{facts_text}"),
        HumanMessage(question),
    ])

    print(f"Diary size: ~{count_tokens_approximately([diary_text()])} tokens")
    print(f"Facts size: ~{count_tokens_approximately([facts_text])} tokens\n")
    print(f"Q: {question}")
    print(f"A: {reply.content}")


if __name__ == "__main__":
    main()