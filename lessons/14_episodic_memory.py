"""
Topic 14: Episodic Memory (remembering past sessions with dates)

Easy example:
    "What is 2 + 3?" -> "5." (not episodic memory)
    "What did we discuss last time?" -> "We discussed Python."
                                      (episodes: past events)

Episodic memory = a diary of past chat sessions.
    Each entry stores WHAT happened and WHEN it happened.
    New information adds a new entry. Old entries stay saved.

Profile (Topic 12) vs Episodes:
    Profile  -> stores current facts. New values replace old ones.
                Answers: "What is true now?"
    Episodes -> stores past events with dates. New entries are added.
                Answers: "What happened before, and when?"
    Why useful? A profile stores current facts, but episodes remember
    past events and conversations.

Flow:
    Write  -> a session ends -> model summarizes what happened and the result
              -> add the date -> save the entry in the store
    Recall -> choose a time range -> load matching entries, oldest first
              -> add those entries to the prompt so the model can answer

Time-aware recall:
    Memory is filtered by DATE, not just by meaning.
    Example: "What did we discuss recently?"
    -> load only recent entries, such as the last 7 days.

Good:
    Keeps a history of past events.
    Answers questions about what happened, when it happened, and what changed.

Bad:
    The diary keeps growing (Topic 21: forgetting old memories).
    The model may create an incomplete or incorrect summary.
    Here, the code manually chooses the time range.
    An agent should work out the correct time range from the question
    (Topic 19: routing).
"""

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from agent_memory.config import get_model

TODAY = date(2026, 10, 14)  # fixed, so results are repeatable
NAMESPACE = ("users", "drv", "episodes")

class Episode(BaseModel):
    # One diary entry for a chat session.
    summary: str = Field(description="1-2 sentences: what happened in this session")
    outcome: str = Field(description="Short result, e.g. 'finished', 'stuck on X', 'decided Y'")


WRITE_PROMPT = """Write a diary entry for this chat session.
Focus on what the user did, decided, finished or got stuck on.

Session:
{transcript}"""

# Past sessions: (date, transcript)
SESSIONS = [
    (date(2026, 8, 20),
     "User: I live in Delhi. Starting an expense tracker app with FastAPI.\n"
     "AI: Nice! Start with a simple /expenses endpoint."),
    (date(2026, 9, 15),
     "User: The expense tracker is done and deployed!\n"
     "AI: Congrats! Consider adding monthly reports next."),
    (date(2026, 10, 5),
     "User: Moved to Mumbai this week, settling in.\n"
     "AI: Hope the move went smoothly!"),
    (date(2026, 10, 10),
     "User: Started a chatbot project. Stuck on streaming the responses.\n"
     "AI: Try the model's astream() method to send tokens as they arrive."),
]

def write_episodes(store: InMemoryStore, writer) -> None:
    # Turn each session into one diary entry and save it.
    for day, transcript in SESSIONS:
        ep = writer.invoke(WRITE_PROMPT.format(transcript=transcript))
        store.put(NAMESPACE, day.isoformat(), {"date": day.isoformat(), **ep.model_dump()})
        print(f"  {day}  {ep.summary}  [{ep.outcome}]")

def recall(store: InMemoryStore, last_days: int | None = None) -> list[dict]:
    # Load entries, optionally only the last N days, oldest first.
    entries = [item.value for item in store.search(NAMESPACE, limit=100)]
    if last_days is not None:
        entries = [e for e in entries
                   if (TODAY - date.fromisoformat(e["date"])).days <= last_days]
    return sorted(entries, key=lambda e: e["date"])

def ask(model, question: str, entries: list[dict]) -> None:
    diary = "\n".join(
        f"- {e['date']} ({(TODAY - date.fromisoformat(e['date'])).days} days ago): "
        f"{e['summary']} Outcome: {e['outcome']}"
        for e in entries
    )
    reply = model.invoke([
        SystemMessage(f"Today is {TODAY}. Past sessions with the user:\n{diary}\n"
                      "Answer in two lines."),
        HumanMessage(question),
    ])
    print(f"Q: {question}")
    print(f"   (sent {len(entries)} of {len(SESSIONS)} entries)")
    print(f"A: {reply.content}\n")

def main() -> None:
    model = get_model()
    writer = model.with_structured_output(Episode, method="function_calling")
    store = InMemoryStore()

    print("Writing diary entries:")
    write_episodes(store, writer)
    print()

    # The history the profile card threw away
    ask(model, "What have I built before, and where did I used to live?", recall(store))

    # Time-aware: only the last 7 days
    ask(model, "What was I stuck on recently?", recall(store, last_days=7))

if __name__ == "__main__":
    main()