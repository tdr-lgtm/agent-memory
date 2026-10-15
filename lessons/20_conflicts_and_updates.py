"""
Topic 20: Conflicts and Updates

Two problems that make memory messy:
    Clash     -> old "lives in Delhi" vs new "moved to Mumbai"
    Duplicate -> old "is vegetarian" vs new "doesn't eat meat"

Fix: before saving, compare the new fact with saved facts.
Pick ONE action:
    ADD    -> brand new fact                     ("started learning Rust")
    UPDATE -> replaces an old fact               (Delhi -> Mumbai)
    NOOP   -> duplicate, do nothing              ("doesn't eat meat")
    DELETE -> old fact no longer true, nothing replaces it ("quit the gym")

NOOP = "no operation" = do nothing.

Old facts from UPDATE/DELETE go to HISTORY -> nothing is truly lost.
(Fixes Topic 12, where "lived in Delhi" vanished.)

Same pattern used by memory tools like Mem0.

Good:
    No clashes, no copies. Memory stays small and current.
Bad:
    One extra model call per new fact.
    A wrong UPDATE overwrites a fact that was still true.
"""

from typing import Literal

from pydantic import BaseModel, Field

from agent_memory.config import get_model

class Decision(BaseModel):
    action: Literal["ADD", "UPDATE", "NOOP", "DELETE"]
    target_id: int | None = Field(None, description="ID of the saved fact to UPDATE or DELETE")
    text: str | None = Field(None, description="New fact text for ADD or UPDATE")
    reason: str = Field(description="One short sentence: why")


DECIDE_PROMPT = """You manage a user's memory. Decide what to do with a NEW fact.

Saved facts:
{facts}

NEW fact: {new}

Actions:
- ADD: new information, not covered by any saved fact
- UPDATE: replaces or changes a saved fact (give its target_id and the new text)
- NOOP: means the same as a saved fact (duplicate)
- DELETE: makes a saved fact untrue, and nothing replaces it (give its target_id)"""

NEW_FACTS = [
    "Drv moved to Mumbai.",
    "Drv doesn't eat meat.",
    "Drv started learning Rust.",
    "Drv quit going to the gym.",
]

def main() -> None:
    model = get_model()
    decider = model.with_structured_output(Decision, method="function_calling")

    facts = {
        0: "Drv lives in Delhi.",
        1: "Drv is vegetarian.",
        2: "Drv goes to the gym on weekday mornings.",
    }
    history = []
    next_id = 3

    for new in NEW_FACTS:
        facts_text = "\n".join(f"[{i}] {t}" for i, t in facts.items())
        d = decider.invoke(DECIDE_PROMPT.format(facts=facts_text, new=new))

        print(f"NEW: {new}")
        print(f"  -> {d.action}  {('target ' + str(d.target_id)) if d.target_id is not None else ''}")
        print(f"     ({d.reason})")

        if d.action == "ADD":
            facts[next_id] = d.text or new
            next_id += 1
        elif d.action == "UPDATE" and d.target_id in facts:
            history.append(facts[d.target_id])
            facts[d.target_id] = d.text or new
        elif d.action == "DELETE" and d.target_id in facts:
            history.append(facts.pop(d.target_id))
        # NOOP: do nothing

        print()

    print("Memory now:")
    for i, t in facts.items():
        print(f"  [{i}] {t}")
    print("History (old facts):")
    for t in history:
        print(f"  - {t}")

if __name__ == "__main__":
    main()