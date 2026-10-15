"""
Topic 16: Procedural Memory (learning rules for how to behave)

Easy example:
    Day 1: Waiter serves tea with sugar.
           Customer: "No sugar, please."
    Day 2: Waiter remembers the customer's preference.
           (semantic memory: remembers a fact)
    Day 30: Waiter automatically serves tea without sugar.
            (procedural memory: follows a rule)

Procedural memory = rules that change how an agent behaves.

For an AI agent, the system prompt contains its instructions.
Procedural memory updates these instructions based on feedback.

Semantic (Topic 15) vs Procedural:
    Semantic   -> remembers a fact.
                  Example: "The user prefers short answers."
    Procedural -> changes the agent's behaviour.
                  Example: "Always answer in under 3 sentences."

Flow:
    1. Save the initial instructions in the store.
    2. Agent answers using the current instructions.
    3. User gives feedback about the answer.
    4. Model rewrites the instructions using the feedback.
    5. Save the new instructions as a new version.
       Keep old versions so they can be checked or restored.
    6. The next answer uses the updated instructions.

Risk:
    The model may accidentally remove old rules when rewriting instructions.
    Bad feedback may create bad rules that affect every future answer.

Good:
    The agent changes how it behaves, not just what it remembers.

Bad:
    Important rules may disappear or change incorrectly.
    Bad rules may affect all future answers (Topic 22).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from agent_memory.config import get_model

NAMESPACE = ("agent", "instructions")

START_INSTRUCTIONS = "You are a helpful Python tutor."

UPDATE_PROMPT = """You maintain the instructions for an AI assistant.

Current instructions:
{instructions}

New feedback from the user:
{feedback}

Rewrite the instructions so the assistant follows this feedback.
KEEP every existing rule unless the feedback directly changes it.
Return ONLY the new instructions as a short bullet list."""

# (question, feedback given AFTER the answer)
SCRIPT = [
    ("What is a Python decorator?",
     "Too long. Keep answers under 3 sentences and always show a tiny code example."),
    ("What is a Python generator?",
     "I'm a beginner. If you use a technical word, explain it in brackets."),
    ("What is a Python closure?",
     None),
]

def save_version(store: InMemoryStore, version: int, text: str) -> None:
    store.put(NAMESPACE, f"v{version}", {"version": version, "text": text})

def load_latest(store: InMemoryStore) -> dict:
    items = [item.value for item in store.search(NAMESPACE, limit=100)]
    return max(items, key=lambda v: v["version"])

def update_instructions(model, instructions: str, feedback: str) -> str:
    """Current instructions + feedback -> rewritten instructions."""
    prompt = UPDATE_PROMPT.format(instructions=instructions, feedback=feedback)
    return model.invoke(prompt).content.strip()

def main() -> None:
    model = get_model()
    store = InMemoryStore()
    save_version(store, 1, START_INSTRUCTIONS)

    for question, feedback in SCRIPT:
        current = load_latest(store)

        reply = model.invoke([
            SystemMessage(current["text"]),
            HumanMessage(question),
        ])
        answer = reply.content
        words = len(answer.split())
        has_code = "```" in answer

        print("=" * 60)
        print(f"Instructions v{current['version']}:\n{current['text']}\n")
        print(f"Q: {question}")
        print(f"A ({words} words, code example: {'yes' if has_code else 'no'}):")
        print(answer, "\n")

        if feedback:
            print(f"FEEDBACK: {feedback}")
            new_text = update_instructions(model, current["text"], feedback)
            save_version(store, current["version"] + 1, new_text)
            print(f"-> saved instructions v{current['version'] + 1}\n")

    print("=" * 60)
    print("All versions in the store:")
    for item in sorted(store.search(NAMESPACE, limit=100), key=lambda i: i.value["version"]):
        print(f"\n[v{item.value['version']}]\n{item.value['text']}")

if __name__ == "__main__":
    main()