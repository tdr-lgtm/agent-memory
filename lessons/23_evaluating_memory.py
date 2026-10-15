"""
Topic 23: Evaluating Memory

Problem:
    So far we ran each lesson ONCE and eyeballed it.
    - Models vary every run -> one good run can be LUCK
    - "Looks right" is not a number -> can't compare versions

Example (cricket coach):
    Don't judge a new technique by ONE ball.
    Watch many balls, count good ones, compare with the old technique.

Flow:
    1. test questions, each with a CHECK for a good answer
    2. run each WITH memory and WITHOUT memory (baseline)
    3. repeat RUNS times -> averages out luck
    4. count passes, compare scores

Jargon:
    Eval     -> a repeatable test that gives a score
    Baseline -> the version you compare against (here: no memory)

Checks here are simple code (keywords, word count): cheap and exact.
For fuzzy quality, real systems use "LLM-as-judge" (a model grades answers).

Good:
    Real numbers. Catches luck. Lets you compare any two memory setups.
Bad:
    Only as good as your tests. Costs many model calls.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agent_memory.config import get_model

RUNS = 3  # repeat each test to average out luck (set to 1 for a faster run)

MEMORY = [
    "Name: Drv",
    "Lives in Mumbai",
    "Is vegetarian",
    "Prefers Python",
    "Prefers short answers",
    "2026-01-15: finished and deployed an expense tracker",
]

MEAT = ["chicken", "fish", "salmon", "mutton", "beef", "pork", "prawn", "egg", "meat"]

# (question, check(answer) -> pass?)
TESTS = [
    ("Where do I live?",
     lambda a: "mumbai" in a.lower()),
    ("Suggest one dinner dish for me tonight.",
     lambda a: not any(w in a.lower() for w in MEAT)),
    ("Which programming language should I use for a small script?",
     lambda a: "python" in a.lower()),
    ("Explain what recursion is.",
     lambda a: len(a.split()) <= 60),
    ("What did I finish in September?",
     lambda a: "expense" in a.lower()),
]

def ask(model, question: str, use_memory: bool) -> str:
    messages = []
    if use_memory:
        notes = "\n".join(f"- {m}" for m in MEMORY)
        messages.append(SystemMessage(f"What you know about the user:\n{notes}"))
    messages.append(HumanMessage(question))
    return model.invoke(messages).content

def main() -> None:
    model = get_model()
    totals = {"without": 0, "with": 0}

    print(f"{'question':<62}{'without':>9}{'with':>7}")
    print("-" * 78)

    for question, check in TESTS:
        scores = {}
        for mode, use_memory in [("without", False), ("with", True)]:
            passed = sum(check(ask(model, question, use_memory)) for _ in range(RUNS))
            scores[mode] = passed
            totals[mode] += passed
        print(f"{question:<62}{scores['without']:>6}/{RUNS}{scores['with']:>5}/{RUNS}")

    total = len(TESTS) * RUNS
    print("-" * 78)
    print(f"{'TOTAL':<62}{totals['without']:>6}/{total}{totals['with']:>5}/{total}")
    print(f"\nWithout memory: {totals['without'] / total:.0%}")
    print(f"With memory:    {totals['with'] / total:.0%}")

if __name__ == "__main__":
    main()