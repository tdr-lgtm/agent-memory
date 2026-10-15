"""
Topic 17: Self-Reflection Memory (learning from mistakes)

Easy example:
    A cricket team loses a match.
    The coach asks, "What went wrong?"
    They discover that players struggled against short balls.
    Lesson: "Practise playing short balls."
    In the next match, they use this lesson to improve.

Self-reflection memory = an AI agent learns lessons from its own mistakes.

Postmortem = reviewing a failed attempt to understand what went wrong
             and how to avoid the same mistake next time.

Procedural (Topic 16) vs Self-reflection:
    Procedural   -> user gives feedback.
                    Example: "Your answer is too long."
                    Agent updates its instructions.
    Self-reflection -> agent detects its own mistake using tests.
                       Agent writes a lesson for future tasks.

Flow:
    1. Attempt  -> agent writes code using any saved lessons.
    2. Check    -> run tests to see whether the code works.
    3. Reflect -> if tests fail, the agent identifies a general lesson.
    4. Save     -> store the lesson in memory and try again.
    5. Transfer -> use the saved lesson on a different task.

Important idea:
    The lesson should teach a general rule, not just fix one specific bug.
    This idea is demonstrated in the Reflexion paper (Shinn et al., 2023).

Example of transfer:
    Task 1: Write normalize_email().
            The agent learns: "Return None for invalid input."
    Task 2: Write normalize_phone().
            The agent uses the same lesson for invalid phone numbers.

    We test Task 2 twice:
        Without lessons -> the agent starts without previous lessons.
        With lessons    -> the agent receives lessons from Task 1.

    Comparing both results helps us see whether the lesson transfers.

Good:
    The agent can learn from failed attempts.
    Lessons can help with different tasks in the future.

Bad:
    It needs tests or another checker to detect mistakes.
    Without a checker, the agent may not know that it failed.
    A wrong lesson can harm many future tasks.

Safety:
    exec() runs Python code dynamically.
    It is acceptable for a small, trusted local demonstration, but running
    model-generated code in a real application requires a secure sandbox.
"""

import re

from langgraph.store.memory import InMemoryStore

from agent_memory.config import get_model

NAMESPACE = ("agent", "lessons")
MAX_ATTEMPTS = 3

TASKS = {
    "normalize_email": {
        "spec": "Write a Python function normalize_email(email) that cleans up an email address (lowercase the whole address, remove surrounding spaces).",
        "tests": [
            ("  Drv@Example.COM ", "drv@example.com"),
            ("test@site.org", "test@site.org"),
            ("not-an-email", None),
            ("", None),
        ],
    },
    "normalize_phone": {
        "spec": "Write a Python function normalize_phone(phone) that cleans up a phone number.",
        "tests": [
            ("+91 98765-43210", "+919876543210"),
            ("98765 43210", "9876543210"),
            ("abc", None),
            ("   ", None),
        ],
    },
}

ATTEMPT_PROMPT = """{spec}

{lessons}Return ONLY the function inside a ```python code block."""

REFLECT_PROMPT = """You wrote this code:
{code}

It failed these tests:
{failures}

Write ONE short, GENERAL lesson (one sentence) that would help you
avoid this kind of mistake on FUTURE, DIFFERENT tasks.
Do not mention this function's name. Return only the lesson."""

def extract_code(text: str) -> str:
    # Pull the code out of a ```python block (or use the whole text).
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text

def run_tests(code: str, func_name: str, tests: list) -> list[str]:
    """Run the model's code against the tests. Return a list of failures."""
    namespace = {}
    try:
        exec(code, namespace)
        func = namespace[func_name]
    except Exception as e:
        return [f"code did not run: {e!r}"]

    failures = []
    for arg, expected in tests:
        try:
            got = func(arg)
        except Exception as e:
            got = f"error: {e!r}"
        if got != expected:
            failures.append(f"{func_name}({arg!r}) returned {got!r}, expected {expected!r}")
    return failures


def load_lessons(store: InMemoryStore) -> list[str]:
    return [item.value["lesson"] for item in store.search(NAMESPACE, limit=100)]

def solve(model, store: InMemoryStore, name: str, use_lessons: bool, learn: bool) -> None:
    task = TASKS[name]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        lessons = load_lessons(store) if use_lessons else []
        lessons_text = (
            "Lessons from your past mistakes:\n" + "\n".join(f"- {l}" for l in lessons) + "\n\n"
            if lessons else ""
        )

        reply = model.invoke(ATTEMPT_PROMPT.format(spec=task["spec"], lessons=lessons_text))
        code = extract_code(reply.content)
        failures = run_tests(code, name, task["tests"])

        passed = len(task["tests"]) - len(failures) if not failures or "did not run" not in failures[0] else 0
        print(f"  attempt {attempt} (lessons: {len(lessons)}): {passed}/{len(task['tests'])} tests passed")
        for f in failures:
            print(f"     x {f}")

        if not failures:
            print("  -> PASSED\n")
            return

        if not learn:
            print("  -> stopping (learning off for this run)\n")
            return

        # reflect: write and save a general lesson
        lesson = model.invoke(REFLECT_PROMPT.format(
            code=code, failures="\n".join(failures)
        )).content.strip()
        store.put(NAMESPACE, f"lesson-{len(lessons)}", {"lesson": lesson})
        print(f"  reflection -> {lesson}")

    print("  -> gave up after max attempts\n")

def main() -> None:
    model = get_model()
    store = InMemoryStore()

    print("TASK 1: normalize_email (learning ON)")
    solve(model, store, "normalize_email", use_lessons=True, learn=True)

    print("TASK 2: normalize_phone WITHOUT lessons (one attempt)")
    solve(model, store, "normalize_phone", use_lessons=False, learn=False)

    print("TASK 2: normalize_phone WITH lessons from task 1 (one attempt)")
    solve(model, store, "normalize_phone", use_lessons=True, learn=False)

    print("Lessons in the store:")
    for lesson in load_lessons(store):
        print(f"  - {lesson}")

if __name__ == "__main__":
    main()