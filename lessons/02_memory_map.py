"""
Topic 2: The Memory Map

Memory helps an AI agent retain and use information.

Four types of memory:
- Working: Information needed for the current task.
- Episodic: Past events and experiences.
- Semantic: Facts and general knowledge.
- Procedural: Instructions, methods, and skills.

Short-term memory:
- Temporarily holds information for the current task.

Long-term memory:
- Retains information for future tasks, usually using persistent storage.

Example: Cooking khaman
- Working: user just asked about Python.
- Episodic: Tuesday: code failed on import.
- Semantic: user's name is Drv.
- Procedural: explain with example first.
"""

SEMANTIC = [
    "User's name is drv.",
    "User uses OpenRouter with gpt-oss-120b.",
]

EPISODIC = [
    "2026-10-14: User's setup failed because files were empty; fixed by re-creating them.",
]

PROCEDURAL = [
    "Explain with a short example before theory.",
    "Use plain English and flag jargon.",
]

# Short-term (working) memory: the current conversation
WORKING = [
    ("human", "Why did my uv sync fail earlier?"),
]

def build_prompt() -> list[tuple[str, str]]:
    """
    Build a prompt containing all four types of agent memory.

    Returns:
        A list of message tuples containing the system prompt with
        procedural, semantic, and episodic memory, followed by
        working memory messages.
    """
    system_text = (
        "## How to behave (procedural)\n"
        + "\n".join(f"- {rule}" for rule in PROCEDURAL)
        + "\n\n## Facts about the user (semantic)\n"
        + "\n".join(f"- {fact}" for fact in SEMANTIC)
        + "\n\n## Past events (episodic)\n"
        + "\n".join(f"- {event}" for event in EPISODIC)
    )
    return [("system", system_text)] + WORKING

def main() -> None:
    for role, text in build_prompt():
        print(f"===== {role} =====")
        print(text)
        print()

if __name__ == "__main__":
    main()