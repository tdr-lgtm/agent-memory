
"""
Topic 12: Profile (Entity) Memory

Example:
    Think of a shopkeeper's customer card.

    Name: Drv
    City: Mumbai
    Diet: No spicy food

    If the user moves from Delhi to Mumbai, the city is updated.
    The new value replaces the old value.

Entity:
    The person or thing the agent keeps information about.

Schema:
    The fixed fields used to store information, like name, city,
    and diet.

Structured output:
    The model returns information in a fixed format instead
    of normal text.

Pydantic:
    A Python library used to define and validate the fields.

How it works:
    1. Extract -> Find facts mentioned in the new message.
    2. Merge   -> Update fields with new values. Keep unchanged fields.
    3. Use     -> Add the current profile to the prompt.

Compared to vector memory:
    Vector memory  -> Stores many facts and searches for relevant ones.
    Profile memory -> Stores facts in fixed fields and updates old values.

Good:
    Simple, organized, and easy to update.
    Keeps one current value per field.

Bad:
    Can only store information covered by its fields.
    The model may miss or misunderstand facts.
    Old values are lost unless history is stored separately.
"""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agent_memory.config import get_model


class Profile(BaseModel):
    """Facts about the user. Leave a field empty if not mentioned."""

    name: str | None = Field(None, description="The user's name")
    city: str | None = Field(None, description="City where the user currently lives")
    diet: str | None = Field(None, description="Food preferences or restrictions")
    language: str | None = Field(None, description="Programming language the user prefers")
    project: str | None = Field(None, description="What the user is currently building")


EXTRACT_PROMPT = """Extract facts the user states about THEMSELVES in this message.
Only fill a field if this message clearly mentions it. Otherwise leave it empty.

Message: {message}"""

# Messages from different chats, over time
USER_MESSAGES = [
    "Hi, I'm Drv. I live in Delhi and I'm building an expense tracker.",
    "I don't eat spicy food. Also, I mostly code in Python.",
    "Big news, I just moved to Mumbai!",
    "Finished the expense tracker. Now I'm building a chatbot.",
]

def merge(profile: Profile, update: Profile) -> Profile:
    """Copy only the filled fields from update onto profile."""
    filled = {k: v for k, v in update.model_dump().items() if v is not None}
    return profile.model_copy(update=filled)

def show(profile: Profile) -> None:
    for field, value in profile.model_dump().items():
        print(f"    {field:<9} {value or '-'}")

def main() -> None:
    model = get_model()
    extractor = model.with_structured_output(Profile, method="function_calling")
    profile = Profile()

    # 1 + 2: extract and merge after each message
    for message in USER_MESSAGES:
        update = extractor.invoke(EXTRACT_PROMPT.format(message=message))
        profile = merge(profile, update)

        print(f"User: {message}")
        print(f"  extracted: { {k: v for k, v in update.model_dump().items() if v} }")
        print("  profile now:")
        show(profile)
        print()

    # 3: use the card in a new chat
    question = "Where do I live, and what am I working on? Answer in one line."
    card = "\n".join(f"- {k}: {v}" for k, v in profile.model_dump().items() if v)
    reply = model.invoke(
        [
            SystemMessage(f"What you know about the user:\n{card}"),
            HumanMessage(question),
        ]
    )
    print(f"Q: {question}")
    print(f"A: {reply.content}")

if __name__ == "__main__":
    main()