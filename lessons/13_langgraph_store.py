
"""
Topic 13: LangGraph Store (Long-Term Memory Across Chats)

Example:
    Chat history -> Belongs to one conversation (checkpointer).
    Contact card -> Belongs to the user and can be used in any chat (store).

Checkpointer vs. Store:
    Checkpointer -> Saves conversation state for a specific thread_id.
    Store        -> Saves user information using a user_id.
                   The information can be used in new conversations.

Important terms:
    Store      -> Saves long-term memory as key-value data.
    Namespace  -> A folder that groups memories, such as ("users", "drv").
                  Each user has a separate namespace.
    put()      -> Saves a memory item.
    get()      -> Loads a memory item.

How it works:
    START -> chatbot -> update_profile -> END

    chatbot:
        Reads the user's profile from the store.
        Adds the profile to the prompt and generates a reply.

    update_profile:
        Extracts facts from the conversation.
        Updates the profile and saves it back to the store.

    Nodes can access the store through a "store" parameter.
    LangGraph provides it when the node runs.
    The user_id comes from the configuration, alongside thread_id.

Storage:
    InMemoryStore -> Keeps data in memory while the program runs.
                     Data is lost when the program ends.
    PostgresStore  -> Saves data in a database so it can persist
                      across program runs.

Good:
    Memory can be reused across different conversations.
    Each user's information stays in a separate namespace.

Bad:
    Extracting facts requires an additional model call.
    The agent can only store the information defined in its
    profile schema.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from agent_memory.config import get_model


class Profile(BaseModel):
    # Facts about the user. Leave a field empty if not mentioned.
    name: str | None = Field(None, description="The user's name")
    city: str | None = Field(None, description="City where the user currently lives")
    diet: str | None = Field(None, description="Food preferences or restrictions")
    language: str | None = Field(None, description="Programming language the user prefers")
    project: str | None = Field(None, description="What the user is currently building")

EXTRACT_PROMPT = """Extract facts the user states about THEMSELVES in this message.
Only fill a field if this message clearly mentions it. Otherwise leave it empty.

Message: {message}"""

model = get_model()
extractor = model.with_structured_output(Profile, method="function_calling")

def load_profile(store: BaseStore, user_id: str) -> Profile:
    # Read this user's card from the store (empty card if none yet)
    item = store.get(("users", user_id), "profile")
    return Profile(**item.value) if item else Profile()

def chatbot(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    # Read the profile, put it in the prompt, reply.
    user_id = config["configurable"]["user_id"]
    profile = load_profile(store, user_id)

    card = "\n".join(f"- {k}: {v}" for k, v in profile.model_dump().items() if v)
    system = SystemMessage(
        f"What you know about the user:\n{card or '(nothing yet)'}\n"
        "Answer in one or two lines."
    )
    reply = model.invoke([system] + state["messages"])
    return {"messages": [reply]}

def update_profile(state: MessagesState, config: RunnableConfig, *, store: BaseStore) -> dict:
    # Extract facts from the newest user message and save them.
    user_id = config["configurable"]["user_id"]
    latest_user_msg = [m for m in state["messages"] if m.type == "human"][-1]

    update = extractor.invoke(EXTRACT_PROMPT.format(message=latest_user_msg.content))
    filled = {k: v for k, v in update.model_dump().items() if v is not None}

    if filled:
        profile = load_profile(store, user_id).model_copy(update=filled)
        store.put(("users", user_id), "profile", profile.model_dump())
        print(f"    (saved to store: {filled})")
    return {}

def build_graph(store: BaseStore):
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("update_profile", update_profile)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", "update_profile")
    builder.add_edge("update_profile", END)
    return builder.compile(checkpointer=InMemorySaver(), store=store)

def chat(graph, user_id: str, thread_id: str, text: str) -> None:
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    print(f"[{user_id} / {thread_id}] You: {text}")
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    print(f"[{user_id} / {thread_id}] AI:  {result['messages'][-1].content}\n")

def main() -> None:
    store = InMemoryStore()
    graph = build_graph(store)

    # Chat 1: Drv shares facts
    chat(graph, "drv", "chat-1", "Hi, I'm Drv. I live in Delhi and I'm building an expense tracker.")
    chat(graph, "drv", "chat-1", "Update: I just moved to Mumbai.")

    # Chat 2: SAME user, BRAND-NEW thread -> no old messages, but the store remembers
    chat(graph, "drv", "chat-2", "Where do I live and what am I building?")

    # Chat 3: DIFFERENT user -> different namespace -> knows nothing
    chat(graph, "guest", "chat-3", "Where do I live and what am I building?")

    print("What's in the store:")
    for item in store.search(("users",)):
        print(f"  {item.namespace} / {item.key}: {item.value}")

if __name__ == "__main__":
    main()