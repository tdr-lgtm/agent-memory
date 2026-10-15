
"""
Topic 8: Trimming by Tokens (Token Buffer)

Example:
    Think of packing a bag.

    Bag A: Maximum 5 items.
            Counts messages. (Topic 4)

    Bag B: Maximum 7 kg.
            Counts how much space or weight is used.
            This topic limits tokens, not messages.

Token buffer:
    Keeps the newest messages until the token limit is reached.
    A long message uses more tokens than a short message.

trim_messages (LangChain):
    strategy="last"  -> Keep messages starting from the newest.
    max_tokens=500   -> Maximum number of tokens to keep.
    token_counter    -> Counts the tokens in messages.
    count_tokens_approximately -> Estimates token usage without
                                  calling the model API.
    start_on="human" -> Makes the kept history start with a
                        human message.

Important:
    The checkpointer still stores the full conversation.
    Trimming only decides which messages are sent to the model.

Watch out:
    Trimming keeps one continuous block of recent messages.
    It does not skip a large message to keep older messages.
    A large message can push many older messages out.

Good:
    Limits the size of the conversation sent to the model.
    Long messages use more of the token budget.

Bad:
    Older information may be lost from the messages sent to the model.
    Important facts, such as the user's name, may be forgotten.

"""

from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from agent_memory.config import get_model

MAX_TOKENS = 500  # budget for what we send each turn

USER_TURNS = [
    "Hi, my name is Drv and I live in India.",
    "I'm learning about agent memory.",
    "Explain what a token is in two sentences.",
    "Give me one tip for writing clean Python.",
    "Suggest one name for a snack shop.",
    "What's my name and where do I live?",
]

model = get_model()

def chatbot(state: MessagesState) -> dict:
    # Trim stored messages to the token budget, then call the model.
    stored = state["messages"]
    to_send = trim_messages(
        stored,
        strategy="last",
        max_tokens=MAX_TOKENS,
        token_counter=count_tokens_approximately,
        start_on="human",
    )

    stored_tokens = count_tokens_approximately(stored)
    sent_tokens = count_tokens_approximately(to_send)
    print(
        f"stored: {len(stored):>2} msgs / ~{stored_tokens:>5} tokens   "
        f"sent: {len(to_send):>2} msgs / ~{sent_tokens:>4} tokens"
    )

    reply = model.invoke(to_send)
    return {"messages": [reply]}

def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=InMemorySaver())

def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "chat-A"}}

    for turn, text in enumerate(USER_TURNS, start=1):
        print(f"turn {turn}  ", end="")
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)

    print("\nLast answer:", result["messages"][-1].content)

if __name__ == "__main__":
    main()