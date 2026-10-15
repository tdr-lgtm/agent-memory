"""
Topic 7: LangGraph Checkpointer and Threads

Example (WhatsApp):
    Separate chats. Open one -> the app shows old messages.
    You don't store them yourself. Chats never mix.

Until now WE were the app: we kept a Python list and passed it in.
Now LangGraph does it.

Checkpointer:
    Stores the conversation after every turn, loads it back next turn.
    InMemorySaver = stored in RAM, gone when the script ends.
    (Real apps use SqliteSaver / PostgresSaver to keep it on disk.)

Thread:
    One separate conversation, named by a thread_id.
    Same thread_id -> same history. New thread_id -> fresh start.

Big change:
    We only send the NEW message. LangGraph loads the old ones.

MessagesState:
    Built-in state with one key, "messages".
    New messages are APPENDED to the list, not replacing it.

Note:
    This is still buffer memory (Topic 3) underneath: every stored
    message is sent every turn. Topic 8 adds trimming.
"""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from agent_memory.config import get_model

model = get_model()

def chatbot(state: MessagesState) -> dict:
    # One node: send all stored messages to the model, return its reply.
    reply = model.invoke(state["messages"])
    return {"messages": [reply]}

def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=InMemorySaver())

def ask(graph, thread_id: str, text: str) -> None:
    # Send ONE new message to a thread and print the reply.
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    stored = len(result["messages"])
    print(f"[{thread_id}] You: {text}")
    print(f"[{thread_id}] AI:  {result['messages'][-1].content}")
    print(f"[{thread_id}] messages stored in this thread: {stored}\n")


def main() -> None:
    graph = build_graph()

    # Thread A: introduce, then ask
    ask(graph, "chat-A", "Hi, my name is Drv and I live in India. Reply in one line.")
    ask(graph, "chat-A", "What's my name and where do I live?")

    # Thread B: brand-new conversation, same question
    ask(graph, "chat-B", "What's my name and where do I live?")


if __name__ == "__main__":
    main()