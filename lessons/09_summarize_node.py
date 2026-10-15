
"""
Topic 9: Summarize Node (Summary Buffer in LangGraph)

Example:
    Your notebook is full.
    You write important points on a sticky note,
    then remove the old pages.
    The notebook becomes smaller, but important information stays.

What this combines:
    Token budget   -> Limits the size of messages.
    Summary buffer -> Saves important old information in a summary.
    Delete messages -> Removes old messages from storage.

Flow:
    START -> chatbot -> Is the token limit exceeded?
                         Yes -> Summarize old messages -> END
                         No  -> END

New LangGraph concepts:
    Custom state -> Adds a "summary" field to MessagesState.
    Conditional edge -> Chooses the next step based on a condition.
    RemoveMessage -> Deletes messages from stored graph state.

Summarize node:
    1. trim_messages keeps recent messages within the token budget.
    2. Older messages are combined into a short summary.
    3. RemoveMessage deletes the old messages from storage.

Edge case:
    If a new reply is too large to fit the budget,
    it may be summarized and removed from storage.
    The summary keeps its important information.

Good:
    Limits message size, preserves important old facts,
    and keeps stored conversation history small.

Bad:
    Summaries can lose exact details.
    Updating the summary requires an extra model call.
"""

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_memory.config import get_model

MAX_TOKENS = 500  # storage budget before summarizing

USER_TURNS = [
    "Hi, my name is Drv and I live in India.",
    "I'm learning about agent memory.",
    "Explain what a token is in two sentences.",
    "Give me one tip for writing clean Python.",
    "Suggest one name for a snack shop.",
    "What's my name and where do I live?",
]

SUMMARY_PROMPT = """Update the conversation summary.

Current summary:
{summary}

Old messages to fold in:
{transcript}

Write the new summary in under 60 words.
Always keep facts about the user (name, location, goals).
Shorten long explanations to one line."""

model = get_model()


class State(MessagesState):
    summary: str

def chatbot(state: State) -> dict:
    # Send summary (if any) + stored messages to the model.
    summary = state.get("summary", "")
    to_send = []
    if summary:
        to_send.append(SystemMessage(f"Summary of the earlier conversation:\n{summary}"))
    to_send.extend(state["messages"])

    print(
        f"stored: {len(state['messages']):>2} msgs   "
        f"summary: {'yes' if summary else 'no ':<3}   "
        f"sent: ~{count_tokens_approximately(to_send):>4} tokens"
    )

    reply = model.invoke(to_send)
    return {"messages": [reply]}

def over_budget(state: State) -> str:
    # The fork: summarize if storage is too big, otherwise finish.
    if count_tokens_approximately(state["messages"]) > MAX_TOKENS:
        return "summarize"
    return END

def summarize(state: State) -> dict:
    # Fold old messages into the summary, then delete them from storage.
    messages = state["messages"]

    keep = trim_messages(
        messages,
        strategy="last",
        max_tokens=MAX_TOKENS,
        token_counter=count_tokens_approximately,
        start_on="human",
    )
    keep_ids = {m.id for m in keep}
    old = [m for m in messages if m.id not in keep_ids]

    transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
    prompt = SUMMARY_PROMPT.format(
        summary=state.get("summary", "") or "(empty)",
        transcript=transcript,
    )
    new_summary = model.invoke(prompt).content

    print(f"         -> folded {len(old)} old msgs into summary, kept {len(keep)}")

    return {
        "summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in old],
    }

def build_graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", over_budget, ["summarize", END])
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=InMemorySaver())

def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "chat-A"}}

    for turn, text in enumerate(USER_TURNS, start=1):
        print(f"turn {turn}  ", end="")
        result = graph.invoke({"messages": [HumanMessage(text)]}, config)

    print("\nFinal summary:", result.get("summary", ""))
    print("\nLast answer:", result["messages"][-1].content)

if __name__ == "__main__":
    main()