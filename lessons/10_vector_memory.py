"""
Topic 10: Vector Store Memory

Example:
    You ask "What should I cook for dinner?"
    A friend remembers "you don't eat spicy food" -- found by MEANING,
    even though the words don't match.
    A keyword search ("spicy" in text) would miss it.

Phase 2 shift:
    Memory now lives OUTSIDE the chat and is pulled in when relevant.

Jargon:
    Embedding          -> turn text into numbers that capture its meaning
    Vector             -> that list of numbers
    Cosine similarity  -> closeness score between two vectors (higher = closer meaning)
    Top-k              -> keep only the k best matches

Flow:
    save:   memory text -> embedding -> store vector
    recall: question -> embedding -> compare with all stored vectors
            -> take top-k -> put ONLY those into the prompt

Built by hand with numpy to show there's no magic.
Real vector databases (Chroma, pgvector...) do the same thing, faster.

Embedding model runs locally (fastembed). First run downloads ~70 MB.

Good:
    Finds relevant memories by meaning. Prompt only gets what matters.
    Works across chats -- memory is not tied to one thread.
Bad:
    "Similar meaning" isn't always "useful". Can return near-misses.
    No sense of time or importance (Topic 11 adds that).
"""

import numpy as np
from fastembed import TextEmbedding
from langchain_core.messages import HumanMessage, SystemMessage

from agent_memory.config import get_model

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 2

# Facts saved from past chats (long-term memory)
MEMORIES = [
    "Drv lives in India.",
    "Drv prefers Python over JavaScript.",
    "Drv doesn't eat spicy food.",
    "Drv's laptop has 8 GB of RAM and no GPU.",
    "Drv is building an expense tracker app.",
    "Drv's favourite drink is masala chai.",
    "Drv goes to the gym on weekday mornings.",
]

# New questions
QUERIES = [
    "What should I cook for dinner tonight?",
    "Can I run a big AI model on my computer?",
    "Which language should I use for my new project?",
]


class VectorMemory:
    # A tiny vector store: a list of texts and their vectors.
    def __init__(self, embedder: TextEmbedding):
        self.embedder = embedder
        self.texts: list[str] = []
        self.vectors: list[np.ndarray] = []

    def add(self, texts: list[str]) -> None:
        # Embed each text and store it.
        for text, vec in zip(texts, self.embedder.embed(texts)):
            self.texts.append(text)
            self.vectors.append(vec / np.linalg.norm(vec))

    def search(self, query: str, k: int = TOP_K) -> list[tuple[float, str]]:
        # Return the k memories closest in meaning to the query.
        q = next(iter(self.embedder.embed([query])))
        q = q / np.linalg.norm(q)
        scores = np.array(self.vectors) @ q  # one score per memory
        best = np.argsort(scores)[::-1][:k]  # highest scores first
        return [(float(scores[i]), self.texts[i]) for i in best]


def main() -> None:
    print("Loading embedding model (first run downloads it)...\n")
    memory = VectorMemory(TextEmbedding(EMBED_MODEL))
    memory.add(MEMORIES)

    # Part 1: what does retrieval find?
    for query in QUERIES:
        print(f"Q: {query}")
        for score, text in memory.search(query):
            print(f"   {score:.3f}  {text}")
        print()

    # Part 2: use the retrieved memories to answer
    query = QUERIES[0]
    found = [text for _, text in memory.search(query)]
    to_send = [
        SystemMessage("Relevant facts about the user:\n" + "\n".join(f"- {t}" for t in found)),
        HumanMessage(query + " Answer in two lines."),
    ]
    reply = get_model().invoke(to_send)

    print("Sent to model:", found)
    print("Answer:", reply.content)


if __name__ == "__main__":
    main()