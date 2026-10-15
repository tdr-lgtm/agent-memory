
"""
Topic 11: Retrieval Scoring (Relevance + Recency + Importance)

Example:
    User asks: "Where do I live?"

    Memory A (400 days ago): "I live in Delhi."
    Memory B (3 days ago):   "I moved to Mumbai."

    Both match the question, but Memory B is newer.

Three scores for each memory (0 to 1):
    Relevance   -> How closely the memory matches the question.
    Recency     -> How recent the memory is.
                   Older memories get lower scores.
    Importance  -> How important the memory is.
                   A rating from 1 to 10, divided by 10.

Recency formula:
    0.5 ** (days_ago / HALF_LIFE_DAYS)

    If HALF_LIFE_DAYS = 30, the score becomes 0.5 after 30 days.

Idea:
    Based on the "Generative Agents" paper (Park et al., 2023).

Problem with scoring everything together:
    A fresh but unrelated memory can rank above a relevant memory.

Examples:
    "Where do I live?" -> A recent pizza memory ranked above Delhi.
    "Cook dinner?"     -> "I moved to Mumbai" ranked above
                          "I don't eat spicy food."

Why does this happen?
    Relevance has a maximum score of 1.
    Recency and importance can add up to 2.
    So, a recent and important memory can win even when
    it is not relevant to the question.

    Also, old facts lose their recency score.
    For example, "I don't eat spicy food" may still be true
    after 200 days, even though its recency score is low.

Fix: Two-stage retrieval
    Stage 1: Find candidate memories using relevance only.
    Stage 2: Reorder those candidates using relevance,
             recency, and importance.

    This ensures that recency and importance can reorder
    relevant memories but cannot introduce unrelated ones.

Reranking:
    Sorting a shortlist of retrieved memories again using
    additional scores or information.
"""

from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HALF_LIFE_DAYS = 30
CANDIDATES = 3  # stage 1: how many to keep by relevance
TOP_K = 2  # stage 2: how many to send after reranking

W_RELEVANCE = 1.0
W_RECENCY = 1.0
W_IMPORTANCE = 1.0

@dataclass
class Memory:
    text: str
    days_ago: float
    importance: int  # 1 (trivial) to 10 (critical)

MEMORIES = [
    Memory("Drv lives in Delhi.", days_ago=400, importance=7),
    Memory("Drv just moved to Mumbai.", days_ago=3, importance=8),
    Memory("Drv had pizza for lunch yesterday.", days_ago=1, importance=2),
    Memory("Drv doesn't eat spicy food.", days_ago=200, importance=8),
    Memory("Drv's favourite drink is masala chai.", days_ago=150, importance=3),
    Memory("Drv prefers Python over JavaScript.", days_ago=90, importance=6),
    Memory("Drv's laptop has 8 GB of RAM and no GPU.", days_ago=60, importance=6),
]

QUERIES = [
    "Where do I live?",
    "What should I cook for dinner tonight?",
]

def embed_all(embedder: TextEmbedding, texts: list[str]) -> np.ndarray:
    # Embed texts and scale each vector to length 1.
    vecs = np.array(list(embedder.embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def score(query_vec: np.ndarray, memory_vecs: np.ndarray) -> dict[str, np.ndarray]:
    # Return relevance, recency, importance and total for every memory.
    cosine = memory_vecs @ query_vec
    relevance = (cosine - cosine.min()) / (cosine.max() - cosine.min())

    days = np.array([m.days_ago for m in MEMORIES])
    recency = 0.5 ** (days / HALF_LIFE_DAYS)

    importance = np.array([m.importance for m in MEMORIES]) / 10

    total = W_RELEVANCE * relevance + W_RECENCY * recency + W_IMPORTANCE * importance
    return {"rel": relevance, "rec": recency, "imp": importance, "total": total}

def one_stage(s: dict[str, np.ndarray]) -> np.ndarray:
    # Score everything, sort by total. (My first version.)
    return np.argsort(s["total"])[::-1][:TOP_K]

def two_stage(s: dict[str, np.ndarray]) -> np.ndarray:
    # Filter by relevance first, then rerank by total.
    candidates = np.argsort(s["rel"])[::-1][:CANDIDATES]
    reranked = sorted(candidates, key=lambda i: s["total"][i], reverse=True)
    return np.array(reranked[:TOP_K])

def show(title: str, picks: np.ndarray, s: dict[str, np.ndarray]) -> None:
    print(f"  {title}")
    print(f"    {'rel':>5} {'rec':>5} {'imp':>5} {'total':>6}  memory")
    for i in picks:
        print(
            f"    {s['rel'][i]:>5.2f} {s['rec'][i]:>5.2f} {s['imp'][i]:>5.2f} "
            f"{s['total'][i]:>6.2f}  {MEMORIES[i].text}"
        )

def main() -> None:
    print("Loading embedding model...\n")
    embedder = TextEmbedding(EMBED_MODEL)
    memory_vecs = embed_all(embedder, [m.text for m in MEMORIES])

    for query in QUERIES:
        query_vec = embed_all(embedder, [query])[0]
        s = score(query_vec, memory_vecs)

        print(f"Q: {query}")
        show("One stage (score all, sort):", one_stage(s), s)
        show("Two stage (filter by relevance, then rerank):", two_stage(s), s)
        print()


if __name__ == "__main__":
    main()