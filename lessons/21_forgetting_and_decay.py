"""
Topic 21: Forgetting and Decay (Ebbinghaus curve)

Problem:
    Memory keeps growing. After a year: thousands of facts, mostly junk
    ("had pizza for lunch", "asked about weather once").

Example:
    Your old phone number -> used every day for years -> still remember it.
    A shop's number dialed once -> forgotten in a week.
    Rule: forget what you never use, keep what you use often.

Strength (0 to 1) for each memory:
    strength  = e ^ (-days_since_used / stability)
    stability = BASE_DAYS * (1 + times_used)
    - drops every day it isn't used
    - drops SLOWER the more it has been used
    - below FORGET_BELOW -> removed

Ebbinghaus curve: forgetting curve discovered by Hermann Ebbinghaus (1880s).
Exponential decay: drops fast at first, then slower.

Topic 11 vs this:
    Topic 11 -> old memories only RANKED lower
    Topic 21 -> weak memories are REMOVED, and USING a memory makes it stronger

No model call -- pure maths.

Good:
    Memory stays small. Junk fades on its own. Useful facts survive.
Bad:
    A fact that's still TRUE but rarely used gets forgotten
    (e.g. "laptop has 8 GB RAM"). Important-but-rare facts need protection.
"""

import math
from dataclasses import dataclass

BASE_DAYS = 7  # a never-used memory lasts about a week
FORGET_BELOW = 0.2

@dataclass
class Memory:
    text: str
    days_since_used: int
    times_used: int

    def stability(self) -> float:
        return BASE_DAYS * (1 + self.times_used)

    def strength(self) -> float:
        return math.exp(-self.days_since_used / self.stability())

    def use(self) -> None:
        # Recalling a memory resets the clock and makes it stronger.
        self.days_since_used = 0
        self.times_used += 1

MEMORIES = [
    Memory("Had pizza for lunch.", days_since_used=20, times_used=0),
    Memory("Asked about the weather in Pune.", days_since_used=30, times_used=0),
    Memory("Is vegetarian.", days_since_used=60, times_used=8),
    Memory("Prefers short answers.", days_since_used=90, times_used=12),
    Memory("Started a chatbot project.", days_since_used=4, times_used=1),
    Memory("Laptop has 8 GB RAM.", days_since_used=120, times_used=2),
]

def show(memories: list[Memory]) -> None:
    print(f"  {'strength':>8} {'used':>5} {'days':>5}  memory")
    for m in sorted(memories, key=lambda m: m.strength(), reverse=True):
        flag = "  <- forget" if m.strength() < FORGET_BELOW else ""
        print(f"  {m.strength():>8.2f} {m.times_used:>5} {m.days_since_used:>5}  {m.text}{flag}")

def main() -> None:
    print("All memories:")
    show(MEMORIES)

    kept = [m for m in MEMORIES if m.strength() >= FORGET_BELOW]
    forgotten = [m for m in MEMORIES if m.strength() < FORGET_BELOW]

    print(f"\nKept {len(kept)}, forgot {len(forgotten)}:")
    for m in forgotten:
        print(f"  - {m.text}")

    # Using a memory saves it
    laptop = next(m for m in MEMORIES if "Laptop" in m.text)
    before = laptop.strength()
    laptop.use()
    print(f"\nUser asks about their laptop -> memory used once:")
    print(f"  'Laptop has 8 GB RAM.' strength {before:.2f} -> {laptop.strength():.2f}")
    print(f"  stability now {laptop.stability():.0f} days (was {BASE_DAYS * 3})")

if __name__ == "__main__":
    main()