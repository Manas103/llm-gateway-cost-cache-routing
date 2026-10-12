# A difficulty-based router over a noisy proxy signal, not the task's own hidden true
# difficulty: generate_tasks gives every task a true_difficulty (what actually determines
# each backend's pass probability) and a separate proxy_score (what the router actually
# sees, correlated with true_difficulty but noisy), the same separation
# generate_oracle_corpus's grounded-vs-proposed split uses elsewhere in this portfolio, so
# the router's classification errors are a measured property of this benchmark, not an
# assumption.
from __future__ import annotations

import random
from dataclasses import dataclass

from .backends import Difficulty

DIFFICULTY_WEIGHTS: dict[Difficulty, float] = {"easy": 0.40, "medium": 0.35, "hard": 0.25}
PROXY_MEAN: dict[Difficulty, float] = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
PROXY_STD = 0.55


@dataclass(frozen=True)
class Task:
    task_id: int
    true_difficulty: Difficulty
    proxy_score: float
    input_tokens: int


def generate_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    difficulties = list(DIFFICULTY_WEIGHTS)
    weights = list(DIFFICULTY_WEIGHTS.values())
    tasks = []
    for i in range(n):
        true_difficulty = rng.choices(difficulties, weights=weights, k=1)[0]
        proxy_score = rng.gauss(PROXY_MEAN[true_difficulty], PROXY_STD)
        input_tokens = max(20, int(rng.gauss(220, 60)))
        tasks.append(Task(task_id=i, true_difficulty=true_difficulty, proxy_score=proxy_score, input_tokens=input_tokens))
    return tasks


def classify_difficulty(proxy_score: float, thresholds: tuple[float, float]) -> Difficulty:
    low, high = thresholds
    if proxy_score < low:
        return "easy"
    if proxy_score < high:
        return "medium"
    return "hard"
