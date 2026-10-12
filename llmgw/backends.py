# 3 backends behind one interface: 2 simulated providers with different price/latency/
# accuracy profiles (fast-sim is cheap and quick but degrades on hard tasks; frontier-sim
# is expensive and slower but holds accuracy on hard tasks), and one real backend that
# genuinely shells out to the `claude` CLI, exercised on a small live sample only
# (tests/test_live_claude_backend.py), never at the 10,000-request or 300-task scale a
# real subprocess call would make slow and non-reproducible to benchmark.
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from typing import Protocol

Difficulty = str  # "easy" | "medium" | "hard"


@dataclass(frozen=True)
class BackendResponse:
    output_tokens: int
    latency_ms: float
    passed: bool | None  # None when no ground truth is available (the replay benchmark)


class Backend(Protocol):
    name: str
    price_per_1k_input: float
    price_per_1k_cached_input: float
    price_per_1k_output: float

    def complete(self, prompt: str, input_tokens: int, cached_prefix_tokens: int, difficulty: Difficulty, rng: random.Random) -> BackendResponse: ...


@dataclass(frozen=True)
class SimulatedBackend:
    """A fast stand-in for a real model API: latency and output length are drawn from a
    fixed distribution, and pass/fail against a task is drawn from this backend's own
    accuracy-by-difficulty table, grounded in a disclosed design choice (fast-sim trades
    accuracy on hard tasks for low price and latency), not a live model's real output."""

    name: str
    price_per_1k_input: float
    price_per_1k_cached_input: float
    price_per_1k_output: float
    latency_ms_mean: float
    latency_ms_std: float
    accuracy_by_difficulty: dict[Difficulty, float]
    output_tokens_mean: float = 180.0
    output_tokens_std: float = 40.0

    def complete(self, prompt: str, input_tokens: int, cached_prefix_tokens: int, difficulty: Difficulty, rng: random.Random) -> BackendResponse:
        latency = max(5.0, rng.gauss(self.latency_ms_mean, self.latency_ms_std))
        output_tokens = max(1, int(rng.gauss(self.output_tokens_mean, self.output_tokens_std)))
        accuracy = self.accuracy_by_difficulty.get(difficulty, 0.5)
        passed = rng.random() < accuracy
        return BackendResponse(output_tokens=output_tokens, latency_ms=latency, passed=passed)


FRONTIER_SIM = SimulatedBackend(
    name="frontier-sim",
    price_per_1k_input=0.0100,
    price_per_1k_cached_input=0.0010,
    price_per_1k_output=0.0300,
    latency_ms_mean=620.0,
    latency_ms_std=140.0,
    accuracy_by_difficulty={"easy": 0.98, "medium": 0.95, "hard": 0.90},
)

FAST_SIM = SimulatedBackend(
    name="fast-sim",
    price_per_1k_input=0.0005,
    price_per_1k_cached_input=0.00005,
    price_per_1k_output=0.0015,
    latency_ms_mean=130.0,
    latency_ms_std=45.0,
    accuracy_by_difficulty={"easy": 0.96, "medium": 0.80, "hard": 0.55},
)


@dataclass(frozen=True)
class ClaudeCLIBackend:
    """Genuinely shells out to `claude -p`; priced the same as frontier-sim for cost
    tracing purposes since this project does not have real per-token billing data for it,
    and used only on a small live sample, never at bulk benchmark scale."""

    name: str = "claude-cli"
    price_per_1k_input: float = 0.0100
    price_per_1k_cached_input: float = 0.0010
    price_per_1k_output: float = 0.0300

    def complete(self, prompt: str, input_tokens: int, cached_prefix_tokens: int, difficulty: Difficulty, rng: random.Random) -> BackendResponse:
        import time

        start = time.monotonic()
        result = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=60)
        latency_ms = (time.monotonic() - start) * 1000.0
        output_tokens = max(1, len(result.stdout.split()))
        return BackendResponse(output_tokens=output_tokens, latency_ms=latency_ms, passed=None)
