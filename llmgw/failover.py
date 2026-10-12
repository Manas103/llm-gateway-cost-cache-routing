# Simulated primary-backend failure plus fallback to a secondary backend. Fault injection
# only (no real backend process is killed, consistent with this project's process-safety
# rules): fail_rate decides whether this call's primary attempt is a simulated fault, in
# which case a detection delay (the time a real gateway would spend noticing the primary
# call errored or timed out) is added before the secondary call runs.
from __future__ import annotations

import random
from dataclasses import dataclass

from .backends import Backend, Difficulty


@dataclass(frozen=True)
class FailoverResult:
    used_backend: str
    failed_over: bool
    total_latency_ms: float


def call_with_failover(
    primary: Backend,
    secondary: Backend,
    *,
    prompt: str,
    input_tokens: int,
    difficulty: Difficulty,
    fail_rate: float,
    detect_ms_mean: float,
    detect_ms_std: float,
    rng: random.Random,
) -> FailoverResult:
    if rng.random() < fail_rate:
        detect_ms = max(1.0, rng.gauss(detect_ms_mean, detect_ms_std))
        response = secondary.complete(prompt, input_tokens, 0, difficulty, rng)
        return FailoverResult(used_backend=secondary.name, failed_over=True, total_latency_ms=detect_ms + response.latency_ms)
    response = primary.complete(prompt, input_tokens, 0, difficulty, rng)
    return FailoverResult(used_backend=primary.name, failed_over=False, total_latency_ms=response.latency_ms)
