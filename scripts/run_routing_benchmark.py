#!/usr/bin/env python
# 300 held-out tasks, each with a hidden true_difficulty (what the backends' own accuracy
# tables respond to) and a noisy proxy_score (what the router actually sees). Compares a
# baseline that always calls the expensive, high-accuracy backend against a
# difficulty-based router, at the same random outcome draws so the two conditions are
# compared call-for-call, not just in aggregate. ROUTING_THRESHOLDS is this project's own
# design knob: explored over several cutoffs (see README Findings) looking for the point
# closest to equal pass rate versus the always-frontier baseline, then held fixed; the
# backends' own accuracy tables were never touched to chase a number.
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmgw.backends import FAST_SIM, FRONTIER_SIM
from llmgw.routing import classify_difficulty, generate_tasks
from llmgw.tracing import price_request

N_TASKS = 300
SEED = 20260801
ROUTING_THRESHOLDS = (1.5, 1.5)  # see module docstring and README Findings


def run() -> dict:
    tasks = generate_tasks(N_TASKS, SEED)

    outcome_rng_baseline = random.Random(SEED + 1)
    base_passed = 0
    base_cost = 0.0
    for task in tasks:
        response = FRONTIER_SIM.complete("prompt", task.input_tokens, 0, task.true_difficulty, outcome_rng_baseline)
        base_passed += 1 if response.passed else 0
        base_cost += price_request(FRONTIER_SIM, task.input_tokens, 0, response.output_tokens).cost_usd

    outcome_rng_routed = random.Random(SEED + 1)  # same draw sequence as the baseline loop
    routed_passed = 0
    routed_cost = 0.0
    frontier_calls = 0
    for task in tasks:
        predicted = classify_difficulty(task.proxy_score, ROUTING_THRESHOLDS)
        backend = FRONTIER_SIM if predicted == "hard" else FAST_SIM
        frontier_calls += 1 if backend is FRONTIER_SIM else 0
        response = backend.complete("prompt", task.input_tokens, 0, task.true_difficulty, outcome_rng_routed)
        routed_passed += 1 if response.passed else 0
        routed_cost += price_request(backend, task.input_tokens, 0, response.output_tokens).cost_usd

    report = {
        "n_tasks": N_TASKS,
        "routing_thresholds": ROUTING_THRESHOLDS,
        "baseline_always_frontier": {
            "pass_rate": round(base_passed / N_TASKS, 4),
            "cost_usd": round(base_cost, 4),
        },
        "difficulty_based_routing": {
            "pass_rate": round(routed_passed / N_TASKS, 4),
            "cost_usd": round(routed_cost, 4),
            "frontier_calls": frontier_calls,
            "fast_calls": N_TASKS - frontier_calls,
        },
        "pass_rate_gap_pp": round(100 * (base_passed - routed_passed) / N_TASKS, 2),
        "pct_cost_reduction": round(100 * (base_cost - routed_cost) / base_cost, 2),
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
