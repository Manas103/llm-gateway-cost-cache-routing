#!/usr/bin/env python
# 10,000-request replay across 5 simulated teams and the 2 simulated backends, half of it
# under the cache-aware layout (static instruction block first) and half under the naive
# layout (variable context first), so the two conditions are measured in the same run
# against the same request mix. Every request is traced into a real DuckDB table and the
# per-team report is a real SQL query against it, not a Python-side running total.
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmgw.backends import FAST_SIM, FRONTIER_SIM
from llmgw.cache import PrefixCache, cache_aware_layout, naive_layout
from llmgw.tracing import connect, per_team_report, price_request, record_request

N_REQUESTS = 10_000
SEED = 20260801
TEAMS = ["ads-platform", "search-infra", "growth", "fraud-risk", "data-platform"]
BACKENDS = [FAST_SIM, FRONTIER_SIM]
STATIC_BLOCK_TOKENS = 300
VARIABLE_TOKENS_MEAN = 200
VARIABLE_TOKENS_STD = 60


def run() -> dict:
    rng = random.Random(SEED)
    con = connect(":memory:")
    cache_aware_cache = PrefixCache()
    naive_cache = PrefixCache()

    cache_aware_billed = 0
    naive_billed = 0

    for i in range(N_REQUESTS):
        team = rng.choice(TEAMS)
        backend = rng.choice(BACKENDS)
        difficulty = rng.choice(["easy", "medium", "hard"])
        variable_tokens = max(10, int(rng.gauss(VARIABLE_TOKENS_MEAN, VARIABLE_TOKENS_STD)))

        # Cache-aware condition: static block first, same signature for this team+backend
        # every time, so every call after the first is a cache hit on it.
        layout = cache_aware_layout(f"{team}-instructions", STATIC_BLOCK_TOKENS, variable_tokens)
        cached = cache_aware_cache.consume(team, backend.name, layout)
        input_tokens = layout.prefix_tokens + layout.suffix_tokens
        response = backend.complete("prompt", input_tokens, cached, difficulty, rng)
        cost = price_request(backend, input_tokens, cached, response.output_tokens)
        cache_aware_billed += cost.billed_input_tokens
        record_request(
            con, team=team, backend_name=backend.name, layout="cache_aware", difficulty=difficulty,
            input_tokens=input_tokens, cached_prefix_tokens=cached, output_tokens=response.output_tokens,
            latency_ms=response.latency_ms, cost_usd=cost.cost_usd, passed=response.passed,
        )

        # Naive condition: the same team+backend+static-block content, but laid out with
        # the variable context first, so the static block never gets to be a cache hit.
        layout_n = naive_layout(f"req-{i}", STATIC_BLOCK_TOKENS, variable_tokens)
        cached_n = naive_cache.consume(team, backend.name, layout_n)
        input_tokens_n = layout_n.prefix_tokens + layout_n.suffix_tokens
        response_n = backend.complete("prompt", input_tokens_n, cached_n, difficulty, rng)
        cost_n = price_request(backend, input_tokens_n, cached_n, response_n.output_tokens)
        naive_billed += cost_n.billed_input_tokens
        record_request(
            con, team=team, backend_name=backend.name, layout="naive", difficulty=difficulty,
            input_tokens=input_tokens_n, cached_prefix_tokens=cached_n, output_tokens=response_n.output_tokens,
            latency_ms=response_n.latency_ms, cost_usd=cost_n.cost_usd, passed=response_n.passed,
        )

    pct_reduction = round(100 * (naive_billed - cache_aware_billed) / naive_billed, 2)
    report = {
        "n_requests_per_condition": N_REQUESTS,
        "static_block_tokens": STATIC_BLOCK_TOKENS,
        "variable_tokens_mean": VARIABLE_TOKENS_MEAN,
        "naive_billed_input_tokens": naive_billed,
        "cache_aware_billed_input_tokens": cache_aware_billed,
        "pct_billed_input_tokens_reduction": pct_reduction,
        "per_team_report_cache_aware": per_team_report(con, layout="cache_aware"),
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
