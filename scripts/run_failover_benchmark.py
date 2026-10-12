#!/usr/bin/env python
# 2,000 calls with a 5% simulated primary-backend fault rate; on a fault, the gateway
# fails over to the fast simulated backend after a simulated detection delay. Measures the
# p99 total latency of the failed-over calls specifically (detection delay plus the
# secondary backend's own latency), since that is what "failover recovered under 900ms at
# p99" is about, not the normal-path latency of calls that never failed.
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmgw.backends import FAST_SIM, FRONTIER_SIM
from llmgw.failover import call_with_failover

N_CALLS = 2000
SEED = 20260801
FAIL_RATE = 0.05
DETECT_MS_MEAN = 60.0
DETECT_MS_STD = 20.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(p * len(ordered)))
    return ordered[idx]


def run() -> dict:
    rng = random.Random(SEED)
    failover_latencies = []
    normal_latencies = []
    for _ in range(N_CALLS):
        result = call_with_failover(
            FRONTIER_SIM, FAST_SIM, prompt="prompt", input_tokens=220, difficulty="medium",
            fail_rate=FAIL_RATE, detect_ms_mean=DETECT_MS_MEAN, detect_ms_std=DETECT_MS_STD, rng=rng,
        )
        if result.failed_over:
            failover_latencies.append(result.total_latency_ms)
        else:
            normal_latencies.append(result.total_latency_ms)

    report = {
        "n_calls": N_CALLS,
        "fail_rate": FAIL_RATE,
        "n_failed_over": len(failover_latencies),
        "failover_p50_ms": round(percentile(failover_latencies, 0.50), 1),
        "failover_p99_ms": round(percentile(failover_latencies, 0.99), 1),
        "failover_max_ms": round(max(failover_latencies), 1) if failover_latencies else None,
        "normal_p99_ms": round(percentile(normal_latencies, 0.99), 1),
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
