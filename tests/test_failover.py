import random

from llmgw.backends import FAST_SIM, FRONTIER_SIM
from llmgw.failover import call_with_failover


def test_failover_never_triggers_at_zero_fail_rate():
    rng = random.Random(1)
    for _ in range(200):
        result = call_with_failover(
            FRONTIER_SIM, FAST_SIM, prompt="p", input_tokens=100, difficulty="easy",
            fail_rate=0.0, detect_ms_mean=50, detect_ms_std=10, rng=rng,
        )
        assert not result.failed_over
        assert result.used_backend == FRONTIER_SIM.name


def test_failover_always_triggers_at_full_fail_rate():
    rng = random.Random(1)
    result = call_with_failover(
        FRONTIER_SIM, FAST_SIM, prompt="p", input_tokens=100, difficulty="easy",
        fail_rate=1.0, detect_ms_mean=50, detect_ms_std=10, rng=rng,
    )
    assert result.failed_over
    assert result.used_backend == FAST_SIM.name


def test_failover_latency_includes_detection_delay_plus_secondary_latency():
    rng = random.Random(1)
    result = call_with_failover(
        FRONTIER_SIM, FAST_SIM, prompt="p", input_tokens=100, difficulty="easy",
        fail_rate=1.0, detect_ms_mean=50, detect_ms_std=0.0001, rng=rng,
    )
    assert result.total_latency_ms > 50.0
