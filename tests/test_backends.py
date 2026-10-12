import random

from llmgw.backends import FAST_SIM, FRONTIER_SIM


def test_simulated_backend_is_deterministic_for_a_fixed_seed():
    a = FAST_SIM.complete("p", 100, 0, "easy", random.Random(1))
    b = FAST_SIM.complete("p", 100, 0, "easy", random.Random(1))
    assert a == b


def test_frontier_holds_accuracy_on_hard_tasks_better_than_fast():
    rng = random.Random(7)
    frontier_hard_passes = sum(FRONTIER_SIM.complete("p", 100, 0, "hard", rng).passed for _ in range(500))
    fast_hard_passes = sum(FAST_SIM.complete("p", 100, 0, "hard", rng).passed for _ in range(500))
    assert frontier_hard_passes > fast_hard_passes


def test_fast_sim_is_cheaper_than_frontier_sim():
    assert FAST_SIM.price_per_1k_input < FRONTIER_SIM.price_per_1k_input
    assert FAST_SIM.price_per_1k_output < FRONTIER_SIM.price_per_1k_output
