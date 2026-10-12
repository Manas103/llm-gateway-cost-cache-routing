from llmgw.cache import PrefixCache, cache_aware_layout, naive_layout


def test_cache_aware_layout_hits_on_second_call_same_team_backend():
    cache = PrefixCache()
    layout1 = cache_aware_layout("team-a-instructions", 300, 150)
    layout2 = cache_aware_layout("team-a-instructions", 300, 220)  # same static id, different variable size

    first = cache.consume("team-a", "fast-sim", layout1)
    second = cache.consume("team-a", "fast-sim", layout2)

    assert first == 0  # nothing to hit on the very first call
    assert second == 300  # the static prefix matched, so it is a hit regardless of variable length


def test_naive_layout_never_hits_because_the_prefix_changes_every_call():
    cache = PrefixCache()
    layout1 = naive_layout("req-1", 300, 150)
    layout2 = naive_layout("req-2", 300, 220)

    cache.consume("team-a", "fast-sim", layout1)
    second = cache.consume("team-a", "fast-sim", layout2)

    assert second == 0


def test_cache_is_isolated_per_team_and_per_backend():
    cache = PrefixCache()
    layout = cache_aware_layout("shared-instructions", 300, 150)

    cache.consume("team-a", "fast-sim", layout)
    cross_team = cache.consume("team-b", "fast-sim", layout)
    cross_backend = cache.consume("team-a", "frontier-sim", layout)

    assert cross_team == 0
    assert cross_backend == 0
