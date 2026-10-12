from llmgw.backends import FAST_SIM, FRONTIER_SIM
from llmgw.tracing import connect, per_team_report, price_request, record_request


def test_price_request_discounts_cached_tokens():
    full_price = price_request(FRONTIER_SIM, input_tokens=500, cached_prefix_tokens=0, output_tokens=100)
    with_cache = price_request(FRONTIER_SIM, input_tokens=500, cached_prefix_tokens=300, output_tokens=100)
    assert with_cache.cost_usd < full_price.cost_usd
    assert with_cache.billed_input_tokens < full_price.billed_input_tokens


def test_record_request_and_per_team_report_round_trip():
    con = connect(":memory:")
    record_request(
        con, team="ads-platform", backend_name=FAST_SIM.name, layout="cache_aware", difficulty="easy",
        input_tokens=400, cached_prefix_tokens=300, output_tokens=120, latency_ms=85.0, cost_usd=0.01, passed=True,
    )
    record_request(
        con, team="ads-platform", backend_name=FAST_SIM.name, layout="cache_aware", difficulty="medium",
        input_tokens=410, cached_prefix_tokens=300, output_tokens=130, latency_ms=90.0, cost_usd=0.012, passed=True,
    )
    report = per_team_report(con)
    assert len(report) == 1
    row = report[0]
    assert row["team"] == "ads-platform"
    assert row["n_requests"] == 2
    assert row["total_cached_prefix_tokens"] == 600


def test_per_team_report_filters_by_layout():
    con = connect(":memory:")
    record_request(
        con, team="growth", backend_name=FAST_SIM.name, layout="cache_aware", difficulty="easy",
        input_tokens=400, cached_prefix_tokens=300, output_tokens=120, latency_ms=85.0, cost_usd=0.01, passed=True,
    )
    record_request(
        con, team="growth", backend_name=FAST_SIM.name, layout="naive", difficulty="easy",
        input_tokens=400, cached_prefix_tokens=0, output_tokens=120, latency_ms=85.0, cost_usd=0.02, passed=True,
    )
    cache_aware_only = per_team_report(con, layout="cache_aware")
    assert cache_aware_only[0]["n_requests"] == 1
    assert cache_aware_only[0]["total_cached_prefix_tokens"] == 300
