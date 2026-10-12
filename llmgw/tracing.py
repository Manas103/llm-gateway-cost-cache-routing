# Real DuckDB (embedded, no service to stand up, unlike the Postgres/Kafka precedent
# elsewhere in this portfolio): every traced call is a row in requests.duckdb, queried
# with real SQL by the per-team cost/latency reports and by the replay benchmark's own
# aggregate numbers.
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id VARCHAR PRIMARY KEY,
    team VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,
    layout VARCHAR NOT NULL,
    difficulty VARCHAR,
    input_tokens INTEGER NOT NULL,
    cached_prefix_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms DOUBLE NOT NULL,
    cost_usd DOUBLE NOT NULL,
    passed BOOLEAN,
    created_at DOUBLE NOT NULL
);
"""


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database)
    con.execute(SCHEMA)
    return con


@dataclass(frozen=True)
class CostBreakdown:
    billed_input_tokens: float
    cost_usd: float


def price_request(backend, input_tokens: int, cached_prefix_tokens: int, output_tokens: int) -> CostBreakdown:
    """billed_input_tokens is a full-price-equivalent count, not a raw token count: a
    cached prefix token is billed at price_per_1k_cached_input, a fraction of the
    uncached rate, so it is weighted down by that same ratio rather than counted as a
    full token or dropped entirely. This is what "cut billed input tokens" means as a
    cost-bearing quantity, not a literal word count of the prompt."""
    uncached_tokens = input_tokens - cached_prefix_tokens
    cached_price_ratio = backend.price_per_1k_cached_input / backend.price_per_1k_input
    billed_input_tokens = uncached_tokens + cached_prefix_tokens * cached_price_ratio
    cost = (
        cached_prefix_tokens / 1000 * backend.price_per_1k_cached_input
        + uncached_tokens / 1000 * backend.price_per_1k_input
        + output_tokens / 1000 * backend.price_per_1k_output
    )
    return CostBreakdown(billed_input_tokens=billed_input_tokens, cost_usd=cost)


def record_request(
    con: duckdb.DuckDBPyConnection,
    *,
    team: str,
    backend_name: str,
    layout: str,
    difficulty: str | None,
    input_tokens: int,
    cached_prefix_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cost_usd: float,
    passed: bool | None,
) -> str:
    request_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            request_id, team, backend_name, layout, difficulty,
            input_tokens, cached_prefix_tokens, output_tokens,
            latency_ms, cost_usd, passed, time.time(),
        ],
    )
    return request_id


def per_team_report(con: duckdb.DuckDBPyConnection, *, layout: str | None = None) -> list[dict]:
    where = "WHERE layout = ?" if layout is not None else ""
    params = [layout] if layout is not None else []
    rows = con.execute(
        f"""
        SELECT team,
               COUNT(*) AS n_requests,
               SUM(input_tokens) AS total_input_tokens,
               SUM(cached_prefix_tokens) AS total_cached_prefix_tokens,
               SUM(output_tokens) AS total_output_tokens,
               AVG(latency_ms) AS avg_latency_ms,
               SUM(cost_usd) AS total_cost_usd
        FROM requests
        {where}
        GROUP BY team
        ORDER BY team
        """,
        params,
    ).fetchall()
    cols = ["team", "n_requests", "total_input_tokens", "total_cached_prefix_tokens", "total_output_tokens", "avg_latency_ms", "total_cost_usd"]
    return [dict(zip(cols, row)) for row in rows]
