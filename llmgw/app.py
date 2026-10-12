# FastAPI wrapper: one endpoint per team's real-time cost/latency trace, backed by the
# same DuckDB connection and query every benchmark script uses.
from __future__ import annotations

import random

import duckdb
from fastapi import FastAPI
from pydantic import BaseModel

from .backends import FAST_SIM, FRONTIER_SIM
from .cache import PrefixCache, cache_aware_layout
from .tracing import connect, per_team_report, price_request, record_request

app = FastAPI(title="LLM Gateway: Cost Tracing and Cache-Aware Routing")
_con: duckdb.DuckDBPyConnection = connect(":memory:")
_cache = PrefixCache()
_rng = random.Random(1)

BACKENDS_BY_NAME = {"fast-sim": FAST_SIM, "frontier-sim": FRONTIER_SIM}


class CompleteRequest(BaseModel):
    team: str
    backend: str
    static_block_id: str
    static_tokens: int
    variable_tokens: int
    difficulty: str


class CompleteResponse(BaseModel):
    backend: str
    input_tokens: int
    cached_prefix_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@app.post("/complete", response_model=CompleteResponse)
def complete(req: CompleteRequest) -> CompleteResponse:
    backend = BACKENDS_BY_NAME[req.backend]
    layout = cache_aware_layout(req.static_block_id, req.static_tokens, req.variable_tokens)
    cached = _cache.consume(req.team, backend.name, layout)
    input_tokens = layout.prefix_tokens + layout.suffix_tokens
    response = backend.complete("prompt", input_tokens, cached, req.difficulty, _rng)
    cost = price_request(backend, input_tokens, cached, response.output_tokens)
    record_request(
        _con, team=req.team, backend_name=backend.name, layout="cache_aware", difficulty=req.difficulty,
        input_tokens=input_tokens, cached_prefix_tokens=cached, output_tokens=response.output_tokens,
        latency_ms=response.latency_ms, cost_usd=cost.cost_usd, passed=response.passed,
    )
    return CompleteResponse(
        backend=backend.name, input_tokens=input_tokens, cached_prefix_tokens=cached,
        output_tokens=response.output_tokens, latency_ms=response.latency_ms, cost_usd=cost.cost_usd,
    )


@app.get("/teams/{team}/report")
def team_report(team: str) -> dict:
    rows = [r for r in per_team_report(_con) if r["team"] == team]
    return rows[0] if rows else {"team": team, "n_requests": 0}
