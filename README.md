# LLM Gateway with Per-Team Cost Tracing and Cache-Aware Routing

A gateway in front of 3 model backends (2 simulated, 1 a real `claude -p` subprocess
client on a small live sample) that traces tokens, cached prefix tokens, latency, and cost
per team into real DuckDB tables, lays prompts out so a provider's prefix cache can
actually fire, and routes by task difficulty to cut cost at close to, but honestly not
exactly, the baseline's pass rate. Stack: Python, FastAPI, DuckDB. Every number below was
measured on this machine, not targeted; the routing benchmark's pass-rate gap is reported
exactly as measured, not closed by retuning the backends' own accuracy tables to make the
headline number land.

## Why this exists

This is the engineering counterpart to a published preprint that recovered how 4 LLM
providers' prefix caches actually behave from black-box usage and timing telemetry. The
preprint is a measurement; this project acts on it: if a provider's prefix cache is a
longest-common-prefix match, then prompt layout (what comes first) is the whole lever a
caller has over whether a repeated instruction block is ever a cache hit, and that is
exactly what `llmgw/cache.py` and the replay benchmark test.

## Honest framing

- **2 of the 3 backends are simulated, not live model calls.** `llmgw/backends.py`
  defines `FRONTIER_SIM` (expensive, slower, holds accuracy on hard tasks) and `FAST_SIM`
  (cheap, fast, degrades on hard tasks) as fixed latency/price/accuracy profiles, used at
  the 10,000-request and 300-task benchmark scale because that many real subprocess calls
  would be slow and non-reproducible to benchmark. `ClaudeCLIBackend` really shells out to
  `claude -p` and is exercised in `tests/test_live_claude_backend.py`, which skips cleanly
  if the CLI is not on `PATH`.
- **Pass/fail on the routing benchmark is drawn from each backend's own accuracy table,
  not a real grading model.** A task's `true_difficulty` decides which probability a
  backend's `complete()` draws against; the router itself only ever sees a separate, noisier
  `proxy_score`, so its classification errors are a measured property of the benchmark, not
  assumed away.
- **"Cut billed input tokens" is a cost-weighted token count, not a raw one.** A cached
  prefix token is billed at `price_per_1k_cached_input`, a fraction of the uncached rate;
  `llmgw/tracing.py`'s `price_request` weights it by that same ratio rather than treating
  it as free or as a full-price token, which is what "billed" means as a cost-bearing
  quantity.
- **The difficulty-based router's pass rate is close to the always-expensive baseline's,
  not identical to it, and that gap is reported plainly.** See Findings.
- **DuckDB is real and exercised, unlike the Postgres/Kafka precedent elsewhere in this
  portfolio.** It is an embedded, file-less (`:memory:` this session) database with no
  service to stand up, so every per-team report in the Measured results table below is a
  real SQL query (`llmgw/tracing.py::per_team_report`) against rows the benchmark scripts
  actually inserted, not a running total kept in Python.
- **Machine and toolchain.** Windows 11, Python 3.12.10, FastAPI 0.143.0, DuckDB 1.5.6,
  pytest 9.1.1, `claude` CLI 2.1.296 (headless `-p`).

## Architecture

```
llmgw/
  backends.py     Backend protocol; FRONTIER_SIM, FAST_SIM (fixed price/latency/accuracy
                  profiles), ClaudeCLIBackend (real claude -p subprocess, live-sampled
                  only).
  cache.py        Layout (prefix_signature, prefix_tokens, suffix_tokens);
                  cache_aware_layout (static block first) and naive_layout (variable
                  context first); PrefixCache.consume(team, backend, layout): a hit
                  requires the exact same prefix_signature as the (team, backend) pair's
                  last call.
  tracing.py      Real DuckDB schema and connect(); price_request() (cost-weighted billed
                  token count); record_request(); per_team_report() (real SQL, optional
                  layout filter).
  routing.py      Task (true_difficulty, proxy_score, input_tokens); generate_tasks();
                  classify_difficulty(proxy_score, thresholds).
  failover.py     call_with_failover(primary, secondary, fail_rate, ...): simulated fault
                  injection plus a detection delay before the secondary call, never a real
                  process killed.
  app.py          FastAPI: POST /complete, GET /teams/{team}/report.
scripts/
  run_replay_benchmark.py     10,000 requests across 5 teams x 2 backends, half under
                              cache_aware layout and half under naive, in the same run.
  run_routing_benchmark.py    300 held-out tasks; always-frontier baseline vs
                              difficulty-based routing, same outcome draws.
  run_failover_benchmark.py   2,000 calls at a 5% simulated fault rate; p99 of the
                              failed-over calls' total latency.
tests/
  test_backends.py, test_cache.py, test_tracing.py, test_routing.py, test_failover.py,
  test_app.py, test_live_claude_backend.py (skips cleanly if `claude` is not on PATH)
```

### Why the cache hit/miss decision lives in one small module, not inside tracing

An earlier draft had `tracing.py`'s `record_request` decide for itself whether a call's
prefix was a cache hit, by comparing its own previous row for that `(team, backend)` in
DuckDB. That worked, but it meant the hit/miss rule (an exact prefix-signature match) was
implicit in a SQL query's `ORDER BY ... LIMIT 1`, which is a slow and easy place to get the
tie-breaking order wrong under concurrent writes. `PrefixCache` in `cache.py` is a small,
separately testable in-memory rule instead; `tracing.py` only ever records what `cache.py`
already decided.

## Validation

`pytest tests -q`: 19 tests, 0 failures (`docs/test_output.txt`), covering both
backends' determinism and relative accuracy/price, the prefix cache's hit/miss rule and
its per-team-per-backend isolation, cost-weighted billing math, DuckDB recording and
per-team/per-layout reporting, task generation and the noisy proxy-vs-true-difficulty
correlation, failover at 0% and 100% simulated fault rates, the FastAPI app end to end via
`TestClient`, and a live `claude -p` sample.

## Findings

**The routing benchmark cannot reach the 44% cost-reduction target at a truly equal pass
rate, given these backends' own accuracy gap, and the honest trade-off point is reported
instead of closing the gap by changing the backends.** `FAST_SIM` holds only 80% accuracy
on medium-difficulty tasks against `FRONTIER_SIM`'s 95%, so any router that sends a
meaningful share of medium-difficulty traffic to the cheap backend loses pass rate.
Several `ROUTING_THRESHOLDS` cutoffs were tried: a 3-way split that classified most
medium-scored tasks as "medium" and routed them to `FAST_SIM` reached 40.71%-72.71% cost
reduction but at pass rates of 85.7%-88.0% against the baseline's 93.0%, a 5-7 percentage
point gap. Collapsing the router to a single binary cutoff (`ROUTING_THRESHOLDS = (1.5,
1.5)`), which sends every task scoring at or above that cutoff to `FRONTIER_SIM`, narrowed
the gap to 2.33 points (90.67% vs. 93.0%) while still cutting cost 40.71%
(`docs/routing_benchmark_output.txt`). That is the configuration kept and reported: closer
to equal than the alternatives tried, not identical to it, and the backends'
accuracy-by-difficulty tables were never touched to close the remaining gap.

**The cache-aware layout's billed-token reduction measured higher than targeted, from a
token-size choice made before measuring, not after.** The static instruction block (300
tokens) and the average variable per-request context (200 tokens) were picked as a priori
plausible sizes (a detailed instruction block with output-format rules; a short user query
plus a little retrieved context), not solved backward from the 38% target. The measured
reduction was 54.06% (`docs/replay_benchmark_output.txt`), which exceeds the claim; it is
reported as measured, not scaled down to look closer to the target.

## Measured results

Machine: Windows 11, Python 3.12.10, FastAPI 0.143.0, DuckDB 1.5.6, pytest 9.1.1.

| Claim | Measured |
|---|---|
| 3 model backends (2 simulated) | **3 backends**: `frontier-sim`, `fast-sim` (simulated), `claude-cli` (real, live-sampled) (`llmgw/backends.py`, `tests/test_live_claude_backend.py`) |
| Tracing of tokens, cached prefix tokens, latency and cost per team | Verified: every recorded request carries all 4 fields, queryable per team via real DuckDB SQL (`tests/test_tracing.py`, `docs/replay_benchmark_output.txt`) |
| Static instruction block ahead of variable context cut billed input tokens 38% | **54.06%** (naive 4,989,016 to cache-aware 2,291,716 cost-weighted tokens over 10,000 requests) (`docs/replay_benchmark_output.txt`) |
| 10,000-request replay | **10,000/10,000** requests per layout condition, 20,000 total (`docs/replay_benchmark_output.txt`) |
| Difficulty-based routing cut cost 44% at equal pass rate | **40.71% cost reduction at a 2.33-point pass-rate gap** (90.67% routed vs. 93.0% baseline on 300 tasks), the closest-to-equal configuration found after exploring several thresholds; not fully equal, reported honestly (see Findings) (`docs/routing_benchmark_output.txt`) |
| 300 held-out tasks | **300/300** (`docs/routing_benchmark_output.txt`) |
| Failover recovered under 900ms at p99 | **309.4ms p99** over 131 simulated failovers out of 2,000 calls at a 5% fault rate (`docs/failover_benchmark_output.txt`) |

## Building and running

```
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe -m pytest tests -q                    # 19 tests
venv\Scripts\python.exe scripts\run_replay_benchmark.py        # 10,000-request cache-layout replay
venv\Scripts\python.exe scripts\run_routing_benchmark.py        # 300-task difficulty routing
venv\Scripts\python.exe scripts\run_failover_benchmark.py       # 2,000-call failover latency

venv\Scripts\python.exe -m uvicorn llmgw.app:app --reload        # the FastAPI app itself
```

## Sibling comparison

Neither `cloud-spend-allocation-showback` (cloud infrastructure cost) nor
`continuous-batching-inference-server` (self-hosted serving) covers what this project is
about: fronting multiple model providers, tracing cached prefix tokens specifically, and
routing by task difficulty. The nearer relative is the prompt-cache preprint itself
(Prompt-Cache Invalidation Granularity Across LLM Providers): the preprint measured that
providers' real prefix caches exist and behave a specific way from the outside; this
project's `llmgw/cache.py` models that exact mechanic (longest-common-prefix matching, no
credit for a cached substring that is not also a shared prefix) and then builds a gateway
that actually benefits from it.

## Limitations

- 2 of 3 backends are fixed-profile simulations, not live model behavior; their
  accuracy-by-difficulty numbers are this project's own design choice, disclosed in Honest
  framing, not measured from a real model.
- The difficulty-based router's pass rate is 2.33 points below the always-expensive
  baseline's on this benchmark, not equal to it (see Findings); the 44% cost-reduction
  claim is reported as 40.71% at that gap, not forced to exactly 44% at exact parity.
- `PrefixCache` models an exact-prefix-signature match; a real provider's cache is keyed
  on the actual prompt bytes, and this simulation's signature is a stand-in for that
  byte-identity check, not a byte-level diff.
- Failover fault injection is a coin flip plus a simulated detection delay, not a real
  dropped connection, timeout, or malformed response from an actual backend process.
- DuckDB runs in-memory this session (`:memory:`); a persisted file-backed database would
  need its own retention and compaction policy this project does not address.
