# Track 2 Cerebras Fast-Reasoning Harness Patterns

Track 2 agents use direct Cerebras-hosted `gpt-oss` inference through the
Cerebras Python SDK or direct Cerebras API calls. The reference templates keep
the public A2A boundary identical to Track 1 while giving participants a
starting point for compute-aware harnesses. The Track 2 reference templates use
Cerebras directly.

## Reference Agent Map

| Agent | Package | Local Scenario | Internal Strategy |
|-------|---------|----------------|-------------------|
| Direct Cerebras agent | [`src/track_2_agent_under_test_cerebras/`](../src/track_2_agent_under_test_cerebras/) | [`scenarios/track_2_agent_under_test_cerebras/local_smoke.toml`](../scenarios/track_2_agent_under_test_cerebras/local_smoke.toml) | Cerebras `gpt-oss` executor returns schema-constrained next-action JSON. |
| Cerebras planner/executor | [`src/track_2_agent_under_test_cerebras_planner/`](../src/track_2_agent_under_test_cerebras_planner/) | [`scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml`](../scenarios/track_2_agent_under_test_cerebras_planner/local_smoke.toml) | Cerebras `gpt-oss` planner writes a compact plan with high reasoning effort; Cerebras `gpt-oss` executor returns normal A2A output with medium reasoning effort. |

## Model Selection

The direct executor defaults to:

```env
CEREBRAS_API_KEY=...
TRACK2_EXECUTOR_MODEL=gpt-oss-120b
TRACK2_EXECUTOR_REASONING_EFFORT=medium
```

The planner/executor template additionally defaults to:

```env
TRACK2_PLANNER_MODEL=gpt-oss-120b
TRACK2_PLANNER_REASONING_EFFORT=high
TRACK2_PLANNER_MAX_COMPLETION_TOKENS=4096
TRACK2_PLANNER_TEMPERATURE=
```

Use Cerebras-hosted `gpt-oss` models for submitted Track 2 agents. Leave
`TRACK2_PLANNER_TEMPERATURE` and `TRACK2_TEMPERATURE` unset to use provider
defaults, or set them explicitly when needed.

## Public-Tier Development Logistics

Most participants will use the Cerebras public tier during development. Rate
limits can be strict, so use smaller smoke scenarios first, keep completion
budgets tight, and schedule large public-tier runs externally rather than
launching many at once. The reference templates do not do proactive local
request/token quota pacing before calls. They retry reactively only after a
provider-visible Cerebras 429:

```env
TRACK2_CEREBRAS_QUEUE_BACKOFF_SECONDS=60
TRACK2_CEREBRAS_QUEUE_BACKOFF_INITIAL_JITTER_RATIO=0.1
TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MIN_SECONDS=90
TRACK2_CEREBRAS_QUEUE_BACKOFF_SECOND_MAX_SECONDS=120
TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MIN_SECONDS=180
TRACK2_CEREBRAS_QUEUE_BACKOFF_CAP_MAX_SECONDS=300
TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS=1
```

The reference templates compute `avg_llm_call_time_ms` only from successful
provider calls. Failed 429 attempts are visible in logs/reports but do not enter
submitted LLM latency.

Organizers will provide a few elevated-rate/priority test windows for Track 2
teams. Use those windows to validate speed-sensitive harness behavior and
larger public validation runs. Participants may also self-host the open-source
models used by the Cerebras `gpt-oss` executor during development, then switch
to the Cerebras endpoint for test-window validation.

Codex Pro plans are allocated by June 15 for selected Track 2 participants to
speed up harness engineering and development. Codex with
`gpt-5.3-codex-spark` is not the runtime used by the new submitted-agent
templates.

## Pattern 1: Direct Next-Action Baseline

Each CAR-bench assistant step becomes one Cerebras SDK call:

```text
A2A input from evaluator
  -> build transcript and task-filtered tool prompt
  -> Cerebras next-action JSON
  -> parse JSON
  -> return text Part or data Part(tool_calls) to evaluator
```

This is the lowest-latency and easiest-to-debug Track 2 template. It is the best
starting point before adding planners, verifiers, rerankers, or ensembles.

## Pattern 2: Cerebras High-Effort Planner Plus Medium-Effort Executor

Use a high-reasoning Cerebras `gpt-oss` planner to write compact private
guidance, then let a medium-reasoning Cerebras `gpt-oss` executor produce the
benchmark-visible next action. The reference planner runs when a new user turn
arrives. Tool-result continuation turns reuse the active private plan until the
executor returns a final user response.

The private plan is not a CAR-bench tool call and is never returned to the
evaluator. If the executor needs benchmark-visible planning behavior, it can
call CAR-bench's supplied `planning_tool` like any other available tool.

## Rate-Limit Accounting

The Cerebras templates write an audit report when a known Cerebras 429 shape is
observed. Reports are written to
`CAR_BENCH_CEREBRAS_RATE_LIMIT_REPORT_DIR`, falling back to
`CAR_BENCH_RATE_LIMIT_REPORT_DIR`, then `/tmp/car-bench-rate-limit-reports`.

The observed Cerebras shapes are:

- `queue_exceeded` with `param: "queue"` on the original provider error. This
  means the provider queue is saturated. If Cerebras sends `retry-after`, the
  template respects it; otherwise it applies jittered local backoff: roughly
  60 seconds on the first queue retry, 90-120 seconds on the second, and
  180-300 seconds on later queue retries.
- `token_quota_exceeded` with `param: "quota"` and `retry-after` or
  `x-ratelimit-reset-*` headers. This means a request/token quota window has a
  provider-visible reset hint, so the template applies that wait plus
  `TRACK2_CEREBRAS_RATE_LIMIT_RETRY_BUFFER_SECONDS` before retrying.

For quota 429s, the client prefers `x-ratelimit-reset-tokens-minute` because it
directly names the token window that usually bites first. If that header is
missing, the log and JSON report call it out and the client falls back to
`x-ratelimit-reset-requests-day` or `retry-after` when present. The terminal
shows the wait duration, resume time, reason, source header, available reset
values, and missing expected headers.

Rate-limit reports include session start time, wall time until the limit, wall
time since the previous rate-limit and retry markers, successful-call token
usage, estimated attempted request tokens, the failed call shape, wait decision,
and raw provider payloads. These files are for diagnosis, reproducibility, and
future audit.

Final time-budget and quota-wait accounting details will be announced before
the official evaluation. Until then, treat rate-limit reports as evidence for
understanding provider behavior rather than as a final scoring-policy contract.
Do not fabricate timing metadata. Successful planner/executor calls still count
as LLM calls for `num_llm_calls` and `avg_llm_call_time_ms`.
