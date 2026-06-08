#!/usr/bin/env python3
"""Plot completed CAR-bench full test-set results.

The script reads saved result payloads from ``output/<agent-name>/*.json`` and
creates a structured plot report under ``outputs/plots``. Each chart uses 2x2
subfigures:

* overall
* base-only
* hallucination-only
* disambiguation-only

By default it plots Pass^3, CAR-bench's main consistency metric. Bars inside
each panel are ordered so the highest score is on the right.

It also writes latency charts from evaluator-measured A2A timing. That timing is
measured around evaluator-to-agent calls, so it is the best default for comparing
how long the agent under test takes without trusting participant-reported values.

Agent token charts are based on standard ``turn_metrics`` token fields reported
by the agent client and persisted per task trial. Consumed-token charts split
input tokens from output tokens, where output includes completion and thinking
tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "output"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_OUTPUT_PATH = DEFAULT_REPORT_DIR / "scores" / "pass_power_3.png"
DEFAULT_LATENCY_OUTPUT_PATH = DEFAULT_REPORT_DIR / "latency" / "a2a_per_task_mean.png"
DEFAULT_TOKEN_OUTPUT_PATH = DEFAULT_REPORT_DIR / "tokens" / "input_output_per_task.png"
SPLITS = ("base", "hallucination", "disambiguation")
TOKEN_KINDS = ("total", "input", "output", "prompt", "completion", "thinking")
EXPECTED_PUBLIC_TEST_COUNTS = {
    "base": 100,
    "hallucination": 98,
    "disambiguation": 56,
}


@dataclass(frozen=True)
class RunResult:
    path: Path
    final_result: dict[str, Any]
    model: str
    reasoning_effort: str | None
    completed_at: datetime
    metrics: dict[str, float]
    latencies: dict[str, float]
    tokens: dict[str, float]
    token_breakdown: dict[str, dict[str, float]]
    token_summary: dict[str, dict[str, float]]
    wall_time_seconds: float | None
    counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an ordered CAR-bench plot report from full-test output JSON files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing saved result JSON files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=(
            "Directory for the full generated plot report. "
            f"Default: {DEFAULT_REPORT_DIR}"
        ),
    )
    parser.add_argument(
        "--single-plot",
        action="store_true",
        help=(
            "Only write the selected score, latency, and token plots using "
            "--output, --latency-output, and --token-output."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Plot path. Extension controls format. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--latency-output",
        type=Path,
        default=DEFAULT_LATENCY_OUTPUT_PATH,
        help=(
            "Latency plot path. Extension controls format. "
            f"Default: {DEFAULT_LATENCY_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Only write the score plot; skip the latency plot.",
    )
    parser.add_argument(
        "--token-output",
        type=Path,
        default=DEFAULT_TOKEN_OUTPUT_PATH,
        help=(
            "Token-consumption plot path. Extension controls format. "
            f"Default: {DEFAULT_TOKEN_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--skip-tokens",
        action="store_true",
        help="Skip the token-consumption plot.",
    )
    parser.add_argument(
        "--latency-aggregation",
        choices=("mean", "median", "p95", "total"),
        default="mean",
        help=(
            "How to aggregate latency values. mean/median/p95 are seconds; "
            "total is minutes. Default: mean"
        ),
    )
    parser.add_argument(
        "--latency-source",
        choices=("a2a", "llm"),
        default="a2a",
        help=(
            "Latency source. a2a uses evaluator-measured agent response time; "
            "llm uses agent-reported total_llm_latency_ms. Default: a2a"
        ),
    )
    parser.add_argument(
        "--latency-scope",
        choices=("per-turn", "per-task", "total"),
        default="per-task",
        help=(
            "A2A latency view: per-turn agent response time, per-task summed "
            "agent response time, or total summed agent response time. "
            "LLM latency only supports per-task. Default: per-task"
        ),
    )
    parser.add_argument(
        "--token-kind",
        choices=TOKEN_KINDS,
        default="total",
        help=(
            "Token component for single-plot mode. total is input + output. "
            "Default: total"
        ),
    )
    parser.add_argument(
        "--token-scope",
        choices=("per-task", "total", "per-hour"),
        default="per-task",
        help=(
            "Token view: mean tokens per task trial, total tokens, or tokens per "
            "benchmark wall-clock hour. Default: per-task"
        ),
    )
    parser.add_argument(
        "--metric",
        default="Pass^3",
        help="Metric to plot: Pass^3, Pass@3, Pass^1, Pass@1, or pass_rate. Default: Pass^3",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep repeated runs for the same model label instead of plotting only the latest.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional chart title. Default includes the selected metric.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Figure width in inches. Defaults to a size based on the number of models.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=9.0,
        help="Figure height in inches. Default: 9.0",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI for raster formats. Default: 200",
    )
    parser.add_argument(
        "--expected-base",
        type=int,
        default=EXPECTED_PUBLIC_TEST_COUNTS["base"],
        help="Expected public full-test unique base tasks. Default: 100",
    )
    parser.add_argument(
        "--expected-hallucination",
        type=int,
        default=EXPECTED_PUBLIC_TEST_COUNTS["hallucination"],
        help="Expected public full-test unique hallucination tasks. Default: 98",
    )
    parser.add_argument(
        "--expected-disambiguation",
        type=int,
        default=EXPECTED_PUBLIC_TEST_COUNTS["disambiguation"],
        help="Expected public full-test unique disambiguation tasks. Default: 56",
    )
    return parser.parse_args()


def canonical_metric(metric: str) -> str:
    compact = metric.strip().replace(" ", "")
    lower = compact.lower()
    if lower in {"pass_rate", "passrate", "overall_pass_rate"}:
        return "pass_rate"
    if lower.startswith("pass@"):
        suffix = lower.removeprefix("pass@")
        if suffix.isdigit():
            return f"Pass@{suffix}"
    if lower.startswith("pass^"):
        suffix = lower.removeprefix("pass^")
        if suffix.isdigit():
            return f"Pass^{suffix}"
    if lower.startswith("pass") and lower.removeprefix("pass").isdigit():
        return f"Pass^{lower.removeprefix('pass')}"
    raise ValueError(
        f"Unsupported metric {metric!r}. Use Pass^3, Pass@3, Pass^1, Pass@1, or pass_rate."
    )


def metric_filename(metric: str) -> str:
    if metric == "pass_rate":
        return "pass_rate"
    return (
        metric.lower()
        .replace("^", "_power_")
        .replace("@", "_at_")
        .replace(" ", "_")
    )


def expected_counts_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "base": args.expected_base,
        "hallucination": args.expected_hallucination,
        "disambiguation": args.expected_disambiguation,
    }


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable JSON: {path} ({exc})")
        return None

    if not isinstance(data, dict):
        print(f"Skipping non-payload JSON: {path}")
        return None
    return data


def final_result_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    final_result = payload.get("final_result")
    if isinstance(final_result, dict):
        return final_result
    if all(key in payload for key in ("score", "max_score", "pass_rate")):
        return payload
    return None


def parse_datetime(value: object, path: Path) -> datetime:
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.min.replace(tzinfo=timezone.utc)


def safe_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def positive_number(value: object) -> float | None:
    number = safe_number(value)
    if number is not None and number > 0:
        return number
    return None


def wall_time_seconds_from_payload(
    payload: dict[str, Any],
    final_result: dict[str, Any],
    path: Path,
) -> float | None:
    del path
    metadata = payload.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}

    raw_wall = positive_number(metadata.get("raw_wall_time_seconds"))
    if raw_wall is not None:
        return raw_wall

    started_at = metadata.get("started_at")
    completed_at = metadata.get("completed_at")
    if isinstance(started_at, str) and isinstance(completed_at, str):
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            elapsed = (completed - started).total_seconds()
            if elapsed > 0:
                return elapsed
        except ValueError:
            pass

    for key in (
        "wall_time_seconds",
        "raw_client_wall_time_seconds",
        "client_wall_time_seconds",
    ):
        wall_time = positive_number(metadata.get(key))
        if wall_time is not None:
            return wall_time

    for key in ("raw_time_used", "time_used"):
        wall_time = positive_number(final_result.get(key))
        if wall_time is not None:
            return wall_time

    return None


def split_counts(final_result: dict[str, Any]) -> dict[str, int]:
    detailed = final_result.get("detailed_results_by_split")
    if isinstance(detailed, dict):
        counts = {}
        for split in SPLITS:
            rows = detailed.get(split)
            if isinstance(rows, list):
                task_ids = {
                    str(row.get("task_id"))
                    for row in rows
                    if isinstance(row, dict) and row.get("task_id") is not None
                }
                counts[split] = len(task_ids)
        if counts:
            return counts

    rewards = final_result.get("task_rewards_by_split")
    if isinstance(rewards, dict):
        counts = {}
        for split in SPLITS:
            split_rewards = rewards.get(split)
            if isinstance(split_rewards, dict):
                counts[split] = len(split_rewards)
        if counts:
            return counts

    return {}


def config_marks_full_test_set(config: dict[str, Any]) -> bool | None:
    task_split = str(config.get("task_split", "test")).lower()
    if task_split != "test":
        return False

    saw_split_key = False
    for split in SPLITS:
        filter_value = config.get(f"tasks_{split}_task_id_filter")
        if filter_value:
            return False

        num_key = f"tasks_{split}_num_tasks"
        if num_key not in config:
            continue
        saw_split_key = True
        try:
            if int(config[num_key]) != -1:
                return False
        except (TypeError, ValueError):
            return False

    return True if saw_split_key else None


def task_label_marks_full_test_set(task_selection: object) -> bool | None:
    if not isinstance(task_selection, str):
        return None
    label = task_selection.lower()
    if "train" in label:
        return False
    if all(piece in label for piece in ("baseall", "hallall", "disall")):
        return True
    if all(piece in label for piece in ("base100", "hall98", "dis56")):
        return True
    return None


def is_full_test_set(
    payload: dict[str, Any],
    final_result: dict[str, Any],
    counts: dict[str, int],
    expected_counts: dict[str, int],
) -> bool:
    metadata = payload.get("metadata", {})
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if isinstance(config, dict):
        config_status = config_marks_full_test_set(config)
        if config_status is not None:
            return config_status

    task_selection = metadata.get("task_selection") if isinstance(metadata, dict) else None
    label_status = task_label_marks_full_test_set(task_selection)
    if label_status is not None:
        return label_status

    if counts:
        return all(counts.get(split, 0) >= expected_counts[split] for split in SPLITS)

    return False


def normalize_percent(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    score = float(value)
    if score <= 1.0:
        return score * 100.0
    return score


def overall_pass_rate(final_result: dict[str, Any]) -> float | None:
    score = final_result.get("score")
    max_score = final_result.get("max_score")
    if isinstance(score, (int, float)) and isinstance(max_score, (int, float)) and max_score:
        return float(score) / float(max_score) * 100.0

    pass_rate = final_result.get("pass_rate")
    if isinstance(pass_rate, (int, float)):
        return float(pass_rate)

    return None


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_ms(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def a2a_turn_times_from_row(row: dict[str, Any]) -> list[float]:
    row_values = row.get("a2a_turn_times_ms")
    if isinstance(row_values, list):
        values = [
            float(value)
            for value in row_values
            if isinstance(value, (int, float))
        ]
        if values:
            return values

    scalar_value = numeric_ms(row.get("a2a_turn_time_ms"))
    if scalar_value is not None:
        return [scalar_value]

    values = []
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list):
        return values

    for message in trajectory:
        if not isinstance(message, dict):
            continue

        evaluator_metrics = message.get("evaluator_metrics")
        if isinstance(evaluator_metrics, dict):
            value = numeric_ms(evaluator_metrics.get("a2a_turn_time_ms"))
            if value is not None:
                values.append(value)
                continue

        turn_metrics = message.get("turn_metrics")
        if isinstance(turn_metrics, dict):
            value = numeric_ms(turn_metrics.get("raw_turn_time_ms"))
            if value is None:
                value = numeric_ms(turn_metrics.get("turn_time_ms"))
            if value is not None:
                values.append(value)

    return values


def a2a_task_time_from_row(row: dict[str, Any]) -> float | None:
    total = numeric_ms(row.get("total_a2a_time_ms"))
    if total is not None:
        return total

    turn_times = a2a_turn_times_from_row(row)
    if turn_times:
        return sum(turn_times)

    return None


def latency_values_from_rows(
    rows: list[Any],
    source: str,
    scope: str,
) -> list[float]:
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        if source == "llm":
            value = numeric_ms(row.get("total_llm_latency_ms"))
            if value is not None:
                values.append(value)
            continue

        if scope == "per-turn":
            values.extend(a2a_turn_times_from_row(row))
        else:
            value = a2a_task_time_from_row(row)
            if value is not None:
                values.append(value)

    return values


def aggregate_latency(values_ms: list[float], aggregation: str) -> float | None:
    if not values_ms:
        return None
    if aggregation == "mean":
        return sum(values_ms) / len(values_ms) / 1000.0
    if aggregation == "median":
        return percentile(values_ms, 0.5) / 1000.0
    if aggregation == "p95":
        return percentile(values_ms, 0.95) / 1000.0
    if aggregation == "total":
        return sum(values_ms) / 1000.0 / 60.0
    raise ValueError(f"Unsupported latency aggregation: {aggregation}")


def latency_dict(
    final_result: dict[str, Any],
    aggregation: str,
    source: str,
    scope: str,
) -> dict[str, float]:
    detailed = final_result.get("detailed_results_by_split")
    if not isinstance(detailed, dict):
        return {}

    latencies: dict[str, float] = {}
    overall_values: list[float] = []
    effective_aggregation = "total" if scope == "total" else aggregation

    for split in SPLITS:
        rows = detailed.get(split)
        if not isinstance(rows, list):
            continue
        values_ms = latency_values_from_rows(rows, source, scope)
        overall_values.extend(values_ms)
        aggregate = aggregate_latency(values_ms, effective_aggregation)
        if aggregate is not None:
            latencies[split] = aggregate

    overall = aggregate_latency(overall_values, effective_aggregation)
    if overall is not None:
        latencies["overall"] = overall

    return latencies


def token_components_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    input_tokens = numeric_ms(row.get("agent_input_tokens"))
    output_tokens = numeric_ms(row.get("agent_output_tokens"))
    prompt = numeric_ms(row.get("agent_prompt_tokens"))
    completion = numeric_ms(row.get("agent_completion_tokens"))
    thinking = numeric_ms(row.get("agent_thinking_tokens"))
    total = numeric_ms(row.get("agent_total_tokens"))

    if any(
        value is not None
        for value in (
            input_tokens,
            output_tokens,
            prompt,
            completion,
            thinking,
            total,
        )
    ):
        prompt = prompt or 0.0
        completion = completion or 0.0
        thinking = thinking or 0.0
        input_tokens = input_tokens if input_tokens is not None else prompt
        output_tokens = (
            output_tokens
            if output_tokens is not None
            else completion + thinking
        )
        total = total if total is not None else input_tokens + output_tokens
        return {
            "input": input_tokens,
            "output": output_tokens,
            "prompt": prompt,
            "completion": completion,
            "thinking": thinking,
            "total": total,
        }

    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list):
        return None

    prompt = 0.0
    completion = 0.0
    thinking = 0.0
    saw_metrics = False
    for message in trajectory:
        if not isinstance(message, dict):
            continue
        metrics = message.get("turn_metrics")
        if not isinstance(metrics, dict):
            continue
        saw_metrics = True
        prompt += numeric_ms(metrics.get("prompt_tokens")) or 0.0
        completion += numeric_ms(metrics.get("completion_tokens")) or 0.0
        thinking += numeric_ms(metrics.get("thinking_tokens")) or 0.0

    if not saw_metrics:
        return None

    return {
        "input": prompt,
        "output": completion + thinking,
        "prompt": prompt,
        "completion": completion,
        "thinking": thinking,
        "total": prompt + completion + thinking,
    }


def token_values_from_rows(rows: list[Any], kind: str) -> list[float]:
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        components = token_components_from_row(row)
        if components is None:
            continue
        values.append(components[kind])
    return values


def aggregate_tokens(
    values: list[float],
    scope: str,
    wall_time_seconds: float | None,
) -> float | None:
    if not values:
        return None
    if scope == "per-task":
        return sum(values) / len(values)
    if scope == "total":
        return sum(values)
    if scope == "per-hour":
        if wall_time_seconds is None:
            return None
        hours = wall_time_seconds / 3600.0
        if hours <= 0:
            return None
        return sum(values) / hours
    raise ValueError(f"Unsupported token scope: {scope}")


def token_dict(
    final_result: dict[str, Any],
    kind: str,
    scope: str,
    wall_time_seconds: float | None,
) -> dict[str, float]:
    detailed = final_result.get("detailed_results_by_split")
    if not isinstance(detailed, dict):
        return {}

    tokens: dict[str, float] = {}
    overall_values: list[float] = []

    for split in SPLITS:
        rows = detailed.get(split)
        if not isinstance(rows, list):
            continue
        values = token_values_from_rows(rows, kind)
        overall_values.extend(values)
        aggregate = aggregate_tokens(values, scope, wall_time_seconds)
        if aggregate is not None:
            tokens[split] = aggregate

    overall = aggregate_tokens(overall_values, scope, wall_time_seconds)
    if overall is not None:
        tokens["overall"] = overall

    return tokens


def token_breakdown_dict(
    final_result: dict[str, Any],
    scope: str,
    wall_time_seconds: float | None,
) -> dict[str, dict[str, float]]:
    input_tokens = token_dict(final_result, "input", scope, wall_time_seconds)
    output_tokens = token_dict(final_result, "output", scope, wall_time_seconds)
    total_tokens = token_dict(final_result, "total", scope, wall_time_seconds)
    panels = set(input_tokens) | set(output_tokens) | set(total_tokens)
    return {
        panel: {
            "input": input_tokens.get(panel, 0.0),
            "output": output_tokens.get(panel, 0.0),
            "total": total_tokens.get(
                panel,
                input_tokens.get(panel, 0.0) + output_tokens.get(panel, 0.0),
            ),
        }
        for panel in panels
    }


def overall_consumed_token_summary(
    final_result: dict[str, Any],
    wall_time_seconds: float | None,
) -> dict[str, dict[str, float]]:
    summary = {}
    for scope in ("per-task", "total", "per-hour"):
        values = token_breakdown_dict(
            final_result,
            scope,
            wall_time_seconds,
        ).get("overall")
        if values:
            summary[scope] = values
    return summary


def metric_dict(final_result: dict[str, Any], metric: str) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if metric == "pass_rate":
        overall = overall_pass_rate(final_result)
        if overall is not None:
            metrics["overall"] = overall

        detailed = final_result.get("detailed_results_by_split")
        if isinstance(detailed, dict):
            for split in SPLITS:
                rows = detailed.get(split)
                if not isinstance(rows, list) or not rows:
                    continue
                rewards = [
                    float(row.get("reward", 0.0))
                    for row in rows
                    if isinstance(row, dict) and isinstance(row.get("reward", 0.0), (int, float))
                ]
                if rewards:
                    metrics[split] = sum(rewards) / len(rewards) * 100.0
        return metrics

    if metric.startswith("Pass^"):
        overall_scores = final_result.get("pass_power_k_scores")
        split_scores = final_result.get("pass_power_k_scores_by_split")
    else:
        overall_scores = final_result.get("pass_at_k_scores")
        split_scores = final_result.get("pass_at_k_scores_by_split")

    if isinstance(overall_scores, dict):
        overall = normalize_percent(overall_scores.get(metric))
        if overall is not None:
            metrics["overall"] = overall

    split_values = []
    if isinstance(split_scores, dict):
        for split in SPLITS:
            split_metric = split_scores.get(split)
            if not isinstance(split_metric, dict):
                continue
            value = normalize_percent(split_metric.get(metric))
            if value is None:
                continue
            metrics[split] = value
            split_values.append(value)

    if "overall" not in metrics and split_values:
        metrics["overall"] = sum(split_values) / len(split_values)

    return metrics


def base_model_label(payload: dict[str, Any], path: Path) -> tuple[str, str | None]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return path.parent.name, None

    model = metadata.get("model")
    if not model:
        agent_metadata = metadata.get("agent_metadata")
        if isinstance(agent_metadata, dict):
            model = agent_metadata.get("result_model")
    if not model:
        model = metadata.get("agent_name")
    if not model:
        model = path.parent.name

    reasoning_effort = metadata.get("reasoning_effort")
    if not reasoning_effort:
        agent_metadata = metadata.get("agent_metadata")
        if isinstance(agent_metadata, dict):
            reasoning_effort = agent_metadata.get("result_reasoning_effort")

    return str(model), str(reasoning_effort) if reasoning_effort else None


def read_runs(
    input_dir: Path,
    metric: str,
    expected_counts: dict[str, int],
    latency_aggregation: str,
    latency_source: str,
    latency_scope: str,
    token_kind: str,
    token_scope: str,
) -> list[RunResult]:
    runs = []
    if not input_dir.exists():
        print(f"No input directory found: {input_dir}")
        return runs

    for path in sorted(input_dir.rglob("*.json")):
        payload = load_json(path)
        if payload is None:
            continue

        final_result = final_result_from_payload(payload)
        if final_result is None:
            print(f"Skipping JSON without a final result: {path}")
            continue

        counts = split_counts(final_result)
        if not is_full_test_set(payload, final_result, counts, expected_counts):
            print(f"Skipping non-full-test result: {path}")
            continue

        metrics = metric_dict(final_result, metric)
        if not metrics:
            print(f"Skipping result without {metric}: {path}")
            continue
        latencies = latency_dict(
            final_result,
            latency_aggregation,
            latency_source,
            latency_scope,
        )
        wall_time_seconds = wall_time_seconds_from_payload(
            payload,
            final_result,
            path,
        )
        tokens = token_dict(
            final_result,
            token_kind,
            token_scope,
            wall_time_seconds,
        )
        token_breakdown = token_breakdown_dict(
            final_result,
            token_scope,
            wall_time_seconds,
        )
        token_summary = overall_consumed_token_summary(
            final_result,
            wall_time_seconds,
        )

        metadata = payload.get("metadata", {})
        completed_at = parse_datetime(
            metadata.get("completed_at") if isinstance(metadata, dict) else None,
            path,
        )
        model, reasoning_effort = base_model_label(payload, path)
        runs.append(
            RunResult(
                path=path,
                final_result=final_result,
                model=model,
                reasoning_effort=reasoning_effort,
                completed_at=completed_at,
                metrics=metrics,
                latencies=latencies,
                tokens=tokens,
                token_breakdown=token_breakdown,
                token_summary=token_summary,
                wall_time_seconds=wall_time_seconds,
                counts=counts,
            )
        )

    return runs


def disambiguated_label(run: RunResult, duplicate_models: set[str]) -> str:
    if run.model not in duplicate_models:
        return run.model
    if run.reasoning_effort:
        return f"{run.model}\n{run.reasoning_effort}"
    return f"{run.model}\n{run.path.parent.name}"


def latest_per_label(runs: list[RunResult]) -> list[RunResult]:
    latest: dict[tuple[str, str | None], RunResult] = {}
    for run in runs:
        key = (run.model, run.reasoning_effort)
        previous = latest.get(key)
        if previous is None or run.completed_at > previous.completed_at:
            latest[key] = run
    return sorted(latest.values(), key=lambda run: (run.model.lower(), run.reasoning_effort or ""))


def apply_latest_filter(runs: list[RunResult], keep_duplicates: bool) -> list[RunResult]:
    if keep_duplicates:
        return runs
    before = len(runs)
    latest_runs = latest_per_label(runs)
    dropped = before - len(latest_runs)
    if dropped:
        print(f"Using latest run per model/effort label; dropped {dropped} older duplicate(s).")
    return latest_runs


def runs_with_metric(runs: list[RunResult], metric: str) -> list[RunResult]:
    metric_runs = []
    for run in runs:
        metrics = metric_dict(run.final_result, metric)
        if metrics:
            metric_runs.append(replace(run, metrics=metrics))
    return metric_runs


def runs_with_latency(
    runs: list[RunResult],
    aggregation: str,
    source: str,
    scope: str,
) -> list[RunResult]:
    return [
        replace(
            run,
            latencies=latency_dict(run.final_result, aggregation, source, scope),
        )
        for run in runs
    ]


def runs_with_tokens(runs: list[RunResult], kind: str, scope: str) -> list[RunResult]:
    return [
        replace(
            run,
            tokens=token_dict(
                run.final_result,
                kind,
                scope,
                run.wall_time_seconds,
            ),
            token_breakdown=token_breakdown_dict(
                run.final_result,
                scope,
                run.wall_time_seconds,
            ),
        )
        for run in runs
    ]


def configure_matplotlib(output_path: Path) -> Any:
    mpl_config_dir = output_path.parent / ".matplotlib-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    xdg_cache_dir = output_path.parent / ".cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to create the plot. Run this from the repo virtualenv, "
            "for example: .venv/bin/python outputs/plot_results.py"
        ) from exc

    return plt


def plot_results(
    runs: list[RunResult],
    metric: str,
    output_path: Path,
    title: str | None,
    width: float | None,
    height: float,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(output_path)

    duplicate_models = {
        model
        for model in {run.model for run in runs}
        if sum(1 for run in runs if run.model == model) > 1
    }
    labels_by_path = {run.path: disambiguated_label(run, duplicate_models) for run in runs}
    max_panel_size = max(
        sum(1 for run in runs if panel in run.metrics)
        for panel in ("overall", *SPLITS)
    )
    figure_width = width or max(12.0, min(26.0, 8.0 + max_panel_size * 0.65))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(figure_width, height), constrained_layout=True)
    fig.suptitle(
        title or f"CAR-bench full test-set results ({metric})",
        fontsize=16,
        fontweight="bold",
    )

    panels = [
        ("overall", "Overall", "#334155"),
        ("base", "Base Only", "#2563eb"),
        ("hallucination", "Hallucination Only", "#d97706"),
        ("disambiguation", "Disambiguation Only", "#059669"),
    ]

    for ax, (key, panel_title, color) in zip(axes.flat, panels, strict=True):
        panel_runs = sorted(
            (run for run in runs if key in run.metrics),
            key=lambda run: (run.metrics[key], labels_by_path[run.path].lower()),
        )
        values = [run.metrics[key] for run in panel_runs]
        labels = [labels_by_path[run.path] for run in panel_runs]

        ax.set_title(panel_title, pad=10, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"{metric} (%)")
        ax.grid(axis="y", color="#d9dde7", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")

        if not panel_runs:
            ax.text(
                0.5,
                0.5,
                f"No {metric} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#64748b",
            )
            ax.set_xticks([])
            continue

        x_positions = list(range(len(panel_runs)))
        bars = ax.bar(
            x_positions,
            values,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            width=0.72,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=3,
            fontsize=8,
            color="#111827",
        )
        ax.set_xticks(x_positions)
        label_rotation = 35 if len(labels) > 1 else 0
        label_alignment = "right" if len(labels) > 1 else "center"
        ax.set_xticklabels(labels, rotation=label_rotation, ha=label_alignment)
        ax.margins(x=0.03)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def latency_axis_label(source: str, scope: str, aggregation: str) -> str:
    effective_aggregation = "total" if scope == "total" else aggregation
    if effective_aggregation == "total":
        if source == "llm":
            return "Total agent-reported LLM latency (min)"
        return "Total evaluator-measured A2A response time (min)"

    statistic = {
        "mean": "Mean",
        "median": "Median",
        "p95": "P95",
        "total": "Total",
    }[effective_aggregation]

    if source == "llm":
        return f"{statistic} agent-reported LLM latency per task trial (s)"
    if scope == "per-turn":
        return f"{statistic} evaluator-measured A2A response time per turn (s)"
    return f"{statistic} summed A2A response time per task trial (s)"


def latency_chart_title(source: str, scope: str, aggregation: str) -> str:
    effective_aggregation = "total" if scope == "total" else aggregation
    if source == "llm":
        return (
            "Agent-reported LLM latency per task: "
            f"{effective_aggregation} (advisory)"
        )
    if effective_aggregation == "total":
        return "Evaluator-measured total A2A response time (lower is better)"
    scope_label = {
        "per-turn": "A2A response time per turn",
        "per-task": "summed A2A response time per task",
        "total": "total A2A response time",
    }[scope]
    return f"CAR-bench {scope_label}: {effective_aggregation} (lower is better)"


def format_latency_label(value: float, source: str, scope: str, aggregation: str) -> str:
    del source
    effective_aggregation = "total" if scope == "total" else aggregation
    if effective_aggregation == "total":
        return f"{value:.1f}m"
    if value >= 100:
        return f"{value:.0f}s"
    return f"{value:.1f}s"


def plot_latency_results(
    runs: list[RunResult],
    aggregation: str,
    source: str,
    scope: str,
    output_path: Path,
    title: str | None,
    width: float | None,
    height: float,
    dpi: int,
) -> None:
    latency_runs = [run for run in runs if run.latencies]
    if not latency_runs:
        print(f"No {source} {scope} latency data found; skipped latency plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(output_path)

    duplicate_models = {
        model
        for model in {run.model for run in latency_runs}
        if sum(1 for run in latency_runs if run.model == model) > 1
    }
    labels_by_path = {
        run.path: disambiguated_label(run, duplicate_models)
        for run in latency_runs
    }
    max_panel_size = max(
        sum(1 for run in latency_runs if panel in run.latencies)
        for panel in ("overall", *SPLITS)
    )
    max_value = max(
        value
        for run in latency_runs
        for value in run.latencies.values()
    )
    figure_width = width or max(12.0, min(26.0, 8.0 + max_panel_size * 0.65))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(figure_width, height), constrained_layout=True)
    fig.suptitle(
        title or latency_chart_title(source, scope, aggregation),
        fontsize=16,
        fontweight="bold",
    )

    panels = [
        ("overall", "Overall", "#475569"),
        ("base", "Base Only", "#0ea5e9"),
        ("hallucination", "Hallucination Only", "#f59e0b"),
        ("disambiguation", "Disambiguation Only", "#10b981"),
    ]
    y_label = latency_axis_label(source, scope, aggregation)
    y_max = max(max_value * 1.18, 1.0)

    for ax, (key, panel_title, color) in zip(axes.flat, panels, strict=True):
        panel_runs = sorted(
            (run for run in latency_runs if key in run.latencies),
            key=lambda run: (run.latencies[key], labels_by_path[run.path].lower()),
        )
        values = [run.latencies[key] for run in panel_runs]
        labels = [labels_by_path[run.path] for run in panel_runs]

        ax.set_title(panel_title, pad=10, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#d9dde7", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")

        if not panel_runs:
            ax.text(
                0.5,
                0.5,
                "No latency data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#64748b",
            )
            ax.set_xticks([])
            continue

        x_positions = list(range(len(panel_runs)))
        bars = ax.bar(
            x_positions,
            values,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            width=0.72,
        )
        ax.bar_label(
            bars,
            labels=[
                format_latency_label(value, source, scope, aggregation)
                for value in values
            ],
            padding=3,
            fontsize=8,
            color="#111827",
        )
        ax.set_xticks(x_positions)
        label_rotation = 35 if len(labels) > 1 else 0
        label_alignment = "right" if len(labels) > 1 else "center"
        ax.set_xticklabels(labels, rotation=label_rotation, ha=label_alignment)
        ax.margins(x=0.03)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def token_kind_label(kind: str) -> str:
    return {
        "total": "total",
        "input": "input",
        "output": "output",
        "prompt": "prompt",
        "completion": "completion",
        "thinking": "thinking",
    }[kind]


def token_axis_label(kind: str, scope: str) -> str:
    label = token_kind_label(kind)
    if scope == "per-task":
        return f"Mean agent {label} tokens per task trial"
    if scope == "total":
        return f"Total agent {label} tokens"
    return f"Agent {label} tokens per benchmark wall-clock hour"


def token_chart_title(kind: str, scope: str) -> str:
    label = token_kind_label(kind)
    scope_label = {
        "per-task": "per task",
        "total": "total",
        "per-hour": "per benchmark wall-clock hour",
    }[scope]
    return f"CAR-bench agent {label} tokens: {scope_label} (lower is less)"


def compact_token_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def format_token_label(value: float, scope: str) -> str:
    suffix = "/h" if scope == "per-hour" else ""
    return f"{compact_token_label(value)}{suffix}"


def print_token_summary(runs: list[RunResult]) -> None:
    summary_runs = [run for run in runs if run.token_summary]
    if not summary_runs:
        return

    duplicate_models = {
        model
        for model in {run.model for run in summary_runs}
        if sum(1 for run in summary_runs if run.model == model) > 1
    }
    labels_by_path = {
        run.path: disambiguated_label(run, duplicate_models)
        for run in summary_runs
    }

    print("Agent consumed token summary (overall):")
    for run in sorted(summary_runs, key=lambda item: labels_by_path[item.path].lower()):
        pieces = []
        labels = {
            "per-task": "per task",
            "total": "total",
            "per-hour": "per wall-clock hour",
        }
        for scope, label in labels.items():
            values = run.token_summary.get(scope)
            if not values:
                continue
            pieces.append(
                f"{compact_token_label(values.get('input', 0.0))} in + "
                f"{compact_token_label(values.get('output', 0.0))} out "
                f"({compact_token_label(values.get('total', 0.0))} {label})"
            )
        if pieces:
            print(f"  {labels_by_path[run.path]}: {', '.join(pieces)}")


def token_breakdown_axis_label(scope: str) -> str:
    if scope == "per-task":
        return "Mean consumed tokens per task trial"
    if scope == "total":
        return "Total consumed tokens"
    return "Consumed tokens per benchmark wall-clock hour"


def token_breakdown_chart_title(scope: str) -> str:
    scope_label = {
        "per-task": "per task",
        "total": "total",
        "per-hour": "per benchmark wall-clock hour",
    }[scope]
    return f"CAR-bench consumed tokens: input vs output ({scope_label})"


def plot_consumed_token_results(
    runs: list[RunResult],
    scope: str,
    output_path: Path,
    title: str | None,
    width: float | None,
    height: float,
    dpi: int,
) -> None:
    token_runs = [run for run in runs if run.token_breakdown]
    if not token_runs:
        print("No input/output token data found; skipped consumed-token plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(output_path)

    duplicate_models = {
        model
        for model in {run.model for run in token_runs}
        if sum(1 for run in token_runs if run.model == model) > 1
    }
    labels_by_path = {
        run.path: disambiguated_label(run, duplicate_models)
        for run in token_runs
    }
    max_panel_size = max(
        sum(1 for run in token_runs if panel in run.token_breakdown)
        for panel in ("overall", *SPLITS)
    )
    max_value = max(
        values.get("total", 0.0)
        for run in token_runs
        for values in run.token_breakdown.values()
    )
    figure_width = width or max(12.0, min(26.0, 8.0 + max_panel_size * 0.65))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(figure_width, height), constrained_layout=True)
    fig.suptitle(
        title or token_breakdown_chart_title(scope),
        fontsize=16,
        fontweight="bold",
    )

    panels = [
        ("overall", "Overall"),
        ("base", "Base Only"),
        ("hallucination", "Hallucination Only"),
        ("disambiguation", "Disambiguation Only"),
    ]
    y_label = token_breakdown_axis_label(scope)
    y_max = max(max_value * 1.18, 1.0)
    input_color = "#2563eb"
    output_color = "#f97316"

    for ax, (key, panel_title) in zip(axes.flat, panels, strict=True):
        panel_runs = sorted(
            (run for run in token_runs if key in run.token_breakdown),
            key=lambda run: (
                run.token_breakdown[key].get("total", 0.0),
                labels_by_path[run.path].lower(),
            ),
        )
        labels = [labels_by_path[run.path] for run in panel_runs]
        input_values = [
            run.token_breakdown[key].get("input", 0.0)
            for run in panel_runs
        ]
        output_values = [
            run.token_breakdown[key].get("output", 0.0)
            for run in panel_runs
        ]
        total_values = [
            run.token_breakdown[key].get(
                "total",
                run.token_breakdown[key].get("input", 0.0)
                + run.token_breakdown[key].get("output", 0.0),
            )
            for run in panel_runs
        ]

        ax.set_title(panel_title, pad=10, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#d9dde7", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")

        if not panel_runs:
            ax.text(
                0.5,
                0.5,
                "No token data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#64748b",
            )
            ax.set_xticks([])
            continue

        x_positions = list(range(len(panel_runs)))
        input_bars = ax.bar(
            x_positions,
            input_values,
            color=input_color,
            edgecolor="white",
            linewidth=0.9,
            width=0.72,
            label="Input",
        )
        output_bars = ax.bar(
            x_positions,
            output_values,
            bottom=input_values,
            color=output_color,
            edgecolor="white",
            linewidth=0.9,
            width=0.72,
            label="Output",
        )
        del input_bars, output_bars
        for x_position, total in zip(x_positions, total_values, strict=True):
            ax.text(
                x_position,
                total,
                format_token_label(total, scope),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111827",
            )
        ax.set_xticks(x_positions)
        label_rotation = 35 if len(labels) > 1 else 0
        label_alignment = "right" if len(labels) > 1 else "center"
        ax.set_xticklabels(labels, rotation=label_rotation, ha=label_alignment)
        ax.margins(x=0.03)
        ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_token_results(
    runs: list[RunResult],
    kind: str,
    scope: str,
    output_path: Path,
    title: str | None,
    width: float | None,
    height: float,
    dpi: int,
) -> None:
    token_runs = [run for run in runs if run.tokens]
    if not token_runs:
        print(f"No {kind} token data found; skipped token plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(output_path)

    duplicate_models = {
        model
        for model in {run.model for run in token_runs}
        if sum(1 for run in token_runs if run.model == model) > 1
    }
    labels_by_path = {
        run.path: disambiguated_label(run, duplicate_models)
        for run in token_runs
    }
    max_panel_size = max(
        sum(1 for run in token_runs if panel in run.tokens)
        for panel in ("overall", *SPLITS)
    )
    max_value = max(
        value
        for run in token_runs
        for value in run.tokens.values()
    )
    figure_width = width or max(12.0, min(26.0, 8.0 + max_panel_size * 0.65))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(figure_width, height), constrained_layout=True)
    fig.suptitle(
        title or token_chart_title(kind, scope),
        fontsize=16,
        fontweight="bold",
    )

    panels = [
        ("overall", "Overall", "#374151"),
        ("base", "Base Only", "#7c3aed"),
        ("hallucination", "Hallucination Only", "#dc2626"),
        ("disambiguation", "Disambiguation Only", "#0891b2"),
    ]
    y_label = token_axis_label(kind, scope)
    y_max = max(max_value * 1.18, 1.0)

    for ax, (key, panel_title, color) in zip(axes.flat, panels, strict=True):
        panel_runs = sorted(
            (run for run in token_runs if key in run.tokens),
            key=lambda run: (run.tokens[key], labels_by_path[run.path].lower()),
        )
        values = [run.tokens[key] for run in panel_runs]
        labels = [labels_by_path[run.path] for run in panel_runs]

        ax.set_title(panel_title, pad=10, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#d9dde7", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")

        if not panel_runs:
            ax.text(
                0.5,
                0.5,
                "No token data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#64748b",
            )
            ax.set_xticks([])
            continue

        x_positions = list(range(len(panel_runs)))
        bars = ax.bar(
            x_positions,
            values,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            width=0.72,
        )
        ax.bar_label(
            bars,
            labels=[format_token_label(value, scope) for value in values],
            padding=3,
            fontsize=8,
            color="#111827",
        )
        ax.set_xticks(x_positions)
        label_rotation = 35 if len(labels) > 1 else 0
        label_alignment = "right" if len(labels) > 1 else "center"
        ax.set_xticklabels(labels, rotation=label_rotation, ha=label_alignment)
        ax.margins(x=0.03)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_base_runs(args: argparse.Namespace, metric: str) -> list[RunResult]:
    runs = read_runs(
        args.input_dir,
        metric,
        expected_counts_from_args(args),
        "mean",
        "a2a",
        "per-task",
        "total",
        "per-task",
    )
    return apply_latest_filter(runs, args.keep_duplicates)


def plot_score_report(
    runs: list[RunResult],
    metric: str,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    metric_runs = runs_with_metric(runs, metric)
    if not metric_runs:
        print(f"No {metric} data found; skipped score plot.")
        return
    plot_results(
        runs=metric_runs,
        metric=metric,
        output_path=output_path,
        title=args.title,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
    )
    print(f"Plotted {len(metric_runs)} full-test result(s) to {output_path}")


def write_report_notes(report_dir: Path, metric: str) -> None:
    notes = f"""# CAR-bench Plot Report

This report only includes completed full-test-set result JSON files.
Every plot uses the same four panels: Overall, Base Only, Hallucination Only,
and Disambiguation Only. Bars are ordered left-to-right by value.

## Scores

`scores/{metric_filename(metric)}.png` plots `{metric}` as a percentage.
Higher is better.

## Latency

Latency is shown in `latency/`.

- `a2a_per_turn_mean.png`: evaluator-measured wall time around each A2A call
  from the evaluator to the agent under test, averaged over agent turns.
- `a2a_per_task_mean.png`: sum of evaluator-measured A2A response times within
  each task trial, averaged over task trials.
- `a2a_total.png`: total evaluator-measured A2A response time across all task
  trials, shown in minutes. This is active waiting time for agent responses, not
  full benchmark wall-clock time.
- `llm_per_task_mean_agent_reported.png`: agent-reported LLM latency metadata
  averaged per task trial. This is useful for profiling but is not as robust as
  evaluator-measured A2A timing.

## Tokens

Token charts are shown in `tokens/` and split consumed tokens into input and
output. Input means prompt/input tokens. Output means completion/output tokens
plus thinking/reasoning tokens when reported.

- `input_output_per_task.png`: mean input/output tokens per task trial.
- `input_output_total.png`: total input/output tokens over the full run.
- `input_output_per_wall_hour.png`: input/output tokens divided by benchmark
  wall-clock hours. This is the rate-limit-style view. Overall is total run
  tokens divided by total run wall-clock time. Split panels show each split's
  token contribution divided by the same full-run wall-clock time.

Token counts come from agent/provider-reported metadata. The per-hour
denominator uses true run wall-clock time when available: `raw_wall_time_seconds`
first, then `completed_at - started_at`, then older wall-time fallbacks.
"""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "README.md").write_text(notes, encoding="utf-8")


def generate_full_report(
    runs: list[RunResult],
    metric: str,
    args: argparse.Namespace,
) -> None:
    report_dir = args.report_dir
    print(f"Writing CAR-bench plot report to {report_dir}")
    write_report_notes(report_dir, metric)

    plot_score_report(
        runs,
        metric,
        report_dir / "scores" / f"{metric_filename(metric)}.png",
        args,
    )

    if not args.skip_latency:
        latency_specs = [
            ("a2a", "per-turn", "mean", "a2a_per_turn_mean.png"),
            ("a2a", "per-task", "mean", "a2a_per_task_mean.png"),
            ("a2a", "total", "total", "a2a_total.png"),
            ("llm", "per-task", "mean", "llm_per_task_mean_agent_reported.png"),
        ]
        for source, scope, aggregation, filename in latency_specs:
            latency_runs = runs_with_latency(runs, aggregation, source, scope)
            output_path = report_dir / "latency" / filename
            plot_latency_results(
                runs=latency_runs,
                aggregation=aggregation,
                source=source,
                scope=scope,
                output_path=output_path,
                title=None,
                width=args.width,
                height=args.height,
                dpi=args.dpi,
            )
            if any(run.latencies for run in latency_runs):
                print(f"Plotted latency report to {output_path}")

    if not args.skip_tokens:
        token_specs = [
            ("per-task", "input_output_per_task.png"),
            ("total", "input_output_total.png"),
            ("per-hour", "input_output_per_wall_hour.png"),
        ]
        for scope, filename in token_specs:
            token_runs = runs_with_tokens(runs, "total", scope)
            output_path = report_dir / "tokens" / filename
            plot_consumed_token_results(
                runs=token_runs,
                scope=scope,
                output_path=output_path,
                title=None,
                width=args.width,
                height=args.height,
                dpi=args.dpi,
            )
            if any(run.token_breakdown for run in token_runs):
                print(f"Plotted consumed-token report to {output_path}")

    print_token_summary(runs)


def generate_single_plots(
    runs: list[RunResult],
    metric: str,
    args: argparse.Namespace,
) -> None:
    plot_score_report(runs, metric, args.output, args)

    if not args.skip_latency:
        if args.latency_source == "llm" and args.latency_scope != "per-task":
            raise SystemExit("--latency-source llm only supports --latency-scope per-task.")
        latency_runs = runs_with_latency(
            runs,
            aggregation=args.latency_aggregation,
            source=args.latency_source,
            scope=args.latency_scope,
        )
        plot_latency_results(
            runs=latency_runs,
            aggregation=args.latency_aggregation,
            source=args.latency_source,
            scope=args.latency_scope,
            output_path=args.latency_output,
            title=None,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
        if any(run.latencies for run in latency_runs):
            effective_latency_aggregation = (
                "total"
                if args.latency_scope == "total"
                else args.latency_aggregation
            )
            latency_description = (
                f"{args.latency_source} total latency"
                if args.latency_scope == "total"
                else (
                    f"{args.latency_source} {args.latency_scope} "
                    f"{effective_latency_aggregation} latency"
                )
            )
            print(
                f"Plotted {latency_description} for "
                f"{sum(1 for run in latency_runs if run.latencies)} full-test result(s) "
                f"to {args.latency_output}"
            )

    if not args.skip_tokens:
        token_runs = runs_with_tokens(runs, args.token_kind, args.token_scope)
        if args.token_kind == "total":
            plot_consumed_token_results(
                runs=token_runs,
                scope=args.token_scope,
                output_path=args.token_output,
                title=None,
                width=args.width,
                height=args.height,
                dpi=args.dpi,
            )
        else:
            plot_token_results(
                runs=token_runs,
                kind=args.token_kind,
                scope=args.token_scope,
                output_path=args.token_output,
                title=None,
                width=args.width,
                height=args.height,
                dpi=args.dpi,
            )
        if any(run.tokens for run in token_runs):
            print(
                f"Plotted {args.token_kind} token {args.token_scope} summary for "
                f"{sum(1 for run in token_runs if run.tokens)} full-test result(s) "
                f"to {args.token_output}"
            )

    print_token_summary(runs)


def main() -> None:
    args = parse_args()
    metric = canonical_metric(args.metric)
    runs = load_base_runs(args, metric)
    if not runs:
        raise SystemExit(
            "No completed full-test result JSONs with the requested metric were found."
        )

    if args.single_plot:
        generate_single_plots(runs, metric, args)
    else:
        generate_full_report(runs, metric, args)


if __name__ == "__main__":
    main()
