"""Gate-first Track 2 executor: planner/executor + parallel verifier gate + aggregator.

Per benchmark step the executor branch (private planner -> executor next-action) and the
grounding gate run CONCURRENTLY. The gate depends only on the transcript + tool schemas, not
on the executor's output, so it is a PARALLEL branch: it adds ZERO sequential-call depth
(the ≤5-sequential-call budget is spent by planner->executor only). A deterministic aggregator
then decides — abstaining with an explicit "I can't do X" when a majority of gate votes find a
required tool missing and the executor was about to act. This is the paper's "stable
activation mechanism" for the completion-compliance constraint.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__:
    from .aggregator import aggregate
    from .gates import (
        AmbiguityGate,
        AmbiguityGateSettings,
        AmbiguityVerdict,
        GateSettings,
        GroundingGate,
        GroundingVerdict,
        PolicyGate,
        PolicyGateSettings,
        PolicyResponseRepairer,
        PolicyVerdict,
    )
    from .planner_agent import PlannerExecutorCARBenchAgentExecutor
else:  # pragma: no cover - direct-script import fallback
    from aggregator import aggregate
    from gates import (
        AmbiguityGate,
        AmbiguityGateSettings,
        AmbiguityVerdict,
        GateSettings,
        GroundingGate,
        GroundingVerdict,
        PolicyGate,
        PolicyGateSettings,
        PolicyResponseRepairer,
        PolicyVerdict,
    )
    from planner_agent import PlannerExecutorCARBenchAgentExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))
from track_2_agent_under_test_cerebras.car_bench_agent import (  # noqa: E402
    AgentInferenceResult,
)
from track_2_agent_under_test_cerebras.cerebras_client import (  # noqa: E402
    add_token_usage,
)
sys.path.pop(0)


class GatedPlannerExecutorCARBenchAgentExecutor(PlannerExecutorCARBenchAgentExecutor):
    """Planner/executor agent with a parallel grounding gate and asymmetric aggregation."""

    def __init__(
        self,
        *,
        enable_grounding_gate: bool = True,
        enable_ambiguity_gate: bool = True,
        enable_policy_gate: bool = False,
        gate_settings: GateSettings | None = None,
        ambiguity_gate_settings: AmbiguityGateSettings | None = None,
        policy_gate_settings: PolicyGateSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.enable_grounding_gate = enable_grounding_gate
        self.enable_ambiguity_gate = enable_ambiguity_gate
        self.enable_policy_gate = enable_policy_gate
        self._gate: GroundingGate | None = None
        self._ambiguity_gate: AmbiguityGate | None = None
        self._policy_gate: PolicyGate | None = None
        self._policy_repairer: PolicyResponseRepairer | None = None
        logger = cerebras_agent_logger_for(self)
        if enable_grounding_gate:
            settings = gate_settings or GateSettings(
                api_base=kwargs.get("api_base"),
                service_tier=kwargs.get("service_tier"),
            )
            self._gate = GroundingGate(settings=settings, logger=logger)
        if enable_ambiguity_gate:
            a_settings = ambiguity_gate_settings or AmbiguityGateSettings(
                api_base=kwargs.get("api_base"),
                service_tier=kwargs.get("service_tier"),
            )
            self._ambiguity_gate = AmbiguityGate(settings=a_settings, logger=logger)
        if enable_policy_gate:
            p_settings = policy_gate_settings or PolicyGateSettings(
                api_base=kwargs.get("api_base"),
                service_tier=kwargs.get("service_tier"),
            )
            self._policy_gate = PolicyGate(settings=p_settings, logger=logger)
            self._policy_repairer = PolicyResponseRepairer(
                settings=p_settings, logger=logger
            )

    def _call_model_with_retries(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
    ) -> AgentInferenceResult:
        run_grounding = self.enable_grounding_gate and self._gate is not None and bool(tools)
        run_ambiguity = (
            self.enable_ambiguity_gate and self._ambiguity_gate is not None and bool(tools)
        )
        run_policy = (
            self.enable_policy_gate and self._policy_gate is not None and bool(tools)
        )
        if not run_grounding and not run_ambiguity and not run_policy:
            return super()._call_model_with_retries(
                context_id=context_id,
                messages=messages,
                tools=tools,
                ctx_logger=ctx_logger,
            )

        instruction = _instruction_from_messages(messages)
        policy_wiki = _policy_wiki_from_messages(messages)
        gathered_results, gathered_tools = _gathered_from_messages(messages)

        # Executor branch + all gates run truly in parallel (the gates read only the transcript +
        # tool schemas + prior tool results, never the executor's output -> zero sequential depth).
        with ThreadPoolExecutor(max_workers=4) as pool:
            executor_future = pool.submit(
                super()._call_model_with_retries,
                context_id=context_id,
                messages=messages,
                tools=tools,
                ctx_logger=ctx_logger,
            )
            grounding_future = (
                pool.submit(
                    self._gate.evaluate,
                    instruction=instruction,
                    tools=tools,
                    policy_wiki=policy_wiki,
                )
                if run_grounding
                else None
            )
            ambiguity_future = (
                pool.submit(
                    self._ambiguity_gate.evaluate,
                    instruction=instruction,
                    tools=tools,
                    policy_wiki=policy_wiki,
                    gathered_results=gathered_results,
                    gathered_tools=gathered_tools,
                )
                if run_ambiguity
                else None
            )
            policy_future = (
                pool.submit(
                    self._policy_gate.evaluate,
                    instruction=instruction,
                    tools=tools,
                    policy_wiki=policy_wiki,
                    gathered_results=gathered_results,
                )
                if run_policy
                else None
            )
            # Surface an executor failure even if a gate finishes first.
            executor_result = executor_future.result()
            grounding = (
                grounding_future.result() if grounding_future is not None else GroundingVerdict()
            )
            ambiguity = (
                ambiguity_future.result() if ambiguity_future is not None else None
            )
            policy = policy_future.result() if policy_future is not None else None

        decision = aggregate(
            executor_result.next_action,
            grounding,
            ambiguity=ambiguity,
            policy=policy,
            tools=tools,
            gathered_tools=gathered_tools,
        )

        # Policy repair: one grounded rewrite of a respond that omitted a wiki informing obligation.
        # Sequential (depth +1), but only when the policy gate fires on such a respond -> the common
        # path is untouched. Falls back to the executor's own respond if the repair call fails.
        repair_result = None
        final_action = decision.action
        if (
            decision.repair_prompt
            and self._policy_repairer is not None
            and final_action.get("action") == "respond"
        ):
            repaired, repair_result = self._policy_repairer.repair(
                draft=str(final_action.get("content") or ""),
                obligation=decision.repair_prompt,
                gathered_results=gathered_results,
            )
            if repaired:
                final_action = {"action": "respond", "content": repaired}

        ctx_logger.info(
            "Gate aggregation",
            overridden=decision.overridden or final_action is not decision.action,
            reason=decision.reason,
            executor_action=executor_result.next_action.get("action"),
            policy_repaired=repair_result is not None,
            **decision.gate_summary,
        )

        return _merge_result(
            executor_result, grounding, ambiguity, policy, repair_result, final_action
        )


def _merge_result(
    executor_result: AgentInferenceResult,
    grounding: GroundingVerdict,
    ambiguity: AmbiguityVerdict | None,
    policy: PolicyVerdict | None,
    repair_result: Any,
    final_action: dict[str, Any],
) -> AgentInferenceResult:
    """Roll gate cost into the step result. Gate wall-clock is parallel (max); tokens/calls sum.

    The policy REPAIR call (when it fires) is SEQUENTIAL, so its duration ADDS to the parallel gate
    duration rather than max-ing with it -- this keeps the reported per-step latency honest.
    """
    gate_duration = grounding.duration_ms
    gate_tokens = grounding.token_usage
    gate_cost = grounding.cost
    gate_calls = grounding.internal_calls
    gate_quota_wait = grounding.quota_wait_ms
    if ambiguity is not None:
        gate_duration = max(gate_duration, ambiguity.duration_ms)
        gate_tokens = add_token_usage(gate_tokens, ambiguity.token_usage)
        gate_cost += ambiguity.cost
        gate_calls += ambiguity.internal_calls
        gate_quota_wait = max(gate_quota_wait, ambiguity.quota_wait_ms)
    if policy is not None:
        gate_duration = max(gate_duration, policy.duration_ms)
        gate_tokens = add_token_usage(gate_tokens, policy.token_usage)
        gate_cost += policy.cost
        gate_calls += policy.internal_calls
        gate_quota_wait = max(gate_quota_wait, policy.quota_wait_ms)
    repair_duration = 0.0
    if repair_result is not None:
        repair_duration = repair_result.duration_ms  # sequential -> adds below
        gate_tokens = add_token_usage(gate_tokens, repair_result.token_usage)
        gate_cost += repair_result.cost
        gate_calls += 1
        gate_quota_wait = max(gate_quota_wait, repair_result.quota_wait_ms)
    return AgentInferenceResult(
        next_action=final_action,
        elapsed_ms=max(executor_result.elapsed_ms, gate_duration) + repair_duration,
        token_usage=add_token_usage(executor_result.token_usage, gate_tokens),
        cost=executor_result.cost + gate_cost,
        internal_calls=max(executor_result.internal_calls, 1) + gate_calls,
        quota_wait_ms=max(executor_result.quota_wait_ms, gate_quota_wait),
    )


def _instruction_from_messages(messages: list[dict[str, Any]]) -> str:
    """Join user-role turns — the request (plus any clarifications) the gate reasons over."""
    parts = [
        str(m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user" and m.get("content")
    ]
    return "\n".join(p for p in parts if p)


def _policy_wiki_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "system" and message.get("content"):
            return str(message["content"])
    return None


def _tool_call_names(message: dict[str, Any]) -> list[str]:
    """Extract tool names from an assistant message's tool_calls, tolerating shape variants."""
    names: list[str] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        name = (fn or {}).get("name") if isinstance(fn, dict) else None
        name = name or call.get("tool_name") or call.get("name")
        if name:
            names.append(str(name))
    return names


def _gathered_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, set[str]]:
    """Compact rendering of tool calls + results already in the transcript, for the ambiguity gate.

    Returns (results_text, tool_names). The gate is TRANSCRIPT-AWARE: it needs to know which reads
    (get_user_preferences, get_weather, ...) already happened and what they returned, so it can
    route needs_gather -> resolved/needs_ask instead of asking on an internal task.
    """
    lines: list[str] = []
    tool_names: set[str] = set()
    pending: list[str] = []  # names from the most recent assistant tool_calls, awaiting results
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            called = _tool_call_names(message)
            tool_names.update(called)
            pending = called
        elif role == "tool":
            content = str(message.get("content") or "")
            name = pending.pop(0) if pending else str(message.get("name") or "tool")
            if content:
                lines.append(f"{name} -> {content[:600]}")
    return "\n".join(lines), tool_names


def cerebras_agent_logger_for(agent: Any):
    """Bind a gate-scoped logger off the executor's Cerebras client logger if present."""
    client = getattr(agent, "client", None)
    logger = getattr(client, "logger", None)
    return logger
