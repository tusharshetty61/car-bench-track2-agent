"""Server entry point for the Track 2 Cerebras planner/executor agent."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

if __package__:
    from .gated_agent import GatedPlannerExecutorCARBenchAgentExecutor
    from .gates import AmbiguityGateSettings, GateSettings, PolicyGateSettings
    from .planner_agent import (
        DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        DEFAULT_PLANNER_REASONING_EFFORT,
    )
else:
    from gated_agent import GatedPlannerExecutorCARBenchAgentExecutor
    from gates import AmbiguityGateSettings, GateSettings, PolicyGateSettings
    from planner_agent import (
        DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        DEFAULT_EXECUTOR_REASONING_EFFORT,
        DEFAULT_PLANNER_MODEL,
        DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        DEFAULT_PLANNER_REASONING_EFFORT,
    )

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from track_2_agent_under_test_cerebras.cerebras_client import (
    DEFAULT_CEREBRAS_API_BASE,
    DEFAULT_EXECUTOR_MODEL,
)
sys.path.pop(0)

# Shipped default: GREEDY (temp 0) for both planner and executor. Ablation
# (disamb .200->.333, Base->1.000) proved per-trial consistency was the bottleneck
# and that the Cerebras default (~1.0, applied when temperature is omitted) was the
# cause. Override per-run via --temperature / TRACK2_TEMPERATURE etc.
DEFAULT_TRACK2_TEMPERATURE = 0.0

logger = configure_logger(role="agent_under_test", context="server")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_float(name: str, default: float | None = None) -> float | None:
    value = _env_or_default(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = _env_or_default(name)
    if value is None:
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_or_default(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def prepare_agent_card(url: str) -> AgentCard:
    card = AgentCard(
        name="car_bench_agent_cerebras_planner",
        description=(
            "In-car voice assistant using private planning and direct "
            "Cerebras SDK execution"
        ),
        version="1.0.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )

    iface = card.supported_interfaces.add()
    iface.url = url
    iface.protocol_binding = "JSONRPC"
    iface.protocol_version = "1.0"

    card.capabilities.streaming = False
    card.capabilities.push_notifications = False
    card.capabilities.extended_agent_card = False

    skill = card.skills.add()
    skill.id = "car_assistant"
    skill.name = "In-Car Voice Assistant (Cerebras Planner/Executor)"
    skill.description = "Privately plans and returns CAR-bench A2A text or tool calls"
    skill.tags.extend(["benchmark", "car-bench", "voice-assistant", "cerebras"])

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CAR-bench Track 2 Cerebras planner/executor agent."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--planner-model", type=str, default=None)
    parser.add_argument("--executor-model", type=str, default=None)
    parser.add_argument("--service-tier", type=str, default=None)
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Cerebras-compatible API base URL (env: TRACK2_CEREBRAS_API_BASE / CEREBRAS_API_BASE).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--planner-temperature", type=float, default=None)
    parser.add_argument("--executor-temperature", type=float, default=None)
    parser.add_argument("--planner-reasoning-effort", type=str, default=None)
    parser.add_argument("--executor-reasoning-effort", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--planner-max-completion-tokens", type=int, default=None)
    parser.add_argument("--executor-max-completion-tokens", type=int, default=None)
    parser.add_argument("--malformed-retries", type=int, default=None)
    parser.add_argument(
        "--grounding-gate",
        dest="grounding_gate",
        action="store_true",
        default=None,
        help="Enable the parallel grounding gate (default on).",
    )
    parser.add_argument(
        "--no-grounding-gate",
        dest="grounding_gate",
        action="store_false",
        help="Disable the grounding gate (raw planner/executor baseline).",
    )
    parser.add_argument("--gate-votes", type=int, default=None)
    parser.add_argument("--gate-reasoning-effort", type=str, default=None)
    parser.add_argument("--gate-max-completion-tokens", type=int, default=None)
    parser.add_argument(
        "--ambiguity-gate",
        dest="ambiguity_gate",
        action="store_true",
        default=None,
        help="Enable the parallel disambiguation gate (default on).",
    )
    parser.add_argument(
        "--no-ambiguity-gate",
        dest="ambiguity_gate",
        action="store_false",
        help="Disable the disambiguation gate.",
    )
    parser.add_argument("--ambiguity-gate-votes", type=int, default=None)
    parser.add_argument("--ambiguity-reasoning-effort", type=str, default=None)
    parser.add_argument("--ambiguity-max-completion-tokens", type=int, default=None)
    parser.add_argument(
        "--policy-gate",
        dest="policy_gate",
        action="store_true",
        default=None,
        help="Enable the parallel policy-obligation gate + repair (default OFF; experimental).",
    )
    parser.add_argument(
        "--no-policy-gate",
        dest="policy_gate",
        action="store_false",
        help="Disable the policy-obligation gate.",
    )
    parser.add_argument("--policy-gate-votes", type=int, default=None)
    parser.add_argument("--policy-reasoning-effort", type=str, default=None)
    parser.add_argument("--policy-max-completion-tokens", type=int, default=None)
    args = parser.parse_args()

    if not _env_or_default("CEREBRAS_API_KEY"):
        raise SystemExit("CEREBRAS_API_KEY must be set for Track 2 Cerebras runs.")

    planner_model = (
        args.planner_model
        if args.planner_model is not None
        else _env_or_default("TRACK2_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
    )
    planner_reasoning_effort = (
        args.planner_reasoning_effort
        if args.planner_reasoning_effort is not None
        else _env_or_default(
            "TRACK2_PLANNER_REASONING_EFFORT",
            DEFAULT_PLANNER_REASONING_EFFORT,
        )
    )

    executor_model = (
        args.executor_model
        if args.executor_model is not None
        else _env_or_default("TRACK2_EXECUTOR_MODEL", DEFAULT_EXECUTOR_MODEL)
    )
    service_tier = (
        args.service_tier
        if args.service_tier is not None
        else _env_or_default("TRACK2_CEREBRAS_SERVICE_TIER")
    )
    api_base = (
        args.api_base
        if args.api_base is not None
        else _env_or_default(
            "TRACK2_CEREBRAS_API_BASE",
            _env_or_default("CEREBRAS_API_BASE", DEFAULT_CEREBRAS_API_BASE),
        )
    )
    planner_temperature = (
        args.planner_temperature
        if args.planner_temperature is not None
        else _env_float("TRACK2_PLANNER_TEMPERATURE", DEFAULT_TRACK2_TEMPERATURE)
    )
    executor_temperature = (
        args.executor_temperature
        if args.executor_temperature is not None
        else (
            args.temperature
            if args.temperature is not None
            else _env_float("TRACK2_TEMPERATURE", DEFAULT_TRACK2_TEMPERATURE)
        )
    )
    executor_reasoning_effort = (
        args.executor_reasoning_effort
        if args.executor_reasoning_effort is not None
        else (
            args.reasoning_effort
            if args.reasoning_effort is not None
            else _env_or_default(
                "TRACK2_EXECUTOR_REASONING_EFFORT",
                DEFAULT_EXECUTOR_REASONING_EFFORT,
            )
        )
    )
    planner_max_completion_tokens = (
        args.planner_max_completion_tokens
        if args.planner_max_completion_tokens is not None
        else _env_int(
            "TRACK2_PLANNER_MAX_COMPLETION_TOKENS",
            DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
        )
    )
    executor_max_completion_tokens = (
        args.executor_max_completion_tokens
        if args.executor_max_completion_tokens is not None
        else _env_int(
            "TRACK2_MAX_COMPLETION_TOKENS",
            DEFAULT_EXECUTOR_MAX_COMPLETION_TOKENS,
        )
    )
    malformed_retries = (
        args.malformed_retries
        if args.malformed_retries is not None
        else _env_int("TRACK2_LLM_MALFORMED_RETRIES", 1)
    )
    enable_grounding_gate = (
        args.grounding_gate
        if args.grounding_gate is not None
        else _env_bool("TRACK2_ENABLE_GROUNDING_GATE", True)
    )
    gate_settings = GateSettings(
        model=executor_model or DEFAULT_EXECUTOR_MODEL,
        reasoning_effort=(
            args.gate_reasoning_effort
            if args.gate_reasoning_effort is not None
            else _env_or_default("TRACK2_GATE_REASONING_EFFORT", "low")
        ),
        votes=(
            args.gate_votes
            if args.gate_votes is not None
            else _env_int("TRACK2_GATE_VOTES", 3)
        ),
        max_completion_tokens=(
            args.gate_max_completion_tokens
            if args.gate_max_completion_tokens is not None
            else _env_int("TRACK2_GATE_MAX_COMPLETION_TOKENS", 2048)
        ),
        api_base=api_base,
        service_tier=service_tier,
    )
    enable_ambiguity_gate = (
        args.ambiguity_gate
        if args.ambiguity_gate is not None
        else _env_bool("TRACK2_ENABLE_AMBIGUITY_GATE", True)
    )
    ambiguity_gate_settings = AmbiguityGateSettings(
        model=executor_model or DEFAULT_EXECUTOR_MODEL,
        reasoning_effort=(
            args.ambiguity_reasoning_effort
            if args.ambiguity_reasoning_effort is not None
            else _env_or_default("TRACK2_AMBIGUITY_REASONING_EFFORT", "medium")
        ),
        votes=(
            args.ambiguity_gate_votes
            if args.ambiguity_gate_votes is not None
            else _env_int("TRACK2_AMBIGUITY_GATE_VOTES", 3)
        ),
        max_completion_tokens=(
            args.ambiguity_max_completion_tokens
            if args.ambiguity_max_completion_tokens is not None
            else _env_int("TRACK2_AMBIGUITY_MAX_COMPLETION_TOKENS", 2048)
        ),
        api_base=api_base,
        service_tier=service_tier,
    )
    # Policy-obligation gate: EXPERIMENTAL, default OFF (validate on TRAIN before shipping). Targets
    # the cross-split route-informing miss (LLM-POL:021/022); see notes/STRATEGY.md 2026-07-10.
    enable_policy_gate = (
        args.policy_gate
        if args.policy_gate is not None
        else _env_bool("TRACK2_ENABLE_POLICY_GATE", False)
    )
    policy_gate_settings = PolicyGateSettings(
        model=executor_model or DEFAULT_EXECUTOR_MODEL,
        reasoning_effort=(
            args.policy_reasoning_effort
            if args.policy_reasoning_effort is not None
            else _env_or_default("TRACK2_POLICY_REASONING_EFFORT", "low")
        ),
        votes=(
            args.policy_gate_votes
            if args.policy_gate_votes is not None
            else _env_int("TRACK2_POLICY_GATE_VOTES", 3)
        ),
        max_completion_tokens=(
            args.policy_max_completion_tokens
            if args.policy_max_completion_tokens is not None
            else _env_int("TRACK2_POLICY_MAX_COMPLETION_TOKENS", 2048)
        ),
        api_base=api_base,
        service_tier=service_tier,
    )

    logger.info(
        "Starting CAR-bench agent (Cerebras planner/executor)",
        planner_model=planner_model,
        executor_model=executor_model,
        service_tier=service_tier,
        planner_temperature=planner_temperature,
        executor_temperature=executor_temperature,
        planner_reasoning_effort=planner_reasoning_effort,
        executor_reasoning_effort=executor_reasoning_effort,
        planner_max_completion_tokens=planner_max_completion_tokens,
        executor_max_completion_tokens=executor_max_completion_tokens,
        malformed_retries=malformed_retries,
        grounding_gate=enable_grounding_gate,
        gate_votes=gate_settings.votes,
        gate_reasoning_effort=gate_settings.reasoning_effort,
        ambiguity_gate=enable_ambiguity_gate,
        ambiguity_gate_votes=ambiguity_gate_settings.votes,
        ambiguity_reasoning_effort=ambiguity_gate_settings.reasoning_effort,
        policy_gate=enable_policy_gate,
        policy_gate_votes=policy_gate_settings.votes,
        policy_reasoning_effort=policy_gate_settings.reasoning_effort,
        host=args.host,
        port=args.port,
    )

    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=GatedPlannerExecutorCARBenchAgentExecutor(
            enable_grounding_gate=enable_grounding_gate,
            enable_ambiguity_gate=enable_ambiguity_gate,
            enable_policy_gate=enable_policy_gate,
            gate_settings=gate_settings,
            ambiguity_gate_settings=ambiguity_gate_settings,
            policy_gate_settings=policy_gate_settings,
            planner_model=planner_model,
            executor_model=executor_model or DEFAULT_EXECUTOR_MODEL,
            planner_max_completion_tokens=planner_max_completion_tokens,
            executor_max_completion_tokens=executor_max_completion_tokens,
            api_base=api_base,
            service_tier=service_tier,
            planner_temperature=planner_temperature,
            executor_temperature=executor_temperature,
            planner_reasoning_effort=planner_reasoning_effort,
            executor_reasoning_effort=executor_reasoning_effort,
            malformed_retries=malformed_retries,
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_jsonrpc_routes(request_handler, "/", enable_v0_3_compat=True)
    card_routes = create_agent_card_routes(card)
    app = Starlette(routes=routes + card_routes)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=1000,
    )


if __name__ == "__main__":
    main()
