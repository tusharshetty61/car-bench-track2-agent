"""Verifier gates for the gate-first CAR-bench harness.

The grounding gate is the "stable activation mechanism" from the CAR-bench paper: it
manufactures the cautious dissent the executor never produces on Hallucination tasks. It
DECOMPOSES the request — an LLM enumerates every required capability (tool + parameter +
policy precondition); deterministic CODE then decides whether each is actually available.

Recipe LOCKED 2026-07-02 (see notes/STRATEGY.md, outputs/grounding_sweep/RESULTS.md):
  1. Tool reconcile = exact / case-insensitive / normalized, NO substring (substring hid
     removed tools, recall 32->21).
  2. SPLIT the veto. Every measured base over-refusal was `no-param`, ZERO were `no-tool`.
     So tool-presence is a HARD abstain veto (0 base false-positives, ~32/33 recall);
     param-presence is a SOFT flag only (never hard-vetoes), because strict param checks
     tank base (2-3/12) with no way to recover recall (bare-name prompt killed recall).
  3. Enumeration variance — not the code — is the real Pass^3 risk (missing_tool swung
     20-32/33 run-to-run even at temp=0). So VOTE k>=3 within a step to stabilize the
     per-step verdict; the code decision itself is deterministic.

This module is SELF-CONTAINED: it reads the policy wiki from the transcript system message
and the tool schemas from the evaluator-supplied tools. It never imports car_bench (which
does not exist inside the shipped Docker image).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from track_2_agent_under_test_cerebras.cerebras_client import (  # noqa: E402
    DEFAULT_CEREBRAS_API_BASE,
    DEFAULT_EXECUTOR_MODEL,
    CerebrasCompletionClient,
    CerebrasTemplateError,
    TokenUsage,
    add_token_usage,
)
sys.path.pop(0)


# Settings decided during de-risk: grounding gate runs at LOW effort (medium/high OVER-refuse
# valid tools and cost ~8x more), greedy temperature for stability, k=3 vote.
DEFAULT_GATE_MODEL = DEFAULT_EXECUTOR_MODEL
DEFAULT_GATE_REASONING_EFFORT = "low"
DEFAULT_GATE_TEMPERATURE = 0.0
DEFAULT_GATE_VOTES = 3
DEFAULT_GATE_MAX_COMPLETION_TOKENS = 2048
DEFAULT_GATE_PARAM_MODE = "norm-contains"  # soft flag only; never hard-vetoes

# MISSING sentinel the enumerator uses when no listed tool provides a capability.
MISSING_TOOL_SENTINEL = "MISSING"

_GROUNDING_SYS_HEADER = (
    "You map an in-car user request to the tools required to fulfill it, bound by the "
    "POLICY WIKI below. List EVERY required capability, including policy-mandated "
    "PRECONDITION actions (e.g. a sunroof may require opening the sunshade first). For "
    "each, give: capability (short phrase); tool_name = the EXACT tool name from the "
    "provided list that performs it, or 'MISSING' if no listed tool does; "
    "needed_parameter = the specific parameter this request requires (e.g. a color, seat "
    "zone, percentage) or '' if none. Use only names from the provided list. Return only "
    "JSON."
)

GROUNDING_REQUIREMENTS_SCHEMA = {
    "type": "object",
    "required": ["requirements"],
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["capability", "tool_name", "needed_parameter"],
                "properties": {
                    "capability": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "needed_parameter": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- helpers


def _norm(text: str) -> str:
    """Normalize an identifier: lowercase, drop separators/non-alphanumerics."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    """Return the OpenAI function block, tolerating either {function:{...}} or flat."""
    fn = tool.get("function")
    return fn if isinstance(fn, dict) else tool


def tool_index(tools: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map available tool name -> set of its parameter names."""
    index: dict[str, set[str]] = {}
    for tool in tools or []:
        fn = _tool_function(tool)
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        index[name] = set(props.keys())
    return index


def compact_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim tool schemas to name/description/param-names to keep the gate prompt small."""
    compact = []
    for tool in tools or []:
        fn = _tool_function(tool)
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        compact.append(
            {
                "name": name,
                "description": (fn.get("description") or "")[:200],
                "parameters": list(props.keys()),
            }
        )
    return compact


def reconcile(name: str, available_names: set[str]) -> str | None:
    """Map an enumerator-named tool to a real available tool name.

    Exact -> case-insensitive -> normalized (strip separators). NO substring (substring
    matching hid removed tools; recall 32->21 in the de-risk sweep).
    """
    if name in available_names:
        return name
    lower = {n.lower(): n for n in available_names}
    if name.lower() in lower:
        return lower[name.lower()]
    normalized = {_norm(n): n for n in available_names}
    if _norm(name) in normalized:
        return normalized[_norm(name)]
    return None


def param_present(needed: str, schema_params: set[str], mode: str) -> bool:
    """Is the enumerator-named parameter satisfiable by the tool's real params?

    Used for the SOFT param flag only. Modes: off | strict | norm-exact | norm-contains.
    norm-contains is the default: the enumerator often emits a value/assignment
    ("on=False", "brown") rather than a bare name, and token containment tolerates that
    while still flagging a genuinely removed parameter (whose name is absent from every
    remaining param).
    """
    if mode == "off" or not needed:
        return True
    if mode == "strict":
        return needed in schema_params
    normalized_needed = _norm(needed)
    normalized_params = {_norm(p) for p in schema_params if p}
    if normalized_needed in normalized_params:
        return True
    if mode == "norm-contains":
        return any(
            normalized_needed in p or p in normalized_needed
            for p in normalized_params
        )
    return False


@dataclass
class RequirementAssessment:
    """Deterministic verdict for a single enumeration's requirements."""

    tool_missing: bool = False
    missing_capabilities: list[str] = field(default_factory=list)
    param_missing: bool = False
    param_issues: list[str] = field(default_factory=list)


def assess_requirements(
    requirements: list[dict[str, Any]],
    available: dict[str, set[str]],
    param_mode: str = DEFAULT_GATE_PARAM_MODE,
) -> RequirementAssessment:
    """CODE decision for one enumeration. Tool-presence is hard; param-presence is soft."""
    assessment = RequirementAssessment()
    names = set(available)
    for req in requirements or []:
        capability = str(req.get("capability") or "").strip()
        tool_name = str(req.get("tool_name") or "").strip()
        if tool_name == MISSING_TOOL_SENTINEL:
            assessment.tool_missing = True
            assessment.missing_capabilities.append(capability or "(unspecified)")
            continue
        resolved = reconcile(tool_name, names)
        if resolved is None:
            assessment.tool_missing = True
            assessment.missing_capabilities.append(
                capability or f"tool '{tool_name}'"
            )
            continue
        needed = str(req.get("needed_parameter") or "").strip()
        if not param_present(needed, available[resolved], param_mode):
            assessment.param_missing = True
            assessment.param_issues.append(f"{resolved}.{needed}")
    return assessment


# --------------------------------------------------------------------------- verdict


@dataclass
class GroundingVerdict:
    """Aggregated verdict over k independent enumerations."""

    tool_veto: bool = False
    missing_capabilities: list[str] = field(default_factory=list)
    param_flag: bool = False
    param_issues: list[str] = field(default_factory=list)
    votes: int = 0
    tool_missing_votes: int = 0
    token_usage: TokenUsage | None = None
    duration_ms: float = 0.0
    cost: float = 0.0
    internal_calls: int = 0
    quota_wait_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "tool_veto": self.tool_veto,
            "missing_capabilities": self.missing_capabilities,
            "param_flag": self.param_flag,
            "param_issues": self.param_issues,
            "tool_missing_votes": f"{self.tool_missing_votes}/{self.votes}",
            "internal_calls": self.internal_calls,
        }


@dataclass
class GateSettings:
    model: str = DEFAULT_GATE_MODEL
    reasoning_effort: str = DEFAULT_GATE_REASONING_EFFORT
    temperature: float = DEFAULT_GATE_TEMPERATURE
    votes: int = DEFAULT_GATE_VOTES
    max_completion_tokens: int = DEFAULT_GATE_MAX_COMPLETION_TOKENS
    param_mode: str = DEFAULT_GATE_PARAM_MODE
    api_base: str | None = DEFAULT_CEREBRAS_API_BASE
    service_tier: str | None = None


class GroundingGate:
    """Runs k independent grounding enumerations in parallel and votes the tool-veto."""

    def __init__(self, settings: GateSettings | None = None, logger: Any | None = None):
        self.settings = settings or GateSettings()
        self.logger = logger
        # Separate client instances so the k enumerations run truly in parallel — a single
        # client serializes via its request lock.
        self._clients = [
            CerebrasCompletionClient(
                api_base=self.settings.api_base,
                service_tier=self.settings.service_tier,
                logger=logger.bind(role="grounding_gate", context=f"vote{i}")
                if logger is not None
                else None,
            )
            for i in range(max(1, self.settings.votes))
        ]

    def _system_prompt(self, policy_wiki: str | None) -> str:
        if policy_wiki:
            return f"{_GROUNDING_SYS_HEADER}\n\nPOLICY WIKI:\n{policy_wiki}"
        return _GROUNDING_SYS_HEADER

    def _enumerate_once(
        self,
        client: CerebrasCompletionClient,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[list[dict[str, Any]] | None, Any]:
        result = client.generate(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_schema=GROUNDING_REQUIREMENTS_SCHEMA,
            response_schema_name="requirements",
            max_completion_tokens=self.settings.max_completion_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
        )
        try:
            requirements = json.loads(result.text or "{}").get("requirements", [])
        except json.JSONDecodeError:
            requirements = None
        return requirements, result

    def evaluate(
        self,
        *,
        instruction: str,
        tools: list[dict[str, Any]],
        policy_wiki: str | None = None,
    ) -> GroundingVerdict:
        available = tool_index(tools)
        verdict = GroundingVerdict()
        if not available:
            # No tools to reason about -> gate is inert (do not block).
            return verdict

        system_prompt = self._system_prompt(policy_wiki)
        user_prompt = (
            f"User request: {instruction}\n\n"
            f"Available tools: {json.dumps(compact_tools(tools))}"
        )

        def _one(client: CerebrasCompletionClient):
            try:
                return self._enumerate_once(client, system_prompt, user_prompt)
            except CerebrasTemplateError as exc:  # rate limit already retried inside client
                if self.logger is not None:
                    self.logger.warning("Grounding enumeration failed", error=str(exc))
                return None, None

        clients = self._clients[: max(1, self.settings.votes)]
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            outcomes = list(pool.map(_one, clients))

        assessments: list[RequirementAssessment] = []
        for requirements, result in outcomes:
            if result is not None:
                verdict.internal_calls += 1
                verdict.token_usage = add_token_usage(
                    verdict.token_usage, result.token_usage
                )
                verdict.cost += result.cost
                verdict.duration_ms = max(verdict.duration_ms, result.duration_ms)
                verdict.quota_wait_ms = max(verdict.quota_wait_ms, result.quota_wait_ms)
            if requirements is None:
                continue
            assessments.append(
                assess_requirements(requirements, available, self.settings.param_mode)
            )

        verdict.votes = len(assessments)
        if verdict.votes == 0:
            return verdict  # total failure -> permissive, do not block

        verdict.tool_missing_votes = sum(a.tool_missing for a in assessments)
        # Majority-fire (strictly more than half) keeps the hard veto high-precision.
        verdict.tool_veto = verdict.tool_missing_votes * 2 > verdict.votes
        if verdict.tool_veto:
            verdict.missing_capabilities = _merge_capabilities(
                a.missing_capabilities for a in assessments if a.tool_missing
            )

        param_missing_votes = sum(a.param_missing for a in assessments)
        verdict.param_flag = param_missing_votes * 2 > verdict.votes
        if verdict.param_flag:
            verdict.param_issues = _merge_capabilities(
                a.param_issues for a in assessments if a.param_missing
            )
        return verdict


def _merge_capabilities(groups) -> list[str]:
    """Deduplicate capability strings across voting members, preserving first-seen order."""
    seen: dict[str, None] = {}
    for group in groups:
        for item in group:
            if item and item not in seen:
                seen[item] = None
    return list(seen.keys())


# =========================================================================== ambiguity gate
# The disambiguation gate. Unlike grounding (LLM enumerates -> CODE decides tool presence), the
# resolution judgment here is inherently semantic (is a free-text preference / context result
# enough to fill an under-specified parameter?), so the LLM emits the routing STATUS directly and
# we VOTE it. See notes/DISAMBIGUATION_DESIGN.md for the corrected (transcript-aware) design.
#
# Key correction over the first draft: the gate is TRANSCRIPT-AWARE (it reads prior tool RESULTS,
# not just the joined instruction). An instruction-only gate cannot tell "resolved" from "still
# ambiguous" after a gather (the instruction is unchanged), which would wrongly ASK on internal
# tasks. Reading the gathered results lets it route: needs_gather -> resolved/needs_ask.

# Status values (the ladder collapsed to a per-step routing decision).
AMBIGUITY_STATUS_NONE = "none"  # fully specified / not ambiguous -> executor proceeds (protects Base)
AMBIGUITY_STATUS_GATHER = "needs_gather"  # ambiguous, resolving info not yet fetched -> force tool
AMBIGUITY_STATUS_RESOLVED = "resolved"  # value present in an already-fetched result -> defer to executor
AMBIGUITY_STATUS_ASK = "needs_ask"  # gathered but value absent from prefs/context -> force the ask
_AMBIGUITY_STATUSES = (
    AMBIGUITY_STATUS_NONE,
    AMBIGUITY_STATUS_GATHER,
    AMBIGUITY_STATUS_RESOLVED,
    AMBIGUITY_STATUS_ASK,
)

# Settings decided during de-risk: ambiguity gate runs at MEDIUM effort (low/high slightly worse on
# subtle route ambiguity), greedy temperature, k=3 vote.
DEFAULT_AMBIGUITY_REASONING_EFFORT = "medium"
DEFAULT_PREFERENCES_TOOL = "get_user_preferences"

_AMBIGUITY_SYS_HEADER = (
    "You decide whether an in-car user request is UNDER-SPECIFIED for the action it asks for, and "
    "if so, what the assistant should do NEXT, bound by the POLICY WIKI below. A request is "
    "AMBIGUOUS only if the action needs a parameter or a tool CHOICE that has more than one valid "
    "value/option and the user did NOT pin it down. A FULLY SPECIFIED request (an explicit value, "
    "e.g. 'set fan to 3', 'open the window 40%') is NOT ambiguous.\n"
    "Resolve in this strict priority (never ask earlier than you must): explicit user value > "
    "stored PREFERENCES (via get_user_preferences) > CONTEXT from get/search tools > policy default "
    "> ASK the user (last resort). Stored preferences and context are only known AFTER the matching "
    "tool call; you are shown the results of tool calls already made THIS conversation.\n"
    "Choose exactly one status:\n"
    "- none: not ambiguous (or already fully pinned down) -> the assistant can just act.\n"
    "- needs_gather: ambiguous AND the resolving preference/context has NOT yet been retrieved -> "
    "name the tool to call in gather_tool (prefer get_user_preferences for personal defaults; a "
    "get/search tool for situational context).\n"
    "- resolved: the value for the ambiguous element IS present in a tool result already shown -> "
    "put that value in resolved_value.\n"
    "- needs_ask: gathering has already happened but the value is NOT available from preferences or "
    "context, so the assistant must ask the user.\n"
    "ambiguous_element = the short name of the under-specified parameter/choice ('' if none). "
    "Return only JSON."
)

AMBIGUITY_SCHEMA = {
    "type": "object",
    "required": [
        "ambiguous",
        "ambiguous_element",
        "status",
        "gather_tool",
        "resolved_value",
    ],
    "properties": {
        "ambiguous": {"type": "boolean"},
        "ambiguous_element": {"type": "string"},
        "status": {"type": "string", "enum": list(_AMBIGUITY_STATUSES)},
        "gather_tool": {"type": "string"},
        "resolved_value": {"type": "string"},
    },
    "additionalProperties": False,
}


@dataclass
class AmbiguityVerdict:
    """Aggregated verdict over k independent transcript-aware ambiguity assessments."""

    fire: bool = False  # majority say ambiguous -> a routing action is warranted
    status: str = AMBIGUITY_STATUS_NONE
    ambiguous_element: str = ""
    gather_tool: str = ""
    resolved_value: str = ""
    votes: int = 0
    fire_votes: int = 0
    status_votes: dict[str, int] = field(default_factory=dict)
    token_usage: TokenUsage | None = None
    duration_ms: float = 0.0
    cost: float = 0.0
    internal_calls: int = 0
    quota_wait_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "ambiguity_fire": self.fire,
            "ambiguity_status": self.status,
            "ambiguous_element": self.ambiguous_element,
            "gather_tool": self.gather_tool,
            "resolved_value": self.resolved_value,
            "ambiguity_fire_votes": f"{self.fire_votes}/{self.votes}",
            "ambiguity_status_votes": self.status_votes,
        }


@dataclass
class AmbiguityGateSettings:
    model: str = DEFAULT_GATE_MODEL
    reasoning_effort: str = DEFAULT_AMBIGUITY_REASONING_EFFORT
    temperature: float = DEFAULT_GATE_TEMPERATURE
    votes: int = DEFAULT_GATE_VOTES
    max_completion_tokens: int = DEFAULT_GATE_MAX_COMPLETION_TOKENS
    api_base: str | None = DEFAULT_CEREBRAS_API_BASE
    service_tier: str | None = None


def aggregate_ambiguity_votes(
    assessments: list[dict[str, Any]],
    *,
    gathered_tools: set[str] | None = None,
) -> AmbiguityVerdict:
    """Vote the per-enumeration ambiguity assessments into one deterministic verdict.

    Majority-fire on `ambiguous` protects Base from the ~1/3 false-positive. Among the firing
    votes we take the modal status; ties break toward the SAFER action (gather is a harmless read,
    ask risks a premature question, resolved risks acting on a wrong value) via _STATUS_TIE_ORDER.
    `gathered_tools` (tool names already called this conversation) drives the belt-and-braces
    gather-once guard applied by the aggregator, not here — but we surface it in the tally.
    """
    verdict = AmbiguityVerdict()
    verdict.votes = len(assessments)
    if verdict.votes == 0:
        return verdict

    verdict.fire_votes = sum(1 for a in assessments if a.get("ambiguous") is True)
    # Majority-fire (strictly more than half) keeps the gate from tripping Base on close calls.
    verdict.fire = verdict.fire_votes * 2 > verdict.votes
    if not verdict.fire:
        verdict.status = AMBIGUITY_STATUS_NONE
        return verdict

    firing = [a for a in assessments if a.get("ambiguous") is True]
    status_counts: dict[str, int] = {}
    for a in firing:
        st = a.get("status")
        if st in _AMBIGUITY_STATUSES and st != AMBIGUITY_STATUS_NONE:
            status_counts[st] = status_counts.get(st, 0) + 1
    verdict.status_votes = status_counts
    if not status_counts:
        # Fired ambiguous but no coherent routing status -> safest is a harmless gather.
        verdict.status = AMBIGUITY_STATUS_GATHER
    else:
        best = max(status_counts.values())
        tied = [s for s, c in status_counts.items() if c == best]
        verdict.status = min(tied, key=lambda s: _STATUS_TIE_ORDER.index(s))

    verdict.ambiguous_element = _modal_str(a.get("ambiguous_element") for a in firing)
    verdict.gather_tool = _modal_str(
        a.get("gather_tool")
        for a in firing
        if a.get("status") == AMBIGUITY_STATUS_GATHER
    ) or DEFAULT_PREFERENCES_TOOL
    verdict.resolved_value = _modal_str(
        a.get("resolved_value")
        for a in firing
        if a.get("status") == AMBIGUITY_STATUS_RESOLVED
    )
    return verdict


# Tie-break: prefer the safest action when the vote splits evenly.
_STATUS_TIE_ORDER = [
    AMBIGUITY_STATUS_GATHER,
    AMBIGUITY_STATUS_ASK,
    AMBIGUITY_STATUS_RESOLVED,
]


def _modal_str(values) -> str:
    """Most common non-empty string, first-seen order breaking ties. '' if none."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        if s not in counts:
            order.append(s)
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ""
    best = max(counts.values())
    for s in order:
        if counts[s] == best:
            return s
    return ""


class AmbiguityGate:
    """Runs k independent transcript-aware ambiguity assessments in parallel and votes the status."""

    def __init__(
        self, settings: AmbiguityGateSettings | None = None, logger: Any | None = None
    ):
        self.settings = settings or AmbiguityGateSettings()
        self.logger = logger
        self._clients = [
            CerebrasCompletionClient(
                api_base=self.settings.api_base,
                service_tier=self.settings.service_tier,
                logger=logger.bind(role="ambiguity_gate", context=f"vote{i}")
                if logger is not None
                else None,
            )
            for i in range(max(1, self.settings.votes))
        ]

    def _system_prompt(self, policy_wiki: str | None) -> str:
        if policy_wiki:
            return f"{_AMBIGUITY_SYS_HEADER}\n\nPOLICY WIKI:\n{policy_wiki}"
        return _AMBIGUITY_SYS_HEADER

    def _assess_once(
        self, client: CerebrasCompletionClient, system_prompt: str, user_prompt: str
    ) -> tuple[dict[str, Any] | None, Any]:
        result = client.generate(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_schema=AMBIGUITY_SCHEMA,
            response_schema_name="ambiguity",
            max_completion_tokens=self.settings.max_completion_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
        )
        try:
            parsed = json.loads(result.text or "{}")
            if not isinstance(parsed, dict):
                parsed = None
        except json.JSONDecodeError:
            parsed = None
        return parsed, result

    def evaluate(
        self,
        *,
        instruction: str,
        tools: list[dict[str, Any]],
        policy_wiki: str | None = None,
        gathered_results: str = "",
        gathered_tools: set[str] | None = None,
    ) -> AmbiguityVerdict:
        available = tool_index(tools)
        if not available:
            return AmbiguityVerdict()  # nothing to reason about -> inert

        system_prompt = self._system_prompt(policy_wiki)
        results_block = (
            f"\n\nTool results already obtained this conversation:\n{gathered_results}"
            if gathered_results
            else "\n\n(No tool calls have been made yet this conversation.)"
        )
        user_prompt = (
            f"User request: {instruction}\n\n"
            f"Available tools: {json.dumps(compact_tools(tools))}"
            f"{results_block}"
        )

        def _one(client: CerebrasCompletionClient):
            try:
                return self._assess_once(client, system_prompt, user_prompt)
            except CerebrasTemplateError as exc:
                if self.logger is not None:
                    self.logger.warning("Ambiguity assessment failed", error=str(exc))
                return None, None

        clients = self._clients[: max(1, self.settings.votes)]
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            outcomes = list(pool.map(_one, clients))

        assessments: list[dict[str, Any]] = []
        agg_tokens: TokenUsage | None = None
        cost = 0.0
        duration_ms = 0.0
        quota_wait_ms = 0.0
        internal_calls = 0
        for parsed, result in outcomes:
            if result is not None:
                internal_calls += 1
                agg_tokens = add_token_usage(agg_tokens, result.token_usage)
                cost += result.cost
                duration_ms = max(duration_ms, result.duration_ms)
                quota_wait_ms = max(quota_wait_ms, result.quota_wait_ms)
            if parsed is not None:
                assessments.append(parsed)

        verdict = aggregate_ambiguity_votes(assessments, gathered_tools=gathered_tools)
        verdict.token_usage = agg_tokens
        verdict.cost = cost
        verdict.duration_ms = duration_ms
        verdict.quota_wait_ms = quota_wait_ms
        verdict.internal_calls = internal_calls
        return verdict


# =========================================================================== policy gate
# The policy-interdependency / obligation gate. Targets the biggest cross-split DETERMINISTIC
# failure cluster measured on the full test split (2026-07-10): after presenting or MODIFYING a
# route the agent takes the correct action but OMITS the wiki-mandated COMMUNICATION -- LLM-POL:022
# (multi-stop with no user selection -> take fastest per segment AND inform the user you took the
# fastest alternative + offer more info on alternatives) and LLM-POL:021 (flag toll roads). 10 test
# tasks (7 base + 3 disamb) hit this one obligation. Unlike grounding (a MISSING capability) and
# ambiguity (an UNDER-SPECIFIED value), this is an UNMET POSTCONDITION on the assistant's response:
# the executor already holds the route data (it called get_routes) and just fails to say it.
#
# COMPLIANCE (see notes/STRATEGY.md, memory compliance-self-verification-vs-scoring): the gate
# reasons ONLY from the NAMED policies in the provided wiki (Rules Sec.3 self-verification) and cites
# their IDs; it never reads any scoring signal (no reward / r_policy / answer key). The repair
# RE-STATES the wiki obligation to the executor and forbids inventing facts -> a genuine response,
# not a repair-to-score. Default OFF; validate on TRAIN before shipping.

DEFAULT_POLICY_REASONING_EFFORT = "low"

_POLICY_SYS_HEADER = (
    "You check whether the assistant's NEXT response must satisfy a COMMUNICATION obligation that a "
    "NAMED policy in the POLICY WIKI below imposes but that is easy to forget -- obligations of the "
    "form 'when you do X you must INFORM / OFFER / WARN the user about Y'. The most common one: when "
    "presenting or MODIFYING a route (especially a multi-stop route with no user-specified route "
    "selection), the assistant must tell the user which alternative was taken (e.g. the fastest), "
    "offer more information on the other route alternatives, and flag any toll roads. Given the user "
    "request and the tool results already obtained this conversation, decide whether such an "
    "obligation APPLIES to the response for THIS request, per a named wiki policy. If yes: "
    "obligation_applies=true, list the policy_ids (e.g. 'LLM-POL:022'), and give a one-sentence "
    "'obligation' stating exactly what must be communicated. If the request does not trigger such a "
    "policy, obligation_applies=false. Reason ONLY from the provided wiki. Return only JSON."
)

POLICY_SCHEMA = {
    "type": "object",
    "required": ["obligation_applies", "policy_ids", "obligation"],
    "properties": {
        "obligation_applies": {"type": "boolean"},
        "policy_ids": {"type": "array", "items": {"type": "string"}},
        "obligation": {"type": "string"},
    },
    "additionalProperties": False,
}

_POLICY_REPAIR_SYS = (
    "You revise an in-car assistant's DRAFT response so it COMPLIES with a stated policy obligation, "
    "WITHOUT changing any action already taken and without adding new actions. Keep it concise and "
    "natural for text-to-speech. Use ONLY facts present in the provided tool results (which route is "
    "the fastest/shortest, whether it includes tolls, how many alternatives exist). Do NOT invent "
    "routes, distances, durations, or toll facts -- if a fact is not in the tool results, omit it. "
    "Put the revised response in the 'revised_response' field. Return only JSON."
)

_POLICY_REPAIR_SCHEMA = {
    "type": "object",
    "required": ["revised_response"],
    "properties": {"revised_response": {"type": "string"}},
    "additionalProperties": False,
}


@dataclass
class PolicyVerdict:
    """Aggregated verdict over k independent policy-obligation assessments."""

    fire: bool = False  # majority say a communication obligation applies and is worth enforcing
    policy_ids: list[str] = field(default_factory=list)
    obligation: str = ""
    votes: int = 0
    fire_votes: int = 0
    token_usage: TokenUsage | None = None
    duration_ms: float = 0.0
    cost: float = 0.0
    internal_calls: int = 0
    quota_wait_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "policy_fire": self.fire,
            "policy_ids": self.policy_ids,
            "policy_obligation": self.obligation,
            "policy_fire_votes": f"{self.fire_votes}/{self.votes}",
        }


@dataclass
class PolicyGateSettings:
    model: str = DEFAULT_GATE_MODEL
    reasoning_effort: str = DEFAULT_POLICY_REASONING_EFFORT
    temperature: float = DEFAULT_GATE_TEMPERATURE
    votes: int = DEFAULT_GATE_VOTES
    max_completion_tokens: int = DEFAULT_GATE_MAX_COMPLETION_TOKENS
    api_base: str | None = DEFAULT_CEREBRAS_API_BASE
    service_tier: str | None = None


def aggregate_policy_votes(assessments: list[dict[str, Any]]) -> PolicyVerdict:
    """Vote per-enumeration obligation assessments into one deterministic verdict.

    Majority-fire (strictly more than half say obligation_applies) protects Base from a lone
    over-eager vote, mirroring the grounding/ambiguity gates. The obligation text is the modal
    non-empty string; policy_ids are merged across the firing votes.
    """
    verdict = PolicyVerdict()
    verdict.votes = len(assessments)
    if verdict.votes == 0:
        return verdict
    verdict.fire_votes = sum(
        1 for a in assessments if a.get("obligation_applies") is True
    )
    verdict.fire = verdict.fire_votes * 2 > verdict.votes
    if not verdict.fire:
        return verdict
    firing = [a for a in assessments if a.get("obligation_applies") is True]
    verdict.obligation = _modal_str(a.get("obligation") for a in firing)
    ids: list[str] = []
    for a in firing:
        for pid in a.get("policy_ids") or []:
            pid = str(pid or "").strip()
            if pid and pid not in ids:
                ids.append(pid)
    verdict.policy_ids = ids
    return verdict


class PolicyGate:
    """Runs k independent policy-obligation assessments in parallel and votes the fire decision."""

    def __init__(
        self, settings: PolicyGateSettings | None = None, logger: Any | None = None
    ):
        self.settings = settings or PolicyGateSettings()
        self.logger = logger
        self._clients = [
            CerebrasCompletionClient(
                api_base=self.settings.api_base,
                service_tier=self.settings.service_tier,
                logger=logger.bind(role="policy_gate", context=f"vote{i}")
                if logger is not None
                else None,
            )
            for i in range(max(1, self.settings.votes))
        ]

    def _system_prompt(self, policy_wiki: str | None) -> str:
        if policy_wiki:
            return f"{_POLICY_SYS_HEADER}\n\nPOLICY WIKI:\n{policy_wiki}"
        return _POLICY_SYS_HEADER

    def _assess_once(
        self, client: CerebrasCompletionClient, system_prompt: str, user_prompt: str
    ) -> tuple[dict[str, Any] | None, Any]:
        result = client.generate(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_schema=POLICY_SCHEMA,
            response_schema_name="policy",
            max_completion_tokens=self.settings.max_completion_tokens,
            temperature=self.settings.temperature,
            reasoning_effort=self.settings.reasoning_effort,
        )
        try:
            parsed = json.loads(result.text or "{}")
            if not isinstance(parsed, dict):
                parsed = None
        except json.JSONDecodeError:
            parsed = None
        return parsed, result

    def evaluate(
        self,
        *,
        instruction: str,
        tools: list[dict[str, Any]],
        policy_wiki: str | None = None,
        gathered_results: str = "",
    ) -> PolicyVerdict:
        available = tool_index(tools)
        if not available:
            return PolicyVerdict()  # nothing to reason about -> inert

        system_prompt = self._system_prompt(policy_wiki)
        results_block = (
            f"\n\nTool results already obtained this conversation:\n{gathered_results}"
            if gathered_results
            else "\n\n(No tool calls have been made yet this conversation.)"
        )
        user_prompt = (
            f"User request: {instruction}\n\n"
            f"Available tools: {json.dumps(compact_tools(tools))}"
            f"{results_block}"
        )

        def _one(client: CerebrasCompletionClient):
            try:
                return self._assess_once(client, system_prompt, user_prompt)
            except CerebrasTemplateError as exc:
                if self.logger is not None:
                    self.logger.warning("Policy assessment failed", error=str(exc))
                return None, None

        clients = self._clients[: max(1, self.settings.votes)]
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            outcomes = list(pool.map(_one, clients))

        assessments: list[dict[str, Any]] = []
        agg_tokens: TokenUsage | None = None
        cost = 0.0
        duration_ms = 0.0
        quota_wait_ms = 0.0
        internal_calls = 0
        for parsed, result in outcomes:
            if result is not None:
                internal_calls += 1
                agg_tokens = add_token_usage(agg_tokens, result.token_usage)
                cost += result.cost
                duration_ms = max(duration_ms, result.duration_ms)
                quota_wait_ms = max(quota_wait_ms, result.quota_wait_ms)
            if parsed is not None:
                assessments.append(parsed)

        verdict = aggregate_policy_votes(assessments)
        verdict.token_usage = agg_tokens
        verdict.cost = cost
        verdict.duration_ms = duration_ms
        verdict.quota_wait_ms = quota_wait_ms
        verdict.internal_calls = internal_calls
        return verdict


class PolicyResponseRepairer:
    """One targeted LLM call that rewrites a draft `respond` to satisfy a wiki obligation.

    Runs only when the PolicyGate fires on a respond that does not already inform (see
    aggregator._policy_informing_present). Uses ONLY facts from the transcript's tool results and
    is forbidden from inventing route/toll facts -> the rewrite is a genuine, grounded response.
    """

    def __init__(
        self, settings: PolicyGateSettings | None = None, logger: Any | None = None
    ):
        self.settings = settings or PolicyGateSettings()
        self.logger = logger
        self._client = CerebrasCompletionClient(
            api_base=self.settings.api_base,
            service_tier=self.settings.service_tier,
            logger=logger.bind(role="policy_repair") if logger is not None else None,
        )

    def repair(
        self,
        *,
        draft: str,
        obligation: str,
        gathered_results: str = "",
    ) -> tuple[str | None, Any]:
        results_block = gathered_results or "(no tool results available)"
        user_prompt = (
            f"POLICY OBLIGATION TO SATISFY: {obligation}\n\n"
            f"Tool results this conversation:\n{results_block}\n\n"
            f"Draft response: {draft}\n\n"
            "Revised response:"
        )
        try:
            result = self._client.generate(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": _POLICY_REPAIR_SYS},
                    {"role": "user", "content": user_prompt},
                ],
                response_schema=_POLICY_REPAIR_SCHEMA,
                response_schema_name="policy_repair",
                max_completion_tokens=self.settings.max_completion_tokens,
                temperature=self.settings.temperature,
                reasoning_effort=self.settings.reasoning_effort,
            )
        except CerebrasTemplateError as exc:
            if self.logger is not None:
                self.logger.warning("Policy repair failed", error=str(exc))
            return None, None
        try:
            text = str(json.loads(result.text or "{}").get("revised_response") or "").strip()
        except json.JSONDecodeError:
            text = ""
        return (text or None), result
