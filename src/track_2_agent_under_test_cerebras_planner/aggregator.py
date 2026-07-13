"""Deterministic aggregator for the gate-first harness.

Pure code (0 LLM calls) => the SAME executor action + gate verdict always produce the SAME
decision, which is what makes the harness Pass^3-consistent given a fixed verdict. The
aggregation is ASYMMETRIC, as the CAR-bench failure analysis requires:

  * ABSTAIN veto (grounding tool-presence is high-precision): if a majority of gate votes say
    a required TOOL is missing AND the executor is about to ACT (tool_calls), replace the
    action with an explicit abstention that NAMES the missing capability. This converts the
    executor's implicit fabrication (paper error E5a) into the compliant "I can't do X" the
    benchmark rewards.
  * The param-presence flag is SOFT: it is surfaced for logging/telemetry only and never
    vetoes, because a hard param veto tanks Base (measured 2-3/12) with no recoverable recall.

The veto fires against a `tool_calls` action AND against a *fabricating* `respond` action.
A `respond` is fabricating when the executor talks to the user WITHOUT acknowledging the
missing capability — either claiming fulfillment ("Fresh air mode is set") or, more subtly,
asking a capability-clarifying question ("Which airflow direction would you like?") that
implies the ungrounded action is possible. Both are implicit fabrication (paper E1/E5a) and
were the cause of the hall_4 Pass^3 regression (2026-07-05 diagnostic): the executor fabricated
via a clarifying question, which the old aggregator (tool_calls-only veto) let sail through.
A `respond` that ALREADY abstains ("Sorry, I can't ...") is left untouched — we never turn a
correct conversational turn into a redundant refusal. This is safe on Base/Disambiguation
because tool-presence is high-precision (0 Base false-positives measured): when the tools ARE
present, `tool_veto` is false, so legitimate confirmations and disambiguation "ask the user"
turns are never overridden.

SECOND branch — DISAMBIGUATION (added 2026-07-05, orthogonal to grounding; grounding keeps
priority since "can't do it at all" beats "needs clarification"). Driven by the transcript-aware
`AmbiguityVerdict.status` (see notes/DISAMBIGUATION_DESIGN.md):
  * needs_gather -> the executor is about to act/guess on an under-specified value before consulting
    stored PREFERENCES. Force a `get_user_preferences` gather (deterministic fetch-all argument
    built from the tool's own schema). Only preference gathers are forced — the executor already
    gathers situational CONTEXT (weather, light status) on its own, and forcing get_user_preferences
    on a context task would derail it — so a context gather defers to the executor.
  * needs_ask -> preferences/context can't resolve it; force a targeted clarifying question (the
    disambiguation_user path), unless the executor is already asking one in its own words.
  * resolved -> the value is in an already-fetched result; DEFER to the executor (it should now act
    with the value). Never force an ask here — that was the flaw that would fail internal tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

if __package__:
    from .gates import (
        DEFAULT_PREFERENCES_TOOL,
        AMBIGUITY_STATUS_ASK,
        AMBIGUITY_STATUS_GATHER,
        AmbiguityVerdict,
        GroundingVerdict,
        PolicyVerdict,
    )
else:  # pragma: no cover - direct-script import fallback
    from gates import (
        DEFAULT_PREFERENCES_TOOL,
        AMBIGUITY_STATUS_ASK,
        AMBIGUITY_STATUS_GATHER,
        AmbiguityVerdict,
        GroundingVerdict,
        PolicyVerdict,
    )


@dataclass
class AggregationDecision:
    """What the aggregator did, for logging and the report's audit trail."""

    action: dict[str, Any]
    overridden: bool = False
    reason: str | None = None
    gate_summary: dict[str, Any] = field(default_factory=dict)
    # Set by the policy branch: a one-line wiki obligation the executor's `respond` failed to
    # satisfy. The aggregator is pure code so it cannot rewrite the response itself; the gated
    # agent runs a single grounded repair call (PolicyResponseRepairer) when this is present.
    repair_prompt: str | None = None


# Phrases that signal the executor is ALREADY acknowledging an inability. Conservative on
# purpose: a false "already abstaining" only means we keep the executor's own (passing) refusal
# instead of our canonical one; a false "fabricating" means we replace a non-abstaining respond
# with an explicit refusal, which under a solid tool_veto is the correct move anyway.
_ABSTENTION_MARKERS = (
    "can't",
    "cant",
    "cannot",
    "can not",
    "unable",
    "don't have",
    "dont have",
    "do not have",
    "not able",
    "no tool",
    "not available",
    "isn't available",
    "not possible",
    "not supported",
    "i'm not able",
    "i am not able",
)


def _looks_like_abstention(content: str | None) -> bool:
    """True if a `respond` already acknowledges an inability (leave it alone)."""
    if not content:
        return False
    lowered = content.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def _abstention_message(missing_capabilities: list[str]) -> str:
    """Short, TTS-friendly, explicit refusal that names the missing capability."""
    caps = [c for c in missing_capabilities if c]
    if not caps:
        return (
            "Sorry, I can't do that -I don't have a tool available for that request."
        )
    if len(caps) == 1:
        return f"Sorry, I can't do that -I don't have a way to {caps[0]}."
    joined = ", ".join(caps[:-1]) + f", or {caps[-1]}"
    return f"Sorry, I can't do that -I don't have a way to {joined}."


_QUESTION_MARKERS = (
    "which",
    "what",
    "would you",
    "do you want",
    "could you",
    "can you tell",
    "how much",
    "how many",
    "please specify",
    "let me know",
)


def _looks_like_question(content: str | None) -> bool:
    """True if a `respond` is already a clarifying question (leave the executor's wording alone)."""
    if not content:
        return False
    lowered = content.lower()
    return "?" in content or any(marker in lowered for marker in _QUESTION_MARKERS)


def _clarifying_question(ambiguous_element: str) -> str:
    """Short, targeted question naming the under-specified element (elicits the scripted answer)."""
    element = (ambiguous_element or "").strip()
    if not element:
        return "Could you clarify what you'd like exactly?"
    return f"Could you tell me the {element} you'd like?"


def _tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function")
    return fn if isinstance(fn, dict) else tool


def _fetch_all_preferences_args(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Build a valid `get_user_preferences` argument that requests EVERY category/subcategory.

    Deterministic: read the tool's own schema and set every boolean leaf True. Returns None if
    the tool (or a usable schema) is not present, so the caller can defer instead of forcing.
    """
    for tool in tools or []:
        fn = _tool_function(tool)
        if fn.get("name") != DEFAULT_PREFERENCES_TOOL:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        categories = (props.get("preference_categories") or {}).get("properties") or {}
        selector: dict[str, dict[str, bool]] = {}
        for category, cat_schema in categories.items():
            subcats = (cat_schema or {}).get("properties") or {}
            if subcats:
                selector[category] = {sub: True for sub in subcats}
        if selector:
            return {"preference_categories": selector}
        # Schema present but shape unexpected -> still issue a call with an empty selector.
        return {"preference_categories": {}}
    return None


def _preferences_available(tools: list[dict[str, Any]] | None) -> bool:
    return any(
        _tool_function(t).get("name") == DEFAULT_PREFERENCES_TOOL for t in (tools or [])
    )


def _is_preferences_gather(gather_tool: str) -> bool:
    """Does the gate's named gather tool refer to the preferences tool?"""
    name = (gather_tool or "").strip().lower()
    return name == DEFAULT_PREFERENCES_TOOL or "preference" in name


# Forcing a clarifying question is risky: a COHERENT question is usually fine (the user sim answers,
# and the resulting action still scores), but a PREMATURE or GARBLED forced question derails the
# task. Measured 2026-07-06 (disamb_4, both_v2): when the gate emitted a degenerate ambiguous_element
# ("on"), the forced question "Could you tell me the on you'd like?" was unparseable -> user STOPs ->
# 0; the trials where we DEFERRED and the executor asked its own coherent question passed. So: (a)
# only force an ask on STRONG CONSENSUS (unanimous needs_ask) since the gate's resolved-vs-ask
# judgment is unreliable; (b) never force an ask before gathering; (c) if the element is degenerate,
# DEFER to the executor's own (coherent) question. Freeze these; do NOT tune against individual tasks.
_ASK_CONSENSUS_NUMERATOR = 1  # require needs_ask_votes >= votes (unanimous among the k votes)

# Tokens too short / generic to make a grammatical, answerable "the {element} you'd like?" question.
_DEGENERATE_ELEMENTS = {"on", "off", "true", "false", "value", "it", "the", "one", "yes", "no"}


def _is_degenerate_element(element: str | None) -> bool:
    """True if the gate's ambiguous_element can't form a coherent forced question -> defer instead."""
    el = (element or "").strip().lower()
    return not el or len(el) < 4 or el in _DEGENERATE_ELEMENTS


def _force_preferences_gather(
    tools: list[dict[str, Any]] | None,
    gathered: set[str],
    gate_summary: dict[str, Any],
) -> AggregationDecision | None:
    """Force a deterministic fetch-all `get_user_preferences` call, or None if not possible."""
    if DEFAULT_PREFERENCES_TOOL in gathered or not _preferences_available(tools):
        return None
    args = _fetch_all_preferences_args(tools)
    if args is None:
        return None
    return AggregationDecision(
        action={
            "action": "tool_calls",
            "tool_calls": [{"tool_name": DEFAULT_PREFERENCES_TOOL, "arguments": args}],
        },
        overridden=True,
        reason="ambiguity_gather_preferences",
        gate_summary=gate_summary,
    )


def _aggregate_ambiguity(
    executor_action: dict[str, Any],
    ambiguity: AmbiguityVerdict,
    tools: list[dict[str, Any]] | None,
    gathered_tools: set[str] | None,
    gate_summary: dict[str, Any],
) -> AggregationDecision | None:
    """The disambiguation branch. Returns a decision to OVERRIDE, or None to defer to the executor."""
    action_type = executor_action.get("action")
    gathered = gathered_tools or set()

    if ambiguity.status == AMBIGUITY_STATUS_GATHER:
        # Only force the PREFERENCE gather (the thing the executor neglects, and whose argument we
        # can build deterministically). Context gathers + already-gathered prefs defer to executor.
        if _is_preferences_gather(ambiguity.gather_tool):
            forced = _force_preferences_gather(tools, gathered, gate_summary)
            if forced is not None:
                return forced
        return None  # context gather / prefs unavailable / already gathered -> defer

    if ambiguity.status == AMBIGUITY_STATUS_ASK:
        # Ladder discipline: never ask before gathering. If preferences were never fetched, gather
        # first (they might resolve it internally) instead of asking prematurely.
        forced = _force_preferences_gather(tools, gathered, gate_summary)
        if forced is not None:
            return forced
        # Only force the ask on STRONG consensus (see _ASK_CONSENSUS_NUMERATOR): the gate's
        # resolved-vs-ask judgment is unreliable, so a split vote defers to the executor.
        ask_votes = ambiguity.status_votes.get(AMBIGUITY_STATUS_ASK, 0)
        if ambiguity.votes == 0 or ask_votes < ambiguity.votes * _ASK_CONSENSUS_NUMERATOR:
            return None
        # If the element is degenerate ("on"), a canned question would be unparseable -> defer to
        # the executor, which asks a coherent question in its own words (measured: that path passes).
        if _is_degenerate_element(ambiguity.ambiguous_element):
            return None
        # Don't override an executor that is already asking a clarifying question in its own words.
        if action_type == "respond" and _looks_like_question(
            executor_action.get("content")
        ):
            return None
        return AggregationDecision(
            action={
                "action": "respond",
                "content": _clarifying_question(ambiguity.ambiguous_element),
            },
            overridden=True,
            reason="ambiguity_ask",
            gate_summary=gate_summary,
        )

    # resolved (or any other status) -> defer to the executor; it should carry the resolved value.
    return None


# Cues that a `respond` already discharges a route-informing obligation (LLM-POL:021/022): mentions
# the chosen alternative, other alternatives, or tolls. Conservative on purpose -- a false "already
# informed" only skips a repair we might have wanted; it never corrupts a passing response. Freeze
# this list; do NOT tune it against individual task pass/fail (that would be Sec.4 scoring-repair).
_POLICY_INFORMING_MARKERS = (
    "fastest",
    "shortest",
    "alternative",
    "alternate route",
    "other route",
    "toll",
    "route option",
)


def _policy_informing_present(content: str | None) -> bool:
    """True if the respond already communicates the route-informing obligation (skip repair)."""
    if not content:
        return False
    lowered = content.lower()
    return any(marker in lowered for marker in _POLICY_INFORMING_MARKERS)


def _aggregate_policy(
    executor_action: dict[str, Any],
    policy: PolicyVerdict,
    gate_summary: dict[str, Any],
) -> AggregationDecision | None:
    """The policy-obligation branch. Signals a grounded repair of a respond that omits the informing.

    Only touches a `respond` action (an informing obligation is on the assistant's words, not on a
    tool call) and only when the draft does not already inform -> no action sequence is changed and
    already-compliant responses are left alone. Returns None to defer otherwise.
    """
    if executor_action.get("action") != "respond":
        return None
    if _policy_informing_present(executor_action.get("content")):
        return None
    obligation = (policy.obligation or "").strip()
    if not obligation:
        return None
    return AggregationDecision(
        action=executor_action,  # fallback if the repair call fails
        overridden=False,
        reason="policy_repair",
        gate_summary=gate_summary,
        repair_prompt=obligation,
    )


def aggregate(
    executor_action: dict[str, Any],
    grounding: GroundingVerdict,
    *,
    ambiguity: AmbiguityVerdict | None = None,
    policy: PolicyVerdict | None = None,
    tools: list[dict[str, Any]] | None = None,
    gathered_tools: set[str] | None = None,
) -> AggregationDecision:
    """Combine the executor's proposed action with the gate verdicts.

    Priority: grounding (can't do it at all) > ambiguity (needs clarification) > policy (must inform).
    """
    gate_summary = grounding.summary()
    if ambiguity is not None:
        gate_summary = {**gate_summary, **ambiguity.summary()}
    if policy is not None:
        gate_summary = {**gate_summary, **policy.summary()}
    action_type = executor_action.get("action")

    # 1) Grounding abstain veto (highest priority: can't do it at all beats needs-clarification).
    if grounding.tool_veto:
        # Hard abstain veto: override an executor about to ACT ungrounded (tool_calls), OR a
        # `respond` that fabricates (claims fulfillment / asks a capability-clarifying question)
        # instead of acknowledging the missing tool. A respond that already abstains is kept.
        fabricating_respond = action_type == "respond" and not _looks_like_abstention(
            executor_action.get("content")
        )
        if action_type == "tool_calls" or fabricating_respond:
            return AggregationDecision(
                action={
                    "action": "respond",
                    "content": _abstention_message(grounding.missing_capabilities),
                },
                overridden=True,
                reason="grounding_tool_veto"
                if action_type == "tool_calls"
                else "grounding_tool_veto_respond",
                gate_summary=gate_summary,
            )

    # 2) Disambiguation branch (only when ambiguity fires by majority).
    if ambiguity is not None and ambiguity.fire:
        decision = _aggregate_ambiguity(
            executor_action, ambiguity, tools, gathered_tools, gate_summary
        )
        if decision is not None:
            return decision

    # 3) Policy-obligation branch (only when policy fires by majority): flag a respond that omits a
    #    wiki-mandated informing so the gated agent can run one grounded repair. Lowest priority --
    #    an abstain/clarify decision above already returned.
    if policy is not None and policy.fire:
        decision = _aggregate_policy(executor_action, policy, gate_summary)
        if decision is not None:
            return decision

    # 4) No override -> take the executor's action.
    return AggregationDecision(
        action=executor_action,
        overridden=False,
        reason=None,
        gate_summary=gate_summary,
    )
