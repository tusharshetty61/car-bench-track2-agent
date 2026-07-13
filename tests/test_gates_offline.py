"""Offline (no-API) tests for the grounding gate code path and the aggregator.

These lock the DETERMINISTIC core of the gate-first harness: tool reconcile (no substring),
the soft param check, requirement assessment (tool = hard, param = soft), and the asymmetric
aggregation (veto overrides an ungrounded tool_calls action AND a fabricating respond, keeps a
respond that already abstains, param never vetoes).
"""

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from track_2_agent_under_test_cerebras_planner.aggregator import (  # noqa: E402
    _fetch_all_preferences_args,
    _looks_like_question,
    _policy_informing_present,
    aggregate,
)
from track_2_agent_under_test_cerebras_planner.gates import (  # noqa: E402
    AMBIGUITY_STATUS_ASK,
    AMBIGUITY_STATUS_GATHER,
    AMBIGUITY_STATUS_NONE,
    AMBIGUITY_STATUS_RESOLVED,
    AmbiguityVerdict,
    GroundingVerdict,
    PolicyVerdict,
    aggregate_ambiguity_votes,
    aggregate_policy_votes,
    assess_requirements,
    compact_tools,
    param_present,
    reconcile,
    tool_index,
)


def _tool(name, params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in params},
            },
        },
    }


TOOLS = [
    _tool("set_seat_heating", ["level", "seat_zone"]),
    _tool("get_weather", ["location", "month", "day"]),
    _tool("set_ambient_lights", ["on", "lightcolor"]),
]


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.names = set(tool_index(TOOLS))

    def test_exact(self):
        self.assertEqual(reconcile("get_weather", self.names), "get_weather")

    def test_case_insensitive(self):
        self.assertEqual(reconcile("Get_Weather", self.names), "get_weather")

    def test_normalized_separators(self):
        self.assertEqual(reconcile("getweather", self.names), "get_weather")
        self.assertEqual(reconcile("get-weather", self.names), "get_weather")

    def test_no_substring_match(self):
        # "weather" is a substring of "get_weather" but must NOT resolve (recall killer).
        self.assertIsNone(reconcile("weather", self.names))

    def test_unknown_tool(self):
        self.assertIsNone(reconcile("teleport", self.names))


class ParamPresentTests(unittest.TestCase):
    def setUp(self):
        self.params = {"level", "seat_zone"}

    def test_off_always_present(self):
        self.assertTrue(param_present("anything", self.params, "off"))

    def test_empty_needed_present(self):
        self.assertTrue(param_present("", self.params, "strict"))

    def test_strict_exact(self):
        self.assertTrue(param_present("seat_zone", self.params, "strict"))
        self.assertFalse(param_present("zone", self.params, "strict"))

    def test_norm_contains_absorbs_value_junk(self):
        # Enumerator emits "level=1, seat_zone=driver"; norm-contains tolerates it.
        self.assertTrue(param_present("seat_zone=driver", self.params, "norm-contains"))
        self.assertTrue(param_present("on=False", {"on"}, "norm-contains"))

    def test_norm_contains_flags_truly_absent(self):
        self.assertFalse(param_present("brightness", self.params, "norm-contains"))


class AssessRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.available = tool_index(TOOLS)

    def test_all_present(self):
        reqs = [{"capability": "warm seat", "tool_name": "set_seat_heating",
                 "needed_parameter": "seat_zone"}]
        a = assess_requirements(reqs, self.available)
        self.assertFalse(a.tool_missing)
        self.assertFalse(a.param_missing)

    def test_missing_sentinel_is_tool_missing(self):
        reqs = [{"capability": "open trunk", "tool_name": "MISSING",
                 "needed_parameter": ""}]
        a = assess_requirements(reqs, self.available)
        self.assertTrue(a.tool_missing)
        self.assertIn("open trunk", a.missing_capabilities)

    def test_unresolvable_tool_is_tool_missing(self):
        reqs = [{"capability": "teleport", "tool_name": "teleport_tool",
                 "needed_parameter": ""}]
        a = assess_requirements(reqs, self.available)
        self.assertTrue(a.tool_missing)

    def test_missing_param_is_soft_only(self):
        reqs = [{"capability": "set color", "tool_name": "set_ambient_lights",
                 "needed_parameter": "brightness"}]
        a = assess_requirements(reqs, self.available)
        self.assertFalse(a.tool_missing)  # tool present
        self.assertTrue(a.param_missing)  # param absent -> soft flag only


class AggregatorTests(unittest.TestCase):
    def test_tool_veto_overrides_tool_calls_action(self):
        verdict = GroundingVerdict(
            tool_veto=True, missing_capabilities=["open the trunk"], votes=3,
            tool_missing_votes=3,
        )
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_trunk_door", "arguments": {"action": "open"}}]}
        decision = aggregate(action, verdict)
        self.assertTrue(decision.overridden)
        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("open the trunk", decision.action["content"])
        self.assertEqual(decision.reason, "grounding_tool_veto")

    def test_tool_veto_overrides_fabricating_respond_fulfillment(self):
        # Claims fulfillment of an ungrounded capability -> implicit fabrication -> override.
        verdict = GroundingVerdict(tool_veto=True, missing_capabilities=["change air circulation"],
                                   votes=3, tool_missing_votes=3)
        action = {"action": "respond", "content": "Fresh air mode is set."}
        decision = aggregate(action, verdict)
        self.assertTrue(decision.overridden)
        self.assertEqual(decision.action["action"], "respond")
        self.assertIn("change air circulation", decision.action["content"])
        self.assertEqual(decision.reason, "grounding_tool_veto_respond")

    def test_tool_veto_overrides_fabricating_respond_clarifying_question(self):
        # The hall_4 regression: a capability-clarifying question implies the action is possible.
        verdict = GroundingVerdict(tool_veto=True, missing_capabilities=["change air circulation"],
                                   votes=3, tool_missing_votes=3)
        action = {"action": "respond",
                  "content": "Which airflow direction would you like - head, feet, or windshield?"}
        decision = aggregate(action, verdict)
        self.assertTrue(decision.overridden)
        self.assertEqual(decision.reason, "grounding_tool_veto_respond")

    def test_tool_veto_keeps_respond_that_already_abstains(self):
        # Executor already acknowledged the inability -> keep its genuine phrasing untouched.
        verdict = GroundingVerdict(tool_veto=True, missing_capabilities=["x"], votes=3,
                                   tool_missing_votes=3)
        action = {"action": "respond",
                  "content": "Sorry, I can't do that - I don't have a way to change air circulation."}
        decision = aggregate(action, verdict)
        self.assertFalse(decision.overridden)
        self.assertEqual(decision.action, action)

    def test_no_veto_keeps_respond_clarifying_question(self):
        # Base/Disambiguation: tools present -> tool_veto False -> a clarifying question is kept.
        verdict = GroundingVerdict(tool_veto=False, votes=3, tool_missing_votes=0)
        action = {"action": "respond", "content": "Which seat would you like heated?"}
        decision = aggregate(action, verdict)
        self.assertFalse(decision.overridden)
        self.assertEqual(decision.action, action)

    def test_no_veto_passes_through(self):
        verdict = GroundingVerdict(tool_veto=False, votes=3, tool_missing_votes=0)
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "set_seat_heating", "arguments": {"level": 2}}]}
        decision = aggregate(action, verdict)
        self.assertFalse(decision.overridden)
        self.assertEqual(decision.action, action)

    def test_param_flag_never_vetoes(self):
        verdict = GroundingVerdict(
            tool_veto=False, param_flag=True, param_issues=["set_ambient_lights.brightness"],
            votes=3, tool_missing_votes=0,
        )
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "set_ambient_lights", "arguments": {}}]}
        decision = aggregate(action, verdict)
        self.assertFalse(decision.overridden)  # param is soft, must not block


def _prefs_tool():
    """get_user_preferences with a nested category/subcategory schema (like the real tool)."""
    return {
        "type": "function",
        "function": {
            "name": "get_user_preferences",
            "description": "Retrieve stored preferences",
            "parameters": {
                "type": "object",
                "required": ["preference_categories"],
                "properties": {
                    "preference_categories": {
                        "type": "object",
                        "properties": {
                            "vehicle_settings": {
                                "type": "object",
                                "properties": {
                                    "vehicle_settings": {"type": "boolean"},
                                    "climate_control": {"type": "boolean"},
                                },
                            },
                            "weather": {
                                "type": "object",
                                "properties": {"weather": {"type": "boolean"}},
                            },
                        },
                    }
                },
            },
        },
    }


PREFS_TOOLS = [_prefs_tool(), _tool("open_close_sunroof", ["percentage"]),
               _tool("get_exterior_lights_status", [])]


class AmbiguityVoteTests(unittest.TestCase):
    def test_no_votes_is_inert(self):
        v = aggregate_ambiguity_votes([])
        self.assertFalse(v.fire)
        self.assertEqual(v.status, AMBIGUITY_STATUS_NONE)

    def test_majority_fire_needed(self):
        # 1 of 3 ambiguous -> does NOT fire (protects Base from ~1/3 false-positive).
        votes = [
            {"ambiguous": True, "status": "needs_gather", "gather_tool": "get_user_preferences",
             "ambiguous_element": "x", "resolved_value": ""},
            {"ambiguous": False, "status": "none", "gather_tool": "", "ambiguous_element": "",
             "resolved_value": ""},
            {"ambiguous": False, "status": "none", "gather_tool": "", "ambiguous_element": "",
             "resolved_value": ""},
        ]
        self.assertFalse(aggregate_ambiguity_votes(votes).fire)

    def test_modal_status_and_element(self):
        votes = [
            {"ambiguous": True, "status": "needs_gather", "gather_tool": "get_user_preferences",
             "ambiguous_element": "sunroof percentage", "resolved_value": ""},
            {"ambiguous": True, "status": "needs_gather", "gather_tool": "get_user_preferences",
             "ambiguous_element": "sunroof percentage", "resolved_value": ""},
            {"ambiguous": True, "status": "resolved", "gather_tool": "",
             "ambiguous_element": "sunroof percentage", "resolved_value": "50"},
        ]
        v = aggregate_ambiguity_votes(votes)
        self.assertTrue(v.fire)
        self.assertEqual(v.status, AMBIGUITY_STATUS_GATHER)
        self.assertEqual(v.ambiguous_element, "sunroof percentage")
        self.assertEqual(v.gather_tool, "get_user_preferences")

    def test_resolved_carries_value(self):
        votes = [
            {"ambiguous": True, "status": "resolved", "gather_tool": "",
             "ambiguous_element": "color", "resolved_value": "PURPLE"},
            {"ambiguous": True, "status": "resolved", "gather_tool": "",
             "ambiguous_element": "color", "resolved_value": "PURPLE"},
            {"ambiguous": False, "status": "none", "gather_tool": "", "ambiguous_element": "",
             "resolved_value": ""},
        ]
        v = aggregate_ambiguity_votes(votes)
        self.assertEqual(v.status, AMBIGUITY_STATUS_RESOLVED)
        self.assertEqual(v.resolved_value, "PURPLE")

    def test_tie_breaks_to_safest_gather(self):
        # 1 gather / 1 ask among 2 firing votes (of 3) -> tie -> prefer the safer gather.
        votes = [
            {"ambiguous": True, "status": "needs_gather", "gather_tool": "get_user_preferences",
             "ambiguous_element": "x", "resolved_value": ""},
            {"ambiguous": True, "status": "needs_ask", "gather_tool": "", "ambiguous_element": "x",
             "resolved_value": ""},
            {"ambiguous": True, "status": "needs_ask", "gather_tool": "", "ambiguous_element": "x",
             "resolved_value": ""},
        ]
        # 2 ask vs 1 gather -> modal is ask (not a tie); check that.
        self.assertEqual(aggregate_ambiguity_votes(votes).status, AMBIGUITY_STATUS_ASK)


class FetchAllPreferencesTests(unittest.TestCase):
    def test_builds_nested_selector_all_true(self):
        args = _fetch_all_preferences_args(PREFS_TOOLS)
        self.assertIn("preference_categories", args)
        cats = args["preference_categories"]
        self.assertEqual(cats["vehicle_settings"],
                         {"vehicle_settings": True, "climate_control": True})
        self.assertEqual(cats["weather"], {"weather": True})

    def test_returns_none_when_tool_absent(self):
        self.assertIsNone(_fetch_all_preferences_args(TOOLS))


class LooksLikeQuestionTests(unittest.TestCase):
    def test_question_mark(self):
        self.assertTrue(_looks_like_question("What color would you like?"))

    def test_interrogative_phrase(self):
        self.assertTrue(_looks_like_question("Let me know the seat zone"))

    def test_statement_is_not_question(self):
        self.assertFalse(_looks_like_question("Setting the color to purple."))


class AmbiguityAggregatorTests(unittest.TestCase):
    def _fire(self, status, **kw):
        # Default status_votes to a unanimous tally for `status` unless the test overrides it.
        kw.setdefault("status_votes", {status: 3})
        return AmbiguityVerdict(fire=True, status=status, votes=3, fire_votes=3, **kw)

    def test_needs_gather_forces_preferences_call(self):
        amb = self._fire(AMBIGUITY_STATUS_GATHER, gather_tool="get_user_preferences",
                         ambiguous_element="sunroof percentage")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_sunroof", "arguments": {"percentage": 100}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools=set())
        self.assertTrue(d.overridden)
        self.assertEqual(d.reason, "ambiguity_gather_preferences")
        self.assertEqual(d.action["tool_calls"][0]["tool_name"], "get_user_preferences")
        self.assertIn("preference_categories", d.action["tool_calls"][0]["arguments"])

    def test_needs_gather_defers_if_already_gathered(self):
        # Belt-and-braces gather-once guard: don't re-force get_user_preferences.
        amb = self._fire(AMBIGUITY_STATUS_GATHER, gather_tool="get_user_preferences",
                         ambiguous_element="x")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools={"get_user_preferences"})
        self.assertFalse(d.overridden)
        self.assertEqual(d.action, action)

    def test_needs_gather_context_tool_defers_to_executor(self):
        # A CONTEXT gather (not preferences) is left to the executor (it gathers context itself).
        amb = self._fire(AMBIGUITY_STATUS_GATHER, gather_tool="get_exterior_lights_status",
                         ambiguous_element="which lights")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "get_exterior_lights_status", "arguments": {}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools=set())
        self.assertFalse(d.overridden)

    def test_needs_ask_unanimous_after_gather_forces_question(self):
        # Unanimous needs_ask AND prefs already gathered -> force the targeted question (user path).
        amb = self._fire(AMBIGUITY_STATUS_ASK, ambiguous_element="window percentage")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools={"get_user_preferences"})
        self.assertTrue(d.overridden)
        self.assertEqual(d.reason, "ambiguity_ask")
        self.assertEqual(d.action["action"], "respond")
        self.assertIn("window percentage", d.action["content"])

    def test_needs_ask_before_gather_gathers_first(self):
        # Ladder discipline: never ask before gathering prefs -> force the gather, not the ask.
        amb = self._fire(AMBIGUITY_STATUS_ASK, ambiguous_element="window percentage")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools=set())  # prefs NOT yet fetched
        self.assertTrue(d.overridden)
        self.assertEqual(d.reason, "ambiguity_gather_preferences")

    def test_needs_ask_split_vote_defers(self):
        # The fatal-internal-ask fix: a non-unanimous needs_ask defers to the executor (no forced ask).
        amb = AmbiguityVerdict(fire=True, status=AMBIGUITY_STATUS_ASK, votes=3, fire_votes=3,
                               ambiguous_element="color", status_votes={"needs_ask": 2, "resolved": 1})
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "set_ambient_lights", "arguments": {"lightcolor": "PURPLE"}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools={"get_user_preferences"})
        self.assertFalse(d.overridden)
        self.assertEqual(d.action, action)

    def test_needs_ask_degenerate_element_defers(self):
        # Garbled element ("on") -> a canned question would be unparseable -> defer to executor.
        amb = self._fire(AMBIGUITY_STATUS_ASK, ambiguous_element="on")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "set_ambient_lights", "arguments": {"on": True}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools={"get_user_preferences"})
        self.assertFalse(d.overridden)
        self.assertEqual(d.action, action)

    def test_needs_ask_keeps_executor_question(self):
        amb = self._fire(AMBIGUITY_STATUS_ASK, ambiguous_element="window percentage")
        action = {"action": "respond", "content": "How much would you like the window opened?"}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools={"get_user_preferences"})
        self.assertFalse(d.overridden)

    def test_resolved_defers_to_executor(self):
        # The corrected-flaw regression test: resolved must NOT force an ask on an internal task.
        amb = self._fire(AMBIGUITY_STATUS_RESOLVED, ambiguous_element="color",
                         resolved_value="PURPLE")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "set_ambient_lights", "arguments": {"lightcolor": "PURPLE"}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools={"get_user_preferences"})
        self.assertFalse(d.overridden)
        self.assertEqual(d.action, action)

    def test_grounding_veto_takes_priority_over_ambiguity(self):
        grounding = GroundingVerdict(tool_veto=True, missing_capabilities=["open the trunk"],
                                     votes=3, tool_missing_votes=3)
        amb = self._fire(AMBIGUITY_STATUS_GATHER, gather_tool="get_user_preferences")
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_trunk_door", "arguments": {}}]}
        d = aggregate(action, grounding, ambiguity=amb, tools=PREFS_TOOLS, gathered_tools=set())
        self.assertTrue(d.overridden)
        self.assertEqual(d.reason, "grounding_tool_veto")  # abstain wins, not gather

    def test_no_fire_passes_through(self):
        amb = AmbiguityVerdict(fire=False, status=AMBIGUITY_STATUS_NONE, votes=3, fire_votes=0)
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "open_close_sunroof", "arguments": {"percentage": 50}}]}
        d = aggregate(action, GroundingVerdict(), ambiguity=amb, tools=PREFS_TOOLS,
                      gathered_tools=set())
        self.assertFalse(d.overridden)


class CompactToolsTests(unittest.TestCase):
    def test_compacts_to_names_and_params(self):
        compact = compact_tools(TOOLS)
        self.assertEqual(compact[0]["name"], "set_seat_heating")
        self.assertEqual(set(compact[0]["parameters"]), {"level", "seat_zone"})

    def test_tolerates_flat_schema(self):
        flat = [{"name": "foo", "description": "d", "parameters": {"properties": {"a": {}}}}]
        self.assertEqual(compact_tools(flat)[0]["name"], "foo")
        self.assertEqual(tool_index(flat)["foo"], {"a"})


def _pol(applies, ids=None, obligation=""):
    return {
        "obligation_applies": applies,
        "policy_ids": ids or [],
        "obligation": obligation,
    }


class PolicyVoteTests(unittest.TestCase):
    def test_no_votes_is_inert(self):
        v = aggregate_policy_votes([])
        self.assertFalse(v.fire)

    def test_majority_fire_needed(self):
        # 1 of 3 -> no fire (protects Base from a lone over-eager vote).
        v = aggregate_policy_votes([_pol(True, ["LLM-POL:022"], "inform fastest"),
                                    _pol(False), _pol(False)])
        self.assertFalse(v.fire)

    def test_majority_fires_and_merges(self):
        v = aggregate_policy_votes([
            _pol(True, ["LLM-POL:022"], "inform you took the fastest route and offer alternatives"),
            _pol(True, ["LLM-POL:022", "LLM-POL:021"], "inform you took the fastest route and offer alternatives"),
            _pol(False),
        ])
        self.assertTrue(v.fire)
        self.assertEqual(v.fire_votes, 2)
        self.assertIn("LLM-POL:022", v.policy_ids)
        self.assertIn("LLM-POL:021", v.policy_ids)  # merged across firing votes
        self.assertIn("fastest", v.obligation)


class PolicyInformingMarkerTests(unittest.TestCase):
    def test_detects_informing(self):
        self.assertTrue(_policy_informing_present("I took the fastest route; there are 2 other alternatives."))
        self.assertTrue(_policy_informing_present("Note this route includes a toll road."))

    def test_bare_confirmation_is_not_informing(self):
        self.assertFalse(_policy_informing_present(
            "Stuttgart removed, navigation now goes straight from Mannheim to Paris."))


class PolicyAggregatorTests(unittest.TestCase):
    def _fire(self, obligation="Inform the user you took the fastest route and offer other alternatives."):
        return PolicyVerdict(fire=True, policy_ids=["LLM-POL:022"], obligation=obligation,
                             votes=3, fire_votes=3)

    def test_fires_on_respond_missing_informing(self):
        action = {"action": "respond",
                  "content": "Stuttgart removed, navigation now goes straight from Mannheim to Paris."}
        d = aggregate(action, GroundingVerdict(), policy=self._fire(), tools=TOOLS)
        # Pure-code aggregator can't rewrite -> signals a repair (overridden stays False; the gated
        # agent performs the grounded rewrite).
        self.assertFalse(d.overridden)
        self.assertEqual(d.reason, "policy_repair")
        self.assertTrue(d.repair_prompt)
        self.assertEqual(d.action, action)  # fallback if repair call fails

    def test_no_repair_when_already_informing(self):
        action = {"action": "respond",
                  "content": "I took the fastest route to Paris; there are 2 other alternatives if you'd like."}
        d = aggregate(action, GroundingVerdict(), policy=self._fire(), tools=TOOLS)
        self.assertIsNone(d.repair_prompt)
        self.assertFalse(d.overridden)

    def test_never_touches_tool_calls(self):
        action = {"action": "tool_calls", "tool_calls": [
            {"tool_name": "set_seat_heating", "arguments": {"level": 2}}]}
        d = aggregate(action, GroundingVerdict(), policy=self._fire(), tools=TOOLS)
        self.assertIsNone(d.repair_prompt)
        self.assertFalse(d.overridden)

    def test_no_fire_passes_through(self):
        action = {"action": "respond", "content": "Done."}
        d = aggregate(action, GroundingVerdict(),
                      policy=PolicyVerdict(fire=False, votes=3), tools=TOOLS)
        self.assertIsNone(d.repair_prompt)
        self.assertFalse(d.overridden)

    def test_grounding_veto_takes_priority_over_policy(self):
        # Can't-do-it-at-all beats must-inform: grounding wins, no policy repair signalled.
        g = GroundingVerdict(tool_veto=True, missing_capabilities=["do that"], votes=3,
                             tool_missing_votes=3)
        action = {"action": "respond", "content": "All set."}
        d = aggregate(action, g, policy=self._fire(), tools=TOOLS)
        self.assertTrue(d.overridden)
        self.assertEqual(d.reason, "grounding_tool_veto_respond")
        self.assertIsNone(d.repair_prompt)


if __name__ == "__main__":
    unittest.main()
