import json

import pytest
from conftest import ScriptedLLM

from logtriage.agent import TriageAgent
from logtriage.llm import OfflineRuleLLM, extract_json, get_llm

SAMPLE = (
    "2026-01-01 10:00:00.000 [INFO] [NAS] UE state: EMM-NULL\n"
    "2026-01-01 10:00:01.000 [ERROR] [PHY] Downlink sync lost, RSRP=-90dBm SINR=-3.0dB below threshold\n"
    "2026-01-01 10:00:02.000 [ERROR] [RRC] Radio link failure detected, timer T310 expired\n"
    "2026-01-01 10:00:03.000 [ERROR] [TEST] Test case result: FAIL\n"
)


def good(**over):
    d = dict(fault_class="PHY_SYNC_LOSS", root_cause="x", severity="high", root_line=2,
             evidence=[dict(line=2, text="Downlink sync lost")], duplicate_of=None, summary="s", next_steps=["n"])
    d.update(over)
    return json.dumps(d)


def agent(llm, stats=None, **kw):
    return TriageAgent(llm=llm, stats=stats, **kw)


def test_passing_log_skips_the_llm(dataset, stats):
    llm = ScriptedLLM("should never be called")
    r = next(r for r in dataset if r["label"] == "PASS")
    out = agent(llm, stats).run(r["text"])
    assert out["verdict"] == "PASS" and llm.calls == 0


def test_cascade_is_classified_by_root_not_last_error():
    out = agent(OfflineRuleLLM()).run(SAMPLE)
    assert out["fault_class"] == "PHY_SYNC_LOSS" and out["root_line"] == 2


def test_hallucinated_evidence_triggers_retry_and_is_fixed():
    bad = good(evidence=[dict(line=2, text="Kernel panic in modem")])
    llm = ScriptedLLM(bad, good())
    out = agent(llm).run(SAMPLE)
    assert llm.calls == 2 and out["attempts"] == 2 and not out["needs_human_review"]
    assert out["evidence"][0]["text"] == "Downlink sync lost"


def test_invalid_json_then_valid():
    llm = ScriptedLLM("sorry, here is my answer: no json", good())
    out = agent(llm).run(SAMPLE)
    assert out["attempts"] == 2 and out["fault_class"] == "PHY_SYNC_LOSS"


def test_persistent_failure_falls_back_and_flags_human_review():
    llm = ScriptedLLM(good(evidence=[dict(line=99, text="nope")], duplicate_of="DEF-9999", root_line=77))
    out = agent(llm, max_attempts=2).run(SAMPLE)
    assert out["needs_human_review"] and out["fallback_used"]
    assert out["evidence"] == [] and out["duplicate_of"] is None and out["root_line"] == 2


def test_duplicate_must_come_from_retrieved_candidates():
    llm = ScriptedLLM(good(duplicate_of="DEF-1420"))
    out = agent(llm, max_attempts=1).run(SAMPLE)
    assert out["duplicate_of"] is None


def test_llm_exception_is_handled():
    class Boom:
        name = "boom"
        def complete(self, system, user):
            raise RuntimeError("network down")
    out = agent(Boom(), max_attempts=2).run(SAMPLE)
    assert out["needs_human_review"] and out["root_line"] == 2


def test_langgraph_and_plain_runner_agree(dataset, stats):
    fails = [r for r in dataset if r["label"] != "PASS"][:15]
    g, p = agent(OfflineRuleLLM(), stats), agent(OfflineRuleLLM(), stats, use_langgraph=False)
    for r in fails:
        a, b = g.run(r["text"], r["run_id"]), p.run(r["text"], r["run_id"])
        assert (a["fault_class"], a["root_line"], a["duplicate_of"]) == (b["fault_class"], b["root_line"], b["duplicate_of"])


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nhope that helps') == {"a": 1}
    assert extract_json("no json here") is None
    assert extract_json("[1, 2]") is None


def test_get_llm_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_llm("nope")
