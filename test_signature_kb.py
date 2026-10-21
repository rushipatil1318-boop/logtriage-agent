from logtriage.defects import DEFECT_BY_FAULT
from logtriage.kb import DefectIndex
from logtriage.parser import parse_log
from logtriage.signature import TemplateStats, extract_signature


def test_stats_learn_benign_errors(stats):
    text = "2026-01-01 10:00:00.000 [ERROR] [NVM] NVRAM read retry succeeded after 2 attempt(s)\n"
    assert stats.is_benign(parse_log(text)[0])
    real = "2026-01-01 10:00:00.000 [ERROR] [PHY] Downlink sync lost, RSRP=-90dBm SINR=-3.0dB below threshold\n"
    assert not stats.is_benign(parse_log(real)[0])


def test_first_suspect_is_the_root_cause_line(dataset, stats):
    fails = [r for r in dataset if r["label"] != "PASS"]
    hits = sum(extract_signature(parse_log(r["text"]), stats).first_suspect.line_no == r["root_line"] for r in fails)
    assert hits / len(fails) > 0.95


def test_passing_runs_have_no_suspects(dataset, stats):
    passes = [r for r in dataset if r["label"] == "PASS"]
    assert all(not extract_signature(parse_log(r["text"]), stats).has_failure for r in passes)


def test_without_stats_benign_errors_become_false_suspects(dataset):
    passes = [r for r in dataset if r["label"] == "PASS"]
    assert any(extract_signature(parse_log(r["text"]), TemplateStats()).has_failure for r in passes)


def test_kb_retrieves_known_defect_from_its_marker():
    kb = DefectIndex()
    for fault, d in DEFECT_BY_FAULT.items():
        query = d["marker"][2].format(band=3, n=5, ms=900, task="rrc_ctrl")
        assert kb.search(query, k=1)[0]["id"] == d["id"], fault


def test_kb_empty_query():
    assert DefectIndex().search("   ") == []
