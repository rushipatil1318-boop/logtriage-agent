import asyncio

from fastapi.testclient import TestClient

from logtriage.api import app
from logtriage.evaluate import run_config
from logtriage.generator import generate_dataset
from logtriage.mcp_server import mcp

client = TestClient(app)
LOG = generate_dataset(40, 5)
FAIL = next(r for r in LOG if r["label"] != "PASS")


def test_health_and_triage():
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/triage", json={"log_text": FAIL["text"]})
    assert r.status_code == 200 and r.json()["verdict"] == "FAIL"
    assert r.json()["root_line"] == FAIL["root_line"]


def test_api_validation_errors():
    assert client.post("/triage", json={"log_text": ""}).status_code == 422
    assert client.post("/triage", json={"log_text": "x", "backend": "nope"}).status_code == 422
    assert client.post("/triage", json={"log_text": "x", "prompt_version": "v9"}).status_code == 422


def test_defect_search_endpoint():
    r = client.get("/defects/search", params={"q": "watchdog stack overflow canary", "k": 2})
    assert r.json()[0]["id"] == "DEF-1420"


def test_mcp_tools_registered_and_callable():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {"triage_log", "search_defects", "list_error_events", "get_log_context"}
    out = asyncio.run(mcp.call_tool("search_defects", {"query": "AGC saturation", "k": 1}))
    assert "DEF-1042" in str(out)


def test_evaluation_smoke():
    m = run_config("offline", "v2", hard=False, n=40)
    assert m["n"] == 40 and 0.9 <= m["fault_class_accuracy"] <= 1.0
    assert m["citation_validity"] == 1.0


def test_http_backends_build_correct_requests(monkeypatch):
    from unittest.mock import MagicMock

    from logtriage import llm as llm_mod

    seen = {}

    def fake_post(url, **kw):
        seen["url"], seen["kw"] = url, kw
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = ({"content": [{"type": "text", "text": "{}"}]} if "anthropic" in url
                                  else {"choices": [{"message": {"content": "{}"}}]})
        return resp

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    assert llm_mod.AnthropicLLM(model="m", api_key="k").complete("sys", "usr") == "{}"
    assert seen["url"].endswith("/v1/messages") and seen["kw"]["headers"]["x-api-key"] == "k"
    assert seen["kw"]["json"]["system"] == "sys" and seen["kw"]["json"]["temperature"] == 0
    assert llm_mod.OpenAICompatLLM(model="m", base_url="http://localhost:11434/v1").complete("sys", "usr") == "{}"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["kw"]["json"]["messages"][0] == {"role": "system", "content": "sys"}
