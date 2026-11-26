from __future__ import annotations

import json
import os
import re

import requests

from .prompts import FAULT_CLASSES


def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class OfflineRuleLLM:
    name = "offline-rules"
    RULES = [
        (r"Downlink sync lost", "PHY_SYNC_LOSS", "high"),
        (r"Radio link failure", "RRC_RLF", "high"),
        (r"T3410 expired|Registration failed after max retries", "NAS_REG_TIMEOUT", "medium"),
        (r"Attach reject", "ATTACH_REJECT_PLMN", "medium"),
        (r"Authentication failure|AUTN MAC check failed", "AUTH_MAC_FAILURE", "high"),
        (r"Handover (command|failure)", "HANDOVER_FAILURE", "medium"),
        (r"PDN connectivity reject", "PDN_REJECT_NO_IP", "medium"),
        (r"Watchdog timeout|Assertion failed", "MODEM_WATCHDOG_RESET", "critical"),
    ]

    def complete(self, system: str, user: str) -> str:
        ev = json.loads(re.search(r"<evidence>(.*?)</evidence>", user, re.S).group(1))
        suspects = ev["suspect_events"]
        fault, sev, hit = "OTHER", "medium", None
        for s in suspects:
            for pattern, cls, severity in self.RULES:
                if re.search(pattern, s["text"]):
                    fault, sev, hit = cls, severity, s
                    break
            if hit:
                break
        root = hit or suspects[0]
        dup = None
        for c in ev["candidate_defects"]:
            if c["fault_class"] == fault and c["score"] >= ev["dup_threshold"]:
                dup = c["id"]
                break
        cause = next((c["root_cause"] for c in ev["candidate_defects"] if c["id"] == dup), "No matching known defect")
        return json.dumps(dict(
            fault_class=fault, root_cause=cause, severity=sev, root_line=root["line"],
            evidence=[dict(line=s["line"], text=s["text"]) for s in suspects[:3]], duplicate_of=dup,
            summary=f"Rule-based triage: {fault} first seen at line {root['line']}.",
            next_steps=["Inspect the lines before the first suspect", "Compare with the linked defect" if dup else "File a new defect"]))


class AnthropicLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: int = 60):
        self.model = model or os.environ.get("LOGTRIAGE_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.timeout = timeout
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, user: str) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages", timeout=self.timeout,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 1000, "temperature": 0, "system": system,
                  "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class OpenAICompatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, timeout: int = 120):
        self.model = model or os.environ.get("LOGTRIAGE_MODEL", "llama3.1")
        self.base_url = (base_url or os.environ.get("LOGTRIAGE_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LOGTRIAGE_API_KEY", "ollama")
        self.timeout = timeout
        self.name = f"openai-compat:{self.model}"

    def complete(self, system: str, user: str) -> str:
        r = requests.post(
            f"{self.base_url}/chat/completions", timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "temperature": 0,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def get_llm(backend: str = "offline", **kw):
    backend = backend.lower()
    if backend == "offline":
        return OfflineRuleLLM()
    if backend == "anthropic":
        return AnthropicLLM(**kw)
    if backend in ("openai", "ollama"):
        return OpenAICompatLLM(**kw)
    raise ValueError(f"unknown backend {backend!r}; use offline | anthropic | openai")


__all__ = ["get_llm", "extract_json", "OfflineRuleLLM", "AnthropicLLM", "OpenAICompatLLM", "FAULT_CLASSES"]
