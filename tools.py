from __future__ import annotations

from functools import lru_cache

from .agent import TriageAgent
from .generator import generate_dataset
from .kb import DefectIndex
from .llm import get_llm
from .parser import parse_log
from .signature import SEVERE, TemplateStats, extract_signature


@lru_cache(maxsize=1)
def default_stats() -> TemplateStats:
    runs = generate_dataset(200, seed=7, hard=False, prefix="base")
    return TemplateStats().fit([r["text"] for r in runs if r["label"] == "PASS"])


@lru_cache(maxsize=1)
def default_index() -> DefectIndex:
    return DefectIndex()


def triage_log(log_text: str, backend: str = "offline", prompt_version: str = "v2") -> dict:
    agent = TriageAgent(llm=get_llm(backend), kb=default_index(), stats=default_stats(), prompt_version=prompt_version)
    return agent.run(log_text, run_id="adhoc")


def search_defects(query: str, k: int = 3) -> list[dict]:
    return default_index().search(query, k=k)


def list_error_events(log_text: str, only_suspects: bool = True) -> list[dict]:
    events = parse_log(log_text)
    if only_suspects:
        events = extract_signature(events, default_stats()).suspects
    else:
        events = [e for e in events if e.level in SEVERE]
    return [dict(line=e.line_no, level=e.level, component=e.comp, text=e.msg) for e in events]


def get_log_context(log_text: str, line: int, radius: int = 5) -> str:
    lines = log_text.splitlines()
    lo, hi = max(1, line - radius), min(len(lines), line + radius)
    return "\n".join(f"{i:>4}: {lines[i - 1]}" for i in range(lo, hi + 1))
