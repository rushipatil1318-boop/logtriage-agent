from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .parser import Event, parse_log

SEVERE = {"ERROR", "FATAL"}


@dataclass
class TemplateStats:
    n_pass: int = 0
    pass_df: Counter = field(default_factory=Counter)
    benign_threshold: float = 0.05

    def fit(self, passing_logs: list[str]) -> "TemplateStats":
        self.n_pass = len(passing_logs)
        self.pass_df = Counter()
        for text in passing_logs:
            self.pass_df.update({e.tid for e in parse_log(text) if e.level in SEVERE})
        return self

    def pass_rate(self, tid: str) -> float:
        return self.pass_df[tid] / self.n_pass if self.n_pass else 0.0

    def is_benign(self, e: Event) -> bool:
        return self.pass_rate(e.tid) >= self.benign_threshold


@dataclass
class Signature:
    suspects: list[Event]
    first_suspect: Event | None
    context: list[Event]
    text: str

    @property
    def has_failure(self) -> bool:
        return bool(self.suspects)


def extract_signature(events: list[Event], stats: TemplateStats | None = None, context_lines: int = 3) -> Signature:
    suspects = [e for e in events if e.level in SEVERE and not (stats and stats.is_benign(e))]
    suspects = [e for e in suspects if not e.msg.startswith("Test case result")]
    if not suspects:
        return Signature([], None, [], "")
    first = suspects[0]
    idx = events.index(first)
    context = events[max(0, idx - context_lines): idx]
    seen, parts = set(), []
    for e in context + suspects:
        if e.tid not in seen:
            seen.add(e.tid)
            parts.append(e.template)
    return Signature(suspects, first, context, " | ".join(parts))
