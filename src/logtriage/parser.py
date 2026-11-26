from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \[(?P<level>[A-Z]+)\] \[(?P<comp>[A-Za-z0-9_]+)\] (?P<msg>.*)$"
)
MASKS = [
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{12,}\b"), "<HEX>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"(?<![#\w])-?\d+(?:\.\d+)?(?![\d.])"), "<NUM>"),
]


@dataclass(frozen=True)
class Event:
    line_no: int
    ts: str
    level: str
    comp: str
    msg: str
    template: str
    tid: str


def mask(msg: str) -> str:
    for rx, token in MASKS:
        msg = rx.sub(token, msg)
    return msg


def template_id(template: str) -> str:
    return hashlib.md5(template.encode()).hexdigest()[:8]


def parse_log(text: str) -> list[Event]:
    events = []
    for i, raw in enumerate(text.splitlines(), start=1):
        m = LINE_RE.match(raw)
        if m:
            tpl = mask(m["msg"])
            events.append(Event(i, m["ts"], m["level"], m["comp"], m["msg"], tpl, template_id(tpl)))
        elif raw.strip():
            tpl = mask(raw.strip())
            events.append(Event(i, "", "RAW", "", raw.strip(), tpl, template_id(tpl)))
    return events
