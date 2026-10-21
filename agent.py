from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from .kb import DefectIndex
from .llm import extract_json, get_llm
from .parser import parse_log
from .prompts import FAULT_CLASSES, SEVERITIES, SYSTEMS, build_user_prompt
from .signature import TemplateStats, extract_signature

try:
    from langgraph.graph import END, START, StateGraph
    HAVE_LANGGRAPH = True
except ImportError:
    HAVE_LANGGRAPH = False


class TriageState(TypedDict, total=False):
    log_text: str
    run_id: str
    events: list
    signature: Any
    candidates: list
    draft: dict | None
    raw: str
    issues: list
    attempts: int
    result: dict


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


class TriageAgent:
    def __init__(self, llm=None, kb: DefectIndex | None = None, stats: TemplateStats | None = None,
                 prompt_version: str = "v2", max_attempts: int = 2, dup_threshold: float = 0.35, use_langgraph: bool = True):
        self.llm = llm or get_llm("offline")
        self.kb = kb or DefectIndex()
        self.stats = stats
        self.prompt_version = prompt_version
        self.max_attempts = max_attempts
        self.dup_threshold = dup_threshold
        self.graph = self._build_graph() if (use_langgraph and HAVE_LANGGRAPH) else None

    def n_parse(self, s: TriageState) -> dict:
        return {"events": parse_log(s["log_text"]), "attempts": 0, "issues": []}

    def n_signature(self, s: TriageState) -> dict:
        return {"signature": extract_signature(s["events"], self.stats)}

    def n_retrieve(self, s: TriageState) -> dict:
        return {"candidates": self.kb.search(s["signature"].text, k=3)}

    def _evidence(self, s: TriageState) -> dict:
        sig = s["signature"]
        fmt = lambda e: dict(line=e.line_no, level=e.level, component=e.comp, text=e.msg)
        return dict(run_id=s["run_id"], suspect_events=[fmt(e) for e in sig.suspects[:12]],
                    context_events=[fmt(e) for e in sig.context], candidate_defects=s["candidates"], dup_threshold=self.dup_threshold)

    def n_hypothesize(self, s: TriageState) -> dict:
        prompt = build_user_prompt(self._evidence(s), s["issues"] if s["attempts"] else None)
        try:
            raw = self.llm.complete(SYSTEMS[self.prompt_version], prompt)
        except Exception as exc:
            raw = f"LLM_ERROR: {exc}"
        return {"raw": raw, "draft": extract_json(raw), "attempts": s["attempts"] + 1}

    def n_verify(self, s: TriageState) -> dict:
        d, issues = s["draft"], []
        raw_lines = s["log_text"].splitlines()
        suspect_lines = {e.line_no for e in s["signature"].suspects}
        if d is None:
            issues.append("Answer was not valid JSON. Return only the JSON object.")
        else:
            if d.get("fault_class") not in FAULT_CLASSES:
                issues.append(f"fault_class must be one of {FAULT_CLASSES}")
            if d.get("severity") not in SEVERITIES:
                issues.append(f"severity must be one of {SEVERITIES}")
            if d.get("root_line") not in suspect_lines:
                issues.append(f"root_line {d.get('root_line')!r} is not one of the suspect lines {sorted(suspect_lines)[:12]}")
            for ev in d.get("evidence") or []:
                ln = ev.get("line")
                if not isinstance(ln, int) or not 1 <= ln <= len(raw_lines):
                    issues.append(f"evidence line {ln!r} does not exist")
                elif not ev.get("text") or _norm(ev["text"]) not in _norm(raw_lines[ln - 1]):
                    issues.append(f"evidence text does not match line {ln}")
            dup = d.get("duplicate_of")
            if dup is not None and dup not in {c["id"] for c in s["candidates"]}:
                issues.append(f"duplicate_of {dup!r} is not in candidate_defects")
        return {"issues": issues}

    def n_report(self, s: TriageState) -> dict:
        sig = s["signature"]
        if not sig.has_failure:
            res = dict(verdict="PASS", fault_class="PASS", severity="none", root_line=None, root_cause="", evidence=[],
                       duplicate_of=None, summary="No unexpected ERROR/FATAL events.", next_steps=[], needs_human_review=False,
                       attempts=0, issues=[], fallback_used=False)
        else:
            d, issues = s["draft"] or {}, s["issues"]
            raw_lines = s["log_text"].splitlines()
            ok_ev = [e for e in d.get("evidence") or []
                     if isinstance(e.get("line"), int) and 1 <= e["line"] <= len(raw_lines)
                     and e.get("text") and _norm(e["text"]) in _norm(raw_lines[e["line"] - 1])]
            suspect_lines = {e.line_no for e in sig.suspects}
            fallback = bool(issues)
            res = dict(
                verdict="FAIL",
                fault_class=d.get("fault_class") if d.get("fault_class") in FAULT_CLASSES else "OTHER",
                severity=d.get("severity") if d.get("severity") in SEVERITIES else "medium",
                root_line=d.get("root_line") if d.get("root_line") in suspect_lines else sig.first_suspect.line_no,
                root_cause=str(d.get("root_cause", "")), evidence=ok_ev,
                duplicate_of=d.get("duplicate_of") if d.get("duplicate_of") in {c["id"] for c in s["candidates"]} else None,
                summary=str(d.get("summary", "")), next_steps=list(d.get("next_steps") or []),
                needs_human_review=fallback, attempts=s["attempts"], issues=issues, fallback_used=fallback)
            res["candidates"] = s["candidates"]
            res["evidence_claimed"] = len(d.get("evidence") or [])
            res["evidence_valid"] = len(ok_ev)
        res["run_id"] = s["run_id"]
        res["prompt_version"] = self.prompt_version
        res["backend"] = getattr(self.llm, "name", "custom")
        res["report_md"] = self._markdown(res)
        return {"result": res}

    @staticmethod
    def _markdown(r: dict) -> str:
        lines = [f"# Triage report: {r['run_id']}", f"**Verdict:** {r['verdict']}  |  **Class:** {r['fault_class']}  |  **Severity:** {r['severity']}"]
        if r["verdict"] == "FAIL":
            lines += [f"**Root line:** {r['root_line']}", f"**Duplicate of:** {r['duplicate_of'] or 'none (novel failure)'}",
                      "", f"**Root cause:** {r['root_cause']}", "", r["summary"], "", "**Evidence**"]
            lines += [f"- line {e['line']}: `{e['text']}`" for e in r["evidence"]] or ["- (none verified)"]
            lines += ["", "**Next steps**"] + [f"- {x}" for x in r["next_steps"]]
            if r["needs_human_review"]:
                lines += ["", f"> Needs human review. Verification issues: {'; '.join(r['issues'])}"]
        return "\n".join(lines)

    def r_after_signature(self, s: TriageState) -> str:
        return "retrieve" if s["signature"].has_failure else "report"

    def r_after_verify(self, s: TriageState) -> str:
        return "hypothesize" if s["issues"] and s["attempts"] < self.max_attempts else "report"

    def _build_graph(self):
        g = StateGraph(TriageState)
        for name, fn in [("parse", self.n_parse), ("signature", self.n_signature), ("retrieve", self.n_retrieve),
                         ("hypothesize", self.n_hypothesize), ("verify", self.n_verify), ("report", self.n_report)]:
            g.add_node(name, fn)
        g.add_edge(START, "parse")
        g.add_edge("parse", "signature")
        g.add_conditional_edges("signature", self.r_after_signature, {"retrieve": "retrieve", "report": "report"})
        g.add_edge("retrieve", "hypothesize")
        g.add_edge("hypothesize", "verify")
        g.add_conditional_edges("verify", self.r_after_verify, {"hypothesize": "hypothesize", "report": "report"})
        g.add_edge("report", END)
        return g.compile()

    def _run_plain(self, s: TriageState) -> TriageState:
        s.update(self.n_parse(s)); s.update(self.n_signature(s))
        if self.r_after_signature(s) == "retrieve":
            s.update(self.n_retrieve(s))
            while True:
                s.update(self.n_hypothesize(s)); s.update(self.n_verify(s))
                if self.r_after_verify(s) == "report":
                    break
        s.update(self.n_report(s))
        return s

    def run(self, log_text: str, run_id: str = "adhoc") -> dict:
        t0 = time.perf_counter()
        init: TriageState = {"log_text": log_text, "run_id": run_id}
        out = self.graph.invoke(init) if self.graph else self._run_plain(init)
        out["result"]["latency_s"] = round(time.perf_counter() - t0, 3)
        return out["result"]
