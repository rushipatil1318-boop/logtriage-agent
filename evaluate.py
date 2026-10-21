from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .agent import TriageAgent
from .generator import generate_dataset
from .kb import DefectIndex
from .llm import get_llm
from .parser import parse_log
from .signature import TemplateStats, extract_signature


def tune_threshold(dev: list[dict], stats: TemplateStats, kb: DefectIndex) -> float:
    scored = []
    for r in dev:
        sig = extract_signature(parse_log(r["text"]), stats)
        top = kb.search(sig.text, k=1)
        scored.append((top[0] if top else None, r["known_defect"]))
    best_t, best_f1 = 0.35, -1.0
    for t in np.arange(0.15, 0.80, 0.025):
        tp = sum(1 for top, truth in scored if top and top["score"] >= t and top["id"] == truth)
        fp = sum(1 for top, truth in scored if top and top["score"] >= t and top["id"] != truth)
        fn = sum(1 for top, truth in scored if truth and not (top and top["score"] >= t and top["id"] == truth))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(round(t, 3)), f1
    return best_t


def score(results: list[dict], truths: list[dict]) -> dict:
    y_true = [t["label"] for t in truths]
    y_pred = [r["fault_class"] for r in results]
    fails = [(r, t) for r, t in zip(results, truths) if t["label"] != "PASS"]
    tp = sum(1 for r, t in zip(results, truths) if t["known_defect"] and r["duplicate_of"] == t["known_defect"])
    fp = sum(1 for r, t in zip(results, truths) if r["duplicate_of"] and r["duplicate_of"] != t["known_defect"])
    fn = sum(1 for r, t in zip(results, truths) if t["known_defect"] and r["duplicate_of"] != t["known_defect"])
    claimed = sum(r.get("evidence_claimed", 0) for r in results)
    valid = sum(r.get("evidence_valid", 0) for r in results)
    prec, rec = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(results),
        "fault_class_accuracy": round(float(np.mean([a == b for a, b in zip(y_true, y_pred)])), 4),
        "fault_class_macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "root_line_accuracy": round(float(np.mean([r["root_line"] == t["root_line"] for r, t in fails])), 4),
        "severity_accuracy": round(float(np.mean([r["severity"] == t["severity"] for r, t in fails])), 4),
        "duplicate_precision": round(prec, 4), "duplicate_recall": round(rec, 4),
        "duplicate_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "citation_validity": round(valid / claimed, 4) if claimed else None,
        "retry_rate": round(float(np.mean([r["attempts"] > 1 for r in results])), 4),
        "human_review_rate": round(float(np.mean([r["needs_human_review"] for r in results])), 4),
        "mean_latency_s": round(float(np.mean([r["latency_s"] for r in results])), 3),
        "errors_by_class": dict(Counter(t["label"] for r, t in zip(results, truths) if r["fault_class"] != t["label"])),
    }


def run_config(backend: str, prompt: str, hard: bool, n: int, seed_dev: int = 1, seed_test: int = 2) -> dict:
    dev = generate_dataset(n, seed_dev, hard, prefix="dev")
    test = generate_dataset(n, seed_test, hard, prefix="test")
    stats = TemplateStats().fit([r["text"] for r in dev if r["label"] == "PASS"])
    kb = DefectIndex()
    thr = tune_threshold(dev, stats, kb)
    agent = TriageAgent(llm=get_llm(backend), kb=kb, stats=stats, prompt_version=prompt, dup_threshold=thr)
    results = [agent.run(r["text"], r["run_id"]) for r in test]
    m = score(results, test)
    m.update(backend=agent.llm.name, prompt=prompt, hard=hard, dup_threshold=thr)
    return m


COLS = ["fault_class_accuracy", "fault_class_macro_f1", "root_line_accuracy", "duplicate_f1", "citation_validity", "retry_rate", "mean_latency_s"]


def to_markdown(rows: list[dict]) -> str:
    head = "| backend | prompt | data | " + " | ".join(c.replace("_", " ") for c in COLS) + " |"
    sep = "|" + "---|" * (3 + len(COLS))
    body = ["| {} | {} | {} | {} |".format(r["backend"], r["prompt"], "hard" if r["hard"] else "easy",
                                           " | ".join("-" if r[c] is None else str(r[c]) for c in COLS)) for r in rows]
    return "\n".join([head, sep] + body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="runs per split")
    ap.add_argument("--backend", nargs="+", default=["offline"])
    ap.add_argument("--prompt", nargs="+", default=["v2"])
    ap.add_argument("--hard", action="store_true", help="only the hard (paraphrased) data; default runs easy and hard")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = []
    for backend in a.backend:
        for prompt in (["v2"] if backend == "offline" else a.prompt):
            for hard in ([True] if a.hard else [False, True]):
                print(f"running backend={backend} prompt={prompt} hard={hard} ...", flush=True)
                rows.append(run_config(backend, prompt, hard, a.n))
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    (out / "RESULTS.md").write_text(to_markdown(rows) + "\n")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
