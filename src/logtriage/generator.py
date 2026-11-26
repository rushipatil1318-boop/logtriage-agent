from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from .defects import DEFECT_BY_FAULT

FAULTS = {
    "PHY_SYNC_LOSS": dict(severity="high", symptom="RRC_RLF", lines=[
        ("WARN", "PHY", ["Out-of-sync indication count={n} on cell {pci}", "OOS indications reported: {n} (PCI {pci})"]),
        ("ERROR", "PHY", ["Downlink sync lost, RSRP={rsrp}dBm SINR={sinr}dB below threshold",
                          "DL synchronization lost; signal quality RSRP={rsrp}dBm SINR={sinr}dB too low"])]),
    "RRC_RLF": dict(severity="high", symptom="NAS_REG_TIMEOUT", lines=[
        ("ERROR", "RRC", ["Radio link failure detected, timer T310 expired", "RLF declared after T310 timeout"]),
        ("ERROR", "RRC", ["RRC re-establishment failed, cause=otherFailure",
                          "Connection re-establishment unsuccessful (cause otherFailure)"])]),
    "NAS_REG_TIMEOUT": dict(severity="medium", symptom=None, lines=[
        ("ERROR", "NAS", ["T3410 expired, attempt {k}/5 for attach request",
                          "Attach request attempt {k}/5 timed out waiting for response (T3410)"]),
        ("ERROR", "NAS", ["Registration failed after max retries, returning to EMM-DEREGISTERED",
                          "Giving up registration: retry limit reached, state EMM-DEREGISTERED"])]),
    "ATTACH_REJECT_PLMN": dict(severity="medium", symptom=None, lines=[
        ("ERROR", "NAS", ["Attach reject received, EMM cause #11 (PLMN not allowed)",
                          "Network rejected attach with EMM cause #11: PLMN not allowed"]),
        ("WARN", "NAS", ["Entering limited service, PLMN {plmn} barred", "Limited service entered; PLMN {plmn} is barred"])]),
    "AUTH_MAC_FAILURE": dict(severity="high", symptom=None, lines=[
        ("ERROR", "NAS", ["Authentication failure: MAC failure (EMM cause #20)",
                          "Authentication rejected by USIM, MAC failure, cause #20"]),
        ("ERROR", "USIM", ["AUTN MAC check failed for RAND {hex}", "USIM could not verify AUTN MAC for RAND {hex}"])]),
    "HANDOVER_FAILURE": dict(severity="medium", symptom=None, lines=[
        ("ERROR", "RRC", ["Handover command for target cell {pci} but random access to target failed",
                          "Random access toward target PCI {pci} failed during handover"]),
        ("ERROR", "RRC", ["Handover failure, reverting to source cell configuration",
                          "HO failed; falling back to source cell config"])]),
    "PDN_REJECT_NO_IP": dict(severity="medium", symptom=None, lines=[
        ("ERROR", "ESM", ["PDN connectivity reject, ESM cause #26 insufficient resources",
                          "ESM cause #26 received: PDN connectivity request rejected (insufficient resources)"]),
        ("WARN", "IP", ["No IPv4 address assigned for APN {apn}", "APN {apn}: IPv4 address missing after bearer setup"])]),
    "MODEM_WATCHDOG_RESET": dict(severity="critical", symptom=None, lines=[
        ("FATAL", "FW", ["Watchdog timeout in task {task}, no heartbeat for {ms} ms",
                         "Task {task} unresponsive, watchdog expired after {ms} ms"]),
        ("FATAL", "FW", ["Assertion failed at {file}:{line}, triggering modem subsystem restart",
                         "ASSERT {file}:{line} hit, modem subsystem restarting"])]),
}
FAULT_CLASSES = list(FAULTS)

NOISE = [
    ("INFO", "NAS", ["UE state: {state}"]),
    ("INFO", "RRC", ["Measurement report sent, serving RSRP={rsrp}dBm neighbor PCI={pci}"]),
    ("INFO", "PHY", ["CQI report={cqi} RI={ri}"]),
    ("INFO", "RRC", ["RRC connection reconfiguration complete"]),
    ("INFO", "IP", ["Default bearer active, IPv4 10.{k}.{n}.{ms}"]),
    ("INFO", "PHY", ["Cell {pci} synchronized, PBCH decoded"]),
    ("WARN", "FW", ["Thermal throttling level {k} active"]),
    ("WARN", "IP", ["DNS response slow: {ms} ms"]),
    ("WARN", "TEST", ["Test harness heartbeat delayed {ms} ms"]),
    ("WARN", "PHY", ["Out-of-sync indication count=1 on cell {pci}, in-sync restored"]),
]
RED_HERRINGS = [
    ("ERROR", "NVM", ["NVRAM read retry succeeded after {k} attempt(s)"]),
    ("ERROR", "LOG", ["Log buffer overflow, {k} lines dropped"]),
]
STATES = ["EMM-REGISTERED", "EMM-REGISTERED-INITIATED", "EMM-NULL", "ECM-CONNECTED"]
TASKS = ["rrc_ctrl", "nas_main", "phy_sched", "ip_stack"]
FILES = ["rrc_ctrl.c", "nas_fsm.c", "phy_mac.c"]
APNS = ["internet", "ims", "test.apn"]


def _fill(rng: random.Random, template: str) -> str:
    return template.format(
        n=rng.randint(1, 9), pci=rng.randint(1, 503), rsrp=-rng.randint(70, 125), sinr=round(rng.uniform(-8, 3), 1),
        k=rng.randint(1, 5), plmn=f"262{rng.randint(1, 9):02d}", hex="".join(rng.choice("0123456789abcdef") for _ in range(16)),
        apn=rng.choice(APNS), task=rng.choice(TASKS), ms=rng.randint(120, 9000), file=rng.choice(FILES),
        line=rng.randint(100, 4000), cqi=rng.randint(3, 15), ri=rng.randint(1, 2), state=rng.choice(STATES),
        band=rng.choice([1, 3, 7, 20, 28]),
    )


def _pick(rng, variants, hard):
    return variants[1] if (hard and len(variants) > 1 and rng.random() < 0.5) else variants[0]


def _chain(rng, fault, hard, with_marker):
    out = []
    for level, comp, variants in FAULTS[fault]["lines"]:
        out.append(dict(level=level, comp=comp, msg=_fill(rng, _pick(rng, variants, hard)), root=False))
    first = next(i for i, e in enumerate(out) if e["level"] in ("ERROR", "FATAL"))
    out[first]["root"] = True
    if with_marker:
        level, comp, text = DEFECT_BY_FAULT[fault]["marker"]
        out.insert(first + 1, dict(level=level, comp=comp, msg=_fill(rng, text), root=False))
    return out


def generate_run(rng: random.Random, run_id: str, hard: bool = False, p_pass: float = 0.25) -> dict:
    label = "PASS" if rng.random() < p_pass else rng.choice(FAULT_CLASSES)
    known = None
    body = []
    for _ in range(rng.randint(26, 46)):
        level, comp, variants = rng.choice(NOISE)
        body.append(dict(level=level, comp=comp, msg=_fill(rng, variants[0]), root=False))
    for _ in range(rng.randint(0, 3 if hard else 2)):
        level, comp, variants = rng.choice(RED_HERRINGS)
        body.insert(rng.randint(0, len(body)), dict(level=level, comp=comp, msg=_fill(rng, variants[0]), root=False))

    cascade = []
    if label != "PASS":
        known = DEFECT_BY_FAULT[label]["id"] if rng.random() < 0.5 else None
        chain = _chain(rng, label, hard, with_marker=known is not None)
        nxt, p = FAULTS[label]["symptom"], 0.75
        while nxt and rng.random() < p:
            cascade.append(nxt)
            chain += _chain(rng, nxt, hard, with_marker=False)
            for e in chain[-len(FAULTS[nxt]["lines"]):]:
                e["root"] = False
            nxt, p = FAULTS[nxt]["symptom"], 0.6
        pos = rng.randint(int(len(body) * 0.4), int(len(body) * 0.7))
        body[pos:pos] = chain

    head = [dict(level="INFO", comp="TEST", msg=f"Test case TC-{rng.randint(1000, 9999)} started", root=False),
            dict(level="INFO", comp="NAS", msg=f"UE powered on, RAT=LTE band {rng.choice([1, 3, 7, 20, 28])}", root=False)]
    tail = [dict(level="INFO" if label == "PASS" else "ERROR", comp="TEST",
                 msg=f"Test case result: {'PASS' if label == 'PASS' else 'FAIL'}", root=False)]
    entries = head + body + tail

    t = datetime(2026, rng.randint(1, 9), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    lines, root_line = [], None
    for i, e in enumerate(entries, start=1):
        t += timedelta(milliseconds=rng.randint(20, 900))
        lines.append(f"{t:%Y-%m-%d %H:%M:%S}.{t.microsecond // 1000:03d} [{e['level']}] [{e['comp']}] {e['msg']}")
        if e["root"] and root_line is None:
            root_line = i
    return dict(run_id=run_id, label=label, severity=FAULTS[label]["severity"] if label != "PASS" else "none",
                root_line=root_line, known_defect=known, cascade=cascade, text="\n".join(lines) + "\n")


def generate_dataset(n: int, seed: int, hard: bool = False, prefix: str = "run") -> list[dict]:
    rng = random.Random(seed)
    return [generate_run(rng, f"{prefix}-{i:04d}", hard=hard) for i in range(n)]


def write_dataset(runs: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "w") as fh:
        for r in runs:
            (out / "logs" / f"{r['run_id']}.log").write_text(r["text"])
            fh.write(json.dumps({k: v for k, v in r.items() if k != "text"}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic modem-style test logs")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hard", action="store_true", help="paraphrased wording + more red herrings")
    ap.add_argument("--out", default="data/generated")
    a = ap.parse_args()
    write_dataset(generate_dataset(a.n, a.seed, a.hard), a.out)
    print(f"wrote {a.n} runs to {a.out}")
