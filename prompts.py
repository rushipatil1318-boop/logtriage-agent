from __future__ import annotations

import json

FAULT_CLASSES = [
    "PHY_SYNC_LOSS", "RRC_RLF", "NAS_REG_TIMEOUT", "ATTACH_REJECT_PLMN",
    "AUTH_MAC_FAILURE", "HANDOVER_FAILURE", "PDN_REJECT_NO_IP", "MODEM_WATCHDOG_RESET", "OTHER",
]
SEVERITIES = ["low", "medium", "high", "critical"]

SCHEMA = """Return ONLY a JSON object with these keys:
{"fault_class": one of %s,
 "root_cause": short string,
 "severity": one of %s,
 "root_line": integer line number of the earliest line that shows the root failure,
 "evidence": [{"line": integer, "text": "exact text copied from that log line"}],
 "duplicate_of": a defect id from candidate_defects, or null,
 "summary": 1-2 sentences for an engineer,
 "next_steps": [short strings]}""" % (FAULT_CLASSES, SEVERITIES)

SYSTEM_V1 = "You are a cellular modem test engineer. Triage the failed test run described by the evidence.\n" + SCHEMA

SYSTEM_V2 = """You are a senior cellular modem test engineer triaging a FAILED automated test run.
You get pre-filtered evidence (suspect ERROR/FATAL lines, the lines just before the first one, and candidate known defects).

Rules:
1. Failures cascade. The ROOT cause is usually the EARLIEST suspect; later errors are often symptoms
   (e.g. sync loss -> radio link failure -> registration timeout). Classify by the root, not the last error.
2. root_line must be a line number that appears in suspect_events.
3. evidence must quote text copied exactly from the given lines. Never invent lines or text.
4. Set duplicate_of only if a candidate defect clearly describes the same failure AND its distinguishing detail
   appears in the evidence; otherwise null. A high score alone is not enough.
5. If the failure fits none of the classes, use "OTHER".
6. Wording varies between firmware versions ("DL sync lost" and "Downlink synchronization lost" are the same fault).

Example (abridged):
suspect_events: [{"line": 31, "text": "Downlink sync lost, RSRP=-100dBm"}, {"line": 33, "text": "RLF declared after T310 timeout"}]
answer: {"fault_class": "PHY_SYNC_LOSS", "root_line": 31, "evidence": [{"line": 31, "text": "Downlink sync lost"}], "duplicate_of": null, ...}

""" + SCHEMA

SYSTEMS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}


def build_user_prompt(evidence: dict, feedback: list[str] | None = None) -> str:
    msg = "<evidence>\n" + json.dumps(evidence, indent=1) + "\n</evidence>"
    if feedback:
        msg += "\n\nYour previous answer had problems, fix them:\n- " + "\n- ".join(feedback)
    return msg
