import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st

from logtriage import tools
from logtriage.generator import generate_dataset
from logtriage.parser import parse_log

st.set_page_config(page_title="LogTriage Agent", layout="wide")
st.title("LogTriage Agent")
st.caption("LLM-assisted log analysis and defect triage for modem-style test runs (synthetic data).")

with st.sidebar:
    backend = st.selectbox("Backend", ["offline", "anthropic", "openai"],
                           help="offline = keyword-rule baseline; anthropic needs ANTHROPIC_API_KEY; openai = Ollama/OpenAI-compatible")
    prompt = st.selectbox("Prompt version", ["v2", "v1"])
    hard = st.checkbox("Hard mode samples (paraphrased wording)")
    seed = st.number_input("Sample seed", 0, 9999, 1)

samples = generate_dataset(12, int(seed), hard)
choice = st.selectbox("Sample run", [f"{r['run_id']}  (truth: {r['label']})" for r in samples])
sample = samples[[f"{r['run_id']}  (truth: {r['label']})" for r in samples].index(choice)]
text = st.text_area("Log (paste your own or edit the sample)", sample["text"], height=260)

if st.button("Triage", type="primary"):
    try:
        res = tools.triage_log(text, backend, prompt)
    except Exception as exc:
        st.error(f"Triage failed: {exc!r}")
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", res["verdict"])
    c2.metric("Fault class", res["fault_class"])
    c3.metric("Severity", res["severity"])
    c4.metric("Duplicate of", res["duplicate_of"] or "novel")
    if res["needs_human_review"]:
        st.warning("Verification failed after retries; showing safe fallback values. Needs human review.")
    st.markdown(res["report_md"])
    if res["verdict"] == "FAIL":
        st.subheader("Closest known defects")
        st.dataframe(pd.DataFrame(res["candidates"]), width="stretch")
        st.subheader("Log around the root line")
        st.code(tools.get_log_context(text, res["root_line"], 6), language="text")
    with st.expander("Parsed events"):
        st.dataframe(pd.DataFrame([e.__dict__ for e in parse_log(text)]), width="stretch")
    st.caption(f"backend={res['backend']}  attempts={res['attempts']}  latency={res['latency_s']}s")
