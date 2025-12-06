import json
from typing import List

import pandas as pd
import streamlit as st

from memleak import MembershipTester


st.set_page_config(page_title="MemLeak", layout="wide")
st.title("MemLeak — Membership Inference Detector")
st.markdown("Run loss, confidence, and metric-based membership inference attacks on HuggingFace models.")


@st.cache_data(show_spinner=False)
def parse_uploaded_text(file) -> List[str]:
    content = file.read().decode("utf-8")
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return [line.strip() for line in content.splitlines() if line.strip()]


with st.sidebar:
    model_id = st.text_input("Model id or path", value="distilbert-base-uncased")
    max_length = st.number_input("Max length", min_value=32, max_value=512, value=128, step=8)
    batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=4, step=1)
    run_btn = st.button("Run attacks", type="primary")

st.subheader("Upload data")
col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader("Train (JSON list or newline text)", key="train")
    train_text = st.text_area("Train samples (fallback)", height=120, placeholder="Sample 1\nSample 2")
with col2:
    test_file = st.file_uploader("Test/holdout (JSON list or newline text)", key="test")
    test_text = st.text_area("Test samples (fallback)", height=120, placeholder="Holdout 1\nHoldout 2")


def gather_samples(uploaded, textarea) -> List[str]:
    if uploaded:
        return parse_uploaded_text(uploaded)
    if textarea.strip():
        return [t.strip() for t in textarea.splitlines() if t.strip()]
    return []


train_samples = gather_samples(train_file, train_text) or [
    "The cat sat on the mat.",
    "Healthcare data is sensitive.",
    "Alice loves privacy-preserving ML.",
    "Bob plays guitar on weekends.",
]
test_samples = gather_samples(test_file, test_text) or [
    "A new sample unseen in training.",
    "The dog chased the ball.",
    "Memorization should be low here.",
]

if run_btn:
    with st.spinner("Loading model and running attacks..."):
        tester = MembershipTester(model=model_id, max_length=max_length, batch_size=batch_size)
        report = tester.run_attacks(train_samples, test_samples)
    st.success(f"Completed. Privacy risk score: {report.risk_score}/100")
    figs = report.visualize()
    st.plotly_chart(figs["dashboard"], use_container_width=True)
    st.plotly_chart(figs["probabilities"], use_container_width=True)
    st.plotly_chart(figs["heatmap"], use_container_width=True)
    st.plotly_chart(figs["curves"], use_container_width=True)

    st.subheader("Summary")
    st.dataframe(report.summary)

    st.subheader("Recommendations")
    for tip in report.recommendations():
        st.markdown(f"- {tip}")

