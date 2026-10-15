import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("RESEARCHPILOT_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Retrieval Evaluation", layout="wide")
st.title("Retrieval Evaluation Dashboard")
try:
    response = requests.get(f"{API_URL}/evaluation/results", timeout=30)
    response.raise_for_status()
    rows = response.json()["results"]
except requests.RequestException as exc:
    st.error(f"Could not load results: {exc}")
    st.stop()
if not rows:
    st.info("No results yet. Add labeled questions and run the benchmark command.")
    st.stop()
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
a, b, c = st.columns(3)
a.metric("Best Recall@5", f"{df['recall_at_5'].max():.3f}")
b.metric("Best MRR@5", f"{df['mrr_at_5'].max():.3f}")
c.metric("Fastest mean latency", f"{df['latency_ms'].min():.1f} ms")
quality = df.melt(
    id_vars="method", value_vars=["recall_at_5", "mrr_at_5"], var_name="metric", value_name="score"
)
st.plotly_chart(
    px.bar(
        quality,
        x="method",
        y="score",
        color="metric",
        barmode="group",
        title="Retrieval quality",
        range_y=[0, 1],
    ),
    use_container_width=True,
)
st.plotly_chart(
    px.bar(df, x="method", y="latency_ms", title="Mean retrieval latency"), use_container_width=True
)
st.caption("Scores reflect your manually labeled benchmark; they are not generated examples.")
