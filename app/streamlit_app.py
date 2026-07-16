#!/usr/bin/env python3
"""Streamlit dashboard for the drift + optimization system.

Run with:  streamlit run app/streamlit_app.py

The app lets you pick a drift scenario, runs the end-to-end pipeline, and shows
the drift verdict, per-feature statistics, the remediation decision, and the
monitoring-history trend chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from ml_drift.config import load_config  # noqa: E402
from ml_drift.pipeline.orchestrator import SCENARIOS, run_pipeline  # noqa: E402
from ml_drift.monitoring.dashboard import load_history  # noqa: E402

st.set_page_config(page_title="ML Drift + Optimization", layout="wide")
st.title("🛰️ ML System Drift Detection & Post-Detection Optimization")
st.caption("Evidently AI reports + native drift core + automated remediation")

cfg = load_config()

with st.sidebar:
    st.header("Run controls")
    scenario = st.selectbox("Drift scenario", list(SCENARIOS), index=1)
    strategy = st.selectbox(
        "Remediation strategy",
        ["auto", "retrain", "reweight", "feature_stabilize", "none"],
        index=0,
    )
    run = st.button("Run pipeline", type="primary")

if run:
    # Temporarily override the configured strategy for this run.
    cfg._data["optimization"]["strategy"] = strategy  # type: ignore[attr-defined]
    with st.spinner("Running pipeline..."):
        result = run_pipeline(cfg, scenario=scenario, log_monitoring=True)
    s = result.summary()
    d, o = s["drift"], s["optimization"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset drift", str(d["dataset_drift"]))
    c2.metric("Drift share", f"{d['drift_share']:.2f}")
    c3.metric("Strategy", o["strategy"])
    c4.metric("Model promoted", "Yes" if o["promoted"] else "No")

    st.subheader("Post-detection optimization")
    st.write(o["reason"])
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(f"Champion {o['primary_metric']}", f"{o['champion_metric']:.4f}")
    if o["challenger_metric"] is not None:
        mc2.metric(f"Challenger {o['primary_metric']}", f"{o['challenger_metric']:.4f}",
                   delta=f"{o['improvement']:+.4f}")
    mc3.metric("Champion drop on current", f"{o.get('performance_drop', 0):.4f}")

    st.subheader("Per-feature drift")
    st.dataframe(pd.DataFrame(d["features"]), use_container_width=True)

    st.subheader("Evidently reports")
    if s["evidently_available"]:
        for name, path in s["reports"].items():
            if path:
                st.write(f"**{name}** — `{path}`")
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        st.components.v1.html(fh.read(), height=500, scrolling=True)
                except Exception:
                    st.info(f"Open {path} in a browser to view the report.")
    else:
        st.info("Evidently is not installed — native drift decision shown above.")

st.subheader("📈 Monitoring history")
history_file = cfg.root / cfg["monitoring"]["history_file"]
history = load_history(history_file)
if history:
    hist_df = pd.DataFrame(history)
    st.line_chart(hist_df.set_index(hist_df.index)[["drift_share"]])
    st.dataframe(hist_df, use_container_width=True)
else:
    st.write("No runs logged yet — run the pipeline to populate history.")
