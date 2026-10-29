"""Final Streamlit Smart City showcase integrating Experiments 1-10.

Run with:
    python -m streamlit run practicals/app_showcase.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.config import BOROUGHS, BUS_REASONS, OUTPUTS_DIR, TRAIN_LINES, TRAIN_REASONS

st.set_page_config(
    page_title="Smart City Mobility Showcase",
    page_icon=":cityscape:",
    layout="wide",
)


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def _overview_tab() -> None:
    scorecard = _csv(OUTPUTS_DIR / "exp06" / "scorecard.csv")
    metrics = _json(OUTPUTS_DIR / "exp10" / "ml_metrics.json")

    st.subheader("Executive overview")
    cols = st.columns(5)
    cols[0].metric("Total trips analysed", "100,000")
    cols[1].metric("Bus / Train", "50k / 50k")
    cols[2].metric("Best regression R2", f"{metrics.get('regression', [{}])[0].get('r2', 0):.3f}")
    cols[3].metric("Severe-delay AUC", f"{metrics.get('classification', {}).get('roc_auc', 0):.3f}")
    cols[4].metric("Indexed DB records", f"{_json(OUTPUTS_DIR / 'exp09' / 'nosql_lake_metrics.json').get('documents', 0):,}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Bus vs Train reliability scorecard**")
        st.table(scorecard)
    with right:
        st.markdown("**Pipeline coverage**")
        st.markdown(
            """
            - **Ingestion:** CSV, MapReduce, partitioned Parquet, MongoDB
            - **Processing:** Spark SQL, sliding-window streaming
            - **Serving:** Streamlit dashboards and alert feeds
            - **Data lake:** 4 compound indexes on 100,000 documents
            - **ML:** three regressors + balanced severity classifier
            """
        )

    _show_image(
        OUTPUTS_DIR / "exp10" / "exp10_showcase_architecture.png",
        "End-to-end architecture",
    )
    _show_image(
        OUTPUTS_DIR / "exp06" / "exp6_executive_dashboard.png",
        "Historical batch analytics dashboard",
    )


def _batch_tab() -> None:
    metrics = _json(OUTPUTS_DIR / "exp05" / "batch_metrics.json")
    st.subheader("Batch analytics and storage performance")
    cols = st.columns(5)
    cols[0].metric("Unified rows", f"{metrics.get('unified_rows', 0):,}")
    cols[1].metric("CSV latency", f"{metrics.get('benchmark_ms', {}).get('raw_csv', 0):,.0f} ms")
    cols[2].metric("Parquet latency", f"{metrics.get('benchmark_ms', {}).get('parquet', 0):,.0f} ms")
    cols[3].metric("Cached SQL", f"{metrics.get('benchmark_ms', {}).get('cached', 0):,.0f} ms")
    cols[4].metric("Storage saved", f"{metrics.get('storage_reduction_pct', 0):.1f}%")

    benchmark = _csv(OUTPUTS_DIR / "exp05" / "benchmark_results.csv")
    st.markdown("**Benchmark results**")
    st.table(benchmark)
    _show_image(OUTPUTS_DIR / "exp05" / "exp5_query_benchmark.png", "Parquet and cache speedups")
    _show_image(OUTPUTS_DIR / "exp06" / "exp6_time_series_trends.png", "Delay trends over time")


def _streaming_tab() -> None:
    metrics = _json(OUTPUTS_DIR / "exp08" / "dashboard_metrics.json")
    data = _csv(OUTPUTS_DIR / "exp08" / "live_telemetry_enriched.csv")
    st.subheader("Streaming telemetry and threshold alerts")
    cols = st.columns(5)
    cols[0].metric("Live vehicles", f"{metrics.get('total_vehicles', 0):,}")
    cols[1].metric("Average delay", f"{metrics.get('avg_delay', 0):.2f} min")
    cols[2].metric("Passenger-hours lost", f"{metrics.get('total_passenger_hours', 0):,.0f}")
    cols[3].metric("Critical alerts", f"{metrics.get('critical', 0):,}")
    cols[4].metric("Sensors OK", f"{metrics.get('sensor_health_pct', 0):.1f}%")

    if not data.empty:
        data["event_time"] = pd.to_datetime(data["event_time"])
        traj = (
            data.set_index("event_time")
            .groupby("mode")["delay_minutes"]
            .resample("1min")
            .mean()
            .reset_index()
        )
        fig = go.Figure()
        for mode, color in (("Bus", "#1f77b4"), ("Train", "#d62728")):
            subset = traj[traj["mode"] == mode]
            fig.add_trace(go.Scatter(x=subset["event_time"], y=subset["delay_minutes"], mode="lines+markers", name=mode, line=dict(color=color)))
        fig.update_layout(
            title="Live delay trajectory",
            yaxis_title="Delay (minutes)",
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(fig, width="stretch")
        with right:
            status = data["alert_status"].value_counts().reindex(["CRITICAL", "WARNING", "NORMAL"]).fillna(0)
            donut = go.Figure(
                go.Pie(
                    labels=status.index,
                    values=status.values,
                    hole=0.55,
                    marker=dict(colors=["#c0392b", "#f39c12", "#27ae60"]),
                )
            )
            donut.update_layout(title="Alert mix", height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(donut, width="stretch")

        st.markdown("**Highest-delay incidents**")
        st.table(
            data.sort_values("delay_minutes", ascending=False)[
                ["mode", "location", "delay_minutes", "speed_mph", "passenger_hours_lost", "alert_status", "reason"]
            ].head(12)
        )

    _show_image(OUTPUTS_DIR / "exp08" / "exp8_static_dashboard.png", "Static streaming dashboard snapshot")


def _nosql_tab() -> None:
    metrics = _json(OUTPUTS_DIR / "exp09" / "nosql_lake_metrics.json")
    st.subheader("Indexed NoSQL data lake")
    cols = st.columns(5)
    cols[0].metric("Documents", f"{metrics.get('documents', 0):,}")
    cols[1].metric("Indexes", f"{metrics.get('index_count', 0)}")
    cols[2].metric("Ingest rate", f"{metrics.get('docs_per_second', 0):,.0f} docs/s")
    cols[3].metric("Data size", f"{metrics.get('data_size_mb', 0):.2f} MB")
    cols[4].metric("Index size", f"{metrics.get('index_size_mb', 0):.2f} MB")

    benchmark = _csv(OUTPUTS_DIR / "exp09" / "query_benchmark.csv")
    st.markdown("**Query optimizer benchmark**")
    st.table(benchmark)

    left, right = st.columns([2, 1])
    with left:
        _show_image(OUTPUTS_DIR / "exp09" / "exp9_index_benchmark.png", "Index impact on latency and work")
    with right:
        st.markdown("**Index catalog**")
        st.table(_csv(OUTPUTS_DIR / "exp09" / "index_catalog.csv"))
        st.markdown("**Top corridors by impact**")
        st.table(_csv(OUTPUTS_DIR / "exp09" / "corridor_aggregation.csv").head(8))


def _ml_tab() -> None:
    metrics = _json(OUTPUTS_DIR / "exp10" / "ml_metrics.json")
    regression = pd.DataFrame(metrics.get("regression", []))
    classification = metrics.get("classification", {})

    st.subheader("Machine learning evaluation")
    left, right = st.columns(2)
    with left:
        st.markdown("**Regression: predicting delay minutes**")
        st.table(regression)
    with right:
        st.markdown("**Classification: severe delay (>= 15 minutes)**")
        st.table(
            pd.DataFrame(
                [
                    {"metric": "ROC-AUC", "value": classification.get("roc_auc")},
                    {"metric": "Accuracy", "value": classification.get("accuracy")},
                    {"metric": "Precision (Severe)", "value": classification.get("precision_severe")},
                    {"metric": "Recall (Severe)", "value": classification.get("recall_severe")},
                    {"metric": "F1 (Severe)", "value": classification.get("f1_severe")},
                    {"metric": "Decision threshold", "value": classification.get("threshold")},
                ]
            )
        )

    _show_image(OUTPUTS_DIR / "exp10" / "exp10_model_evaluation.png", "Evaluation curves and confusion matrix")

    st.subheader("Live predictor")
    model_dir = OUTPUTS_DIR / "exp10" / "models"
    reg_path = model_dir / "delay_regressor.joblib"
    clf_path = model_dir / "severity_classifier.joblib"
    meta_path = model_dir / "model_features.json"
    if not (reg_path.exists() and clf_path.exists() and meta_path.exists()):
        st.warning("Model artifacts are missing. Run `python -m practicals.exp10_ml_showcase` first.")
        return

    regressor = joblib.load(reg_path)
    classifier = joblib.load(clf_path)
    meta = _json(meta_path)

    mode = st.selectbox("Mode", ["Bus", "Train"])
    if mode == "Bus":
        locations = BOROUGHS
        reasons = [reason for reason, _ in BUS_REASONS]
        default_passengers = 30
    else:
        locations = TRAIN_LINES
        reasons = [reason for reason, _ in TRAIN_REASONS]
        default_passengers = 140

    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox("Location / line", locations, key=f"location_{mode}")
        reason = st.selectbox("Delay reason", reasons, key=f"reason_{mode}")
    with col2:
        hour = st.slider("Scheduled hour", 0, 23, 8, key=f"hour_{mode}")
        month = st.slider("Month", 1, 12, 6, key=f"month_{mode}")
    with col3:
        day_label = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0,
            key=f"day_{mode}",
        )
        passengers = st.slider(
            "Passengers onboard",
            1 if mode == "Bus" else 40,
            60 if mode == "Bus" else 250,
            default_passengers,
            key=f"passengers_{mode}",
        )

    day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_label)
    row = pd.DataFrame(
        [
            {
                "Hour": hour,
                "Month": month,
                "DayOfWeek": day_of_week,
                "IsWeekend": int(day_of_week >= 5),
                "IsRushHour": int(hour in [6, 7, 8, 9, 16, 17, 18, 19]),
                "Mode": mode,
                "Location": location,
                "Reason": reason,
                "Passengers": passengers,
            }
        ]
    )
    predicted_delay = float(regressor.predict(row)[0])
    probability = float(classifier.predict_proba(row)[0, 1])
    threshold = float(meta.get("classifier_decision_threshold", 0.5))
    status = "SEVERE DELAY RISK" if probability >= threshold else "NORMAL RISK"

    out1, out2, out3 = st.columns(3)
    out1.metric("Predicted delay", f"{max(predicted_delay, 0):.2f} min")
    out2.metric("Severe-delay probability", f"{100 * probability:.1f}%")
    out3.metric("Decision", status)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * probability,
            title={"text": "Severe-delay risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c0392b" if probability >= threshold else "#2ca02c"},
                "threshold": {"line": {"color": "#1f2937", "width": 3}, "thickness": 0.8, "value": 100 * threshold},
            },
        )
    )
    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(gauge, width="stretch")
    st.caption(
        f"The classifier uses a decision threshold of {100 * threshold:.0f}% (F1-tuned on a validation split). "
        "Regression R2 is modest because the synthetic dataset deliberately contains weak predictive signal."
    )


def main() -> None:
    st.title("Smart City Mobility Intelligence - Final Showcase")
    st.caption(
        "Experiments 1-10: EDA | MapReduce | partitioned storage | Spark SQL | streaming | dashboards | "
        "indexed NoSQL | machine learning"
    )
    tabs = st.tabs(["Overview", "Batch analytics", "Streaming alerts", "NoSQL lake", "ML lab"])
    with tabs[0]:
        _overview_tab()
    with tabs[1]:
        _batch_tab()
    with tabs[2]:
        _streaming_tab()
    with tabs[3]:
        _nosql_tab()
    with tabs[4]:
        _ml_tab()


if __name__ == "__main__":
    main()
