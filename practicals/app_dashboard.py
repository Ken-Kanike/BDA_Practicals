"""Streamlit live dashboard for Experiment 8.

Run with:
    python -m streamlit run practicals/app_dashboard.py
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.config import OUTPUTS_DIR
from practicals.exp08_dashboard import load_live_telemetry

st.set_page_config(
    page_title="Urban Mobility Live Streaming Dashboard",
    page_icon=":vertical_traffic_light:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .alert-critical {
        background-color: #fdecea; border-left: 5px solid #c0392b;
        padding: 8px 12px; margin-bottom: 6px; border-radius: 4px;
    }
    .alert-warning {
        background-color: #fff4e5; border-left: 5px solid #f39c12;
        padding: 8px 12px; margin-bottom: 6px; border-radius: 4px;
    }
    .kpi-label { color: #5a6b7b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    csv_path = OUTPUTS_DIR / "exp08" / "live_telemetry_enriched.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["event_time"] = pd.to_datetime(df["event_time"])
        return df
    df, _ = load_live_telemetry()
    return df


def _rolling_plot(data: pd.DataFrame) -> go.Figure:
    traj = (
        data.set_index("event_time")
        .groupby("mode")[["delay_minutes", "speed_mph"]]
        .resample("1min")
        .mean()
        .reset_index()
    )
    fig = go.Figure()
    for mode, color in (("Bus", "#1f77b4"), ("Train", "#d62728")):
        subset = traj[traj["mode"] == mode]
        fig.add_trace(
            go.Scatter(
                x=subset["event_time"],
                y=subset["delay_minutes"],
                mode="lines+markers",
                name=f"{mode} delay",
                line=dict(color=color, width=2),
            )
        )
    for mode, color in (("Bus", "#17becf"), ("Train", "#ff7f0e")):
        subset = traj[traj["mode"] == mode]
        fig.add_trace(
            go.Scatter(
                x=subset["event_time"],
                y=subset["speed_mph"],
                mode="lines",
                name=f"{mode} speed",
                line=dict(color=color, width=1, dash="dot"),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Rolling delay and speed trajectory",
        yaxis=dict(title="Delay (minutes)"),
        yaxis2=dict(title="Speed (mph)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=60, b=10),
        height=380,
    )
    return fig


def main() -> None:
    data = _load_data()

    st.sidebar.title("Operational controls")
    modality_filter = st.sidebar.multiselect(
        "Transit modality", ["Bus", "Train"], default=["Bus", "Train"]
    )
    severe_threshold = st.sidebar.slider(
        "Severe delay alert threshold (minutes)", 10.0, 30.0, 15.0, 1.0
    )
    critical_threshold = st.sidebar.slider(
        "Critical delay threshold (minutes)", 20.0, 45.0, 25.0, 1.0
    )
    auto_refresh = st.sidebar.checkbox("Live auto-refresh (3s interval)", value=False)

    filtered = data[data["mode"].isin(modality_filter)] if modality_filter else data
    filtered = filtered.copy()
    filtered["alert_status"] = pd.cut(
        filtered["delay_minutes"],
        bins=[-1, severe_threshold, critical_threshold, float("inf")],
        labels=["NORMAL", "WARNING", "CRITICAL"],
    )

    st.title("Urban Mobility Live Streaming Dashboard")
    st.caption(
        f"Monitoring {len(filtered):,} live telemetry records across "
        f"{filtered['location'].nunique()} corridors. "
        f"Thresholds: Warning >= {severe_threshold:.1f} min, Critical >= {critical_threshold:.1f} min."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Vehicles", f"{len(filtered):,}")
    col2.metric("Average delay", f"{filtered['delay_minutes'].mean():.2f} min")
    col3.metric("Passenger-hours lost", f"{filtered['passenger_hours_lost'].sum():,.0f}")
    col4.metric("Mean speed", f"{filtered['speed_mph'].mean():.1f} mph")
    col5.metric("Sensors OK", f"{100 * (filtered['sensor_health'] == 'OK').mean():.1f}%")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(_rolling_plot(filtered), width="stretch")
    with right:
        status_counts = (
            filtered["alert_status"].value_counts().reindex(["CRITICAL", "WARNING", "NORMAL"]).fillna(0)
        )
        donut = go.Figure(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.55,
                marker=dict(colors=["#c0392b", "#f39c12", "#27ae60"]),
            )
        )
        donut.update_layout(
            title="Severity mix", height=380, margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(donut, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        corridor = (
            filtered.groupby(["mode", "location"])["passenger_hours_lost"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        corridor["label"] = corridor["mode"] + ": " + corridor["location"]
        bar = go.Figure(
            go.Bar(
                x=corridor["passenger_hours_lost"],
                y=corridor["label"],
                orientation="h",
                marker_color="#4c72b0",
            )
        )
        bar.update_layout(
            title="Corridor passenger-hour burden",
            xaxis_title="Passenger-hours",
            yaxis=dict(autorange="reversed"),
            height=430,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(bar, width="stretch")
    with right:
        st.subheader("Automated incident feed")
        alerts = filtered[filtered["alert_status"] != "NORMAL"].sort_values(
            "delay_minutes", ascending=False
        )
        if alerts.empty:
            st.success("No active alerts for the selected thresholds.")
        for _, alert in alerts.head(8).iterrows():
            css = "alert-critical" if alert["alert_status"] == "CRITICAL" else "alert-warning"
            st.markdown(
                f'<div class="{css}"><strong>[{alert["mode"].upper()} {alert["alert_status"]}]</strong> '
                f'{alert["location"]} &mdash; {alert["delay_minutes"]:.1f} min delay, '
                f'{alert["passenger_hours_lost"]:.1f} pax-hrs, {alert["reason"]}</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        "Data source: Experiment 7 Structured Streaming sink. "
        "The same feed powers the static dashboard and alert diagnostics in outputs/exp08/."
    )

    if auto_refresh:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
