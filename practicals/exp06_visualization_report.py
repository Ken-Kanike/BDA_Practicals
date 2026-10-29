"""Experiment 6 - Batch analytics visualization and executive reporting."""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from common.config import output_dir, rel
from common.data_gen import ensure_data
from common.spark import get_spark
from common.transit import build_unified

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _load_master(spark, out):
    master_path = out.parent / "exp05" / "parquet_benchmarks" / "unified_transit_master.parquet"
    if master_path.exists():
        return spark.read.parquet(str(master_path)), master_path, True
    unified = build_unified(spark)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    unified.write.mode("overwrite").partitionBy("Mode", "Year").parquet(str(master_path))
    return spark.read.parquet(str(master_path)), master_path, False


def _plot_trends(daily: pd.DataFrame, monthly: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    colors = {"Bus": "#1f77b4", "Train": "#d62728"}
    for mode in ("Bus", "Train"):
        subset = daily[daily["Mode"] == mode].sort_values("Date")
        axes[0].plot(subset["Date"], subset["Daily_Avg_Delay"], color=colors[mode], alpha=0.28, linewidth=0.8)
        axes[0].plot(
            subset["Date"],
            subset["Rolling_14Day"],
            color=colors[mode],
            linewidth=2.0,
            label=f"{mode} (14-day rolling)",
        )
    axes[0].set_title("Daily average delay with 14-day rolling average (2023-2024)")
    axes[0].set_ylabel("Delay (minutes)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for mode in ("Bus", "Train"):
        axes[1].plot(monthly.index, monthly[mode], marker="o", color=colors[mode], label=mode)
    axes[1].set_title("Monthly cumulative passenger-hours lost")
    axes[1].set_ylabel("Passenger-hours")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dow_and_heatmaps(df: pd.DataFrame, path_dow, path_heat) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    dow = (
        df.groupby(["DayName", "Mode"])["Delay_Minutes"]
        .mean()
        .unstack()
        .reindex(DOW_ORDER)
    )
    dow.plot(kind="bar", ax=ax, color={"Bus": "#1f77b4", "Train": "#d62728"})
    ax.set_title("Day-of-week congestion profile")
    ax.set_ylabel("Mean delay (minutes)")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_dow, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mode, cmap in ((axes[0], "Bus", "Blues"), (axes[1], "Train", "Reds")):
        subset = df[df["Mode"] == mode]
        pivot = subset.pivot_table(
            index="DayName", columns="Hour", values="Delay_Minutes", aggfunc="mean"
        ).reindex(DOW_ORDER)
        sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.2, cbar_kws={"label": "Mean delay (min)"})
        ax.set_title(f"{mode} - day-of-week x hour intensity")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path_heat, dpi=150)
    plt.close(fig)


def _plot_root_cause(df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, mode, cmap in ((axes[0], "Bus", "YlOrRd"), (axes[1], "Train", "YlGnBu")):
        subset = df[df["Mode"] == mode]
        top_locs = (
            subset.groupby("Location")["Passenger_Delay_Impact_Hrs"].sum().nlargest(8).index
        )
        top_reasons = (
            subset.groupby("Reason")["Passenger_Delay_Impact_Hrs"].sum().nlargest(7).index
        )
        pivot = subset[subset["Location"].isin(top_locs) & subset["Reason"].isin(top_reasons)].pivot_table(
            index="Location", columns="Reason", values="Passenger_Delay_Impact_Hrs", aggfunc="sum"
        ).fillna(0)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".0f",
            linewidths=0.2,
            cbar_kws={"label": "Passenger-hours lost"},
        )
        ax.set_title(f"{mode} - location x reason root-cause burden")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dashboard(df: pd.DataFrame, scorecard: pd.DataFrame, severity: pd.DataFrame, path) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 2.2, 2.2], hspace=0.5, wspace=0.3)

    kpis = [
        ("Total trips", f"{len(df):,}"),
        ("Mean delay", f"{df['Delay_Minutes'].mean():.2f} min"),
        ("Passenger-hours lost", f"{df['Passenger_Delay_Impact_Hrs'].sum():,.0f}"),
        ("Severe (>20 min)", f"{100 * (df['Delay_Minutes'] > 20).mean():.1f}%"),
    ]
    for i, (label, value) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.84, facecolor="#eef4fb", edgecolor="#9bb7d4"))
        ax.text(0.5, 0.68, label, ha="center", va="center", fontsize=11, color="#1f4e79")
        ax.text(0.5, 0.32, value, ha="center", va="center", fontsize=18, fontweight="bold", color="#10334f")

    ax1 = fig.add_subplot(gs[1, :2])
    severity.pivot(index="Mode", columns="Severity", values="Count").plot(
        kind="bar", ax=ax1, color=["#8c2d04", "#d94801", "#fdae6b", "#74c476"]
    )
    ax1.set_title("Severity-tier incident counts")
    ax1.set_ylabel("Trips")
    ax1.tick_params(axis="x", rotation=0)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 2:])
    top_corridors = (
        df.groupby(["Mode", "Location"])["Passenger_Delay_Impact_Hrs"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )
    labels = [f"{r.Mode}: {r.Location}" for r in top_corridors.itertuples()]
    ax2.barh(labels[::-1], top_corridors["Passenger_Delay_Impact_Hrs"][::-1], color="#4c72b0")
    ax2.set_title("Top corridors by passenger-hours lost")
    ax2.set_xlabel("Passenger-hours")
    ax2.grid(axis="x", alpha=0.3)

    ax3 = fig.add_subplot(gs[2, :])
    scorecard.set_index("Mode")[["Mean_Delay", "StdDev", "Severe_20_Pct"]].plot(
        kind="bar", ax=ax3, color=["#4c72b0", "#dd8452", "#c44e52"]
    )
    ax3.set_title("Reliability scorecard comparison")
    ax3.set_ylabel("Minutes / percent")
    ax3.tick_params(axis="x", rotation=0)
    ax3.grid(axis="y", alpha=0.3)
    fig.suptitle("Multimodal Smart City Executive Dashboard", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    meta = ensure_data()
    out = output_dir("exp06")
    spark = get_spark("Exp06_Visualization", shuffle_partitions=8)
    try:
        master, master_path, reused = _load_master(spark, out)
        df = master.toPandas()
        df["Date"] = pd.to_datetime(df["Date"])
        df["DayName"] = df["Date"].dt.day_name()
        df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

        daily = (
            df.groupby(["Date", "Mode"])["Delay_Minutes"]
            .agg(Daily_Avg_Delay="mean", Daily_Incidents="count")
            .reset_index()
        )
        daily["Rolling_14Day"] = daily.groupby("Mode")["Daily_Avg_Delay"].transform(
            lambda series: series.rolling(14, min_periods=1).mean()
        )
        monthly = (
            df.groupby(["YearMonth", "Mode"])["Passenger_Delay_Impact_Hrs"]
            .sum()
            .unstack()
            .fillna(0)
        )

        scorecard = (
            df.groupby("Mode")
            .agg(
                Trips=("Delay_Minutes", "size"),
                Mean_Delay=("Delay_Minutes", "mean"),
                StdDev=("Delay_Minutes", "std"),
                Passenger_Hours_Lost=("Passenger_Delay_Impact_Hrs", "sum"),
                Severe_20_Pct=("Delay_Minutes", lambda s: 100.0 * (s > 20).mean()),
            )
            .reset_index()
        )
        scorecard = scorecard.round(2)
        scorecard.to_csv(out / "scorecard.csv", index=False)

        severity = (
            df.assign(
                Severity=pd.cut(
                    df["Delay_Minutes"],
                    bins=[-1, 10, 20, 30, float("inf")],
                    labels=["Minor", "Moderate", "Severe", "Critical"],
                )
            )
            .groupby(["Mode", "Severity"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        severity.to_csv(out / "severity_tiers.csv", index=False)

        _plot_trends(daily, monthly, out / "exp6_time_series_trends.png")
        _plot_dow_and_heatmaps(
            df,
            out / "exp6_day_of_week_profile.png",
            out / "exp6_heatmaps_dow_hour.png",
        )
        _plot_root_cause(df, out / "exp6_root_cause_heatmaps.png")
        _plot_dashboard(df, scorecard, severity, out / "exp6_executive_dashboard.png")

        bus = df[df["Mode"] == "Bus"]
        train = df[df["Mode"] == "Train"]
        bus_top = (
            bus.groupby(["Location", "Reason"])["Passenger_Delay_Impact_Hrs"]
            .sum()
            .sort_values(ascending=False)
        )
        train_top = (
            train.groupby(["Location", "Reason"])["Passenger_Delay_Impact_Hrs"]
            .sum()
            .sort_values(ascending=False)
        )
        morning = df[df["TimeOfDayBand"] == "Morning Rush (06-09)"]["Passenger_Delay_Impact_Hrs"].sum()
        evening = df[df["TimeOfDayBand"] == "Evening Rush (16-19)"]["Passenger_Delay_Impact_Hrs"].sum()
        night = df[df["TimeOfDayBand"] == "Night Off-Peak (20-05)"]["Passenger_Delay_Impact_Hrs"].sum()
        train_signal = train[train["Reason"] == "Signal Problem"]["Passenger_Delay_Impact_Hrs"].sum()
        train_congestion_track = train[
            train["Reason"].isin(["Congestion", "Track Work"])
        ]["Passenger_Delay_Impact_Hrs"].sum()
        train_total = train["Passenger_Delay_Impact_Hrs"].sum()
        bus_dow_mean = bus.groupby("DayName")["Delay_Minutes"].mean().reindex(DOW_ORDER)

        report_lines = [
            "EXPERIMENT 6 - BATCH ANALYTICS VISUALIZATION & REPORTING",
            "=" * 78,
            f"Generated : {datetime.now().isoformat(timespec='seconds')}",
            f"Source    : {'Experiment 5 Parquet artifacts' if reused else 'rebuilt from raw CSVs'}",
            f"Path      : {rel(master_path)}",
            f"Records   : {len(df):,}",
            "",
            "MULTIMODAL PERFORMANCE & RELIABILITY SCORECARD",
            "-" * 78,
            scorecard.to_string(index=False),
            "",
            "TEMPORAL & SEASONAL URBAN DYNAMICS",
            "-" * 78,
            f"Morning peak (06-09) passenger-hours lost : {morning:,.2f}",
            f"Evening peak (16-19) passenger-hours lost : {evening:,.2f}",
            f"Night off-peak (20-05) passenger-hours lost: {night:,.2f}",
            (
                f"Bus weekday mean delay range              : "
                f"{bus_dow_mean.min():.2f}-{bus_dow_mean.max():.2f} minutes "
                f"(stable across all seven days)"
            ),
            "",
            "GEOSPATIAL BOTTLENECKS & ROOT-CAUSE ATTRIBUTION",
            "-" * 78,
            "Top 5 bus location/reason combinations:",
            bus_top.head(5).to_string(),
            "",
            "Top 5 train location/reason combinations:",
            train_top.head(5).to_string(),
            "",
            f"Train Signal Problem passenger-hours      : {train_signal:,.2f} "
            f"({100 * train_signal / train_total:.1f}% of train impact)",
            f"Train Congestion + Track Work impact      : {train_congestion_track:,.2f} "
            f"({100 * train_congestion_track / train_total:.1f}% of train impact)",
            "",
            "STRATEGIC SMART CITY RECOMMENDATIONS",
            "-" * 78,
            "1. Dynamic Transit Signal Priority on Brooklyn/Manhattan bus corridors at 07:00-09:00 and 16:00-18:00.",
            "2. Predictive IoT fleet maintenance to reduce mechanical-problem delays.",
            "3. Intelligent rail dispatching on East of Hudson and New Haven high-capacity corridors.",
            "4. Scale the partitioned Parquet architecture for municipal pipelines (large storage savings).",
            "",
            "KEY ANALYTICAL FINDINGS",
            "-" * 78,
            (
                f"1. Train mean delay ({scorecard.loc[scorecard.Mode == 'Train', 'Mean_Delay'].iloc[0]:.2f} min) "
                f"is lower than bus ({scorecard.loc[scorecard.Mode == 'Bus', 'Mean_Delay'].iloc[0]:.2f} min), "
                f"but train passenger-hours lost "
                f"({scorecard.loc[scorecard.Mode == 'Train', 'Passenger_Hours_Lost'].iloc[0]:,.0f}) far exceed "
                f"bus ({scorecard.loc[scorecard.Mode == 'Bus', 'Passenger_Hours_Lost'].iloc[0]:,.0f}) due to "
                f"higher onboard capacity."
            ),
            (
                f"2. Severe (>20 min) incidents: bus "
                f"{scorecard.loc[scorecard.Mode == 'Bus', 'Severe_20_Pct'].iloc[0]:.2f}% vs train "
                f"{scorecard.loc[scorecard.Mode == 'Train', 'Severe_20_Pct'].iloc[0]:.2f}% - bus delays are "
                f"proportionally more severe."
            ),
            f"3. Largest bus bottleneck: {bus_top.index[0][0]} / {bus_top.index[0][1]} ({bus_top.iloc[0]:,.0f} passenger-hours).",
            f"4. Largest train bottleneck: {train_top.index[0][0]} / {train_top.index[0][1]} ({train_top.iloc[0]:,.0f} passenger-hours).",
            f"5. Signal Problem alone accounts for {100 * train_signal / train_total:.1f}% of train impact.",
            f"6. Bus weekday averages stay within {bus_dow_mean.min():.2f}-{bus_dow_mean.max():.2f} minutes - no strong weekday/weekend effect.",
            "",
            "CONCLUSION",
            "-" * 78,
            (
                "The Experiment 5 aggregates were transformed into a full visual analytics suite: "
                "rolling time-series trends, monthly passenger-hour curves, day-of-week/hour heatmaps, "
                "root-cause cross-tabulations and an executive KPI dashboard. Bus and train delays fail "
                "in structurally different ways: buses suffer frequent congestion-driven delays, while "
                "trains suffer rarer but much higher-impact signal and track incidents."
            ),
        ]
        report = "\n".join(report_lines)
        (out / "batch_analytics_report.txt").write_text(report, encoding="utf-8")

        metrics = {
            "records": len(df),
            "reused_exp05_parquet": reused,
            "scorecard": scorecard.to_dict(orient="records"),
            "morning_impact_hours": round(float(morning), 2),
            "evening_impact_hours": round(float(evening), 2),
            "night_impact_hours": round(float(night), 2),
            "train_signal_share_pct": round(100 * train_signal / train_total, 2),
            "train_congestion_track_share_pct": round(100 * train_congestion_track / train_total, 2),
            "bus_top_bottleneck": {
                "location": bus_top.index[0][0],
                "reason": bus_top.index[0][1],
                "impact_hours": round(float(bus_top.iloc[0]), 2),
            },
            "train_top_bottleneck": {
                "location": train_top.index[0][0],
                "reason": train_top.index[0][1],
                "impact_hours": round(float(train_top.iloc[0]), 2),
            },
        }
        (out / "visualization_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        assert len(df) == 100_000
        assert len(scorecard) == 2
        assert metrics["train_signal_share_pct"] > 0
        for name in (
            "exp6_time_series_trends.png",
            "exp6_day_of_week_profile.png",
            "exp6_heatmaps_dow_hour.png",
            "exp6_root_cause_heatmaps.png",
            "exp6_executive_dashboard.png",
        ):
            assert (out / name).stat().st_size > 20_000, name

        print(report)
        print("\nEXP06 PASSED")
        return metrics
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
