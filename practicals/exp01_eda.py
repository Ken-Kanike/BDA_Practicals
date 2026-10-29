"""Experiment 1 - Domain selection, environment setup and exploratory data analysis."""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import avg, col, count, desc, max as spark_max, min as spark_min, stddev, sum as spark_sum

from common.config import BUS_CSV, TRAIN_CSV, output_dir
from common.data_gen import ensure_data
from common.spark import get_spark


def _null_counts(df) -> dict:
    row = df.agg(*[spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).first()
    return {c: int(row[c] or 0) for c in df.columns}


def _delay_stats(df) -> dict:
    row = df.select(
        count("*").alias("rows"),
        avg("Delay_Minutes").alias("mean"),
        stddev("Delay_Minutes").alias("stddev"),
        spark_min("Delay_Minutes").alias("min"),
        spark_max("Delay_Minutes").alias("max"),
    ).first()
    median = float(df.approxQuantile("Delay_Minutes", [0.5], 0.001)[0])
    p90 = float(df.approxQuantile("Delay_Minutes", [0.9], 0.001)[0])
    p99 = float(df.approxQuantile("Delay_Minutes", [0.99], 0.001)[0])
    return {
        "rows": int(row["rows"]),
        "mean": round(float(row["mean"]), 3),
        "median": round(median, 3),
        "stddev": round(float(row["stddev"]), 3),
        "min": round(float(row["min"]), 3),
        "max": round(float(row["max"]), 3),
        "p90": round(p90, 3),
        "p99": round(p99, 3),
        "severe_20_pct": round(100.0 * df.filter(col("Delay_Minutes") > 20).count() / row["rows"], 2),
    }


def _plot_reasons(bus_reasons: pd.DataFrame, train_reasons: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, frame, title, color in (
        (axes[0], bus_reasons, "Bus - Top 10 delay reasons", "#1f77b4"),
        (axes[1], train_reasons, "Train - Top 10 delay reasons", "#d62728"),
    ):
        frame = frame.sort_values("count")
        ax.barh(frame["Reason"], frame["count"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Records")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_locations(bus_borough: pd.DataFrame, train_lines: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bus_borough = bus_borough.sort_values("count")
    axes[0].barh(bus_borough["Borough"], bus_borough["count"], color="#2ca02c")
    axes[0].set_title("Bus delay records by borough")
    axes[0].set_xlabel("Records")
    axes[0].grid(axis="x", alpha=0.3)
    train_lines = train_lines.sort_values("count")
    axes[1].barh(train_lines["Line"], train_lines["count"], color="#9467bd")
    axes[1].set_title("Train delay records by line")
    axes[1].set_xlabel("Records")
    axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_delay_box(bus_pdf: pd.DataFrame, train_pdf: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = [bus_pdf["Delay_Minutes"], train_pdf["Delay_Minutes"]]
    bp = axes[0].boxplot(data, labels=["Bus", "Train"], showfliers=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_title("Delay minute distribution and outliers")
    axes[0].set_ylabel("Delay (minutes)")
    axes[0].grid(axis="y", alpha=0.3)

    for frame, label, color in ((bus_pdf, "Bus", "#1f77b4"), (train_pdf, "Train", "#d62728")):
        axes[1].hist(frame["Delay_Minutes"], bins=60, alpha=0.55, label=label, color=color)
    axes[1].set_title("Delay histogram (right-skewed)")
    axes[1].set_xlabel("Delay (minutes)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 60)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    meta = ensure_data()
    out = output_dir("exp01")
    spark = get_spark("Exp01_EDA", shuffle_partitions=4)
    try:
        bus = spark.read.csv(str(BUS_CSV), header=True, inferSchema=True)
        train = spark.read.csv(str(TRAIN_CSV), header=True, inferSchema=True)
        bus.cache()
        train.cache()

        bus_rows, bus_cols = bus.count(), len(bus.columns)
        train_rows, train_cols = train.count(), len(train.columns)
        bus_nulls = _null_counts(bus)
        train_nulls = _null_counts(train)
        bus_dupes = bus_rows - bus.dropDuplicates().count()
        train_dupes = train_rows - train.dropDuplicates().count()

        bus_reasons = bus.groupBy("Reason").count().orderBy(desc("count")).limit(10).toPandas()
        train_reasons = train.groupBy("Reason").count().orderBy(desc("count")).limit(10).toPandas()
        bus_borough = bus.groupBy("Borough").count().orderBy(desc("count")).toPandas()
        train_lines = train.groupBy("Line").count().orderBy(desc("count")).toPandas()

        bus_stats = _delay_stats(bus)
        train_stats = _delay_stats(train)

        bus_pdf = bus.select("Delay_Minutes").toPandas()
        train_pdf = train.select("Delay_Minutes").toPandas()

        _plot_reasons(bus_reasons, train_reasons, out / "exp1_delay_reasons_comparison.png")
        _plot_locations(bus_borough, train_lines, out / "exp1_borough_and_lines.png")
        _plot_delay_box(bus_pdf, train_pdf, out / "exp1_delay_distribution_boxplot.png")

        report_lines = [
            "EXPERIMENT 1 - DOMAIN SELECTION, SETUP & EXPLORATORY DATA ANALYSIS",
            "=" * 78,
            f"Generated : {datetime.now().isoformat(timespec='seconds')}",
            f"Seed      : {meta['seed']}",
            "",
            "DATASET SHAPE",
            "-" * 78,
            f"Bus   : {bus_rows:,} rows x {bus_cols} columns -> {', '.join(bus.columns)}",
            f"Train : {train_rows:,} rows x {train_cols} columns -> {', '.join(train.columns)}",
            "",
            "DATA QUALITY",
            "-" * 78,
            f"Bus null values   : {sum(bus_nulls.values())}  ({bus_nulls})",
            f"Train null values : {sum(train_nulls.values())}  ({train_nulls})",
            f"Bus duplicates    : {bus_dupes}",
            f"Train duplicates  : {train_dupes}",
            "",
            "DELAY STATISTICS (minutes)",
            "-" * 78,
            f"{'Metric':<18}{'Bus':>12}{'Train':>12}",
            *[
                f"{key:<18}{bus_stats[key]:>12}{train_stats[key]:>12}"
                for key in ("rows", "mean", "median", "stddev", "min", "p90", "p99", "max", "severe_20_pct")
            ],
            "",
            "TOP 10 BUS DELAY REASONS",
            "-" * 78,
            bus_reasons.to_string(index=False),
            "",
            "TOP 10 TRAIN DELAY REASONS",
            "-" * 78,
            train_reasons.to_string(index=False),
            "",
            "BUS RECORDS BY BOROUGH",
            "-" * 78,
            bus_borough.to_string(index=False),
            "",
            "TRAIN RECORDS BY LINE",
            "-" * 78,
            train_lines.to_string(index=False),
            "",
            "CONCLUSION",
            "-" * 78,
            (
                f"Both datasets are clean (zero nulls and duplicates) and structurally "
                f"comparable at {bus_rows:,} records each. Bus mean delay is "
                f"{bus_stats['mean']:.2f} min vs {train_stats['mean']:.2f} min for trains; "
                f"both distributions are right-skewed. '{bus_reasons.iloc[0]['Reason']}' "
                f"dominates bus delays and '{train_reasons.iloc[0]['Reason']}' dominates train delays. "
                f"The data is ready for distributed processing."
            ),
        ]
        report = "\n".join(report_lines)
        (out / "eda_summary_report.txt").write_text(report, encoding="utf-8")
        metrics = {
            "bus_rows": bus_rows,
            "train_rows": train_rows,
            "bus_columns": bus_cols,
            "train_columns": train_cols,
            "bus_nulls": sum(bus_nulls.values()),
            "train_nulls": sum(train_nulls.values()),
            "bus_duplicates": bus_dupes,
            "train_duplicates": train_dupes,
            "bus_stats": bus_stats,
            "train_stats": train_stats,
            "top_bus_reason": bus_reasons.iloc[0]["Reason"],
            "top_bus_reason_count": int(bus_reasons.iloc[0]["count"]),
            "top_train_reason": train_reasons.iloc[0]["Reason"],
            "top_train_reason_count": int(train_reasons.iloc[0]["count"]),
        }
        (out / "eda_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        assert bus_rows == 50_000 and train_rows == 50_000
        assert metrics["bus_nulls"] == 0 and metrics["train_nulls"] == 0
        assert metrics["bus_duplicates"] == 0 and metrics["train_duplicates"] == 0
        assert metrics["top_bus_reason"] == "Heavy Traffic"
        assert metrics["top_train_reason"] == "Signal Problem"
        for png in out.glob("*.png"):
            assert png.stat().st_size > 10_000, png

        print(report)
        print("\nEXP01 PASSED")
        return metrics
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
