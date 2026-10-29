"""Experiment 2 - Project proposal and end-to-end architecture definition."""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from common.config import DOCS_DIR, output_dir
from common.data_gen import ensure_data

LAYERS = [
    {
        "layer": "Ingestion Layer",
        "technology": "Pandas / PySpark CSV readers; simulated Kafka producer",
        "purpose": "Load raw bus/train delay logs and live IoT telemetry into the platform.",
        "experiments": "Exp 1, 4, 7",
    },
    {
        "layer": "Storage Layer",
        "technology": "HDFS-style partitioned Parquet data lake; MongoDB document store",
        "purpose": "Persist raw and cleaned data at scale with partition pruning and flexible schemas.",
        "experiments": "Exp 4, 9",
    },
    {
        "layer": "Processing Layer",
        "technology": "Spark RDDs (MapReduce), Spark SQL, Catalyst, window functions",
        "purpose": "Clean, aggregate, join and analyse 100,000 unified transit records.",
        "experiments": "Exp 3, 5, 6",
    },
    {
        "layer": "Streaming Layer",
        "technology": "Spark Structured Streaming, watermarks, sliding windows",
        "purpose": "Compute live rolling delay metrics and severe-delay alerts.",
        "experiments": "Exp 7, 8",
    },
    {
        "layer": "Machine Learning Layer",
        "technology": "scikit-learn Random Forest / GBT / Linear Regression",
        "purpose": "Predict delay duration and severe-congestion risk from trip features.",
        "experiments": "Exp 10",
    },
    {
        "layer": "Serving / Dashboard Layer",
        "technology": "Streamlit + Plotly, Matplotlib/Seaborn reports",
        "purpose": "Give operators live KPIs, incident feeds, model predictions and static reports.",
        "experiments": "Exp 6, 8, 10",
    },
]

PROPOSAL = """# Project Proposal and Architecture Definition

## 1. Project Title & Problem Domain

**Title:** Comparative Study of Transport Modes: Big Data Analysis of Bus vs Train Delays

**Domain:** Transportation / Smart City.

**Context:** Cities depend on both road-based (bus) and rail-based (train) public
transit. The two modes have different delay characteristics, and the volume and
variety of delay records exceed what spreadsheet analysis can handle.

**Problem statement:** Bus and train delays affect commuter trust, congestion and
transit budgets differently depending on cause and mode. Comparing delay patterns
across routes, lines and time periods requires a distributed pipeline that can
ingest, store, process, model and visualise a large number of records.

**Objectives**

1. Ingest and store bus and train delay records in a structured data lake.
2. Process the data at scale using distributed in-memory computing.
3. Predict delay duration and severe-delay risk with machine learning.
4. Provide interactive dashboards comparing bus and train reliability.

## 2. Dataset Allocation

- **Origin:** Synthetic bus and train delay datasets generated to match real-world
  transit open-data schemas.
- **Shape:** 50,000 bus records and 50,000 train records across a two-year window.
- **Bus fields:** `Date`, `Route_ID`, `Borough`, `Scheduled_Time`, `Reason`,
  `Delay_Minutes`, `Number_Of_Students_On_Bus`.
- **Train fields:** `Date`, `Line`, `Scheduled_Time`, `Reason`, `Delay_Minutes`,
  `Passengers_Onboard`.
- **Targets:** `Delay_Minutes` for regression and a derived `Severe_Delay_Flag`
  (delay >= 15 minutes) for classification.
- **Scope:** Bus records cover five boroughs and multiple routes; train records
  cover seven lines, mirroring a city-wide transit network.

## 3. Team Roles & Responsibilities

| Member | Role | Responsibilities |
|---|---|---|
| Chaitanya Jhade (48) | Data Engineer | Ingestion pipeline, HDFS/local storage, schema consistency, ETL. |
| Gaurav Joshi (50) | Data Scientist | ML layer, regression and severity classification, model evaluation. |
| Junaid Shaikh (137) | Data Analyst | EDA, delay trends by reason/route/time, comparative insights. |
| Soham Sonavane (138) | BI & Visualization Engineer | Presentation layer, dashboards and stakeholder reporting. |

## 4. End-to-End System Architecture

The system is organised into decoupled layers so each can scale independently:

1. **Ingestion Layer** - raw CSV logs are loaded with Pandas/PySpark; the streaming
   experiment simulates a Kafka producer emitting newline-delimited JSON telemetry.
2. **Storage Layer** - a data lake stores raw records and partitioned Parquet;
   MongoDB stores flexible document records for staging and NoSQL analytics.
3. **Processing Layer** - PySpark RDDs implement MapReduce text processing and
   Spark SQL performs batch aggregations, window functions and Catalyst-planned
   joins/aggregations.
4. **Streaming Layer** - Spark Structured Streaming applies watermarks and
   sliding windows to live telemetry and emits severe-delay alerts.
5. **Machine Learning Layer** - scikit-learn models predict delay minutes and the
   probability that a trip becomes severely delayed.
6. **Serving / Dashboard Layer** - Streamlit/Plotly dashboards and static reports
   expose KPIs, incident feeds and predictions to transit administrators.

## 5. Version Control & Reproducibility

All pipeline code, configuration and experiment outputs are version-controlled with
Git. Each practical is implemented as a runnable Python module under `practicals/`,
with shared utilities under `common/` and a deterministic synthetic data generator
under `common/data_gen.py`.
"""


def _box(ax, xy, width, height, text, facecolor, fontsize=9.5):
    box = mpatches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return box


def _arrow(ax, start, end, color="#555555"):
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _architecture_diagram(path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Smart City Transit Big Data Platform - End-to-End Architecture",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )

    # Sources
    _box(ax, (0.03, 0.88), 0.27, 0.08, "Bus delay CSV\n50,000 records", "#dbeafe")
    _box(ax, (0.365, 0.88), 0.27, 0.08, "Train delay CSV\n50,000 records", "#fde2e2")
    _box(ax, (0.70, 0.88), 0.27, 0.08, "Live IoT telemetry\nJSON micro-batches", "#e8f5e9")

    layer_colors = ["#cfe8ff", "#d9f2e6", "#fff2cc", "#e6e0f8", "#ffe0e6", "#d8f3f9"]
    y_positions = [0.74, 0.61, 0.48, 0.35, 0.22, 0.09]
    for layer, color, y in zip(LAYERS, layer_colors, y_positions):
        _box(
            ax,
            (0.06, y),
            0.88,
            0.10,
            f"{layer['layer']}\n{layer['technology']}\n{layer['purpose']}",
            color,
            fontsize=9,
        )

    for y_from, y_to in zip(y_positions[:-1], y_positions[1:]):
        _arrow(ax, (0.50, y_from), (0.50, y_to + 0.10))

    # Source-to-ingestion arrows
    _arrow(ax, (0.165, 0.88), (0.28, 0.84))
    _arrow(ax, (0.50, 0.88), (0.50, 0.84))
    _arrow(ax, (0.835, 0.88), (0.72, 0.84))

    # Side annotations
    ax.text(
        0.02,
        0.515,
        "Batch path\nExp 1-6",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#1f4e79",
    )
    ax.text(
        0.985,
        0.515,
        "Real-time path\nExp 7-8",
        rotation=270,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.text(
        0.02,
        0.155,
        "Prediction\n+ serving\nExp 9-10",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#7b1fa2",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _experiment_map(path) -> None:
    experiments = [
        ("Exp 1", "EDA & setup"),
        ("Exp 2", "Proposal & architecture"),
        ("Exp 3", "MapReduce text"),
        ("Exp 4", "Ingestion & NoSQL staging"),
        ("Exp 5", "Spark SQL batch"),
        ("Exp 6", "Visual analytics & report"),
        ("Exp 7", "Structured Streaming"),
        ("Exp 8", "Live dashboard & alerts"),
        ("Exp 9", "Indexed NoSQL data lake"),
        ("Exp 10", "ML & final showcase"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    for i, (label, title) in enumerate(experiments):
        color = "#cfe8ff" if i < 6 else ("#d9f2e6" if i < 8 else "#e6e0f8")
        _box(ax, (i + 0.05, 0.35), 0.90, 0.34, f"{label}\n{title}", color, fontsize=8.5)
        if i < len(experiments) - 1:
            _arrow(ax, (i + 0.95, 0.52), (i + 1.05, 0.52))
    ax.set_title("Experiment pipeline - from raw data to Smart City showcase", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    meta = ensure_data()
    out = output_dir("exp02")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    _architecture_diagram(out / "architecture.png")
    _experiment_map(out / "experiment_pipeline.png")
    (DOCS_DIR / "PROJECT_PROPOSAL.md").write_text(PROPOSAL, encoding="utf-8")

    layers_path = out / "system_layers.json"
    layers_path.write_text(json.dumps(LAYERS, indent=2), encoding="utf-8")

    summary = [
        "EXPERIMENT 2 - PROJECT PROPOSAL & ARCHITECTURE SUMMARY",
        "=" * 72,
        "Title : Comparative Study of Transport Modes - Big Data Analysis of Bus vs Train Delays",
        "Domain: Transportation / Smart City",
        f"Data  : {meta['bus_rows']:,} bus + {meta['train_rows']:,} train records (2-year window)",
        "",
        "OBJECTIVES",
        "- Ingest and store bus/train delay records in a data lake.",
        "- Process the data at scale with distributed in-memory computing.",
        "- Predict delay duration and severe-delay risk.",
        "- Provide interactive reliability dashboards for transit administrators.",
        "",
        "TEAM",
        "- Chaitanya Jhade (48)  : Data Engineer",
        "- Gaurav Joshi (50)     : Data Scientist",
        "- Junaid Shaikh (137)   : Data Analyst",
        "- Soham Sonavane (138)  : BI & Visualization Engineer",
        "",
        "ARCHITECTURE LAYERS",
    ]
    for layer in LAYERS:
        summary.append(f"- {layer['layer']}: {layer['technology']} ({layer['experiments']})")
    summary += [
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "Artifacts: docs/PROJECT_PROPOSAL.md, architecture.png, experiment_pipeline.png, system_layers.json",
    ]
    report = "\n".join(summary)
    (out / "proposal_summary.txt").write_text(report, encoding="utf-8")

    assert len(LAYERS) == 6
    assert (out / "architecture.png").stat().st_size > 40_000
    assert (DOCS_DIR / "PROJECT_PROPOSAL.md").stat().st_size > 2_000
    assert len(json.loads(layers_path.read_text(encoding="utf-8"))) == 6

    print(report)
    print("\nEXP02 PASSED")
    return {"layers": len(LAYERS), "artifacts": 4}


if __name__ == "__main__":
    main()
