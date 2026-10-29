# Project Proposal and Architecture Definition

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
