"""Shared transit DataFrame transformations used by Experiments 5 and 6."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    dayofweek,
    hour,
    lit,
    month,
    round as spark_round,
    to_date,
    to_timestamp,
    when,
    year,
)

from common.config import BUS_CSV, TRAIN_CSV

UNIFIED_COLUMNS = [
    "Date",
    "Mode",
    "Location",
    "Scheduled_Time",
    "Hour",
    "Month",
    "Year",
    "DayOfWeek",
    "TimeOfDayBand",
    "Reason",
    "Delay_Minutes",
    "Passengers",
    "Passenger_Delay_Impact_Hrs",
]


def load_raw(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    bus = spark.read.csv(str(BUS_CSV), header=True, inferSchema=True)
    train = spark.read.csv(str(TRAIN_CSV), header=True, inferSchema=True)
    return bus, train


def _prepared(df: DataFrame, mode: str, location_col: str, passenger_col: str) -> DataFrame:
    time_ts = to_timestamp(col("Scheduled_Time"), "HH:mm:ss")
    prepared = (
        df.withColumn("Mode", lit(mode))
        .withColumn("Location", col(location_col))
        .withColumn("Passengers", col(passenger_col))
        .withColumn("Hour", hour(time_ts))
        .withColumn("Month", month(to_date(col("Date"))))
        .withColumn("Year", year(to_date(col("Date"))))
        .withColumn("DayOfWeek", dayofweek(to_date(col("Date"))))
        .withColumn(
            "TimeOfDayBand",
            when((col("Hour") >= 6) & (col("Hour") <= 9), "Morning Rush (06-09)")
            .when((col("Hour") >= 10) & (col("Hour") <= 15), "Midday Off-Peak (10-15)")
            .when((col("Hour") >= 16) & (col("Hour") <= 19), "Evening Rush (16-19)")
            .otherwise("Night Off-Peak (20-05)"),
        )
        .withColumn(
            "Passenger_Delay_Impact_Hrs",
            spark_round((col("Delay_Minutes") * col(passenger_col)) / 60.0, 2),
        )
    )
    return prepared.select(*UNIFIED_COLUMNS)


def build_unified(spark: SparkSession) -> DataFrame:
    bus, train = load_raw(spark)
    bus_prepared = _prepared(bus, "Bus", "Borough", "Number_Of_Students_On_Bus")
    train_prepared = _prepared(train, "Train", "Line", "Passengers_Onboard")
    return bus_prepared.unionByName(train_prepared)
