"""SparkSession factory with Windows/Hadoop compatibility handling."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_WINUTILS_CANDIDATES = [
    os.environ.get("HADOOP_HOME"),
    str(Path(__file__).resolve().parents[1] / ".hadoop"),
    r"C:\Users\junai\AppData\Local\Temp\opencode\hadoop",
]


def configure_hadoop() -> str | None:
    """Point Spark at winutils.exe when running on Windows without HADOOP_HOME."""
    for candidate in _WINUTILS_CANDIDATES:
        if not candidate:
            continue
        winutils = Path(candidate) / "bin" / "winutils.exe"
        if winutils.exists():
            os.environ["HADOOP_HOME"] = candidate
            os.environ["hadoop.home.dir"] = candidate
            bin_dir = str(winutils.parent)
            if bin_dir.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return candidate
    return None


def get_spark(
    app_name: str,
    master: str = "local[*]",
    shuffle_partitions: int = 8,
    warehouse: str | None = None,
):
    """Create a local SparkSession configured for these practicals."""
    configure_hadoop()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.adaptive.enabled", "true")
    )
    if warehouse:
        Path(warehouse).mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.sql.warehouse.dir", str(Path(warehouse).resolve()))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
