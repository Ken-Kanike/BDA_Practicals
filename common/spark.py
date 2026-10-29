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

_PYTHON_CANDIDATES = [
    os.environ.get("PYSPARK_PYTHON"),
    r"C:\Users\junai\AppData\Local\Temp\opencode\py311\python.exe",
    r"C:\Users\junai\AppData\Local\Programs\Python\Python311\python.exe",
    sys.executable,
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


def configure_python() -> str:
    """Select a Python executable whose path does not break Spark workers.

    On Windows the default interpreter may live under ``C:\\Program Files``.
    Spark launches Python workers through cmd.exe and a space in the path can
    make the worker fail with an EOF error, so a space-free junction is used
    when available.
    """
    for candidate in _PYTHON_CANDIDATES:
        if candidate and Path(candidate).exists():
            os.environ["PYSPARK_PYTHON"] = candidate
            os.environ["PYSPARK_DRIVER_PYTHON"] = candidate
            return candidate
    return sys.executable


def get_spark(
    app_name: str,
    master: str = "local[*]",
    shuffle_partitions: int = 8,
    warehouse: str | None = None,
):
    """Create a local SparkSession configured for these practicals."""
    configure_hadoop()
    python_exe = configure_python()
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
        .config("spark.pyspark.python", python_exe)
        .config("spark.pyspark.driver.python", python_exe)
        .config("spark.python.worker.reuse", "false")
    )
    if warehouse:
        Path(warehouse).mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.sql.warehouse.dir", str(Path(warehouse).resolve()))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
