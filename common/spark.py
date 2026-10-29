"""SparkSession factory with Windows/Hadoop compatibility handling."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_WINUTILS_CANDIDATES = [
    os.environ.get("HADOOP_HOME"),
    str(Path(__file__).resolve().parents[1] / ".hadoop"),
]

_PYTHON_CANDIDATES = [
    os.environ.get("PYSPARK_PYTHON"),
    sys.executable,
]


def _space_free_python_alias() -> str | None:
    """Alias a spaced interpreter path so Spark workers can launch on Windows.

    Spark starts Python workers through cmd.exe, and a space in the
    interpreter path (e.g. ``C:\\Program Files\\...``) can fail with an EOF
    error.  A junction in the OS temp folder (always space-free) is created
    pointing at the interpreter's parent directory.  Never puts a username or
    other machine-specific value into the generated reports.
    """
    if os.name != "nt":
        return None
    exe = Path(sys.executable)
    if " " not in str(exe):
        return None
    alias_dir = Path(tempfile.gettempdir()) / "spark_py_alias"
    python_alias = alias_dir / exe.name
    target = exe.parent
    try:
        if python_alias.exists():
            alias_dir.unlink()
        try:
            import _winapi

            _winapi.CreateJunction(str(target), str(alias_dir))
        except ImportError:
            return None
        if python_alias.exists():
            return str(python_alias)
    except OSError:
        return None
    return None


def _python_candidates() -> list[str]:
    return [
        os.environ.get("PYSPARK_PYTHON"),
        _space_free_python_alias(),
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

    Prefers ``PYSPARK_PYTHON`` from the environment, then a space-free alias
    of the current interpreter, then the interpreter itself.
    """
    for candidate in _python_candidates():
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
