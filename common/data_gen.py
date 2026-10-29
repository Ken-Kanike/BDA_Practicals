"""Deterministic synthetic dataset generator.

The original practical reports used two 50,000-row CSV exports that are not
shipped with the documents.  This module recreates equivalent, reproducible
datasets with the same columns, reason distributions, scale and approximate
delay characteristics so every experiment can be executed end-to-end.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common.config import (
    BOROUGHS,
    BUS_COLUMNS,
    BUS_CSV,
    BUS_REASONS,
    METADATA_JSON,
    TRAIN_COLUMNS,
    TRAIN_CSV,
    TRAIN_LINES,
    TRAIN_REASONS,
    ensure_dirs,
)

ROUTES_BY_BOROUGH = {
    "Brooklyn": ["B1", "B6", "B12", "B21", "B38", "B41", "B44", "B49", "B52", "B60", "B62", "B68", "B82", "B99"],
    "Manhattan": ["M1", "M5", "M15", "M20", "M31", "M34", "M57", "M72", "M79", "M86", "M96", "M101", "M104"],
    "Queens": ["Q1", "Q4", "Q12", "Q18", "Q27", "Q33", "Q44", "Q46", "Q58", "Q66", "Q69", "Q72", "Q88"],
    "Bronx": ["Bx1", "Bx6", "Bx9", "Bx12", "Bx15", "Bx19", "Bx22", "Bx26", "Bx33", "Bx36", "Bx40"],
    "Staten Island": ["S40", "S44", "S46", "S48", "S51", "S52", "S53", "S54", "S56", "S59", "S61", "S74", "S78", "S79"],
}

BOROUGH_PROBS = np.array([0.290, 0.260, 0.254, 0.120, 0.076])
LINE_PROBS = np.array([0.150, 0.150, 0.170, 0.180, 0.120, 0.115, 0.115])

# Small, deliberately weak relationships keep the ML experiment realistic:
# the signal is genuine but swamped by the stochastic delay tail.
BUS_REASON_EFFECT = {
    "Heavy Traffic": 1.0,
    "Mechanical Problem": 2.5,
    "Late Return From Route": 1.5,
    "Road Construction": 1.0,
    "Weather Conditions": 0.5,
    "Accident": 3.0,
    "Driver Shortage": 1.2,
    "Other": 0.0,
}
TRAIN_REASON_EFFECT = {
    "Signal Problem": 3.0,
    "Mechanical Failure": 2.0,
    "Track Work": 1.5,
    "Congestion": 1.0,
    "Weather Conditions": 0.8,
    "Overhead Line Issue": 2.5,
    "Door Fault": 1.0,
    "Other": 0.0,
}


def _reason_column(rng: np.random.Generator, spec: list[tuple[str, int]]) -> np.ndarray:
    values: list[str] = []
    for reason, count in spec:
        values.extend([reason] * count)
    rng.shuffle(values)
    return np.array(values)


def _hours(rng: np.random.Generator, n: int) -> np.ndarray:
    weights = np.full(24, 0.015, dtype=float)
    weights[6:10] = 0.075
    weights[10:16] = 0.25 / 6.0
    weights[16:20] = 0.075
    hours = rng.choice(np.arange(24), size=n, p=weights)
    return hours


def _clock_times(rng: np.random.Generator, n: int) -> list[str]:
    hours = _hours(rng, n)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    return [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hours, minutes, seconds)]


def _dates(rng: np.random.Generator, n: int) -> list[str]:
    start = date(2023, 1, 1)
    offsets = rng.integers(0, 731, size=n)
    return [(start + timedelta(days=int(d))).isoformat() for d in offsets]


def _calibrate_mean(raw: np.ndarray, target: float, low: float, high: float) -> np.ndarray:
    values = np.clip(raw, low, high)
    for _ in range(3):
        values = np.clip(values * (target / values.mean()), low, high)
    return np.round(values, 2)


def _delays(
    rng: np.random.Generator,
    n: int,
    *,
    median: float,
    sigma: float,
    target_mean: float,
    reason_effects: np.ndarray,
    hours: np.ndarray,
    dates: np.ndarray,
    passengers: np.ndarray,
    passenger_weight: float,
) -> np.ndarray:
    base = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    rush = np.isin(hours, [6, 7, 8, 9, 16, 17, 18, 19]).astype(float)
    weekday = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    weekend = (weekday >= 5).astype(float)
    raw = base + reason_effects + (rush * 0.8) - (weekend * 0.3) + passengers * passenger_weight
    return _calibrate_mean(raw, target_mean, 0.5, 120.0)


def generate_data(force: bool = False, seed: int = 42) -> dict:
    """Generate both CSVs and return a metadata dictionary."""
    ensure_dirs()
    if not force and BUS_CSV.exists() and TRAIN_CSV.exists():
        return json.loads(METADATA_JSON.read_text(encoding="utf-8"))

    rng = np.random.default_rng(seed)
    n = 50_000

    # ---------------- Bus ----------------
    bus_borough = rng.choice(BOROUGHS, size=n, p=BOROUGH_PROBS)
    bus_route = [rng.choice(ROUTES_BY_BOROUGH[b]) for b in bus_borough]
    bus_reason = _reason_column(rng, BUS_REASONS)
    bus_dates = np.array(_dates(rng, n))
    bus_times = _clock_times(rng, n)
    bus_hours = np.array([int(t[:2]) for t in bus_times])
    bus_passengers = np.clip(rng.gamma(shape=2.2, scale=8.0, size=n), 1, 60).round().astype(int)
    bus_effect = np.array([BUS_REASON_EFFECT[r] for r in bus_reason])
    bus_delay = _delays(
        rng,
        n,
        median=10.1,
        sigma=0.68,
        target_mean=12.03,
        reason_effects=bus_effect,
        hours=bus_hours,
        dates=bus_dates,
        passengers=bus_passengers,
        passenger_weight=0.02,
    )

    bus_df = pd.DataFrame(
        {
            "Date": bus_dates,
            "Route_ID": bus_route,
            "Borough": bus_borough,
            "Scheduled_Time": bus_times,
            "Reason": bus_reason,
            "Delay_Minutes": bus_delay,
            "Number_Of_Students_On_Bus": bus_passengers,
        }
    )[BUS_COLUMNS]

    # ---------------- Train ----------------
    train_line = rng.choice(TRAIN_LINES, size=n, p=LINE_PROBS)
    train_reason = _reason_column(rng, TRAIN_REASONS)
    train_dates = np.array(_dates(rng, n))
    train_times = _clock_times(rng, n)
    train_hours = np.array([int(t[:2]) for t in train_times])
    train_passengers = np.clip(rng.normal(140, 30, size=n), 40, 250).round().astype(int)
    train_effect = np.array([TRAIN_REASON_EFFECT[r] for r in train_reason])
    train_delay = _delays(
        rng,
        n,
        median=6.4,
        sigma=0.72,
        target_mean=8.03,
        reason_effects=train_effect,
        hours=train_hours,
        dates=train_dates,
        passengers=train_passengers,
        passenger_weight=0.01,
    )

    train_df = pd.DataFrame(
        {
            "Date": train_dates,
            "Line": train_line,
            "Scheduled_Time": train_times,
            "Reason": train_reason,
            "Delay_Minutes": train_delay,
            "Passengers_Onboard": train_passengers,
        }
    )[TRAIN_COLUMNS]

    bus_df.to_csv(BUS_CSV, index=False)
    train_df.to_csv(TRAIN_CSV, index=False)

    metadata = {
        "seed": seed,
        "bus_rows": len(bus_df),
        "train_rows": len(train_df),
        "bus_columns": BUS_COLUMNS,
        "train_columns": TRAIN_COLUMNS,
        "bus_mean_delay": round(float(bus_df["Delay_Minutes"].mean()), 3),
        "train_mean_delay": round(float(train_df["Delay_Minutes"].mean()), 3),
        "bus_median_delay": round(float(bus_df["Delay_Minutes"].median()), 3),
        "train_median_delay": round(float(train_df["Delay_Minutes"].median()), 3),
        "bus_severe_15_rate": round(float((bus_df["Delay_Minutes"] >= 15).mean()), 4),
        "train_severe_15_rate": round(float((train_df["Delay_Minutes"] >= 15).mean()), 4),
        "bus_reason_counts": bus_df["Reason"].value_counts().to_dict(),
        "train_reason_counts": train_df["Reason"].value_counts().to_dict(),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def ensure_data(force: bool = False) -> dict:
    """Generate the CSVs if missing and validate their shape."""
    meta = generate_data(force=force)
    bus = pd.read_csv(BUS_CSV)
    train = pd.read_csv(TRAIN_CSV)
    if len(bus) != 50_000 or len(train) != 50_000:
        raise RuntimeError(f"Unexpected dataset sizes: bus={len(bus)}, train={len(train)}")
    return meta


if __name__ == "__main__":
    info = ensure_data(force=True)
    print(json.dumps(info, indent=2))
