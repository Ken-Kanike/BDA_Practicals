"""Synthetic IoT transit telemetry used by the streaming experiments."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

LOCATIONS_BUS = ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
LOCATIONS_TRAIN = ["Harlem", "Hudson Line", "New Haven", "East of Hudson", "Pascack Valley"]
REASONS = [
    "Heavy Traffic",
    "Mechanical Problem",
    "Signal Problem",
    "Weather Delay",
    "Track Work",
    "Normal Ops",
]


def generate_events(
    batch_idx: int,
    base_time: datetime,
    num_events: int = 120,
    seed: int | None = None,
) -> list[dict]:
    rng = random.Random(seed if seed is not None else 1000 + batch_idx)
    events: list[dict] = []
    for i in range(num_events):
        mode = "Bus" if rng.random() < 0.5 else "Train"
        location = rng.choice(LOCATIONS_BUS if mode == "Bus" else LOCATIONS_TRAIN)
        incident = rng.random() < 0.45
        delay = round(rng.uniform(5.0, 35.0), 2) if incident else round(rng.uniform(0.0, 4.5), 2)
        speed = round(rng.uniform(5.0, 20.0), 1) if incident else round(rng.uniform(30.0, 60.0), 1)
        reason = rng.choice(REASONS[:5]) if incident else "Normal Ops"
        passenger_load = rng.randint(5, 60) if mode == "Bus" else rng.randint(60, 220)
        events.append(
            {
                "event_time": (base_time + timedelta(seconds=rng.randint(0, 59))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "sensor_id": f"IOT-{rng.randint(1000, 9999)}",
                "vehicle_id": f"{mode[:2].upper()}-{batch_idx:03d}-{i:03d}",
                "mode": mode,
                "route_or_line": location,
                "location": location,
                "speed_mph": speed,
                "delay_minutes": delay,
                "passenger_load": passenger_load,
                "reason": reason,
                "sensor_health": "OK" if rng.random() < 0.94 else rng.choice(["DEGRADED", "FAILED"]),
            }
        )
    return events
