#!/usr/bin/env python3
"""Generate a static IGRF-14 + T96 magnetosphere snapshot for GitHub Pages."""

from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from geopack import geopack, t96


ROOT = Path(__file__).resolve().parents[1]
WIND_DIR = ROOT / "data" / "wind-history"
OUTPUT = ROOT / "data" / "magnetosphere" / "latest.json"
NOAA_DST_URLS = (
    "https://services.swpc.noaa.gov/json/geospace/geospace_dst_1_hour.json",
    "https://services.swpc.noaa.gov/json/geospace/geospace_dst_7_day.json",
    "https://services.swpc.noaa.gov/products/kyoto-dst.json",
)
MAX_INPUT_AGE = timedelta(hours=2)

def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def recent_wind(now: datetime) -> dict:
    points: list[dict] = []
    for day in ((now - timedelta(days=i)).date().isoformat() for i in range(3)):
        path = WIND_DIR / f"{day}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(finite(row.get(key)) is not None for key in ("speed", "density", "by", "bz")):
                points.append(row)
    if not points:
        raise RuntimeError("No complete solar-wind records are available")
    points.sort(key=lambda row: row["t"])
    latest_time = datetime.fromtimestamp(points[-1]["t"] / 1000, timezone.utc)
    if now - latest_time > MAX_INPUT_AGE:
        raise RuntimeError(f"Latest solar-wind input is stale: {latest_time.isoformat()}")

    window_start = points[-1]["t"] - 15 * 60 * 1000
    window = [row for row in points if row["t"] >= window_start]
    result = {key: float(np.median([float(row[key]) for row in window])) for key in ("speed", "density", "by", "bz")}
    result["time"] = latest_time
    result["sample_count"] = len(window)
    return result

def fetch_dst(now: datetime) -> tuple[float, datetime]:
    rows = None
    errors = []

    for url in NOAA_DST_URLS:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "KSWRC-GitHub-Actions/1.0"},
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                rows = json.load(response)

            print(f"Loaded Dst input from {url}")
            break

        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            errors.append(f"{url}: {error}")

    if rows is None:
        raise RuntimeError(
            "All NOAA Dst sources failed: " + "; ".join(errors)
        )

    candidates: list[tuple[datetime, float]] = []

    header = (
        rows[0]
        if rows
        and isinstance(rows[0], list)
        and all(isinstance(item, str) for item in rows[0])
        else None
    )

    data_rows = rows[1:] if header else rows

    for row in data_rows:
        if header and isinstance(row, list):
            row = dict(zip(header, row))

        if isinstance(row, dict):
            time_value = (
                row.get("time_tag")
                or row.get("time")
                or row.get("timestamp")
            )
            dst_value = row.get("dst")

        elif isinstance(row, list) and len(row) >= 2:
            time_value = row[0]
            dst_value = row[1]

        else:
            continue

        try:
            when = parse_time(str(time_value))
            value = finite(dst_value)
        except (TypeError, ValueError):
            continue

        if (
            value is not None
            and when <= now + timedelta(minutes=5)
        ):
            candidates.append((when, value))

    if not candidates:
        raise RuntimeError("No valid Dst value is available")

    when, value = max(candidates, key=lambda item: item[0])

    if now - when > timedelta(hours=4):
        raise RuntimeError(
            f"Latest Dst input is stale: {when.isoformat()}"
        )

    return value, when

def shue_parameters(pdyn: float, bz: float) -> tuple[float, float]:
    # Shue et al. (1998), Earth radii. GSM +X points sunward.
    r0 = (10.22 + 1.29 * math.tanh(0.184 * (bz + 8.14))) * pdyn ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * bz) * (1.0 + 0.024 * math.log(pdyn))
    return r0, alpha


def boundary_radius(point: np.ndarray, r0: float, alpha: float) -> float:
    radius = float(np.linalg.norm(point))
    if radius == 0:
        return r0
    denominator = max(1.0 + float(point[0]) / radius, 0.025)
    return r0 * (2.0 / denominator) ** alpha


def inside_boundary(point: np.ndarray, r0: float, alpha: float) -> bool:
    return float(np.linalg.norm(point)) <= boundary_radius(point, r0, alpha)


def boundary_intersection(a: np.ndarray, b: np.ndarray, r0: float, alpha: float) -> np.ndarray:
    lo, hi = a.copy(), b.copy()
    for _ in range(28):
        mid = (lo + hi) * 0.5
        if inside_boundary(mid, r0, alpha):
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def endpoint_kind(point: np.ndarray) -> str:
    radius = float(np.linalg.norm(point))
    if radius <= 1.06:
        return "earth"
    if radius >= 78.0 or point[0] >= 19.0 or math.hypot(float(point[1]), float(point[2])) >= 39.0:
        return "outer"
    return "truncated"


def combined_trace(seed: np.ndarray, parmod: np.ndarray) -> tuple[np.ndarray, tuple[str, str]]:
    traces = []
    ends = []
    for direction in (-1, 1):
        values = geopack.trace(
            float(seed[0]), float(seed[1]), float(seed[2]), direction,
            rlim=80.0, r0=1.015, parmod=parmod,
            exname="t96", inname="igrf", maxloop=2200,
        )
        endpoint = np.array(values[:3], dtype=float)
        points = np.column_stack(values[3:6]).astype(float)
        if len(points) < 2 or not np.isfinite(points).all():
            raise RuntimeError("Invalid field-line trace")
        traces.append(points)
        ends.append(endpoint_kind(endpoint))
    merged = np.vstack((traces[0][::-1], traces[1][1:]))
    return merged, (ends[0], ends[1])


def clip_open_line(points: np.ndarray, ends: tuple[str, str], r0: float, alpha: float) -> np.ndarray | None:
    if ends[0] == "earth":
        ordered = points
    elif ends[1] == "earth":
        ordered = points[::-1]
    else:
        return None
    kept = [ordered[0]]
    for point in ordered[1:]:
        if inside_boundary(point, r0, alpha):
            kept.append(point)
            continue
        kept.append(boundary_intersection(kept[-1], point, r0, alpha))
        break
    result = np.asarray(kept)
    return result if len(result) >= 3 else None


def resample(points: np.ndarray, maximum: int = 120) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum).round().astype(int)
    return points[indices]


def scene_points(points: np.ndarray) -> list[list[float]]:
    # The existing scene points the Sun toward -X, opposite to standard GSM +X.
    output = np.column_stack((-points[:, 0], points[:, 1], points[:, 2]))
    return [[round(float(value), 4) for value in row] for row in output]


def seed_points() -> list[np.ndarray]:
    seeds = []
    for colatitude in np.linspace(7.0, 80.0, 16):
        theta = math.radians(float(colatitude))
        for longitude in np.linspace(0.0, 360.0, 12, endpoint=False):
            phi = math.radians(float(longitude))
            radius = 1.02
            seeds.append(np.array([
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta),
            ]))
    return seeds


def write_atomic(payload: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, OUTPUT)


def main() -> None:
    now = datetime.now(timezone.utc)
    wind = recent_wind(now)
    dst, dst_time = fetch_dst(now)
    pdyn = max(0.1, min(100.0, 1.94e-6 * wind["density"] * wind["speed"] ** 2))
    by = max(-100.0, min(100.0, wind["by"]))
    bz = max(-100.0, min(100.0, wind["bz"]))
    dst = max(-500.0, min(100.0, dst))
    parmod = np.array([pdyn, dst, by, bz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    epoch = wind["time"].timestamp()
    tilt = float(geopack.recalc(epoch, -wind["speed"], 0.0, 0.0))
    r0, alpha = shue_parameters(pdyn, bz)
    closed: list[list[list[float]]] = []
    open_lines: list[list[list[float]]] = []
    rejected = 0

    for seed_geo in seed_points():
        seed_gsm = np.array(geopack.geogsm(*seed_geo, 1), dtype=float)
        try:
            points, ends = combined_trace(seed_gsm, parmod)
        except (RuntimeError, ValueError, FloatingPointError):
            rejected += 1
            continue
        earth_count = ends.count("earth")
        outer_count = ends.count("outer")
        if earth_count == 2:
            closed.append(scene_points(resample(points)))
        elif earth_count == 1 and outer_count == 1:
            clipped = clip_open_line(points, ends, r0, alpha)
            if clipped is not None:
                open_lines.append(scene_points(resample(clipped)))
            else:
                rejected += 1
        else:
            rejected += 1

    if len(closed) < 4 or len(closed) + len(open_lines) < 8:
        raise RuntimeError(f"Insufficient valid traces: {len(closed)} closed, {len(open_lines)} open")

    payload = {
        "schemaVersion": 1,
        "generatedAt": now.isoformat().replace("+00:00", "Z"),
        "model": {
            "internal": "IGRF-14",
            "external": "Tsyganenko 1996 (T96)",
            "coordinates": "GSM transformed to scene (-X sunward)",
            "trace": "two-way endpoint classification",
            "openLineClip": "Shue et al. 1998 magnetopause",
        },
        "inputs": {
            "solarWindTime": wind["time"].isoformat().replace("+00:00", "Z"),
            "dstTime": dst_time.isoformat().replace("+00:00", "Z"),
            "speedKms": round(wind["speed"], 3),
            "densityCm3": round(wind["density"], 3),
            "dynamicPressureNpa": round(pdyn, 4),
            "dstNt": round(dst, 2),
            "byGsmNt": round(by, 3),
            "bzGsmNt": round(bz, 3),
            "dipoleTiltDeg": round(math.degrees(tilt), 4),
            "solarWindSamples": wind["sample_count"],
        },
        "magnetopause": {"model": "Shue 1998", "r0Re": round(r0, 5), "alpha": round(alpha, 6)},
        "lines": {"closed": closed, "open": open_lines},
        "diagnostics": {"closed": len(closed), "open": len(open_lines), "rejected": rejected},
        "sources": {
            "solarWind": "NOAA SWPC RTSW 1-minute wind and magnetic-field products",
            "dst": "NOAA SWPC Kyoto Dst product",
        },
    }
    write_atomic(payload)
    print(f"Wrote {OUTPUT}: {len(closed)} closed, {len(open_lines)} open, {rejected} rejected")


if __name__ == "__main__":
    main()
