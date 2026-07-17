#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rainbow-beam-height")

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import xradar.io as xradar_io
import zarr

from weathermart.archive.extract_rainbow_column_products_daily_zarr import ARCHIVE_ROOT
from weathermart.archive.extract_rainbow_column_products_daily_zarr import OUTPUT_ROOT
from weathermart.archive.extract_rainbow_column_products_daily_zarr import beam_height_m
from weathermart.archive.extract_rainbow_column_products_daily_zarr import (
    choose_scan_dirs,
)
from weathermart.archive.extract_rainbow_column_products_daily_zarr import (
    files_by_rounded_time,
)

DAY_RE = re.compile(r"^\d{8}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append lowest-elevation beam_height(time, cell) to existing Rainbow daily zarrs."
    )
    parser.add_argument("--start", default=None, help="First day, YYYY-MM-DD. Defaults to first existing zarr day.")
    parser.add_argument("--end", default=None, help="Last day, YYYY-MM-DD. Defaults to last existing zarr day.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--scan-patterns", default="*12ele*.vol,*09ele*.vol,*07ele*.vol")
    parser.add_argument("--round-time", default="5min")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def existing_days(output_root: Path, start: str | None, end: str | None) -> list[pd.Timestamp]:
    names = sorted(path.name for path in output_root.iterdir() if path.is_dir() and DAY_RE.match(path.name))
    days = [pd.Timestamp(name) for name in names]
    if start is not None:
        days = [day for day in days if day >= pd.Timestamp(start).normalize()]
    if end is not None:
        days = [day for day in days if day <= pd.Timestamp(end).normalize()]
    return days


def first_sweep_fixed_angle(path: Path) -> tuple[float, float]:
    datatree = xradar_io.open_rainbow_datatree(str(path))
    root = datatree.ds
    radar_altitude = float(root["altitude"].values)
    for child_name in sorted(name for name in datatree.children if name.startswith("sweep_")):
        ds = datatree[child_name].ds
        return float(ds["sweep_fixed_angle"].values), radar_altitude
    raise RuntimeError(f"No sweeps found in {path}")


def fixed_angles_for_day(
    day: pd.Timestamp,
    radars: np.ndarray,
    archive_root: Path,
    scan_patterns: str,
    round_time: str,
) -> dict[str, tuple[float, float]]:
    day_root = archive_root / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d") / "rainbow5"
    fixed_angles: dict[str, tuple[float, float]] = {}
    for radar in sorted(str(radar) for radar in np.unique(radars)):
        scan_dirs = choose_scan_dirs(day_root / radar, scan_patterns)
        files = files_by_rounded_time(scan_dirs, round_time, max_times=1)
        if not files:
            print(f"[WARN] {day:%Y-%m-%d} {radar}: no source dBZ volume found", flush=True)
            continue
        _, path = files[0]
        fixed_angles[radar] = first_sweep_fixed_angle(path)
    return fixed_angles


def add_beam_height(day: pd.Timestamp, args: argparse.Namespace) -> None:
    path = args.output_root / day.strftime("%Y%m%d")
    if not path.exists():
        print(f"[SKIP] {path} missing", flush=True)
        return

    group = zarr.open_group(str(path), mode="a")
    if "beam_height" in group and not args.overwrite:
        print(f"[SKIP] {path} already has beam_height", flush=True)
        return

    ds = xr.open_zarr(path, consolidated=False)
    required = {"radar", "range", "time"}
    missing = sorted(required - set(ds.variables))
    if missing:
        raise KeyError(f"{path} is missing required variables/coords for beam_height: {missing}")

    radars = np.asarray(ds["radar"].values).astype(str)
    ranges = np.asarray(ds["range"].values, dtype=np.float32)
    if "radar_altitude" in ds:
        radar_altitudes = np.asarray(ds["radar_altitude"].values, dtype=np.float32)
    else:
        radar_altitudes = np.full(ds.sizes["cell"], np.nan, dtype=np.float32)

    fixed_angles = fixed_angles_for_day(day, radars, args.archive_root, args.scan_patterns, args.round_time)
    beam_by_cell = np.full(ds.sizes["cell"], np.nan, dtype=np.float32)
    for radar, (fixed_angle, source_altitude) in fixed_angles.items():
        mask = radars == radar
        altitude = radar_altitudes[mask]
        if not np.isfinite(altitude).any():
            altitude = np.full(mask.sum(), source_altitude, dtype=np.float32)
        beam_by_cell[mask] = beam_height_m(ranges[mask], fixed_angle, altitude)
        print(
            f"[INFO] {day:%Y-%m-%d} {radar}: fixed_angle={fixed_angle:.3f} cells={int(mask.sum())}",
            flush=True,
        )

    if args.dry_run:
        finite = np.isfinite(beam_by_cell)
        print(
            f"[DRY] {path}: finite={int(finite.sum())}/{beam_by_cell.size} "
            f"min={float(np.nanmin(beam_by_cell)):.1f} max={float(np.nanmax(beam_by_cell)):.1f}",
            flush=True,
        )
        return

    if "beam_height" in group:
        del group["beam_height"]

    chunks = (min(20000, ds.sizes["cell"]), 1)
    cell_values = da.from_array(beam_by_cell[:, None], chunks=chunks)
    values = da.broadcast_to(cell_values, (ds.sizes["cell"], ds.sizes["time"]))
    beam = xr.DataArray(
        values,
        dims=("cell", "time"),
        coords={"cell": ds["cell"], "time": ds["time"]},
        name="beam_height",
        attrs={
            "long_name": "Lowest-elevation beam height above mean sea level",
            "units": "m",
            "method": "Computed from range, first sweep fixed angle, radar altitude, and an effective Earth radius model.",
        },
    )
    beam.to_dataset().to_zarr(
        path,
        mode="a",
        consolidated=False,
        encoding={"beam_height": {"dtype": "float32", "chunks": chunks}},
    )
    print(f"[DONE] {path}: wrote beam_height shape={beam.shape} chunks={chunks}", flush=True)


def main() -> None:
    args = parse_args()
    days = existing_days(args.output_root, args.start, args.end)
    if not days:
        raise RuntimeError(f"No existing YYYYMMDD zarr directories found in {args.output_root}")
    print(f"[INFO] Updating {len(days)} day(s) in {args.output_root}", flush=True)
    for day in days:
        add_beam_height(day, args)


if __name__ == "__main__":
    main()
