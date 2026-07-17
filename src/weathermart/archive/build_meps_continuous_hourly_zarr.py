#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-meps-continuous")

import numpy as np
import pandas as pd
import xarray as xr

ARCHIVE_ROOT = Path("/lustre/storeB/users/opmir9231/meps_fcst")
DEFAULT_OUTPUT = Path("/tmp/meps_continuous_hourly_smoke.zarr")
PRECIP_VARIABLES = {"tp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a valid-time, continuous hourly xarray-zarr view from daily MEPS "
            "forecast zarr stores. Precipitation accumulations are converted to "
            "one-hour mm h-1 increments before latest-run selection."
        )
    )
    parser.add_argument("--start", required=True, help="First valid time, e.g. 2020-04-15T01:00")
    parser.add_argument("--end", required=True, help="Last valid time, inclusive")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variables", default="tp", help="Comma-separated variable names from the MEPS zarr archive.")
    parser.add_argument("--y-start", type=int, default=0)
    parser.add_argument("--y-size", type=int, default=0, help="Optional y crop size. 0 keeps all y points.")
    parser.add_argument("--x-start", type=int, default=0)
    parser.add_argument("--x-size", type=int, default=0, help="Optional x crop size. 0 keeps all x points.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_variables(value: str) -> list[str]:
    variables = [item.strip() for item in value.split(",") if item.strip()]
    if not variables:
        raise ValueError("At least one variable is required.")
    return variables


def daily_paths(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    # Include the previous day so cycle-time precipitation can use the preceding run.
    first_day = (start - pd.Timedelta(hours=6)).normalize()
    last_day = end.normalize()
    paths = [root / day.strftime("%Y%m%d") for day in pd.date_range(first_day, last_day, freq="D")]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No MEPS zarr days found in {root} for {first_day:%Y-%m-%d}..{last_day:%Y-%m-%d}")
    return existing


def open_archive(
    paths: list[Path],
    variables: list[str],
    y_start: int,
    y_size: int,
    x_start: int,
    x_size: int,
) -> xr.Dataset:
    datasets = []
    for path in paths:
        ds = xr.open_zarr(path, consolidated=False)
        missing = sorted(set(variables).difference(ds.data_vars))
        if missing:
            raise KeyError(f"{path} is missing variable(s): {missing}")
        keep = list(dict.fromkeys([*variables, "latitude", "longitude"]))
        ds = ds[[name for name in keep if name in ds]]
        indexers = {}
        if y_size > 0:
            indexers["y"] = slice(y_start, y_start + y_size)
        if x_size > 0:
            indexers["x"] = slice(x_start, x_start + x_size)
        if indexers:
            ds = ds.isel(indexers)
        datasets.append(ds)
    return xr.concat(datasets, dim="forecast_reference_time", coords="minimal", compat="override").sortby(
        "forecast_reference_time"
    )


def latest_index_for_valid_time(
    forecast_reference_times: pd.DatetimeIndex,
    lead_times: pd.TimedeltaIndex,
    valid_time: pd.Timestamp,
    *,
    require_positive_lead: bool,
) -> tuple[int, int]:
    candidates = []
    for forecast_index, forecast_reference_time in enumerate(forecast_reference_times):
        lead_time = valid_time - forecast_reference_time
        if require_positive_lead and lead_time <= pd.Timedelta(0):
            continue
        matches = np.where(lead_times == lead_time)[0]
        if len(matches):
            candidates.append((forecast_reference_time, forecast_index, int(matches[0])))
    if not candidates:
        raise KeyError(f"No forecast candidate found for valid time {valid_time}")
    _, forecast_index, lead_index = max(candidates, key=lambda item: item[0])
    return forecast_index, lead_index


def select_latest_by_valid_time(
    da: xr.DataArray,
    valid_times: pd.DatetimeIndex,
    *,
    require_positive_lead: bool = False,
) -> xr.DataArray:
    forecast_reference_times = pd.DatetimeIndex(pd.to_datetime(da.forecast_reference_time.values))
    lead_times = pd.TimedeltaIndex(pd.to_timedelta(da.lead_time.values))
    pieces = []
    selected_forecast_reference_times = []
    selected_lead_times = []
    for valid_time in valid_times:
        forecast_index, lead_index = latest_index_for_valid_time(
            forecast_reference_times,
            lead_times,
            pd.Timestamp(valid_time),
            require_positive_lead=require_positive_lead,
        )
        pieces.append(da.isel(forecast_reference_time=forecast_index, lead_time=lead_index, drop=True))
        selected_forecast_reference_times.append(forecast_reference_times[forecast_index].to_datetime64())
        selected_lead_times.append(lead_times[lead_index].to_timedelta64())

    out = xr.concat(pieces, dim=pd.Index(valid_times, name="time"))
    out = out.assign_coords(
        forecast_reference_time=("time", np.array(selected_forecast_reference_times, dtype="datetime64[ns]")),
        lead_time=("time", np.array(selected_lead_times, dtype="timedelta64[ns]")),
    )
    out["time"].attrs.update(standard_name="time", long_name="valid time")
    out["forecast_reference_time"].attrs.update(standard_name="forecast_reference_time")
    out["lead_time"].attrs.update(standard_name="forecast_period", long_name="time elapsed since the start of the forecast")
    return out


def hourly_precipitation_rate(tp: xr.DataArray) -> xr.DataArray:
    hourly = tp.diff("lead_time", label="upper")
    hourly = hourly.clip(min=0)
    hourly.name = tp.name
    hourly.attrs = dict(tp.attrs)
    hourly.attrs.update(
        units="mm h-1",
        long_name="Hourly precipitation rate from accumulated MEPS precipitation",
        standard_name="precipitation_flux",
        accumulation_period="1h",
        source_variable=tp.attrs.get("source_variable", "precipitation_amount_acc"),
    )
    return hourly


def build_dataset(
    ds: xr.Dataset,
    variables: list[str],
    valid_times: pd.DatetimeIndex,
    archive_root: Path,
) -> xr.Dataset:
    data_vars = {}
    for variable in variables:
        da = ds[variable]
        if variable in PRECIP_VARIABLES:
            da = hourly_precipitation_rate(da)
            selected = select_latest_by_valid_time(da, valid_times, require_positive_lead=True)
        else:
            selected = select_latest_by_valid_time(da, valid_times)
        data_vars[variable] = selected.drop_vars(["forecast_reference_time", "lead_time"], errors="ignore")

    out = xr.Dataset(data_vars)
    out = out.assign_coords(
        y=ds["y"],
        x=ds["x"],
        latitude=(("y", "x"), ds["latitude"].data),
        longitude=(("y", "x"), ds["longitude"].data),
    )
    out["latitude"].attrs.update(standard_name="latitude", units="degrees_north")
    out["longitude"].attrs.update(standard_name="longitude", units="degrees_east")
    out.attrs.update(
        source="MEPS deterministic forecast archive continuous hourly latest-run view",
        archive_root=str(archive_root),
        selection="latest forecast_reference_time per valid time; precipitation uses latest available 1h interval",
    )
    return out.transpose("time", "y", "x")


def main() -> None:
    args = parse_args()
    variables = parse_variables(args.variables)
    start = pd.Timestamp(args.start).tz_localize(None)
    end = pd.Timestamp(args.end).tz_localize(None)
    valid_times = pd.DatetimeIndex(pd.date_range(start, end, freq="1h"))
    if len(valid_times) == 0:
        raise ValueError(f"No hourly valid times for {start} -> {end}")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists. Use --overwrite.")
        shutil.rmtree(args.output)

    paths = daily_paths(args.archive_root, start, end)
    archive = open_archive(paths, variables, args.y_start, args.y_size, args.x_start, args.x_size)
    out = build_dataset(archive, variables, valid_times, args.archive_root)
    out = out.chunk({"time": 1, "y": min(out.sizes["y"], 256), "x": min(out.sizes["x"], 256)})
    encoding = {
        name: {"dtype": "float32"}
        for name in out.data_vars
        if np.issubdtype(out[name].dtype, np.floating)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_zarr(args.output, mode="w", consolidated=False, encoding=encoding)
    print(f"Wrote {args.output} with variables={variables} times={valid_times[0]}..{valid_times[-1]}")


if __name__ == "__main__":
    main()
