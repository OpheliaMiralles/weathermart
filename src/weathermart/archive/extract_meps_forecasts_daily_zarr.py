#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-meps")

import numpy as np
import pandas as pd
import xarray as xr

ARCHIVE_ROOT = Path("/lustre/arkivB/projects/metproduction/MEPS")
OUTPUT_ROOT = Path("/lustre/storeB/users/opmir9231/meps_fcst")

VARIABLES = {
    "cape": "specific_convective_available_potential_energy",
    "cin": "atmosphere_convective_inhibition",
    "wind_gust_10m": "wind_speed_of_gust",
    "x_wind_gust_10m": "x_wind_gust_10m",
    "y_wind_gust_10m": "y_wind_gust_10m",
    "u10": "x_wind_10m",
    "v10": "y_wind_10m",
    "t2m": "air_temperature_2m",
    "rh2m": "relative_humidity_2m",
    "z": "surface_geopotential",
    "tp": "precipitation_amount_acc",
    "lsm": "land_area_fraction",
    "mcc": "medium_type_cloud_area_fraction",
    "msl": "air_pressure_at_sea_level",
    "lcc": "low_type_cloud_area_fraction",
}

OUTPUT_ORDER = [
    "cape",
    "cin",
    "wind_gust_10m",
    "u10",
    "v10",
    "t2m",
    "d2m",
    "z",
    "tp",
    "lsm",
    "mcc",
    "msl",
    "lcc",
]

INTERNAL_ORDER = OUTPUT_ORDER + ["rh2m", "x_wind_gust_10m", "y_wind_gust_10m"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MEPS deterministic forecast lead times 0-6h into daily Zarr files."
    )
    parser.add_argument("--start", required=True, help="First day, YYYY-MM-DD")
    parser.add_argument("--end", help="Last day, YYYY-MM-DD. Defaults to --start")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-hours", default="0,6,12,18")
    parser.add_argument("--lead-hours", default="0,1,2,3,4,5,6")
    parser.add_argument("--spatial-stride", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append-missing",
        action="store_true",
        help="For existing daily zarr stores, append only requested variables that are absent.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="For existing daily zarr stores, replace requested variables while preserving other variables.",
    )
    parser.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="Skip missing run hours within a day instead of failing the whole daily zarr.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(OUTPUT_ORDER),
        help=(
            "Comma-separated output variable names to extract. "
            "Use this with --append-missing to add variables without rewriting existing data."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="0-based day index for SGE/SLURM arrays. If set, only this day in the date range is processed.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_variable_list(value: str) -> list[str]:
    variables = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variables).difference(OUTPUT_ORDER))
    if unknown:
        raise ValueError(f"Unknown output variable(s): {unknown}. Available: {OUTPUT_ORDER}")
    return variables


def day_range(start: str, end: str | None) -> list[pd.Timestamp]:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end or start).normalize()
    return list(pd.date_range(start_day, end_day, freq="D"))


def legacy_meps_file(archive_root: Path, run_time: pd.Timestamp) -> Path:
    return (
        archive_root
        / run_time.strftime("%Y")
        / run_time.strftime("%m")
        / run_time.strftime("%d")
        / f"meps_det_2_5km_{run_time.strftime('%Y%m%dT%HZ')}.nc"
    )


def split_sfc_meps_files(
    archive_root: Path,
    run_time: pd.Timestamp,
    lead_hours: list[int],
) -> list[Path]:
    run_dir = (
        archive_root
        / run_time.strftime("%Y")
        / run_time.strftime("%m")
        / run_time.strftime("%d")
        / run_time.strftime("%H")
        / "member_00"
    )
    return [
        run_dir / f"meps_sfc_{lead_hour:02d}_{run_time.strftime('%Y%m%dT%HZ')}.nc"
        for lead_hour in lead_hours
    ]


def meps_files(
    archive_root: Path,
    run_time: pd.Timestamp,
    lead_hours: list[int],
) -> list[Path]:
    legacy_path = legacy_meps_file(archive_root, run_time)
    if legacy_path.exists():
        return [legacy_path]

    split_paths = split_sfc_meps_files(archive_root, run_time, lead_hours)
    missing = [path for path in split_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return split_paths


def source_label(paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return str(paths[0])
    return f"{paths[0]} ... {paths[-1]}"


def open_meps(paths: Sequence[Path]) -> xr.Dataset:
    if len(paths) > 1:
        return xr.open_mfdataset(
            paths,
            engine="netcdf4",
            combine="nested",
            concat_dim="time",
            chunks={"time": 1, "y": 256, "x": 256},
            coords="minimal",
            compat="override",
            data_vars="minimal",
        )
    return xr.open_dataset(
        paths[0],
        engine="netcdf4",
        chunks={"time": 1, "y": 256, "x": 256},
    )


def lead_indices(ds: xr.Dataset, run_time: pd.Timestamp, lead_hours: list[int]) -> tuple[list[int], list[pd.Timedelta]]:
    valid_times = pd.to_datetime(ds["time"].values)
    run_time = pd.Timestamp(run_time).tz_localize(None)
    available = {
        int(round((pd.Timestamp(valid_time) - run_time).total_seconds() / 3600)): idx
        for idx, valid_time in enumerate(valid_times)
    }
    missing = [hour for hour in lead_hours if hour not in available]
    if missing:
        raise RuntimeError(f"Missing lead hours {missing} in {ds.encoding.get('source', '<dataset>')}")
    indices = [available[hour] for hour in lead_hours]
    leads = [pd.Timedelta(hours=hour) for hour in lead_hours]
    return indices, leads


def squeeze_to_time_y_x(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in {"time", "y", "x"}:
            continue
        if da.sizes[dim] != 1:
            raise ValueError(f"{da.name} has non-singleton vertical dimension {dim}={da.sizes[dim]}")
        da = da.isel({dim: 0}, drop=True)
    return da


def dewpoint_from_temperature_rh(t_kelvin: xr.DataArray, rh_fraction: xr.DataArray) -> xr.DataArray:
    # Magnus formula over water. MEPS RH2M is stored as a fraction, not percent.
    t_celsius = t_kelvin - 273.15
    rh = rh_fraction.clip(min=1e-6, max=1.0)
    a = 17.625
    b = 243.04
    gamma = np.log(rh) + (a * t_celsius) / (b + t_celsius)
    d2m = (b * gamma) / (a - gamma) + 273.15
    d2m.attrs.update(
        standard_name="dew_point_temperature",
        long_name="2 metre dew point temperature computed from MEPS T2M and RH2M",
        units="K",
        computed_from="air_temperature_2m, relative_humidity_2m",
    )
    return d2m


def wind_speed_from_components(x_wind: xr.DataArray, y_wind: xr.DataArray) -> xr.DataArray:
    wind_speed = np.hypot(x_wind, y_wind)
    wind_speed.attrs = {
        "standard_name": "wind_speed",
        "long_name": "10 metre wind gust speed computed from x/y gust components",
        "units": x_wind.attrs.get("units", "m s-1"),
        "computed_from": "x_wind_gust_10m, y_wind_gust_10m",
    }
    return wind_speed


def extract_run(
    paths: Sequence[Path],
    run_time: pd.Timestamp,
    lead_hours: list[int],
    spatial_stride: int,
    output_variables: list[str],
) -> xr.Dataset:
    ds = open_meps(paths)
    indices, leads = lead_indices(ds, run_time, lead_hours)
    if spatial_stride > 1:
        ds = ds.isel(y=slice(None, None, spatial_stride), x=slice(None, None, spatial_stride))

    out = xr.Dataset(
        coords={
            "lead_time": ("lead_time", leads),
            "y": ds["y"],
            "x": ds["x"],
            "latitude": (("y", "x"), ds["latitude"].data),
            "longitude": (("y", "x"), ds["longitude"].data),
        }
    )

    direct_variables = [
        name for name in output_variables if name not in {"d2m", "wind_gust_10m"}
    ]
    if "d2m" in output_variables:
        direct_variables.extend(["t2m", "rh2m"])
    if "wind_gust_10m" in output_variables:
        direct_variables.extend(["x_wind_gust_10m", "y_wind_gust_10m"])
    direct_variable_set = set(direct_variables)
    direct_variables = [name for name in INTERNAL_ORDER if name in direct_variable_set]

    for output_name in direct_variables:
        input_name = VARIABLES[output_name]
        if input_name not in ds:
            raise KeyError(f"{input_name} not found in {source_label(paths)}")
        da = squeeze_to_time_y_x(ds[input_name])
        da = da.isel(time=indices).rename(time="lead_time")
        da = da.assign_coords(lead_time=leads)
        da.name = output_name
        da.attrs = dict(ds[input_name].attrs)
        da.attrs["source_variable"] = input_name
        out[output_name] = da

    if "d2m" in output_variables:
        out["d2m"] = dewpoint_from_temperature_rh(out["t2m"], out["rh2m"])
        if "rh2m" not in output_variables:
            out = out.drop_vars("rh2m")
    if "wind_gust_10m" in output_variables:
        out["wind_gust_10m"] = wind_speed_from_components(
            out["x_wind_gust_10m"], out["y_wind_gust_10m"]
        )
        out = out.drop_vars(["x_wind_gust_10m", "y_wind_gust_10m"])
    out = out[[name for name in OUTPUT_ORDER if name in out.data_vars]]
    out = out.expand_dims(forecast_reference_time=[np.datetime64(run_time.to_datetime64())])
    return out


def existing_data_vars(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with xr.open_zarr(path, consolidated=False) as ds:
        return set(ds.data_vars)


def write_daily_zarr(daily: xr.Dataset, path: Path, *, mode: str) -> None:
    daily = daily.chunk({"forecast_reference_time": 1, "lead_time": 1, "y": 256, "x": 256})
    encoding = {
        name: {"dtype": "float32"}
        for name in daily.data_vars
        if np.issubdtype(daily[name].dtype, np.floating)
    }
    print(f"[WRITE] {path} mode={mode}", flush=True)
    daily.to_zarr(path, mode=mode, consolidated=False, encoding=encoding)


def replace_existing_data_vars(daily: xr.Dataset, path: Path) -> None:
    for name in daily.data_vars:
        var_path = path / name
        if var_path.exists():
            shutil.rmtree(var_path)
    write_daily_zarr(daily, path, mode="a")


def process_day(
    day: pd.Timestamp,
    args: argparse.Namespace,
    run_hours: list[int],
    lead_hours: list[int],
    output_variables: list[str],
) -> None:
    out_path = args.output_root / day.strftime("%Y%m%d")
    tmp_path = out_path.with_name(out_path.name + ".partial")
    existing_variables = existing_data_vars(out_path)
    variables_to_extract = output_variables
    if out_path.exists() and args.append_missing and not args.overwrite:
        variables_to_extract = [name for name in output_variables if name not in existing_variables]
        if not variables_to_extract:
            print(f"[SKIP] {out_path} already has {','.join(output_variables)}", flush=True)
            return
        print(f"[APPEND] {out_path}: {','.join(variables_to_extract)}", flush=True)
    elif out_path.exists() and args.replace_existing and not args.overwrite:
        variables_to_extract = output_variables
        print(f"[REPLACE] {out_path}: {','.join(variables_to_extract)}", flush=True)
    elif out_path.exists() and not args.overwrite:
        print(f"[SKIP] {out_path} exists", flush=True)
        return
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if out_path.exists() and args.overwrite:
        shutil.rmtree(out_path)

    run_times = [day + pd.Timedelta(hours=hour) for hour in run_hours]
    if args.max_runs > 0:
        run_times = run_times[: args.max_runs]

    run_datasets = []
    for run_time in run_times:
        try:
            paths = meps_files(args.archive_root, run_time, lead_hours)
        except FileNotFoundError as exc:
            if args.allow_missing_runs:
                print(f"[MISSING_RUN] {run_time:%Y-%m-%dT%HZ}: {exc}", flush=True)
                continue
            raise
        print(f"[READ] {source_label(paths)}", flush=True)
        run_datasets.append(
            extract_run(
                paths,
                run_time=run_time,
                lead_hours=lead_hours,
                spatial_stride=args.spatial_stride,
                output_variables=variables_to_extract,
            )
        )

    if not run_datasets:
        raise RuntimeError(f"No MEPS runs found for {day:%Y-%m-%d}")

    daily = xr.concat(run_datasets, dim="forecast_reference_time")
    daily = daily.assign_attrs(
        source="MEPS deterministic archive",
        archive_root=str(args.archive_root),
        lead_hours=",".join(str(hour) for hour in lead_hours),
        run_hours=",".join(str(hour) for hour in run_hours),
        spatial_stride=args.spatial_stride,
        notes="tp is accumulated total precipitation from the model run; d2m is computed from t2m and rh2m when requested.",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and args.append_missing and not args.overwrite:
        write_daily_zarr(daily, out_path, mode="a")
    elif out_path.exists() and args.replace_existing and not args.overwrite:
        replace_existing_data_vars(daily, out_path)
    else:
        write_daily_zarr(daily, tmp_path, mode="w")
        tmp_path.rename(out_path)
    print(f"[DONE] {day:%Y-%m-%d} -> {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    days = day_range(args.start, args.end)
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(days):
            raise IndexError(f"task-index {args.task_index} outside 0..{len(days) - 1}")
        days = [days[args.task_index]]

    run_hours = parse_int_list(args.run_hours)
    lead_hours = parse_int_list(args.lead_hours)
    output_variables = parse_variable_list(args.variables)
    for day in days:
        try:
            process_day(day, args, run_hours, lead_hours, output_variables)
        except Exception as exc:
            print(f"[ERROR] {day:%Y-%m-%d}: {exc}", flush=True)
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
