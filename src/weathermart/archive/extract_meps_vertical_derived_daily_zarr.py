#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import xarray as xr

ARCHIVE_ROOT = Path("/lustre/arkivB/projects/metproduction/MEPS")
OUTPUT_ROOT = Path("/lustre/storeB/users/opmir9231/meps_vertical_derived")
GRAVITY = 9.80665

PRESSURE_VARIABLES = {
    "t_pl": "air_temperature_pl",
    "u_pl": "x_wind_pl",
    "v_pl": "y_wind_pl",
    "q_pl": "specific_humidity_pl",
    "rh_pl": "relative_humidity_pl",
    "z_pl": "geopotential_pl",
    "w_pl": "upward_air_velocity_pl",
    "cloud_fraction_pl": "cloud_area_fraction_pl",
    "cloud_water_pl": "mass_fraction_of_cloud_condensed_water_in_air_pl",
    "cloud_ice_pl": "mass_fraction_of_cloud_ice_in_air_pl",
    "rain_water_pl": "mass_fraction_of_rain_in_air_pl",
    "snow_pl": "mass_fraction_of_snow_in_air_pl",
    "graupel_pl": "mass_fraction_of_graupel_in_air_pl",
    "tke_pl": "turbulent_kinetic_energy_pl",
    "pv_pl": "ertel_potential_vorticity_pl",
}

DEFAULT_OUTPUT_VARIABLES = [
    "t_pl",
    "u_pl",
    "v_pl",
    "q_pl",
    "rh_pl",
    "z_pl",
    "w_pl",
    "cloud_fraction_pl",
    "cloud_water_pl",
    "cloud_ice_pl",
    "rain_water_pl",
    "snow_pl",
    "graupel_pl",
    "wind_speed_pl",
    "geopotential_height_pl",
    "dewpoint_pl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MEPS pressure-level variables and derived fields for lead times 0-6h "
            "into daily Zarr files."
        )
    )
    parser.add_argument("--start", required=True, help="First day, YYYY-MM-DD")
    parser.add_argument("--end", help="Last day, YYYY-MM-DD. Defaults to --start")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-hours", default="0,6,12,18")
    parser.add_argument("--lead-hours", default="0,1,2,3,4,5,6")
    parser.add_argument(
        "--pressure-levels",
        default="1000,925,850,800,700,500,400,300,250,200",
        help="Comma-separated pressure levels in hPa. Must exist in the archive file.",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_OUTPUT_VARIABLES),
        help="Comma-separated output variables. Derived choices: wind_speed_pl, geopotential_height_pl, dewpoint_pl.",
    )
    parser.add_argument("--spatial-stride", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
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


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def day_range(start: str, end: str | None) -> list[pd.Timestamp]:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end or start).normalize()
    return list(pd.date_range(start_day, end_day, freq="D"))


def meps_file(archive_root: Path, run_time: pd.Timestamp) -> Path:
    return (
        archive_root
        / run_time.strftime("%Y")
        / run_time.strftime("%m")
        / run_time.strftime("%d")
        / f"meps_det_2_5km_{run_time.strftime('%Y%m%dT%HZ')}.nc"
    )


def open_meps(path: Path) -> xr.Dataset:
    return xr.open_dataset(
        path,
        engine="netcdf4",
        chunks={"time": 1, "pressure": 1, "y": 256, "x": 256},
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
        source = ds.encoding.get("source", "<dataset>")
        raise RuntimeError(f"Missing lead hours {missing} in {source}")
    indices = [available[hour] for hour in lead_hours]
    leads = [pd.Timedelta(hours=hour) for hour in lead_hours]
    return indices, leads


def pressure_indices(ds: xr.Dataset, requested_hpa: list[float]) -> tuple[list[int], list[float]]:
    if "pressure" not in ds.coords and "pressure" not in ds.variables:
        raise KeyError("pressure coordinate not found in MEPS file")
    available = np.asarray(ds["pressure"].values, dtype="float64")
    indices = []
    levels = []
    missing = []
    for level in requested_hpa:
        match = np.where(np.isclose(available, level, atol=1e-6))[0]
        if len(match) == 0:
            missing.append(level)
            continue
        indices.append(int(match[0]))
        levels.append(float(available[match[0]]))
    if missing:
        raise RuntimeError(f"Missing pressure levels {missing}; available levels are {available.tolist()}")
    return indices, levels


def extract_pressure_variable(ds: xr.Dataset, source_name: str, indices: list[int], leads: list[pd.Timedelta], pressure_idx: list[int]) -> xr.DataArray:
    if source_name not in ds:
        raise KeyError(f"{source_name} not found in {ds.encoding.get('source', '<dataset>')}")
    da = ds[source_name].isel(time=indices, pressure=pressure_idx)
    da = da.rename(time="lead_time")
    da = da.assign_coords(lead_time=leads)
    return da


def dewpoint_from_temperature_rh(t_kelvin: xr.DataArray, rh_fraction: xr.DataArray) -> xr.DataArray:
    t_celsius = t_kelvin - 273.15
    rh = rh_fraction.clip(min=1e-6, max=1.0)
    a = 17.625
    b = 243.04
    gamma = np.log(rh) + (a * t_celsius) / (b + t_celsius)
    dewpoint = (b * gamma) / (a - gamma) + 273.15
    dewpoint.attrs.update(
        standard_name="dew_point_temperature",
        long_name="dew point temperature computed from MEPS pressure-level T and RH",
        units="K",
        computed_from="air_temperature_pl, relative_humidity_pl",
    )
    return dewpoint


def extract_run(
    path: Path,
    run_time: pd.Timestamp,
    lead_hours: list[int],
    pressure_levels: list[float],
    output_variables: list[str],
    spatial_stride: int,
) -> xr.Dataset:
    ds = open_meps(path)
    indices, leads = lead_indices(ds, run_time, lead_hours)
    pressure_idx, pressure_values = pressure_indices(ds, pressure_levels)
    if spatial_stride > 1:
        ds = ds.isel(y=slice(None, None, spatial_stride), x=slice(None, None, spatial_stride))

    out = xr.Dataset(
        coords={
            "lead_time": ("lead_time", leads),
            "pressure": ("pressure", pressure_values),
            "y": ds["y"],
            "x": ds["x"],
            "latitude": (("y", "x"), ds["latitude"].data),
            "longitude": (("y", "x"), ds["longitude"].data),
        }
    )

    missing = []
    direct_names = [name for name in output_variables if name in PRESSURE_VARIABLES]
    for output_name in direct_names:
        source_name = PRESSURE_VARIABLES[output_name]
        try:
            da = extract_pressure_variable(ds, source_name, indices, leads, pressure_idx)
        except KeyError as exc:
            missing.append(str(exc))
            continue
        da.name = output_name
        da.attrs = dict(ds[source_name].attrs)
        da.attrs["source_variable"] = source_name
        out[output_name] = da

    if "wind_speed_pl" in output_variables:
        if "u_pl" not in out:
            out["u_pl"] = extract_pressure_variable(ds, PRESSURE_VARIABLES["u_pl"], indices, leads, pressure_idx)
        if "v_pl" not in out:
            out["v_pl"] = extract_pressure_variable(ds, PRESSURE_VARIABLES["v_pl"], indices, leads, pressure_idx)
        out["wind_speed_pl"] = np.hypot(out["u_pl"], out["v_pl"])
        out["wind_speed_pl"].attrs.update(
            standard_name="wind_speed",
            units="m/s",
            computed_from="x_wind_pl, y_wind_pl",
        )

    if "geopotential_height_pl" in output_variables:
        if "z_pl" not in out:
            out["z_pl"] = extract_pressure_variable(ds, PRESSURE_VARIABLES["z_pl"], indices, leads, pressure_idx)
        out["geopotential_height_pl"] = out["z_pl"] / GRAVITY
        out["geopotential_height_pl"].attrs.update(
            standard_name="geopotential_height",
            units="m",
            computed_from="geopotential_pl",
        )

    if "dewpoint_pl" in output_variables:
        if "t_pl" not in out:
            out["t_pl"] = extract_pressure_variable(ds, PRESSURE_VARIABLES["t_pl"], indices, leads, pressure_idx)
        if "rh_pl" not in out:
            out["rh_pl"] = extract_pressure_variable(ds, PRESSURE_VARIABLES["rh_pl"], indices, leads, pressure_idx)
        out["dewpoint_pl"] = dewpoint_from_temperature_rh(out["t_pl"], out["rh_pl"])

    unknown = sorted(set(output_variables) - set(PRESSURE_VARIABLES) - {"wind_speed_pl", "geopotential_height_pl", "dewpoint_pl"})
    if unknown:
        missing.append(f"Unknown requested variables: {unknown}")
    if missing:
        raise RuntimeError("; ".join(missing))

    keep = [name for name in output_variables if name in out]
    out = out[keep]
    out = out.expand_dims(forecast_reference_time=[np.datetime64(run_time.to_datetime64())])
    return out


def process_day(day: pd.Timestamp, args: argparse.Namespace, run_hours: list[int], lead_hours: list[int], pressure_levels: list[float], output_variables: list[str]) -> None:
    out_path = args.output_root / day.strftime("%Y%m%d")
    tmp_path = out_path.with_name(out_path.name + ".partial")
    if out_path.exists() and not args.overwrite:
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
        path = meps_file(args.archive_root, run_time)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[READ] {path}", flush=True)
        run_datasets.append(
            extract_run(
                path,
                run_time=run_time,
                lead_hours=lead_hours,
                pressure_levels=pressure_levels,
                output_variables=output_variables,
                spatial_stride=args.spatial_stride,
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
        pressure_levels_hpa=",".join(f"{level:g}" for level in pressure_levels),
        spatial_stride=args.spatial_stride,
        notes="Pressure-level fields are read from *_pl variables. wind_speed_pl, geopotential_height_pl, and dewpoint_pl are derived when requested.",
    )
    daily = daily.chunk({"forecast_reference_time": 1, "lead_time": 1, "pressure": 1, "y": 256, "x": 256})
    encoding = {
        name: {"dtype": "float32"}
        for name in daily.data_vars
        if np.issubdtype(daily[name].dtype, np.floating)
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[WRITE] {tmp_path}", flush=True)
    daily.to_zarr(tmp_path, mode="w", consolidated=False, encoding=encoding)
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
    pressure_levels = parse_float_list(args.pressure_levels)
    output_variables = parse_str_list(args.variables)
    for day in days:
        try:
            process_day(day, args, run_hours, lead_hours, pressure_levels, output_variables)
        except Exception as exc:
            print(f"[ERROR] {day:%Y-%m-%d}: {exc}", flush=True)
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
