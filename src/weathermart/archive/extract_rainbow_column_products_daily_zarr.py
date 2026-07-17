#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rainbow-products")

import numpy as np
import pandas as pd
import xarray as xr
import xradar.io as xradar_io
from pyproj import Geod

ARCHIVE_ROOT = Path("/lustre/arkivB/projects/remotesensing/radar-volumes-rainbow")
NORDIC_RADAR_ROOT = Path("/lustre/storeB/users/opmir9231/nordic_radar")
OUTPUT_ROOT = Path("/lustre/storeB/users/opmir9231/rainbow_column_products")
TIME_RE = re.compile(r"^(\d{14})")
DEFAULT_PRODUCTS = ("czc", "ezc20", "ezc45", "lzc", "hzc")
ALL_PRODUCTS = ("rzc", "czc", "ezc20", "ezc45", "lzc", "hzc", "beam_height", "radar_altitude")
GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class TargetGrid:
    azimuth: np.ndarray
    range: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive RZC/CZC/EZC/LZC/HZC-like column products from Rainbow polar "
            "reflectivity volumes and store daily Zarr files."
        )
    )
    parser.add_argument("--start", default=None, help="First day, YYYY-MM-DD. Defaults to first nordic_radar day.")
    parser.add_argument("--end", default=None, help="Last day, YYYY-MM-DD. Defaults to last nordic_radar day.")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--nordic-radar-root", type=Path, default=NORDIC_RADAR_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--radars", default="all", help="Comma-separated radar IDs or 'all'.")
    parser.add_argument(
        "--products",
        default=",".join(DEFAULT_PRODUCTS),
        help="Comma-separated products to write. Available: rzc,czc,ezc20,ezc45,lzc,hzc,beam_height,radar_altitude.",
    )
    parser.add_argument("--scan-patterns", default="*12ele*.vol,*09ele*.vol,*07ele*.vol")
    parser.add_argument("--azimuth-step-deg", type=float, default=1.0)
    parser.add_argument("--range-step-m", type=float, default=1000.0)
    parser.add_argument("--max-range-m", type=float, default=240000.0)
    parser.add_argument("--z-r-a", type=float, default=200.0, help="Marshall-Palmer Z=aR^b coefficient.")
    parser.add_argument("--z-r-b", type=float, default=1.6, help="Marshall-Palmer Z=aR^b exponent.")
    parser.add_argument("--min-vpr-dbz", type=float, default=5.0)
    parser.add_argument("--vpr-bin-m", type=float, default=500.0)
    parser.add_argument("--vpr-reference-height-m", type=float, default=1000.0)
    parser.add_argument("--max-vpr-correction-db", type=float, default=10.0)
    parser.add_argument("--round-time", default="5min")
    parser.add_argument("--task-index-month", type=int, default=None, help="0-based month index in the selected range.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--max-times-per-radar", type=int, default=0)
    return parser.parse_args()


def parse_products(value: str) -> tuple[str, ...]:
    products = tuple(item.strip().lower() for item in re.split(r"[,\s]+", value) if item.strip())
    unknown = sorted(set(products) - set(ALL_PRODUCTS))
    if unknown:
        raise ValueError(f"Unknown product(s): {','.join(unknown)}. Available: {','.join(ALL_PRODUCTS)}")
    if not products:
        raise ValueError("At least one product must be requested")
    return products


def nordic_archive_bounds(root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    days = sorted(p.name for p in root.iterdir() if p.is_dir() and re.match(r"^\d{8}$", p.name))
    if not days:
        raise RuntimeError(f"No YYYYMMDD directories found in {root}")
    return pd.Timestamp(days[0]), pd.Timestamp(days[-1])


def selected_days(args: argparse.Namespace) -> list[pd.Timestamp]:
    default_start, default_end = nordic_archive_bounds(args.nordic_radar_root)
    start = pd.Timestamp(args.start).normalize() if args.start else default_start
    end = pd.Timestamp(args.end).normalize() if args.end else default_end
    if args.task_index_month is not None:
        months = list(pd.period_range(start, end, freq="M"))
        if args.task_index_month < 0 or args.task_index_month >= len(months):
            raise IndexError(f"task-index-month {args.task_index_month} outside 0..{len(months) - 1}")
        month = months[args.task_index_month]
        month_start = max(start, month.start_time.normalize())
        month_end = min(end, month.end_time.normalize())
        days = list(pd.date_range(month_start, month_end, freq="D"))
    else:
        days = list(pd.date_range(start, end, freq="D"))
    if args.max_days > 0:
        days = days[: args.max_days]
    return days


def make_target_grid(args: argparse.Namespace) -> TargetGrid:
    azimuth = np.arange(0.0, 360.0, args.azimuth_step_deg, dtype=np.float32)
    ranges = np.arange(args.range_step_m / 2.0, args.max_range_m, args.range_step_m, dtype=np.float32)
    return TargetGrid(azimuth=azimuth, range=ranges)


def parse_radars(day_root: Path, radars: str) -> list[str]:
    available = sorted(p.name for p in day_root.iterdir() if p.is_dir())
    if radars.strip().lower() == "all":
        return available
    requested = [item.strip().upper() for item in radars.split(",") if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        print(f"[WARN] Missing requested radars in {day_root}: {','.join(missing)}", flush=True)
    return [radar for radar in requested if radar in available]


def choose_scan_dirs(radar_dir: Path, scan_patterns: str) -> list[Path]:
    scan_dirs: list[Path] = []
    for pattern in [item.strip() for item in scan_patterns.split(",") if item.strip()]:
        candidates = sorted(p for p in radar_dir.glob(pattern) if p.is_dir())
        for candidate in candidates:
            if any(candidate.glob("*dBZ.vol")):
                scan_dirs.append(candidate)
    return scan_dirs


def files_by_rounded_time(scan_dirs: list[Path], round_time: str, max_times: int) -> list[tuple[pd.Timestamp, Path]]:
    out: dict[pd.Timestamp, Path] = {}
    for scan_dir in scan_dirs:
        for path in sorted(scan_dir.glob("*dBZ.vol")):
            match = TIME_RE.match(path.name)
            if not match:
                continue
            timestamp = pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S", utc=True)
            rounded = timestamp.round(round_time)
            out.setdefault(rounded, path)
    items = sorted(out.items())
    if max_times > 0:
        items = items[:max_times]
    return items


def beam_height_m(ranges_m: np.ndarray, elevation_deg: float, radar_altitude_m: float) -> np.ndarray:
    earth_radius_m = 6371000.0
    effective_radius_m = 4.0 * earth_radius_m / 3.0
    elev_rad = np.deg2rad(elevation_deg)
    return (
        np.sqrt(
            ranges_m**2
            + effective_radius_m**2
            + 2.0 * ranges_m * effective_radius_m * np.sin(elev_rad)
        )
        - effective_radius_m
        + radar_altitude_m
    ).astype(np.float32)


def reflectivity_var_name(ds: xr.Dataset) -> str:
    for name in ("DBZH", "DBZ", "TH", "DBZV"):
        if name in ds:
            return name
    numeric = [name for name, da in ds.data_vars.items() if set(da.dims) >= {"azimuth", "range"}]
    if not numeric:
        raise KeyError("No reflectivity-like azimuth/range variable found")
    return numeric[0]


def read_volume_dbz(path: Path, grid: TargetGrid) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    datatree = xradar_io.open_rainbow_datatree(str(path))
    root = datatree.ds
    radar_meta = {
        "latitude": float(root["latitude"].values),
        "longitude": float(root["longitude"].values),
        "altitude": float(root["altitude"].values),
    }

    sweeps: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    for child_name in sorted(name for name in datatree.children if name.startswith("sweep_")):
        ds = datatree[child_name].ds
        var_name = reflectivity_var_name(ds)
        da = ds[var_name].astype("float32")
        da = da.where(da > -31.5)
        _, azimuth_index = np.unique(da["azimuth"].values, return_index=True)
        _, range_index = np.unique(da["range"].values, return_index=True)
        da = da.isel(
            azimuth=np.sort(azimuth_index),
            range=np.sort(range_index),
        ).sortby(["azimuth", "range"])
        da = da.interp(
            azimuth=grid.azimuth,
            range=grid.range,
            kwargs={"fill_value": np.nan},
        )
        sweeps.append(da.values.astype(np.float32))
        fixed_angle = float(ds["sweep_fixed_angle"].values)
        heights.append(beam_height_m(grid.range, fixed_angle, radar_meta["altitude"]))

    if not sweeps:
        raise RuntimeError(f"No sweeps found in {path}")
    return np.stack(sweeps), np.stack(heights), radar_meta


def vpr_correction_db(
    dbz: np.ndarray,
    heights: np.ndarray,
    min_vpr_dbz: float,
    bin_m: float,
    reference_height_m: float,
    max_correction_db: float,
) -> np.ndarray:
    finite = np.isfinite(dbz) & (dbz >= min_vpr_dbz)
    if np.count_nonzero(finite) < 100:
        return np.zeros(dbz.shape[1:], dtype=np.float32)

    height_3d = np.broadcast_to(heights[:, None, :], dbz.shape)
    max_height = float(np.nanmax(height_3d[finite]))
    edges = np.arange(0.0, max(max_height + bin_m, reference_height_m + 2.0 * bin_m), bin_m)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(centers.shape, np.nan, dtype=np.float32)

    height_values = height_3d[finite]
    dbz_values = dbz[finite]
    bin_index = np.digitize(height_values, edges) - 1
    for idx in range(len(centers)):
        values = dbz_values[bin_index == idx]
        if values.size >= 100:
            medians[idx] = np.nanmedian(values)

    valid = np.isfinite(medians)
    if np.count_nonzero(valid) < 2:
        return np.zeros(dbz.shape[1:], dtype=np.float32)

    profile_at_lowest = np.interp(heights[0], centers[valid], medians[valid])
    profile_at_reference = float(np.interp(reference_height_m, centers[valid], medians[valid]))
    correction = profile_at_reference - profile_at_lowest
    correction = np.clip(correction, -max_correction_db, max_correction_db)
    return np.broadcast_to(correction[None, :], dbz.shape[1:]).astype(np.float32)


def derive_products(
    dbz: np.ndarray,
    heights: np.ndarray,
    products: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    valid = np.isfinite(dbz)
    dbz_for_max = np.where(valid, dbz, -np.inf)
    any_valid = np.any(valid, axis=0)
    height_3d = np.broadcast_to(heights[:, None, :], dbz.shape)

    derived: dict[str, np.ndarray] = {}
    if "beam_height" in products:
        derived["beam_height"] = np.broadcast_to(heights[0][None, :], dbz.shape[1:]).astype(np.float32)

    if "czc" in products or "hzc" in products:
        max_index = np.argmax(dbz_for_max, axis=0)
        if "czc" in products:
            czc = np.take_along_axis(dbz, max_index[None, :, :], axis=0)[0].astype(np.float32)
            czc[~any_valid] = np.nan
            derived["czc"] = czc
        if "hzc" in products:
            hzc = np.take_along_axis(height_3d, max_index[None, :, :], axis=0)[0].astype(np.float32)
            hzc[~any_valid] = np.nan
            derived["hzc"] = hzc

    if "ezc20" in products:
        has_echo = np.any(dbz >= 20.0, axis=0)
        ezc20 = np.max(np.where(dbz >= 20.0, height_3d, -np.inf), axis=0).astype(np.float32)
        ezc20[~has_echo] = np.nan
        derived["ezc20"] = ezc20
    if "ezc45" in products:
        has_echo = np.any(dbz >= 45.0, axis=0)
        ezc45 = np.max(np.where(dbz >= 45.0, height_3d, -np.inf), axis=0).astype(np.float32)
        ezc45[~has_echo] = np.nan
        derived["ezc45"] = ezc45

    if "lzc" in products:
        z_linear = np.power(10.0, dbz / 10.0).astype(np.float32)
        liquid_water_content = (3.44e-6 * np.power(z_linear, 4.0 / 7.0)).astype(np.float32)
        liquid_water_content[~valid] = np.nan
        dz = np.gradient(heights, axis=0).astype(np.float32)
        lzc = np.nansum(liquid_water_content * dz[:, None, :], axis=0).astype(np.float32)
        lzc[~any_valid] = np.nan
        derived["lzc"] = lzc

    if "rzc" in products:
        low_level_dbz = dbz[0].copy()
        correction = vpr_correction_db(
            dbz,
            heights,
            min_vpr_dbz=args.min_vpr_dbz,
            bin_m=args.vpr_bin_m,
            reference_height_m=args.vpr_reference_height_m,
            max_correction_db=args.max_vpr_correction_db,
        )
        corrected_dbz = low_level_dbz + correction
        corrected_z = np.power(10.0, corrected_dbz / 10.0).astype(np.float32)
        rzc = np.power(corrected_z / args.z_r_a, 1.0 / args.z_r_b).astype(np.float32)
        rzc[~np.isfinite(low_level_dbz)] = np.nan
        derived["rzc"] = rzc

    return derived


def empty_products(nt: int, naz: int, nrange: int, products: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        name: np.full((nt, naz, nrange), np.nan, dtype=np.float32)
        for name in products
    }


def polar_cell_coordinates(
    radar: str,
    radar_meta: dict[str, float],
    grid: TargetGrid,
) -> dict[str, tuple[str, np.ndarray]]:
    azimuth_2d, range_2d = np.meshgrid(grid.azimuth, grid.range, indexing="ij")
    lon, lat, _ = GEOD.fwd(
        np.full(azimuth_2d.size, radar_meta["longitude"]),
        np.full(azimuth_2d.size, radar_meta["latitude"]),
        azimuth_2d.ravel(),
        range_2d.ravel(),
    )
    ncell = azimuth_2d.size
    return {
        "latitude": ("cell", lat.astype(np.float32)),
        "longitude": ("cell", lon.astype(np.float32)),
        "azimuth": ("cell", azimuth_2d.ravel().astype(np.float32)),
        "range": ("cell", range_2d.ravel().astype(np.float32)),
        "radar": ("cell", np.full(ncell, radar)),
        "radar_latitude": ("cell", np.full(ncell, radar_meta["latitude"], dtype=np.float32)),
        "radar_longitude": ("cell", np.full(ncell, radar_meta["longitude"], dtype=np.float32)),
        "radar_altitude": ("cell", np.full(ncell, radar_meta["altitude"], dtype=np.float32)),
    }


def process_radar(
    radar: str,
    scan_dirs: list[Path],
    grid: TargetGrid,
    products_to_write: tuple[str, ...],
    args: argparse.Namespace,
) -> xr.Dataset | None:
    files = files_by_rounded_time(scan_dirs, args.round_time, args.max_times_per_radar)
    if not files:
        print(f"[WARN] {radar}: no dBZ files in selected scan directories", flush=True)
        return None

    times = [timestamp.to_datetime64() for timestamp, _ in files]
    products = empty_products(len(files), len(grid.azimuth), len(grid.range), products_to_write)
    radar_meta = {"latitude": np.nan, "longitude": np.nan, "altitude": np.nan}

    for time_index, (timestamp, path) in enumerate(files):
        try:
            print(f"[READ] {radar} {timestamp} {path.name}", flush=True)
            dbz, heights, radar_meta = read_volume_dbz(path, grid)
            derived = derive_products(dbz, heights, products_to_write, args)
            if "radar_altitude" in products_to_write:
                derived["radar_altitude"] = np.full(
                    (len(grid.azimuth), len(grid.range)),
                    radar_meta["altitude"],
                    dtype=np.float32,
                )
        except Exception as exc:
            print(f"[ERROR] {radar} {timestamp}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            continue
        for name, values in derived.items():
            products[name][time_index, :, :] = values

    coords = {"time": ("time", times)}
    coords.update(polar_cell_coordinates(radar, radar_meta, grid))
    ds = xr.Dataset(coords=coords)
    for name, values in products.items():
        ds[name] = (("cell", "time"), values.reshape(values.shape[0], -1).T)
    return ds


def process_day(day: pd.Timestamp, args: argparse.Namespace, grid: TargetGrid) -> None:
    day_root = args.archive_root / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d") / "rainbow5"
    out_path = args.output_root / f"{day:%Y%m%d}"
    tmp_path = out_path.with_name(out_path.name + ".partial")

    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] {out_path} exists", flush=True)
        return
    if not day_root.exists():
        print(f"[WARN] Missing Rainbow day: {day_root}", flush=True)
        return

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if out_path.exists() and args.overwrite:
        shutil.rmtree(out_path)

    products_to_write = parse_products(args.products)
    radar_datasets = []
    for radar in parse_radars(day_root, args.radars):
        scan_dirs = choose_scan_dirs(day_root / radar, args.scan_patterns)
        if not scan_dirs:
            print(f"[WARN] {radar}: no scan directory matching {args.scan_patterns}", flush=True)
            continue
        radar_ds = process_radar(radar, scan_dirs, grid, products_to_write, args)
        if radar_ds is not None:
            radar_datasets.append(radar_ds)

    if not radar_datasets:
        raise RuntimeError(f"No radar datasets produced for {day:%Y-%m-%d}")

    daily = xr.concat(radar_datasets, dim="cell", join="outer", compat="no_conflicts", coords="minimal")
    daily = daily.assign_coords(cell=np.arange(daily.sizes["cell"], dtype=np.int64))
    if "rzc" in daily:
        daily["rzc"].attrs.update(
            long_name="VPR-corrected surface precipitation rate derived from Rainbow dBZ volume",
            units="mm h-1",
            method=(
                "Lowest-elevation reflectivity corrected with an observed median vertical profile "
                "of reflectivity, then converted with Z=aR^b."
            ),
        )
    if "czc" in daily:
        daily["czc"].attrs.update(long_name="Maximum column reflectivity", units="dBZ")
    if "ezc20" in daily:
        daily["ezc20"].attrs.update(long_name="Echo-top height at 20 dBZ", units="m")
    if "ezc45" in daily:
        daily["ezc45"].attrs.update(long_name="Echo-top height at 45 dBZ", units="m")
    if "lzc" in daily:
        daily["lzc"].attrs.update(long_name="Vertically integrated liquid water content", units="kg m-2")
    if "hzc" in daily:
        daily["hzc"].attrs.update(long_name="Height of maximum radar echo", units="m")
    if "beam_height" in daily:
        daily["beam_height"].attrs.update(
            long_name="Lowest-elevation beam height above mean sea level",
            units="m",
            method="Computed from range, sweep fixed angle, radar altitude, and an effective Earth radius model.",
        )
    if "radar_altitude" in daily:
        daily["radar_altitude"].attrs.update(long_name="Radar antenna altitude above mean sea level", units="m")
    daily = daily.assign_attrs(
        source="Rainbow polar radar volume archive",
        archive_root=str(args.archive_root),
        scan_patterns=args.scan_patterns,
        azimuth_step_deg=args.azimuth_step_deg,
        range_step_m=args.range_step_m,
        max_range_m=args.max_range_m,
        z_r_relation=f"Z={args.z_r_a}R^{args.z_r_b}",
        vpr_reference_height_m=args.vpr_reference_height_m,
        max_vpr_correction_db=args.max_vpr_correction_db,
        products=",".join(products_to_write),
        note=(
            "These are derived from archived polar reflectivity volumes. Surface rain rate "
            "is omitted by default because nordic_radar already provides the rain-rate composite; "
            "request product rzc explicitly to write a derived VPR-corrected estimate."
        ),
    )
    daily = daily.chunk({"cell": min(20000, daily.sizes["cell"]), "time": 1})
    encoding = {
        name: {"dtype": "float32"}
        for name in daily.data_vars
        if np.issubdtype(daily[name].dtype, np.floating)
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[WRITE] {tmp_path}", flush=True)
    daily.to_zarr(tmp_path, mode="w", consolidated=False, encoding=encoding)
    tmp_path.rename(out_path)
    print(f"[DONE] {day:%Y-%m-%d} -> {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    parse_products(args.products)
    grid = make_target_grid(args)
    days = selected_days(args)
    print(f"[INFO] Processing {len(days)} day(s), grid={len(grid.azimuth)}x{len(grid.range)}", flush=True)
    for day in days:
        try:
            process_day(day, args, grid)
        except Exception as exc:
            print(f"[ERROR] {day:%Y-%m-%d}: {exc}", flush=True)
            if not args.continue_on_error:
                raise


if __name__ == "__main__":
    main()
