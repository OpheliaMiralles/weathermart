import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype


ARCHIVE_ROOT = Path("/lustre/arkivB/projects/metproduction/MEPS")

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

PRECIP_VARIABLES = {"tp"}


class MEPSNetcdfRetriever(BaseRetriever):
    """Retriever for deterministic MEPS NetCDF files in the local archive."""

    sources = ("MEPS-NETCDF", "MEPSNETCDF", "MEPS_FCST", "MEPS")
    variables = OUTPUT_ORDER
    crs = "lambert"
    batch_dates = True

    def __init__(self, archive_root: str | Path = ARCHIVE_ROOT) -> None:
        self.archive_root = Path(archive_root)

    def retrieve(
        self,
        source: str,
        variables: list[str] | str,
        dates: datetime.date | str | pd.Timestamp | list[Any],
        archive_root: str | Path | None = None,
        run_hours: list[int] | str | None = "0,6,12,18",
        lead_hours: list[int] | str = "0,1,2,3,4,5,6",
        spatial_stride: int = 1,
        latest_by_valid_time: bool = False,
        hourly_precip: bool = False,
        continuous_hourly: bool = False,
    ) -> xr.Dataset:
        """Retrieve MEPS fields from NetCDF archive files.

        When ``latest_by_valid_time`` is false, ``dates`` are interpreted as
        forecast reference dates and the output has
        ``forecast_reference_time, lead_time, y, x`` dimensions. If an input
        date is a midnight day and ``run_hours`` is not None, all requested run
        hours for that day are returned.

        When ``latest_by_valid_time`` is true, ``dates`` are interpreted as
        valid datetimes and the latest available forecast for each valid time is
        selected. With ``hourly_precip=True``, accumulated precipitation is
        converted to one-hour ``mm h-1`` increments before latest-run selection.
        ``continuous_hourly=True`` is a shorthand for that continuous valid-time
        view with hourly precipitation enabled.
        """
        dates, variables = checktype(dates, variables)
        unknown = sorted(set(variables).difference(self.variables))
        if unknown:
            raise ValueError(f"Unknown MEPS variable(s): {unknown}. Available: {self.variables}")

        archive = Path(archive_root) if archive_root is not None else self.archive_root
        lead_hours_list = _parse_ints(lead_hours)
        if continuous_hourly:
            latest_by_valid_time = True
            hourly_precip = True
        if latest_by_valid_time:
            return self._retrieve_latest_valid_time(
                archive=archive,
                variables=variables,
                valid_times=pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None),
                lead_hours=lead_hours_list,
                spatial_stride=spatial_stride,
                hourly_precip=hourly_precip,
            )

        run_times = _expand_run_times(dates, run_hours)
        datasets = [
            _extract_run(
                paths=_meps_files(archive, run_time, lead_hours_list),
                run_time=run_time,
                variables=variables,
                lead_hours=lead_hours_list,
                spatial_stride=spatial_stride,
            )
            for run_time in run_times
        ]
        if not datasets:
            return xr.Dataset()
        out = xr.concat(datasets, dim="forecast_reference_time")
        return out.assign_attrs(source="MEPS deterministic NetCDF archive", archive_root=str(archive))

    def _retrieve_latest_valid_time(
        self,
        archive: Path,
        variables: list[str],
        valid_times: pd.DatetimeIndex,
        lead_hours: list[int],
        spatial_stride: int,
        hourly_precip: bool,
    ) -> xr.Dataset:
        if len(valid_times) == 0:
            return xr.Dataset()
        first = pd.Timestamp(valid_times.min()).tz_localize(None)
        last = pd.Timestamp(valid_times.max()).tz_localize(None)
        start_day = (first - pd.Timedelta(hours=max(lead_hours))).normalize()
        end_day = last.normalize()
        run_times = [
            day + pd.Timedelta(hours=hour)
            for day in pd.date_range(start_day, end_day, freq="D")
            for hour in [0, 6, 12, 18]
        ]
        run_times = [rt for rt in run_times if first - pd.Timedelta(hours=max(lead_hours)) <= rt <= last]
        datasets = []
        for run_time in run_times:
            try:
                paths = _meps_files(archive, run_time, lead_hours)
            except FileNotFoundError:
                continue
            datasets.append(
                _extract_run(
                    paths=paths,
                    run_time=run_time,
                    variables=variables,
                    lead_hours=lead_hours,
                    spatial_stride=spatial_stride,
                )
            )
        if not datasets:
            raise FileNotFoundError(f"No MEPS files found in {archive} for {first}..{last}")

        cube = xr.concat(datasets, dim="forecast_reference_time").sortby("forecast_reference_time")
        selected = {}
        for variable in variables:
            da = cube[variable]
            require_positive_lead = False
            if variable in PRECIP_VARIABLES and hourly_precip:
                da = _hourly_precipitation_rate(da)
                require_positive_lead = True
            selected[variable] = _select_latest_by_valid_time(
                da,
                valid_times,
                require_positive_lead=require_positive_lead,
            ).drop_vars(["forecast_reference_time", "lead_time"], errors="ignore")

        out = xr.Dataset(selected)
        out = out.assign_coords(
            y=cube["y"],
            x=cube["x"],
            latitude=(("y", "x"), cube["latitude"].data),
            longitude=(("y", "x"), cube["longitude"].data),
        )
        out["time"].attrs.update(standard_name="time", long_name="valid time")
        out["latitude"].attrs.update(standard_name="latitude", units="degrees_north")
        out["longitude"].attrs.update(standard_name="longitude", units="degrees_east")
        return out.assign_attrs(
            source="MEPS deterministic NetCDF archive latest-valid-time view",
            archive_root=str(archive),
        ).transpose("time", "y", "x")


def _parse_ints(value: list[int] | str | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _expand_run_times(dates: list[pd.Timestamp], run_hours: list[int] | str | None) -> list[pd.Timestamp]:
    hours = _parse_ints(run_hours)
    out = []
    for date in dates:
        ts = pd.Timestamp(date).tz_localize(None)
        if hours and ts == ts.normalize():
            out.extend(ts + pd.Timedelta(hours=hour) for hour in hours)
        else:
            out.append(ts)
    return sorted(set(out))


def _legacy_meps_file(archive_root: Path, run_time: pd.Timestamp) -> Path:
    return (
        archive_root
        / run_time.strftime("%Y")
        / run_time.strftime("%m")
        / run_time.strftime("%d")
        / f"meps_det_2_5km_{run_time.strftime('%Y%m%dT%HZ')}.nc"
    )


def _split_sfc_meps_files(
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


def _meps_files(archive_root: Path, run_time: pd.Timestamp, lead_hours: list[int]) -> list[Path]:
    legacy_path = _legacy_meps_file(archive_root, run_time)
    if legacy_path.exists():
        return [legacy_path]

    split_paths = _split_sfc_meps_files(archive_root, run_time, lead_hours)
    missing = [path for path in split_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return split_paths


def _source_label(paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return str(paths[0])
    return f"{paths[0]} ... {paths[-1]}"


def _open_meps(paths: Sequence[Path]) -> xr.Dataset:
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
    return xr.open_dataset(paths[0], engine="netcdf4", chunks={"time": 1, "y": 256, "x": 256})


def _lead_indices(ds: xr.Dataset, run_time: pd.Timestamp, lead_hours: list[int]) -> tuple[list[int], list[pd.Timedelta]]:
    valid_times = pd.to_datetime(ds["time"].values)
    run_time = pd.Timestamp(run_time).tz_localize(None)
    available = {
        int(round((pd.Timestamp(valid_time) - run_time).total_seconds() / 3600)): idx
        for idx, valid_time in enumerate(valid_times)
    }
    missing = [hour for hour in lead_hours if hour not in available]
    if missing:
        raise RuntimeError(f"Missing lead hours {missing} in {ds.encoding.get('source', '<dataset>')}")
    return [available[hour] for hour in lead_hours], [pd.Timedelta(hours=hour) for hour in lead_hours]


def _squeeze_to_time_y_x(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in {"time", "y", "x"}:
            continue
        if da.sizes[dim] != 1:
            raise ValueError(f"{da.name} has non-singleton vertical dimension {dim}={da.sizes[dim]}")
        da = da.isel({dim: 0}, drop=True)
    return da


def _dewpoint_from_temperature_rh(t_kelvin: xr.DataArray, rh_fraction: xr.DataArray) -> xr.DataArray:
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


def _wind_speed_from_components(x_wind: xr.DataArray, y_wind: xr.DataArray) -> xr.DataArray:
    wind_speed = np.hypot(x_wind, y_wind)
    wind_speed.attrs = {
        "standard_name": "wind_speed",
        "long_name": "10 metre wind gust speed computed from x/y gust components",
        "units": x_wind.attrs.get("units", "m s-1"),
        "computed_from": "x_wind_gust_10m, y_wind_gust_10m",
    }
    return wind_speed


def _extract_run(
    paths: Sequence[Path],
    run_time: pd.Timestamp,
    variables: list[str],
    lead_hours: list[int],
    spatial_stride: int,
) -> xr.Dataset:
    ds = _open_meps(paths)
    indices, leads = _lead_indices(ds, run_time, lead_hours)
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

    direct_variables = [name for name in variables if name not in {"d2m", "wind_gust_10m"}]
    if "d2m" in variables:
        direct_variables.extend(["t2m", "rh2m"])
    if "wind_gust_10m" in variables:
        direct_variables.extend(["x_wind_gust_10m", "y_wind_gust_10m"])
    direct_variable_set = set(direct_variables)
    direct_variables = [name for name in INTERNAL_ORDER if name in direct_variable_set]

    for output_name in direct_variables:
        input_name = VARIABLES[output_name]
        if input_name not in ds:
            raise KeyError(f"{input_name} not found in {_source_label(paths)}")
        da = _squeeze_to_time_y_x(ds[input_name])
        da = da.isel(time=indices).rename(time="lead_time")
        da = da.assign_coords(lead_time=leads)
        da.name = output_name
        da.attrs = dict(ds[input_name].attrs)
        da.attrs["source_variable"] = input_name
        out[output_name] = da

    if "d2m" in variables:
        out["d2m"] = _dewpoint_from_temperature_rh(out["t2m"], out["rh2m"])
        if "rh2m" not in variables:
            out = out.drop_vars("rh2m")
    if "wind_gust_10m" in variables:
        out["wind_gust_10m"] = _wind_speed_from_components(
            out["x_wind_gust_10m"], out["y_wind_gust_10m"]
        )
        out = out.drop_vars(["x_wind_gust_10m", "y_wind_gust_10m"])
    out = out[[name for name in OUTPUT_ORDER if name in out.data_vars]]
    return out.expand_dims(forecast_reference_time=[np.datetime64(run_time.to_datetime64())])


def _latest_index_for_valid_time(
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


def _select_latest_by_valid_time(
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
        forecast_index, lead_index = _latest_index_for_valid_time(
            forecast_reference_times,
            lead_times,
            pd.Timestamp(valid_time),
            require_positive_lead=require_positive_lead,
        )
        pieces.append(da.isel(forecast_reference_time=forecast_index, lead_time=lead_index, drop=True))
        selected_forecast_reference_times.append(forecast_reference_times[forecast_index].to_datetime64())
        selected_lead_times.append(lead_times[lead_index].to_timedelta64())

    out = xr.concat(pieces, dim=pd.Index(valid_times, name="time"))
    return out.assign_coords(
        forecast_reference_time=("time", np.array(selected_forecast_reference_times, dtype="datetime64[ns]")),
        lead_time=("time", np.array(selected_lead_times, dtype="timedelta64[ns]")),
    )


def _hourly_precipitation_rate(tp: xr.DataArray) -> xr.DataArray:
    hourly = tp.diff("lead_time", label="upper").clip(min=0)
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
