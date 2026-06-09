import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.utils import NORTH_LATITUDE_20_DOMAIN_FILTER

POLAR_DOMAIN_FILTER = NORTH_LATITUDE_20_DOMAIN_FILTER

MARS_INSTRUMENTS = ("AMSU-A", "AMSU-B/MHS", "ATMS", "ATMS_ALLSKY", "MWHS-2", "AWS")

MARS_OUTPUT_VARIABLES = [
    "brightness_temperature",
    "channel",
    "reportype",
    "satellite_identifier",
    "satellite_instrument",
    "analysis_date",
    "analysis_time",
    "expver",
    "lsm",
    "seaice",
    "gp_number",
    "windspeed10m",
    "tsfc",
    "snow_depth",
    "fg_rttov_cld_fraction",
    "zenith",
    "azimuth",
    "scanline",
    "scanpos",
    "emis_atlas",
    "emis_retr",
    "emis_fg",
    "datum_tbflag",
    "datum_status",
    "datum_event1",
    "datum_anflag",
    "biascorr",
    "biascorr_fg",
    "fg_depar",
    "an_depar",
    "final_obs_error",
    "obs_error",
]

SCRIPT_ODB_COLUMNS = [
    "expver",
    "andate",
    "antime",
    "reportype",
    "date",
    "lsm",
    "seaice",
    "time",
    "lat",
    "lon",
    "gp_number",
    "satellite_identifier",
    "satellite_instrument",
    "windspeed10m",
    "tsfc",
    "snow_depth",
    "snow_density",
    "fg_rttov_cld_fraction",
    "zenith",
    "azimuth",
    "scanline",
    "scanpos",
    "vertco_reference_1",
    "emis_atlas",
    "emis_retr",
    "emis_fg",
    "datum_tbflag",
    "obsvalue",
    "datum_status",
    "datum_event1",
    "datum_anflag",
    "datum_rdbflag",
    "biascorr",
    "biascorr_fg",
    "fg_depar",
    "an_depar",
    "final_obs_error",
    "obs_error",
    "tausfc",
    "tup",
    "tdown",
    "tausfc_cld",
    "tup_cld",
    "tdown_cld",
]

MINIMAL_RADIANCE_ODB_COLUMNS = [
    "reportype",
    "date",
    "lsm",
    "seaice",
    "time",
    "lat",
    "lon",
    "satellite_identifier",
    "satellite_instrument",
    "zenith",
    "azimuth",
    "scanline",
    "scanpos",
    "vertco_reference_1",
    "obsvalue",
    "datum_status",
    "biascorr_fg",
    "fg_depar",
]

MARS_INSTRUMENT_REPORTYPES: dict[str, dict[int, str]] = {
    "AMSU-A": {
        21001: "NOAA-15",
        21002: "NOAA-16",
        21003: "NOAA-17",
        21004: "NOAA-18",
        21005: "NOAA-19",
        21007: "METOP-A",
        21008: "AQUA",
        21009: "METOP-B",
        21010: "METOP-C",
    },
    "AMSU-B/MHS": {
        43004: "NOAA-18 AMSU-B",
        44001: "NOAA-19 MHS",
        44002: "METOP-A MHS",
        44003: "NOAA-18 MHS",
        44004: "METOP-B MHS",
        44005: "METOP-C MHS",
    },
    "ATMS": {
        34001: "S-NPP",
        34002: "NOAA-20",
        34003: "NOAA-21",
    },
    "ATMS_ALLSKY": {
        49001: "S-NPP",
        49002: "NOAA-20",
        49003: "NOAA-21",
    },
    "MWHS-2": {
        57001: "FY-3C",
        57002: "FY-3D",
        57003: "FY-3E",
    },
    "AWS": {
        74001: "AWS",
    },
}

DEFAULT_ODB_COLUMNS = [
    "expver",
    "andate",
    "antime",
    "reportype",
    "date",
    "lsm",
    "seaice",
    "time",
    "lat",
    "lon",
    "gp_number",
    "satellite_identifier",
    "satellite_instrument",
    "windspeed10m",
    "tsfc",
    "snow_depth",
    "fg_rttov_cld_fraction",
    "zenith",
    "azimuth",
    "scanline",
    "scanpos",
    "vertco_reference_1",
    "emis_atlas",
    "emis_retr",
    "emis_fg",
    "datum_tbflag",
    "obsvalue",
    "datum_status",
    "datum_event1",
    "datum_anflag",
    "biascorr",
    "biascorr_fg",
    "fg_depar",
    "an_depar",
    "final_obs_error",
    "obs_error",
]

ODB_OUTPUT_COLUMNS = {
    "brightness_temperature": ("obsvalue@body", "obsvalue"),
    "channel": ("vertco_reference_1@body", "vertco_reference_1"),
    "reportype": ("reportype@hdr", "reportype"),
    "satellite_identifier": (
        "satellite_identifier@sat",
        "satellite_identifier",
    ),
    "satellite_instrument": (
        "satellite_instrument@sat",
        "satellite_instrument",
    ),
    "analysis_date": ("andate", "andate@desc"),
    "analysis_time": ("antime", "antime@desc"),
    "expver": ("expver",),
    "lsm": ("lsm@modsurf", "lsm"),
    "seaice": ("seaice@modsurf", "seaice"),
    "gp_number": ("gp_number@hdr", "gp_number"),
    "windspeed10m": ("windspeed10m@modsurf", "windspeed10m"),
    "tsfc": ("tsfc@modsurf", "tsfc"),
    "snow_depth": ("snow_depth@modsurf", "snow_depth"),
    "fg_rttov_cld_fraction": ("fg_rttov_cld_fraction@allsky",),
    "zenith": ("zenith@sat", "zenith"),
    "azimuth": ("azimuth@sat", "azimuth"),
    "scanline": ("scanline@radiance", "scanline@sat", "scanline"),
    "scanpos": ("scanpos@radiance", "scanpos@sat", "scanpos"),
    "emis_atlas": ("emis_atlas@radiance_body", "emis_atlas"),
    "emis_retr": ("emis_retr@radiance_body", "emis_retr"),
    "emis_fg": ("emis_fg@radiance_body", "emis_fg"),
    "datum_tbflag": ("datum_tbflag@allsky_body", "datum_tbflag"),
    "datum_status": ("datum_status@body", "datum_status"),
    "datum_event1": ("datum_event1@body", "datum_event1"),
    "datum_anflag": ("datum_anflag@body", "datum_anflag"),
    "datum_rdbflag": ("datum_rdbflag@body", "datum_rdbflag"),
    "biascorr": ("biascorr@body", "biascorr"),
    "biascorr_fg": ("biascorr_fg@body", "biascorr_fg"),
    "fg_depar": ("fg_depar@body", "fg_depar"),
    "an_depar": ("an_depar@body", "an_depar"),
    "final_obs_error": ("final_obs_error@errstat", "final_obs_error"),
    "obs_error": ("obs_error@errstat", "obs_error"),
    "tausfc": ("tausfc@radiance_body", "tausfc"),
    "tup": ("tup@radiance_body", "tup"),
    "tdown": ("tdown@radiance_body", "tdown"),
    "tausfc_cld": ("tausfc_cld@allsky_body", "tausfc_cld"),
    "tup_cld": ("tup_cld@allsky_body", "tup_cld"),
    "tdown_cld": ("tdown_cld@allsky_body", "tdown_cld"),
}

ODB_COORD_COLUMNS = {
    "latitude": ("lat@hdr", "lat"),
    "longitude": ("lon@hdr", "lon"),
    "date": ("date@hdr", "date"),
    "time": ("time@hdr", "time"),
}


def _column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _format_reportypes(reportypes: Any) -> str | None:
    if reportypes is None:
        return None
    if isinstance(reportypes, str):
        return reportypes
    if isinstance(reportypes, int | np.integer):
        return str(reportypes)
    values = [str(item) for item in reportypes if item is not None]
    return "/".join(values) if values else None


def _resolve_mars_executable(mars_executable: str | Path) -> str:
    mars_path = Path(mars_executable).expanduser()
    if mars_path.parent != Path("."):
        return str(mars_path)

    found = shutil.which(str(mars_executable))
    if found is not None:
        return found

    home_mars = Path.home() / "mars" / "bin" / str(mars_executable)
    if home_mars.exists():
        return str(home_mars)

    return str(mars_executable)


def _default_reportypes(channel_group: str) -> list[int]:
    return list(MARS_INSTRUMENT_REPORTYPES[channel_group])


def _resolve_reportypes(
    channel_group: str,
    reportypes: Any,
) -> Any:
    if reportypes is None:
        return _default_reportypes(channel_group)
    if isinstance(reportypes, dict):
        return reportypes.get(channel_group)
    return reportypes


def _split_requested_variables(
    variables: list[str],
    instruments: list[str] | str | None,
) -> tuple[list[str], list[str]]:
    requested_instruments = [v for v in variables if v in MARS_INSTRUMENTS]
    requested_outputs = [v for v in variables if v in MARS_OUTPUT_VARIABLES]
    if instruments is not None:
        _, requested_instruments = checktype([], instruments)
    if not requested_instruments:
        requested_instruments = ["AMSU-A"]
    if not requested_outputs:
        requested_outputs = list(MARS_OUTPUT_VARIABLES)
    unknown = [
        v
        for v in variables
        if v not in MARS_INSTRUMENTS and v not in MARS_OUTPUT_VARIABLES
    ]
    if unknown:
        raise ValueError(
            f"Unsupported MARS ODB variables {unknown}. "
            f"Output variables: {MARS_OUTPUT_VARIABLES}. "
            f"Instrument selectors: {list(MARS_INSTRUMENTS)}"
        )
    for instrument in requested_instruments:
        if instrument not in MARS_INSTRUMENTS:
            raise ValueError(
                f"Unknown MARS ODB instrument {instrument}. "
                f"Options: {list(MARS_INSTRUMENTS)}"
            )
    return requested_outputs, requested_instruments


def _odb_datetimes(frame: pd.DataFrame) -> pd.Series:
    date_column = _column(frame, ODB_COORD_COLUMNS["date"])
    time_column = _column(frame, ODB_COORD_COLUMNS["time"])
    if date_column is None or time_column is None:
        raise ValueError(
            "ODB dataframe must contain date/time columns to build the time coordinate"
        )
    date_values = frame[date_column].astype("Int64").astype(str)
    time_values = frame[time_column].astype("Int64").astype(str).str.zfill(6)
    return pd.to_datetime(date_values + time_values, format="%Y%m%d%H%M%S")


def _time_window_centers(values, window: str | None) -> pd.DatetimeIndex:
    times = pd.to_datetime(values)
    if window is None:
        return pd.DatetimeIndex(times)
    half_window = pd.to_timedelta(window) / 2
    return pd.DatetimeIndex(times + half_window).floor(window)


def _empty_odb_dataset(source: str, aggregation_window: str | None) -> xr.Dataset:
    return xr.Dataset(
        coords={"time": np.array([], dtype="datetime64[ns]"), "cell": []},
        attrs={
            "source": source,
            "retriever": "MarsODBRetriever",
            "aggregation_window": ""
            if aggregation_window is None
            else aggregation_window,
        },
    )


def _analysis_window_bounds(
    analysis_time: pd.Timestamp,
    window: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if window is None:
        return None
    analysis_time = pd.to_datetime(analysis_time)
    half_window = pd.to_timedelta(window) / 2
    return analysis_time - half_window, analysis_time + half_window


def _assign_requested_analysis_times(
    working: pd.DataFrame,
    observation_time_column: str,
    aggregation_window: str | None,
    analysis_times: list[Any] | tuple[Any, ...] | pd.DatetimeIndex | Any,
) -> pd.DataFrame:
    centers = pd.DatetimeIndex(pd.to_datetime(analysis_times))
    chunks: list[pd.DataFrame] = []
    for center in centers:
        bounds = _analysis_window_bounds(center, aggregation_window)
        if bounds is None:
            selected = working.copy()
        else:
            start, end = bounds
            mask = (working[observation_time_column] >= start) & (
                working[observation_time_column] < end
            )
            selected = working.loc[mask].copy()
        if selected.empty:
            continue
        selected["_weathermart_time"] = center
        chunks.append(selected)
    if not chunks:
        return working.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def _assign_per_row_analysis_times(
    working: pd.DataFrame,
    observation_time_column: str,
    aggregation_window: str | None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for center, group in working.groupby("_weathermart_requested_analysis_time"):
        bounds = _analysis_window_bounds(pd.to_datetime(center), aggregation_window)
        if bounds is None:
            selected = group.copy()
        else:
            start, end = bounds
            mask = (group[observation_time_column] >= start) & (
                group[observation_time_column] < end
            )
            selected = group.loc[mask].copy()
        if selected.empty:
            continue
        selected["_weathermart_time"] = pd.to_datetime(center)
        chunks.append(selected)
    if not chunks:
        return working.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def _array_from_series(
    series: pd.Series,
    index: pd.MultiIndex,
    n_times: int,
    n_cells: int,
) -> np.ndarray:
    values = series.reindex(index)
    if pd.api.types.is_integer_dtype(series.dtype):
        values = values.fillna(-1).astype(np.int64)
    elif pd.api.types.is_numeric_dtype(series.dtype):
        values = values.astype(np.float64)
        values = values.where(values.abs() < 1.0e30)
    else:
        values = values.fillna("").to_numpy(dtype=str)
        return values.reshape(n_times, n_cells)
    return values.to_numpy().reshape(n_times, n_cells)


def odb_dataframe_to_xarray(
    frame: pd.DataFrame,
    variables: list[str] | tuple[str, ...] | None = None,
    *,
    source: str = "MARS_ODB",
    aggregation_window: str | None = "3h",
    analysis_times: list[Any] | tuple[Any, ...] | pd.DatetimeIndex | Any = None,
) -> xr.Dataset:
    if frame.empty:
        return _empty_odb_dataset(source, aggregation_window)

    output_variables = list(variables or MARS_OUTPUT_VARIABLES)
    lat_column = _column(frame, ODB_COORD_COLUMNS["latitude"])
    lon_column = _column(frame, ODB_COORD_COLUMNS["longitude"])
    if lat_column is None or lon_column is None:
        raise ValueError("ODB dataframe must contain latitude and longitude columns")

    working = frame.copy()
    working["_weathermart_observation_time"] = _odb_datetimes(working)
    if "_weathermart_requested_analysis_time" in working:
        working = _assign_per_row_analysis_times(
            working,
            "_weathermart_observation_time",
            aggregation_window,
        )
    elif analysis_times is not None:
        if not isinstance(analysis_times, list | tuple | pd.DatetimeIndex):
            analysis_times = [analysis_times]
        working = _assign_requested_analysis_times(
            working,
            "_weathermart_observation_time",
            aggregation_window,
            analysis_times,
        )
    else:
        working["_weathermart_time"] = working["_weathermart_observation_time"]
    if working.empty:
        return _empty_odb_dataset(source, aggregation_window)
    if (
        aggregation_window is not None
        and analysis_times is None
        and "_weathermart_requested_analysis_time" not in frame
    ):
        working["_weathermart_time"] = _time_window_centers(
            working["_weathermart_time"], aggregation_window
        )
    working = working.sort_values(["_weathermart_time", lat_column, lon_column])
    working["_weathermart_cell"] = working.groupby("_weathermart_time").cumcount()

    times = np.array(
        sorted(working["_weathermart_time"].unique()), dtype="datetime64[ns]"
    )
    n_times = len(times)
    n_cells = int(working["_weathermart_cell"].max()) + 1
    full_index = pd.MultiIndex.from_product(
        [times, np.arange(n_cells)], names=["time", "cell"]
    )
    indexed = working.set_index(["_weathermart_time", "_weathermart_cell"])
    indexed.index.names = ["time", "cell"]

    coords: dict[str, Any] = {
        "time": times,
        "cell": np.arange(n_cells, dtype=np.int64),
        "latitude": (
            ("time", "cell"),
            _array_from_series(indexed[lat_column], full_index, n_times, n_cells),
        ),
        "longitude": (
            ("time", "cell"),
            _array_from_series(indexed[lon_column], full_index, n_times, n_cells),
        ),
    }
    data_vars: dict[str, Any] = {}
    missing_variables: list[str] = []
    for variable in output_variables:
        column = _column(frame, ODB_OUTPUT_COLUMNS.get(variable, (variable,)))
        if column is None:
            missing_variables.append(variable)
            continue
        data_vars[variable] = (
            ("time", "cell"),
            _array_from_series(indexed[column], full_index, n_times, n_cells),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "source": source,
            "retriever": "MarsODBRetriever",
            "aggregation_window": ""
            if aggregation_window is None
            else aggregation_window,
            "missing_odb_variables": ",".join(missing_variables),
            "reportype_platform_map": repr(
                {
                    code: platform
                    for mapping in MARS_INSTRUMENT_REPORTYPES.values()
                    for code, platform in mapping.items()
                }
            ),
        },
    )
    if "brightness_temperature" in ds:
        ds["brightness_temperature"].attrs.update(
            {"long_name": "observed brightness temperature", "units": "K"}
        )
    ds["latitude"].attrs.update({"units": "degrees_north"})
    ds["longitude"].attrs.update({"units": "degrees_east"})
    return ds


def read_odb_to_xarray(
    paths: list[str | Path],
    variables: list[str] | tuple[str, ...] | None = None,
    *,
    source: str = "MARS_ODB",
    aggregation_window: str | None = "3h",
    analysis_times: list[Any] | tuple[Any, ...] | pd.DatetimeIndex | None = None,
) -> xr.Dataset:
    try:
        import pyodc
    except ImportError as exc:
        raise ImportError(
            "Reading submitted MARS ODB files requires pyodc. "
            "Install it with `uv add pyodc` or `pip install pyodc`."
        ) from exc

    valid_paths = [Path(path) for path in paths if Path(path).stat().st_size > 0]
    empty_paths = [str(path) for path in paths if Path(path).stat().st_size == 0]
    analysis_time_by_path = {}
    if analysis_times is not None:
        if len(analysis_times) != len(paths):
            raise ValueError("analysis_times must have the same length as paths")
        analysis_time_by_path = {
            Path(path): pd.to_datetime(analysis_time)
            for path, analysis_time in zip(paths, analysis_times, strict=True)
        }
    frames = []
    for path in valid_paths:
        frame = pyodc.read_odb(path, single=True)
        analysis_time = analysis_time_by_path.get(path)
        if analysis_time is not None and not frame.empty:
            frame = frame.copy()
            frame["_weathermart_requested_analysis_time"] = analysis_time
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ds = odb_dataframe_to_xarray(
        frame,
        variables,
        source=source,
        aggregation_window=aggregation_window,
        analysis_times=None,
    )
    ds.attrs["empty_odb_files"] = ",".join(empty_paths)
    return ds


def _mars_request_key(key: str) -> str:
    """Convert Python-friendly request keys to MARS keys."""
    text = str(key)
    if text.endswith("_") and text[:-1].lower() in {"class", "type", "format"}:
        text = text[:-1]
    return text.upper()


def _mars_key_for_lookup(key: str) -> str:
    return _mars_request_key(key).lower()


def _get_mars_request_value(request: Mapping[str, Any], key: str) -> Any | None:
    wanted = _mars_key_for_lookup(key)
    for request_key, value in request.items():
        if _mars_key_for_lookup(str(request_key)) == wanted:
            return value
    return None


def _set_mars_request_value(request: dict[str, Any], key: str, value: Any) -> None:
    wanted = _mars_key_for_lookup(key)
    for request_key in list(request):
        if _mars_key_for_lookup(str(request_key)) == wanted:
            request[request_key] = value
            return
    request[key] = value


def _format_mars_datetime_value(
    key: str, value: datetime.date | datetime.datetime | pd.Timestamp
) -> str:
    key_lookup = _mars_key_for_lookup(key)
    timestamp = pd.to_datetime(value)
    if key_lookup == "date":
        return f"{timestamp:%Y%m%d}"
    if key_lookup == "time":
        return f"{timestamp:%H:%M:%S}"
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return f"{timestamp:%Y%m%d}"
    return f"{timestamp:%Y%m%d%H%M%S}"


def _format_mars_scalar(key: str, value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp | datetime.datetime | datetime.date):
        return _format_mars_datetime_value(key, value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _quote_mars_value(text: str) -> str:
    stripped = text.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped
    return f'"{stripped}"'


def _format_mars_value(key: str, value: Any) -> str:
    if isinstance(value, list | tuple | set | pd.Index | np.ndarray):
        text = "/".join(_format_mars_scalar(key, item) for item in value)
    else:
        text = _format_mars_scalar(key, value)

    if _mars_key_for_lookup(key) in {"target", "filter"}:
        return _quote_mars_value(text)
    return text


def _target_path_from_value(target: str | Path, output_dir: Path) -> Path:
    target_text = str(target).strip().strip('"').strip("'")
    target_path = Path(target_text).expanduser()
    if not target_path.is_absolute():
        target_path = output_dir / target_path
    return target_path


def _safe_mars_request_stem(request: Mapping[str, Any]) -> str:
    pieces = ["mars"]
    for key in ("class", "stream", "type", "levtype", "date", "time"):
        value = _get_mars_request_value(request, key)
        if value is None:
            continue
        formatted = _format_mars_value(key, value)
        for char in ('"', "'", "/", ":", " ", ","):
            formatted = formatted.replace(char, "-")
        pieces.append(formatted.strip("-"))
    return "_".join(piece for piece in pieces if piece) or "mars_request"


def _mars_submission_env(
    rc_credential_path: str | Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    python_bin = Path(sys.executable).parent
    env["PATH"] = f"{python_bin}{os.pathsep}{env.get('PATH', '')}"
    try:
        import certifi

        ca_bundle = certifi.where()
        env.setdefault("SSL_CERT_FILE", ca_bundle)
        env.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
        env.setdefault("CURL_CA_BUNDLE", ca_bundle)
    except ImportError:
        pass
    if rc_credential_path is not None:
        env["ECMWF_API_RC_FILE"] = str(Path(rc_credential_path).expanduser())
    return env


def _ecmwf_api_credentials(
    rc_credential_path: str | Path | None = None,
) -> tuple[str, str, str]:
    try:
        from ecmwfapi.api import get_apikey_values
        from ecmwfapi.api import get_apikey_values_from_rcfile
    except ImportError as exc:
        raise ImportError(
            "Checking ECMWF MARS queue status requires ecmwf-api-client."
        ) from exc

    if rc_credential_path is not None:
        return get_apikey_values_from_rcfile(str(Path(rc_credential_path).expanduser()))
    return get_apikey_values()


def _mars_request_status_counts(
    rc_credential_path: str | Path | None = None,
) -> dict[str, int]:
    key, url, email = _ecmwf_api_credentials(rc_credential_path)
    request = urllib.request.Request(
        f"{url.rstrip('/')}/services/mars/requests",
        headers={
            "Accept": "application/json",
            "From": email,
            "X-ECMWF-KEY": key,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    counts: dict[str, int] = {}
    for item in payload.get("mars", []):
        status = item.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _wait_for_mars_queue_slot(
    *,
    rc_credential_path: str | Path | None = None,
    mars_max_queued_requests: int | None = None,
    mars_queue_poll_seconds: float = 300,
    mars_queue_wait_timeout: float | None = None,
) -> None:
    if mars_max_queued_requests is None:
        return
    if mars_max_queued_requests < 1:
        raise ValueError("mars_max_queued_requests must be positive or None")

    started = time.monotonic()
    while True:
        counts = _mars_request_status_counts(rc_credential_path)
        queued = counts.get("queued", 0)
        active = counts.get("active", 0)
        if queued < mars_max_queued_requests:
            return

        elapsed = time.monotonic() - started
        if mars_queue_wait_timeout is not None and elapsed >= mars_queue_wait_timeout:
            raise TimeoutError(
                "Timed out waiting for ECMWF MARS queue slot: "
                f"{queued} queued, {active} active, "
                f"limit {mars_max_queued_requests}."
            )

        print(
            "Waiting for ECMWF MARS queue slot: "
            f"{queued} queued, {active} active, "
            f"limit {mars_max_queued_requests}.",
            flush=True,
        )
        time.sleep(mars_queue_poll_seconds)


def _run_mars_request(
    request_file: str | Path,
    *,
    mars_executable: str | Path = "mars",
    mars_timeout: float | None = None,
    rc_credential_path: str | Path | None = None,
    mars_max_queued_requests: int | None = None,
    mars_queue_poll_seconds: float = 300,
    mars_queue_wait_timeout: float | None = None,
) -> None:
    _wait_for_mars_queue_slot(
        rc_credential_path=rc_credential_path,
        mars_max_queued_requests=mars_max_queued_requests,
        mars_queue_poll_seconds=mars_queue_poll_seconds,
        mars_queue_wait_timeout=mars_queue_wait_timeout,
    )
    subprocess.run(
        [_resolve_mars_executable(mars_executable), str(request_file)],
        check=True,
        env=_mars_submission_env(rc_credential_path),
        timeout=mars_timeout,
    )


class MarsRetriever(BaseRetriever):
    """
    Generate and optionally submit arbitrary MARS requests.

    This is intentionally format-agnostic: it can write GRIB, ODB, NetCDF or any
    other MARS-supported request. It does not attempt to parse the returned file;
    it returns request metadata and the target path.
    """

    sources = ("MARS", "ECMWF_MARS", "MARS_GRIB")
    variables = ["param"]
    crs = "epsg:4326"
    batch_dates = True

    @classmethod
    def build_request_text(
        cls,
        request: Mapping[str, Any] | None = None,
        *,
        target: str | Path | None = None,
        verb: str = "retrieve",
        **mars_kwargs: Any,
    ) -> str:
        mars_request: dict[str, Any] = dict(request or {})
        mars_request.update(mars_kwargs)
        if target is not None:
            _set_mars_request_value(mars_request, "target", target)
        mars_request = {
            key: value for key, value in mars_request.items() if value is not None
        }
        if not mars_request:
            raise ValueError("A generic MARS request needs at least one request field.")

        items = list(mars_request.items())
        lines = [f"{verb.upper()},"]
        for index, (key, value) in enumerate(items):
            suffix = "" if index == len(items) - 1 else ","
            lines.append(
                f"    {_mars_request_key(str(key))}={_format_mars_value(str(key), value)}{suffix}"
            )
        return "\n".join(lines) + "\n"

    def retrieve(
        self,
        source: str = "MARS",
        variables: list[str] | str | None = None,
        dates: datetime.date | str | pd.Timestamp | list[Any] | None = None,
        *,
        request: Mapping[str, Any] | None = None,
        request_text: str | None = None,
        input_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        target: str | Path | None = None,
        request_name: str | None = None,
        submit: bool = False,
        mars_executable: str | Path = "mars",
        mars_timeout: float | None = None,
        rc_credential_path: str | Path | None = None,
        mars_max_queued_requests: int | None = None,
        mars_queue_poll_seconds: float = 300,
        mars_queue_wait_timeout: float | None = None,
        verb: str = "retrieve",
        **mars_kwargs: Any,
    ) -> xr.Dataset:
        out_dir = (
            Path(output_dir) if output_dir is not None else Path.cwd() / "mars_requests"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        target_path: Path | None = None
        mars_request: dict[str, Any] = dict(request or {})
        mars_request.update(mars_kwargs)
        if (
            variables is not None
            and _get_mars_request_value(mars_request, "param") is None
        ):
            _set_mars_request_value(mars_request, "param", variables)
        if dates is not None and _get_mars_request_value(mars_request, "date") is None:
            _set_mars_request_value(mars_request, "date", dates)

        if target is not None:
            target_path = _target_path_from_value(target, out_dir)
            _set_mars_request_value(mars_request, "target", target_path)
        elif _get_mars_request_value(mars_request, "target") is not None:
            target_path = _target_path_from_value(
                _get_mars_request_value(mars_request, "target"), out_dir
            )
            _set_mars_request_value(mars_request, "target", target_path)

        if input_file is not None:
            request_file = Path(input_file).expanduser()
            request_text_final = request_file.read_text(encoding="utf-8")
        else:
            stem = request_name or _safe_mars_request_stem(mars_request)
            request_file = out_dir / f"{stem}.inp"
            request_text_final = request_text or self.build_request_text(
                mars_request,
                verb=verb,
            )
            request_file.write_text(request_text_final, encoding="utf-8")

        if submit:
            if target_path is None:
                target_value = _get_mars_request_value(mars_request, "target")
                if target_value is not None:
                    target_path = _target_path_from_value(target_value, out_dir)
            _run_mars_request(
                request_file,
                mars_executable=mars_executable,
                mars_timeout=mars_timeout,
                rc_credential_path=rc_credential_path,
                mars_max_queued_requests=mars_max_queued_requests,
                mars_queue_poll_seconds=mars_queue_poll_seconds,
                mars_queue_wait_timeout=mars_queue_wait_timeout,
            )

        ds = xr.Dataset(
            data_vars={
                "submitted": ("request", np.array([int(submit)], dtype=np.int8))
            },
            coords={
                "request": np.array([0], dtype=np.int64),
                "request_file": ("request", np.array([str(request_file)])),
                "target": (
                    "request",
                    np.array(["" if target_path is None else str(target_path)]),
                ),
            },
            attrs={
                "source": source,
                "retriever": "MarsRetriever",
                "request_text": request_text_final,
            },
        )
        return ds


class MarsODBRetriever(MarsRetriever):
    """
    Generate and optionally submit MARS requests for ODB observations.
    """

    sources = ("MARS_ODB", "ECMWF_ODB")
    variables = list(dict.fromkeys([*MARS_OUTPUT_VARIABLES, *MARS_INSTRUMENTS]))
    crs = "epsg:4326"
    batch_dates = True

    @classmethod
    def build_request_text(
        cls,
        *,
        date: pd.Timestamp,
        target: Path,
        channel_group: str,
        reportypes: Any = None,
        stream: str = "lwda",
        class_: str = "od",
        expver: str = "1",
        odb_type: str = "OFB",
        format_: str = "odb",
        time: str | None = None,
        domain_filter: str = POLAR_DOMAIN_FILTER,
        odb_columns: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        if channel_group not in MARS_INSTRUMENTS:
            raise ValueError(
                f"Unknown channel group {channel_group}. Options: {list(MARS_INSTRUMENTS)}"
            )

        columns = ", ".join(odb_columns or DEFAULT_ODB_COLUMNS)
        request_time = time or f"{pd.to_datetime(date):%H}"
        filters = [f"({domain_filter})"]
        filter_text = " and ".join(filters)
        reportype_str = _format_reportypes(reportypes)
        lines = [
            "RETRIEVE,",
            f"    STREAM={stream},",
            f"    CLASS={class_},",
            f"    EXPVER={expver},",
            f"    TYPE={odb_type},",
            f"    FORMAT={format_},",
            f"    DATE={date:%Y%m%d},",
            f"    TIME={request_time},",
        ]
        if reportype_str:
            lines.append(f"    REPORTYPE={reportype_str},")
        lines.extend(
            [
                (f'    FILTER="select distinct {columns} where {filter_text}",'),
                f'    TARGET="{target}"',
            ]
        )
        return "\n".join(lines) + "\n"

    def retrieve(
        self,
        source: str,
        variables: list[str] | str,
        dates: datetime.date | str | pd.Timestamp | list[Any],
        output_dir: str | Path | None = None,
        reportypes: Any = None,
        stream: str = "lwda",
        class_: str = "od",
        expver: str = "1",
        odb_type: str = "OFB",
        format_: str = "odb",
        time: str | None = None,
        domain_filter: str = POLAR_DOMAIN_FILTER,
        odb_columns: list[str] | tuple[str, ...] | None = None,
        submit: bool = False,
        mars_executable: str | Path = "mars",
        mars_timeout: float | None = None,
        rc_credential_path: str | Path | None = None,
        mars_max_queued_requests: int | None = None,
        mars_queue_poll_seconds: float = 300,
        mars_queue_wait_timeout: float | None = None,
        read_odb: bool = True,
        instruments: list[str] | str | None = None,
        aggregation_window: str | None = "3h",
    ) -> xr.Dataset:
        dates, variables = checktype(dates, variables)
        output_variables, selected_instruments = _split_requested_variables(
            variables, instruments
        )

        out_dir = (
            Path(output_dir) if output_dir is not None else Path.cwd() / "mars_odb"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        targets: list[Path] = []
        target_analysis_times: list[pd.Timestamp] = []
        for date in dates:
            analysis_time = pd.to_datetime(date)
            cycle = f"{analysis_time:%Y%m%d%H}"
            for channel_group in selected_instruments:
                safe_channel_group = channel_group.lower().replace("/", "_")
                target = out_dir / f"{safe_channel_group}_{cycle}.odb"
                request_file = out_dir / f"{safe_channel_group}_{cycle}.inp"
                request_reportypes = _resolve_reportypes(channel_group, reportypes)
                request_text = self.build_request_text(
                    date=analysis_time,
                    target=target,
                    channel_group=channel_group,
                    reportypes=request_reportypes,
                    stream=stream,
                    class_=class_,
                    expver=expver,
                    odb_type=odb_type,
                    format_=format_,
                    time=time,
                    domain_filter=domain_filter,
                    odb_columns=odb_columns,
                )
                request_file.write_text(request_text, encoding="utf-8")
                if submit:
                    _run_mars_request(
                        request_file,
                        mars_executable=mars_executable,
                        mars_timeout=mars_timeout,
                        rc_credential_path=rc_credential_path,
                        mars_max_queued_requests=mars_max_queued_requests,
                        mars_queue_poll_seconds=mars_queue_poll_seconds,
                        mars_queue_wait_timeout=mars_queue_wait_timeout,
                    )
                    targets.append(target)
                    target_analysis_times.append(analysis_time)
                rows.append(
                    {
                        "time": analysis_time,
                        "channel_group": channel_group,
                        "target": str(target),
                        "request_file": str(request_file),
                        "submitted": int(submit),
                        "reportypes": _format_reportypes(request_reportypes) or "",
                    }
                )

        if not rows:
            return xr.Dataset()

        if submit and read_odb:
            ds = read_odb_to_xarray(
                targets,
                output_variables,
                source=source,
                aggregation_window=aggregation_window,
                analysis_times=target_analysis_times,
            )
            ds.attrs.update(
                {
                    "stream": stream,
                    "class": class_,
                    "expver": expver,
                    "type": odb_type,
                    "format": format_,
                    "domain_filter": domain_filter,
                    "request_files": ",".join(row["request_file"] for row in rows),
                    "targets": ",".join(row["target"] for row in rows),
                    "instruments": ",".join(selected_instruments),
                }
            )
            return ds

        frame = (
            pd.DataFrame(rows)
            .sort_values(["time", "channel_group"])
            .reset_index(drop=True)
        )
        ds = xr.Dataset(
            data_vars={
                "submitted": ("request", frame["submitted"].to_numpy(dtype=np.int8)),
            },
            coords={
                "request": np.arange(len(frame)),
                "time": ("request", frame["time"].to_numpy(dtype="datetime64[ns]")),
                "channel_group": (
                    "request",
                    frame["channel_group"].astype(str).to_numpy(),
                ),
                "target": ("request", frame["target"].astype(str).to_numpy()),
                "request_file": (
                    "request",
                    frame["request_file"].astype(str).to_numpy(),
                ),
                "reportypes": ("request", frame["reportypes"].astype(str).to_numpy()),
            },
            attrs={
                "source": source,
                "retriever": "MarsODBRetriever",
                "stream": stream,
                "class": class_,
                "expver": expver,
                "type": odb_type,
                "format": format_,
                "domain_filter": domain_filter,
            },
        )
        return ds
