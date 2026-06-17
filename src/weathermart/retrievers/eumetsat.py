import datetime
import functools
import glob
import json
import logging
import os
import re
import shutil
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import urllib3
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.utils import NORTH_LATITUDE_20_BBOX
from weathermart.utils import NORTH_LATITUDE_20_DOMAIN_FILTER
from weathermart.utils import get_nrows_ncols_from_domain_size_and_reskm

max_workers = min(os.cpu_count() or 4, 8)
logger = logging.getLogger(__name__)
RADIANCE_METADATA_VARIABLES = [
    "latitude",
    "longitude",
    "satellite_zenith_angle",
    "satellite_azimuth_angle",
    "solar_zenith_angle",
    "solar_azimuth_angle",
]

RADIANCE_READER_INSTRUMENTS = {
    "atms_l1c_cdr_nc": "ATMS",
    "aws_mwr_l1b_nc": "AWS",
    "coda": "IASI",
    "mhs_l1c_aapp": "MHS",
}
RADIANCE_INSTRUMENT_IDS = {
    "AMSU-A": 1,
    "MHS": 2,
    "ATMS": 3,
    "MWHS-2": 4,
    "AWS": 5,
    "IASI": 6,
}
IASI_SPECTRAL_CHANNELS = [str(channel) for channel in range(1, 8462)]
EUMETSAT_SOURCES = {
    "MSG_SEVIRI": {
        "platform": "MSG (SEVIRI)",
        "type": "geostationary",
        "freq": "15min",
        "native_res": "3km",
        "products": {
            "channels": {
                "code": "EO:EUM:DAT:MSG:HRSEVIRI",
                "variables": [
                    "HRV",
                    "IR_016",
                    "IR_039",
                    "IR_087",
                    "IR_097",
                    "IR_108",
                    "IR_120",
                    "IR_134",
                    "VIS006",
                    "VIS008",
                    "WV_062",
                    "WV_073",
                ],
                "reader": "seviri_l1b_native",
                "format": ".nat",
                "description": (
                    "Core geostationary imager for nowcasting: cloud motion vectors, "
                    "convective initiation, cloud phase/height proxies, fog/low clouds, "
                    "rapid precipitation evolution."
                ),
            },
            "cloud_top_height": {
                "code": "EO:EUM:DAT:MSG:CTH",
                "format": ".grb",
                "reader": "seviri_l2_grib",
                "variables": ["cloud_top_height", "cloud_top_quality"],
                "description": (
                    "NWC SAF cloud top height/pressure/temperature. "
                    "Key for diagnosing deep convection and storm intensity."
                ),
            },
            "cloud_mask": {
                "code": "EO:EUM:DAT:MSG:CLM",
                "format": ".grb",
                "reader": "seviri_l2_grib",
                "variables": ["cloud_mask"],
                "description": (
                    "Each pixel is classified as one"
                    " of the following four types: clear sky over water, "
                    "clear sky over land, cloud, or not processed "
                    "(off Earth disc)."
                ),
            },
        },
    },
    "METOP": {
        "platform": "Metop-A/B/C (IASI and ASCAT)",
        "type": "polar",
        "freq": "2-4 passes/day",
        "products": {
            "avhrr_l1": {
                "code": "EO:EUM:DAT:METOP:AVHRRL1",
                "variables": [
                    "1",
                    "2",
                    "3a",
                    "3b",
                    "4",
                    "5",
                    "cloud_flags",
                    "satellite_azimuth_angle",
                    "satellite_zenith_angle",
                    "solar_azimuth_angle",
                    "solar_zenith_angle",
                ],
                "format": ".nat",
                "native_res": "1km",
                "reader": "avhrr_l1b_eps",
                "description": (
                    "High-resolution polar imager. "
                    "Surface temperature, cloud mask, snow/ice discrimination, "
                    "surface characterization for short-range forecasting."
                ),
            },
            "iasi_radiances": {
                "code": "EO:EUM:DAT:METOP:IASIL1C-ALL",
                "variables": [*IASI_SPECTRAL_CHANNELS],
                "native_res": "12km",
                "reader": "coda",
                "format": ".nat",
                "description": ("Hyperspectral IR radiances. All spectral channels."),
            },
            "ascat_coastal_winds": {
                "code": "EO:EUM:DAT:METOP:OSI-104",
                "native_res": "12.5km",
                "variables": [
                    "wvc_index",
                    "model_speed",
                    "model_dir",
                    "ice_prob",
                    "ice_age",
                    "wvc_quality_flag",
                    "wind_speed",
                    "wind_dir",
                    "bs_distance",
                ],
                "reader": "xarray_ascat_winds",
                "format": ".nc",
                "description": (
                    "High-resolution coastal ASCAT winds (Metop-B). "
                    "Improves near-shore wind and precipitation forecasts."
                ),
            },
            "atms_radiances": {
                "code": "EO:EUM:DAT:0345",
                "variables": [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    *RADIANCE_METADATA_VARIABLES,
                    "quality_pixel_bitmask",
                    "data_quality_bitmask",
                ],
                "native_res": "16km",
                "reader": "atms_l1c_cdr_nc",
                "format": ".nc",
                "select_by_overlap": True,
                "description": (
                    "ATMS Level 1C CDR MW radiances. 22 spectral channels."
                ),
            },
            "mhs_radiances": {
                "code": "EO:EUM:DAT:METOP:MHSL1",
                "variables": ["1", "2", "3", "4", "5"],
                "native_res": "16km",
                "reader": "mhs_l1c_aapp",
                "format": ".nat",
                "description": ("MW radiances. 5 spectral channels."),
            },
        },
    },
    "MTG": {
        "platform": "MTG-I (FCI and LI)",
        "type": "geostationary",
        "freq": "2–10 min",
        "native_res": "3km",
        "products": {
            "li_flashes": {
                "code": "EO:EUM:DAT:0686",
                "variables": ["flash_count"],
                "format": ".nc",
                "reader": "li_l2_nc",
                "description": (
                    "Accumulated lightning flashes. "
                    "Strong indicator of convective intensity and storm lifecycle."
                ),
            },
            "li_flash_area": {
                "code": "EO:EUM:DAT:0687",
                "variables": ["flash_area"],
                "format": ".nc",
                "reader": "li_l2_nc",
                "description": (
                    "Lightning flash area. "
                    "Helps identify organized convection and severe storm evolution."
                ),
            },
            "all_sky_radiance": {
                "code": "EO:EUM:DAT:0799",
                "variables": ["HRV"],
                "reader": "fci_l1c_nc",
                "format": [".bufr", ".nc"],
                "description": (
                    "MTG All Sky Radiance (ASR) product from the Flexible Combined Imager "
                    "Level-1C high frequency geostationary radiances over Europe (≈10 min)."
                ),
            },
        },
    },
    "NOAA": {
        "platform": "NOAA polar sensors",
        "type": "polar/orbiting",
        "freq": "2–6 passes/day",
        "native_res": "2km",
        "products": {
            "all_sky_radiance_polar": {
                "code": "EO:EUM:DAT:0370",
                "variables": ["radiance"],
                "reader": "goes_imager_nc",
                "format": ".nc",
                "description": (
                    "Polar-orbiting NOAA radiance product accessed via EUMETSAT. Provides radiance measurements from polar imagers/sounders."
                ),
            },
        },
    },
    "AWS": {
        "platform": "Arctic Weather Satellite Proto-Flight Model",
        "type": "polar/orbiting",
        "freq": "14-15> passes/day",
        "native_res": "16km",
        "products": {
            "mwr_l1b": {
                "code": "EO:EUM:DAT:0905",
                "variables": [
                    *[str(channel) for channel in range(1, 20)],
                    *RADIANCE_METADATA_VARIABLES,
                    "scan_number",
                    "surface_type",
                ],
                "native_res": "16km",
                "reader": "aws_mwr_l1b_nc",
                "format": ".nc",
                "select_by_overlap": True,
                "description": (
                    "MWR Level 1B brightness temperatures from the Arctic Weather "
                    "Satellite. Channels are selected with string channel numbers."
                ),
            },
        },
    },
}


def extract_all_variables(sources: dict) -> list[str]:
    vars_all = []

    for platform, pdata in sources.items():
        products = pdata.get("products", {})
        for prod_name, p in products.items():
            if "variables" in p:
                vars_all.extend(p["variables"])

    seen = set()
    uniq = []
    for v in vars_all:
        if v not in seen:
            uniq.append(v)
            seen.add(v)

    return uniq


class _EumetsatTokenCache:
    token = None
    expiry = None
    lock = threading.Lock()


def get_cached_eumdac_token(eumdac, key, secret):
    """
    Request token only when expired.
    Caches token for ~50 minutes (safety below true 1h TTL).
    """
    with _EumetsatTokenCache.lock:
        now = datetime.datetime.utcnow()

        if _EumetsatTokenCache.token and _EumetsatTokenCache.expiry > now:
            return _EumetsatTokenCache.token

        # request new token
        token = eumdac.AccessToken((key, secret))
        _EumetsatTokenCache.token = token
        _EumetsatTokenCache.expiry = now + datetime.timedelta(minutes=50)

        return token


class RateLimiter:
    def __init__(self, rate_per_sec=20):
        self.rate = rate_per_sec
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = (1.0 / self.rate) - (now - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


rate_limiter = RateLimiter(rate_per_sec=20)

RE_YYYYMMDD_HHMMSS_FLEX = re.compile(
    r"(?<!\d)(?P<date>\d{8})(?P<sep>[-_]?)(?P<time>\d{6})(?!\d)"
)
RE_AWS_SENSING_WINDOW = re.compile(r"_G_O_(?P<start>\d{14})_(?P<end>\d{14})")


def extract_time_flex(name: str) -> datetime.datetime | None:
    m = RE_YYYYMMDD_HHMMSS_FLEX.search(name)
    if not m:
        return None
    return datetime.datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")


def extract_aws_sensing_start(name: str) -> datetime.datetime | None:
    m = RE_AWS_SENSING_WINDOW.search(name)
    if not m:
        return None
    return datetime.datetime.strptime(m.group("start"), "%Y%m%d%H%M%S")


def extract_aws_sensing_window(
    name: str,
) -> tuple[datetime.datetime, datetime.datetime] | None:
    m = RE_AWS_SENSING_WINDOW.search(name)
    if not m:
        return None
    return (
        datetime.datetime.strptime(m.group("start"), "%Y%m%d%H%M%S"),
        datetime.datetime.strptime(m.group("end"), "%Y%m%d%H%M%S"),
    )


def retry_download(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        delay = 1
        for attempt in range(5):  # max retries
            try:
                rate_limiter.wait()
                return fn(*args, **kwargs)
            except urllib3.exceptions.ProtocolError:
                time.sleep(delay)
                delay *= 2
            except Exception:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("Download failed after retries")

    return wrapper


def _time_window_centers(values, window: str | None) -> pd.DatetimeIndex:
    times = pd.to_datetime(values)
    if window is None:
        return pd.DatetimeIndex(times)
    half_window = pd.to_timedelta(window) / 2
    return pd.DatetimeIndex(times + half_window).floor(window)


def _centered_time_window(
    analysis_time: pd.Timestamp,
    window: str | None,
) -> tuple[datetime.datetime, datetime.datetime]:
    analysis_time = pd.to_datetime(analysis_time)
    if analysis_time.tzinfo is not None:
        analysis_time = analysis_time.tz_convert(None)
    if window is None:
        start = analysis_time
        end = analysis_time
    else:
        half_window = pd.to_timedelta(window) / 2
        start = analysis_time - half_window
        end = analysis_time + half_window
    return start.to_pydatetime(), end.to_pydatetime()


def _stack_dims_to_cell(ds: xr.Dataset, dims: tuple[str, ...]) -> xr.Dataset:
    stack_dims = tuple(dim for dim in dims if dim in ds.dims)
    if not stack_dims:
        return ds
    ds = ds.stack(cell=stack_dims).reset_index("cell")
    return ds.assign_coords(cell=np.arange(ds.sizes["cell"], dtype=np.int64))


def _is_numeric_channel_name(name: Any) -> bool:
    return isinstance(name, str) and name.isdecimal()


def _iasi_wavenumber_cm1(channel: xr.DataArray) -> xr.DataArray:
    return 645.0 + (channel.astype(np.float32) - 1.0) * 0.25


def _flatten_non_time_dims_to_cell(
    array: xr.DataArray,
    *,
    output_dim: str = "cell",
) -> xr.DataArray:
    """Flatten all non-time dimensions to an observation/cell dimension.

    Xarray cannot create a stacked dimension called ``cell`` when ``cell`` is
    already one of the dimensions being stacked.  Several EUMETSAT readers
    already expose swath samples as ``cell`` before we add the synthetic channel
    axis, so temporarily rename the existing sample dimension before stacking.
    """
    stack_dims = [dim for dim in array.dims if dim != "time"]
    if not stack_dims:
        return array

    if output_dim in stack_dims:
        sample_dim = f"_{output_dim}_sample"
        while sample_dim in array.dims:
            sample_dim = f"_{sample_dim}"
        array = array.rename({output_dim: sample_dim})
        stack_dims = [sample_dim if dim == output_dim else dim for dim in stack_dims]

    return array.stack({output_dim: stack_dims}).reset_index(output_dim)


def _stack_dataarray_like_brightness_temperature(
    da: xr.DataArray,
    *,
    reference_dims: tuple[str, ...],
    channel_values: np.ndarray,
) -> xr.DataArray | None:
    """
    Repeat a metadata array along the same synthetic channel axis used for
    brightness_temperature, then flatten all non-time sample dimensions to cell.
    """
    if not set(da.dims).issubset(set(reference_dims)):
        return None
    expanded = da
    if "_radiance_channel" not in expanded.dims:
        expanded = expanded.expand_dims(
            _radiance_channel=xr.IndexVariable("_radiance_channel", channel_values)
        )
    dims = [dim for dim in reference_dims if dim in expanded.dims]
    expanded = expanded.transpose(*dims)
    return _flatten_non_time_dims_to_cell(expanded)


def _stack_radiance_channels_as_observations(
    ds: xr.Dataset,
    requested_variables: list[str],
    *,
    instrument: str,
    output_name: str = "brightness_temperature",
) -> xr.Dataset:
    """
    Convert an EUMETSAT per-channel/wide radiance dataset to the same logical
    layout as MARS ODB radiances: one observation variable plus a companion
    channel variable on the flattened cell axis.

    Input example:
        "6"(cell), "7"(cell), latitude(cell), longitude(cell)

    Output example:
        brightness_temperature(cell), channel(cell), latitude(cell), longitude(cell)

    If a time dimension is already present, time is preserved and all other
    dimensions, including the synthetic channel axis, are flattened into cell.
    """
    channel_vars = [
        str(variable)
        for variable in requested_variables
        if _is_numeric_channel_name(str(variable)) and str(variable) in ds.data_vars
    ]
    if not channel_vars:
        return ds

    first = ds[channel_vars[0]]
    sample_dims = tuple(first.dims)
    incompatible = [
        name for name in channel_vars if tuple(ds[name].dims) != sample_dims
    ]
    if incompatible:
        logger.warning(
            "Cannot stack radiance channels for %s because channel variables have "
            "incompatible dimensions: %s",
            instrument,
            incompatible,
        )
        return ds

    channel_values = np.asarray([int(name) for name in channel_vars], dtype=np.int32)
    channel_axis = xr.IndexVariable("_radiance_channel", channel_values)
    stacked_channels = xr.concat(
        [ds[name].astype(np.float32) for name in channel_vars],
        dim=channel_axis,
    )
    reference_dims = tuple(
        dim for dim in sample_dims if dim in stacked_channels.dims
    ) + ("_radiance_channel",)
    stacked_channels = stacked_channels.transpose(*reference_dims)
    brightness_temperature = _flatten_non_time_dims_to_cell(stacked_channels)

    out = xr.Dataset(
        {output_name: brightness_temperature.reset_coords(drop=True)},
        attrs=dict(ds.attrs),
    )
    out[output_name].attrs.update(
        {
            "long_name": f"{instrument} channel radiance observation",
            "radiance_layout": "mars_odb_like",
        }
    )
    if instrument != "IASI":
        out[output_name].attrs.setdefault("units", "K")
    else:
        out[output_name].attrs.setdefault("units", "native_l1c_radiance")
        out[output_name].attrs["long_name"] = "IASI L1C spectral radiance observation"
        out[output_name].attrs["source_field"] = "GS1cSpect"
        out[output_name].attrs["note"] = (
            "IASI GS1cSpect values are native L1C spectral radiances, not "
            "Planck-inverted brightness temperatures."
        )

    if "_radiance_channel" in brightness_temperature.coords:
        channel = brightness_temperature["_radiance_channel"].astype(np.int32)
    else:
        # Fallback for xarray versions that drop stacked-dimension coordinates.
        # Build a template with the same sample dimensions as the radiance array,
        # filled with the instrument-local channel number, then flatten it with
        # the identical helper used for the radiance values.
        template = xr.zeros_like(stacked_channels, dtype=np.int32) + stacked_channels[
            "_radiance_channel"
        ].astype(np.int32)
        channel = _flatten_non_time_dims_to_cell(template)
    out["channel"] = channel.reset_coords(drop=True)
    out["channel"].attrs.update(
        {
            "long_name": f"{instrument} channel number",
            "description": "Instrument-local channel index; combine with instrument_id to disambiguate.",
        }
    )
    out["instrument_id"] = xr.full_like(
        out["channel"], RADIANCE_INSTRUMENT_IDS.get(instrument, -1)
    )
    out["instrument_id"].attrs.update(
        {
            "long_name": "radiance instrument identifier",
            "instrument_id_map": repr(RADIANCE_INSTRUMENT_IDS),
        }
    )
    if instrument == "IASI":
        out["wavenumber_cm1"] = _iasi_wavenumber_cm1(out["channel"])
        out["wavenumber_cm1"].attrs.update(
            {
                "long_name": "IASI approximate channel-centre wavenumber",
                "units": "cm-1",
                "formula": "645.0 + (channel - 1) * 0.25",
            }
        )

    metadata_dims = tuple(dim for dim in stacked_channels.dims)
    for name, da in ds.data_vars.items():
        if name in channel_vars or name in out:
            continue
        stacked = _stack_dataarray_like_brightness_temperature(
            da,
            reference_dims=metadata_dims,
            channel_values=channel_values,
        )
        if (
            stacked is not None
            and "cell" in stacked.dims
            and stacked.sizes["cell"] == out.sizes["cell"]
        ):
            out[name] = stacked.reset_coords(drop=True)
            out[name].attrs.update(da.attrs)

    for coord_name in ("latitude", "longitude"):
        if coord_name in brightness_temperature.coords and coord_name not in out:
            out[coord_name] = brightness_temperature[coord_name].reset_coords(drop=True)

    out = out.assign_coords(cell=np.arange(out.sizes["cell"], dtype=np.int64))
    out.attrs["radiance_layout"] = "mars_odb_like"
    out.attrs["radiance_instrument"] = instrument
    return out


def _concat_cell_observations_by_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Aggregate a MARS-ODB-like observation dataset by concatenating observations
    inside each requested time window rather than averaging arbitrary cell IDs.
    """
    if "time" not in ds.dims or "cell" not in ds.dims:
        return ds

    groups: list[xr.Dataset] = []
    for time_value in pd.DatetimeIndex(sorted(pd.unique(ds["time"].values))):
        selected = ds.sel(time=time_value)
        if selected.sizes.get("time", 0) == 1:
            flattened = selected.isel(time=0, drop=True)
        else:
            flattened = (
                selected.stack(_obs=("time", "cell"))
                .reset_index("_obs", drop=True)
                .rename({"_obs": "cell"})
            )
        flattened = flattened.assign_coords(
            cell=np.arange(flattened.sizes["cell"], dtype=np.int64)
        )
        flattened = flattened.expand_dims(time=[time_value])
        groups.append(flattened)
    if not groups:
        return ds.isel(time=slice(0, 0))
    out = xr.concat(groups, dim="time", join="outer")
    out.attrs.update(ds.attrs)
    return out


def _encode_satellite_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_wkt"):
        return value.to_wkt()
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _sanitize_dataset_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)

    for coord_name in list(ds.coords):
        coord = ds.coords[coord_name]
        if coord.ndim == 0 and coord.dtype == object:
            coord_value = coord.values.item()
            if coord_name == "crs" and hasattr(coord_value, "to_wkt"):
                ds.attrs["crs_wkt"] = coord_value.to_wkt()
            else:
                ds.attrs[coord_name] = str(coord_value)
            ds = ds.drop_vars(coord_name)
            for var_name in ds.data_vars:
                if ds[var_name].attrs.get("grid_mapping") == coord_name:
                    ds[var_name].attrs.pop("grid_mapping")

    coordinates = ds.attrs.get("coordinates")
    if isinstance(coordinates, list | tuple):
        coordinates = [str(coord) for coord in coordinates if str(coord) in ds.coords]
        if coordinates:
            ds.attrs["coordinates"] = " ".join(coordinates)
        else:
            ds.attrs.pop("coordinates", None)

    for attr_name, attr_value in list(ds.attrs.items()):
        ds.attrs[attr_name] = _encode_satellite_attr_value(attr_value)

    for var_name in ds.data_vars:
        attrs = ds[var_name].attrs
        attrs.pop("metadata", None)
        coordinates = attrs.get("coordinates")
        if isinstance(coordinates, list | tuple):
            coordinates = [
                str(coord) for coord in coordinates if str(coord) in ds[var_name].coords
            ]
            if coordinates:
                attrs["coordinates"] = " ".join(coordinates)
            else:
                attrs.pop("coordinates", None)
        for attr_name, attr_value in list(attrs.items()):
            attrs[attr_name] = _encode_satellite_attr_value(attr_value)

    return ds


def _materialize_dataset_single_threaded(ds: xr.Dataset) -> xr.Dataset:
    if any(
        getattr(ds[var_name].data, "chunks", None) is not None
        for var_name in ds.data_vars
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in arcsin",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in arccos",
                category=RuntimeWarning,
            )
            return ds.compute(scheduler="single-threaded")
    return ds


def _use_scan_time_as_time(ds: xr.Dataset) -> xr.Dataset:
    if "scan_time" not in ds:
        raise ValueError("time_index='scan' requires a scan_time variable")

    scan_dim = ds["scan_time"].dims[0]
    scan_time = pd.to_datetime(ds["scan_time"].values)
    ds = ds.drop_vars("scan_time")
    if scan_dim != "time":
        ds = ds.rename({scan_dim: "time"})
    ds = ds.assign_coords(time=("time", scan_time))
    return ds


def _prepare_eumetsat_dataset_time(
    ds: xr.Dataset,
    t: datetime.datetime,
    *,
    reader: str | None,
    time_index: str,
) -> xr.Dataset:
    if time_index == "scan" and "scan_time" in ds:
        if "cell" in ds["scan_time"].dims:
            raise ValueError(
                "time_index='scan' is not supported after swath dimensions are "
                "stacked to cell; use time_index='granule'."
            )
        ds = _use_scan_time_as_time(ds)
    ds = _sanitize_dataset_for_zarr(ds)
    if "time" not in ds.dims:
        ds = ds.expand_dims(time=[t])
    return ds


class EumetsatRetriever(BaseRetriever):
    """
    Generic EUMETSAT retriever for geostationary + polar satellites.
    """

    crs = "epsg:4326"
    sources = tuple(EUMETSAT_SOURCES.keys())
    variables = extract_all_variables(EUMETSAT_SOURCES)
    batch_dates = True

    def retrieve(
        self,
        source: str,
        variables: list[str] | str,
        dates: datetime.date | str | pd.Timestamp | list[Any],
        *,
        bbox: tuple[float, float, float, float] | str = NORTH_LATITUDE_20_BBOX,
        product: str = "auto",
        resolution: str | float | None = None,
        eumdac_credentials_path: str | None = None,
        test: bool = False,
        resample: bool = True,
        aggregate_time: bool = True,
        aggregation_window: str | None = "3h",
        time_index: str = "granule",
    ) -> xr.Dataset:
        """
        Retrieve EUMETSAT data for specified dates and variables.

        Parameters
        ----------
        source : str
            Source identifier for the data retrieval process.
        variables : list of tuple[str, dict]
            List of tuples containing variable names and associated parameters.
        dates : list of datetime.date or datetime.date
            Date or list of dates for which to retrieve radar data.
        bbox : tuple of float, optional
            Bounding box for the data retrieval in the format (min_lon, min_lat, max_lon, max_lat).
            Default is NORDIC_DOMAIN.
        resolution : str or float, optional
            Desired resolution in km for the resampled data. Can be a string with 'km' suffix (e.g., '1km')
            or a float representing the resolution in kilometers. Default is 1km.
        eumdac_credentials_path : str, optional
            Path to the file containing the EUMETSAT API token. Default is None.
        test : bool, optional
            If True, only the first 30 minutes of data per day will be downloaded for testing purposes.
        resample : bool, optional
            If True, resample products to the requested bbox/resolution grid when the reader supports it.
            If False, keep the native swath/grid representation from the source files.
        aggregate_time : bool, optional
            If True, average datasets that share the same aggregation window. If False,
            keep every downloaded granule.
        aggregation_window : str, optional
            Pandas frequency used to assign granules to centered analysis windows before
            aggregation. The default "3h" corresponds to +/-90 minutes around
            00, 03, 06, ..., 21 UTC analysis times.
        time_index : {"granule", "scan"}, optional
            If "granule", the time dimension indexes downloaded product files.
            If "scan", supported raw swath readers use scanline observation time as
            the time dimension.
        Returns
        -------
        xr.Dataset
            Merged dataset containing the radar data for all specified dates and variables.
        Raises
        ------
        RuntimeError
            If the EUMETSAT API credentials file is not set or if the token file cannot be read.
        FileNotFoundError
            If the token is not found.

        """
        try:
            import eumdac
            from pyresample.geometry import AreaDefinition
            from satpy.readers.core.config import available_readers
            from satpy.scene import Scene
        except ImportError as exc:
            raise ImportError("Requires eumdac, satpy, pyresample") from exc

        if bbox == NORTH_LATITUDE_20_DOMAIN_FILTER:
            bbox = NORTH_LATITUDE_20_BBOX

        if time_index not in {"granule", "scan"}:
            raise ValueError("time_index must be either 'granule' or 'scan'")

        dates, _ = checktype(dates, variables)
        if product == "auto":
            product_names = list(EUMETSAT_SOURCES[source]["products"].keys())
        else:
            product_names = [product]
        metadata = {
            k: v for k, v in EUMETSAT_SOURCES[source].items() if k != "products"
        }
        resolution = resolution or EUMETSAT_SOURCES[source].get("native_res", None)
        if resample and resolution is None:
            raise RuntimeError(
                "No resolution specified and no consistent native_res known for this source."
            )
        if resample:
            if isinstance(resolution, str) and resolution.endswith("km"):
                res_km = float(resolution.replace("km", ""))
            else:
                res_km = float(resolution)
        if eumdac_credentials_path is None:
            key = os.environ.get("EUMDAC_KEY")
            secret = os.environ.get("EUMDAC_SECRET")
            if key and secret:
                logging.warning(
                    "Using EUMDAC_KEY and EUMDAC_SECRET environment variable."
                )
            if not (key and secret):
                raise RuntimeError("Missing EUMDAC credentials")
            token = get_cached_eumdac_token(eumdac, key, secret)
        else:
            try:
                with open(eumdac_credentials_path, encoding="utf-8") as f:
                    cred = json.load(f)
                token = get_cached_eumdac_token(
                    eumdac, cred["consumer_key"], cred["consumer_secret"]
                )
            except KeyError as exc:
                raise RuntimeError(
                    "Please provide a path to a .eumdac_credentials file in kwargs for authentification. "
                    "See https://api.eumetsat.int/api-key/ for instructions and "
                    ".eumdac_credentials_dummy for an example."
                ) from exc

        datastore = eumdac.DataStore(token)
        available_reader_names = set(available_readers())
        area = None
        if resample:
            width, height = get_nrows_ncols_from_domain_size_and_reskm(bbox, res_km)
            area = AreaDefinition(
                "bbox",
                "bbox",
                "epsg4326",
                {"proj": "latlong"},
                width,
                height,
                bbox,
            )

        def process_products(products):
            datasets = []
            format_suffixes = (
                tuple(format) if isinstance(format, list | tuple) else (format,)
            )

            with TemporaryDirectory(
                dir="/lustre/storeB/users/" + os.environ["USER"] + "/tmp"
            ) as tmpdir:

                @retry_download
                def _download(prod):
                    entries = [e for e in prod.entries if e.endswith(format_suffixes)]
                    if reader == "li_l2_nc":
                        body_entries = [e for e in entries if "BODY" in e]
                        entries = body_entries or entries
                    fname = entries[0]
                    with (
                        prod.open(entry=fname) as src,
                        open(f"{tmpdir}/{os.path.basename(fname)}", "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(_download, products))

                files = []
                for suffix in format_suffixes:
                    files.extend(glob.glob(f"{tmpdir}/*{suffix}"))

                for f in files:
                    try:
                        if reader == "li_l2_nc":
                            ds = xr.open_dataset(f)
                            if "x" in ds.data_vars and "y" in ds.data_vars:
                                ds = ds.set_coords(["x", "y"])
                            pixel_id = np.char.add(
                                np.char.mod("%.3f", ds["x"].values.astype(np.float64)),
                                np.char.add(
                                    "_",
                                    np.char.mod(
                                        "%.3f", ds["y"].values.astype(np.float64)
                                    ),
                                ),
                            )
                            ds = ds.assign_coords(pixel_id=("pixels", pixel_id))
                            ds = ds.swap_dims({"pixels": "pixel_id"})
                            rename_map = {
                                "flash_accumulation": "flash_count",
                                "accumulated_flash_area": "flash_area",
                            }
                            selected = {
                                source_name: target_name
                                for source_name, target_name in rename_map.items()
                                if target_name in satpy_vars
                                and source_name in ds.data_vars
                            }
                            if not selected:
                                raise KeyError(
                                    f"No MTG lightning variables found in {f} for {satpy_vars}"
                                )
                            ds = ds[list(selected)].rename(selected)
                            ds = ds.groupby("pixel_id").mean()
                            t = pd.to_datetime(
                                ds.attrs.get("time_coverage_start")
                                or extract_time_flex(Path(f).stem)
                            ).to_pydatetime()
                        elif reader in available_reader_names:
                            scn = Scene(reader=reader, filenames=[f])
                            logger.debug(
                                f"Available variables are: {scn.available_dataset_names()}"
                            )
                            logger.debug(
                                f"Loading variables {satpy_vars} from file {f}"
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message="invalid value encountered in arcsin",
                                    category=RuntimeWarning,
                                    module=r"geotiepoints\.geointerpolator",
                                )
                                warnings.filterwarnings(
                                    "ignore",
                                    message="invalid value encountered in arccos",
                                    category=RuntimeWarning,
                                    module=r"geotiepoints\.geointerpolator",
                                )
                                scn.load(satpy_vars)
                                if resample:
                                    scn = scn.resample(area, resampler="nearest")
                                if reader.startswith("seviri"):
                                    ds = scn.to_xarray().persist()
                                else:
                                    ds = scn.to_xarray_dataset()
                                    ds = ds.astype(
                                        {v: np.float32 for v in ds.data_vars}
                                    )
                            t = scn.start_time or extract_time_flex(Path(f).stem)
                        elif reader == "coda":
                            ds, t = iasi_metop_to_xarray(
                                f,
                                area,
                                satpy_vars,
                                bbox=None if resample else bbox,
                            )
                        elif reader == "atms_l1c_cdr_nc":
                            ds, t = atms_l1c_cdr_to_xarray(f, satpy_vars)
                        elif reader == "aws_mwr_l1b_nc":
                            ds, t = aws_mwr_l1b_to_xarray(f, satpy_vars)
                        else:
                            # assumes it is in xarray compatible format
                            ds = xr.open_dataset(f)
                            if "time" in ds.coords or "time" in ds.data_vars:
                                ds = ds.drop_vars("time")
                            if ds.lon.max() > 180:
                                ds = ds.assign_coords(
                                    lon=(((ds.lon + 180) % 360) - 180)
                                )
                            if reader == "xarray_ascat_winds":
                                if resample:
                                    ds = regrid_swath_to_area(
                                        ds,
                                        area,
                                    )
                            t = extract_time_flex(Path(f).stem)
                        instrument = RADIANCE_READER_INSTRUMENTS.get(reader)
                        if instrument is not None:
                            radiance_output_name = (
                                "spectral_radiance"
                                if instrument == "IASI"
                                else "brightness_temperature"
                            )
                            ds = _stack_radiance_channels_as_observations(
                                ds,
                                satpy_vars,
                                instrument=instrument,
                                output_name=radiance_output_name,
                            )
                        ds = _prepare_eumetsat_dataset_time(
                            ds,
                            t,
                            reader=reader,
                            time_index=time_index,
                        )
                        if reader == "avhrr_l1b_eps":
                            ds = _materialize_dataset_single_threaded(ds)
                        datasets.append(ds)

                    except Exception:
                        logger.exception("Failed to read %s", f)
                        continue

            return datasets

        all_ds = []
        for prod_name in product_names:
            product_cfg = EUMETSAT_SOURCES[source]["products"][prod_name]
            metadata.update({k: v for k, v in product_cfg.items() if k != "variables"})
            collection_id = product_cfg["code"]
            reader = EUMETSAT_SOURCES[source].get(
                "reader", product_cfg.get("reader", None)
            )
            format = product_cfg.get("format", None)
            if reader is None and format is None:
                raise RuntimeError(
                    "No reader or format specified for this product. Available readers are: "
                    + str(available_readers())
                )
            satpy_vars = list(
                set(product_cfg["variables"]).intersection(set(variables))
            )
            if len(satpy_vars) == 0:
                logger.info(
                    f"No requested variables found in product {prod_name}, skipping."
                )
                continue
            select_by_overlap = product_cfg.get("select_by_overlap", False)
            collection = datastore.get_collection(collection_id)
            products_to_process = []

            for date in dates:
                start, end = _centered_time_window(date, aggregation_window)
                if test:
                    end = start + datetime.timedelta(minutes=30)

                products = list(collection.search(dtstart=start, dtend=end))
                if len(products) == 0:
                    continue

                def ptime(p):
                    sensing_window = extract_aws_sensing_window(str(p))
                    t = (
                        sensing_window[0]
                        if sensing_window is not None
                        else getattr(p, "sensing_start", None)
                        or getattr(p, "sensing_end", None)
                    )
                    if t is None:
                        return None
                    return t

                def poverlaps(p):
                    sensing_window = extract_aws_sensing_window(str(p))
                    if sensing_window is None:
                        sensing_start = getattr(p, "sensing_start", None)
                        sensing_end = getattr(p, "sensing_end", None)
                    else:
                        sensing_start, sensing_end = sensing_window
                    if sensing_start is None and sensing_end is None:
                        return True
                    sensing_start = sensing_start or sensing_end
                    sensing_end = sensing_end or sensing_start
                    if sensing_start < start - datetime.timedelta(days=1):
                        return False
                    return sensing_start <= end and sensing_end >= start

                products.sort(key=lambda p: (ptime(p) or datetime.datetime.max, str(p)))
                chosen = {}
                for p in products:
                    t = ptime(p)
                    if t is None:
                        continue
                    if select_by_overlap:
                        if not poverlaps(p):
                            continue
                    elif t < start or t > end:
                        continue
                    chosen.setdefault(t, p)
                products = list(chosen.values())
                logger.info(
                    "[%s/%s] %s -> %d after time-dedup",
                    source,
                    prod_name,
                    date,
                    len(products),
                )
                products_to_process.extend(products)

            if products_to_process:
                ds_list = process_products(products_to_process)
                if ds_list:
                    all_ds.append(xr.concat(ds_list, dim="time", join="outer"))

            if not all_ds:
                return xr.Dataset()

        ds = xr.concat(all_ds, dim="time").sortby("time")
        ds = ds.assign_attrs(metadata)
        if aggregate_time:
            print(ds.time.values)
            ds = ds.assign_coords(
                time=_time_window_centers(ds["time"].values, aggregation_window)
            )
            if ds.attrs.get("radiance_layout") == "mars_odb_like" and "cell" in ds.dims:
                ds = _concat_cell_observations_by_time(ds).sel(time=dates)
            else:
                ds = ds.groupby("time").mean(
                    skipna=True
                )  # .sel(time=dates, method="nearest")
                print(ds.time.values)
        ds["time"] = pd.to_datetime(ds["time"].values).tz_localize(None)
        ds = _sanitize_dataset_for_zarr(ds)
        return ds


def regrid_swath_to_area(
    ds: xr.Dataset,
    area,
    *,
    vars_to_grid: list[str] | None = None,
    radius_of_influence: float = 30_000,
) -> xr.Dataset:
    """
    Swath -> grid regridding using the provided pyresample AreaDefinition.

    - ds must contain lon/lat per sample (1D or 2D)
    - returns dataset on (y, x) with lon/lat coords for the target grid
    """
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_gauss

    radius_of_influence = 30_000
    sigmas = 15_000
    lons, lats = ds["lon"].values, ds["lat"].values
    swath = SwathDefinition(lons=lons, lats=lats)
    tgt_lons, tgt_lats = area.get_lonlats()
    out = {}
    vars_to_grid = vars_to_grid or list(ds.data_vars.keys())
    for v in vars_to_grid:
        da = ds[v]
        a = np.asarray(da)
        a = np.squeeze(a)
        gridded = resample_gauss(
            swath,
            a,
            area,
            radius_of_influence=radius_of_influence,
            sigmas=sigmas,
            fill_value=np.nan,
        )
        out[v] = (("y", "x"), gridded)
    ds_out = xr.Dataset(
        out,
        coords={
            "lon": (("y", "x"), np.asarray(tgt_lons)),
            "lat": (("y", "x"), np.asarray(tgt_lats)),
        },
        attrs=dict(ds.attrs),
    )
    return ds_out


def aws_mwr_l1b_to_xarray(
    nc_file: str,
    channels: list[str],
) -> tuple[xr.Dataset, datetime.datetime]:
    """
    Read AWS MWR L1B grouped NetCDF and expose requested channels as variables.
    """
    selected_channels = sorted(
        int(channel) for channel in channels if channel.isdigit()
    )
    channel_indices = [channel - 1 for channel in selected_channels]
    calibration = xr.open_dataset(nc_file, group="/data/calibration")
    navigation = xr.open_dataset(nc_file, group="/data/navigation")
    measurement = xr.open_dataset(nc_file, group="/data/measurement")

    brightness_temperature = calibration["aws_toa_brightness_temperature"].isel(
        n_channels=channel_indices
    )
    brightness_temperature = brightness_temperature.rename(
        {"n_scans": "scan", "n_fovs": "fov", "n_channels": "channel"}
    )
    brightness_temperature = brightness_temperature.assign_coords(
        channel=selected_channels
    )

    def geo(name: str) -> xr.DataArray:
        return (
            navigation[name]
            .isel(n_geo_groups=0)
            .rename({"n_scans": "scan", "n_fovs": "fov"})
        )

    data_vars = {
        str(channel): brightness_temperature.sel(channel=channel).drop_vars("channel")
        for channel in selected_channels
    }
    data_vars.update(
        {
            "scan_number": ("scan", measurement["aws_scan_number"].values),
            "latitude": geo("aws_lat"),
            "longitude": geo("aws_lon"),
            "satellite_zenith_angle": geo("aws_satellite_zenith_angle"),
            "satellite_azimuth_angle": geo("aws_satellite_azimuth_angle"),
            "solar_zenith_angle": geo("aws_solar_zenith_angle"),
            "solar_azimuth_angle": geo("aws_solar_azimuth_angle"),
            "surface_type": geo("aws_surface_type"),
            "scan_time": ("scan", navigation["time_startscan_utc_earthview"].values),
        }
    )
    ds = xr.Dataset(data_vars)
    ds = ds.assign_coords(
        scan=np.arange(brightness_temperature.sizes["scan"]),
        fov=np.arange(brightness_temperature.sizes["fov"]),
    )
    ds = _stack_dims_to_cell(ds, ("scan", "fov"))
    for channel in selected_channels:
        ds[str(channel)].attrs.update(
            {
                "units": "K",
                "long_name": f"AWS MWR channel {channel} brightness temperature",
            }
        )

    t = extract_aws_sensing_start(Path(nc_file).stem) or extract_time_flex(
        Path(nc_file).stem
    )
    if t is None:
        t = pd.to_datetime(navigation["time_startscan_utc_earthview"].values[0])
        t = t.to_pydatetime()
    return ds, t.replace(tzinfo=None)


def atms_l1c_cdr_to_xarray(
    nc_file: str,
    channels: list[str],
) -> tuple[xr.Dataset, datetime.datetime]:
    """
    Read EUMETSAT ATMS L1C CDR NetCDF and expose requested channels as variables.
    """
    selected_channels = sorted(
        int(channel) for channel in channels if channel.isdigit()
    )
    ds_in = xr.open_dataset(nc_file)
    brightness_temperature = ds_in["btemps"].sel(channel=selected_channels)
    data_vars = {
        str(channel): brightness_temperature.sel(channel=channel).drop_vars("channel")
        for channel in selected_channels
    }

    metadata_vars = [
        "latitude",
        "longitude",
        "satellite_zenith_angle",
        "satellite_azimuth_angle",
        "solar_zenith_angle",
        "solar_azimuth_angle",
        "quality_pixel_bitmask",
        "data_quality_bitmask",
    ]
    for var_name in metadata_vars:
        if var_name in ds_in:
            data_vars[var_name] = ds_in[var_name]
        elif var_name in ds_in.coords:
            data_vars[var_name] = ds_in.coords[var_name].reset_coords(drop=True)
    if "time" in ds_in:
        data_vars["scan_time"] = ds_in["time"]

    ds = xr.Dataset(data_vars, attrs=ds_in.attrs)
    coords_to_keep_as_variables = [
        name for name in ("latitude", "longitude") if name in ds.coords
    ]
    if coords_to_keep_as_variables:
        ds = ds.reset_coords(coords_to_keep_as_variables, drop=False)
    ds = _stack_dims_to_cell(ds, ("y", "x"))
    for channel in selected_channels:
        ds[str(channel)].attrs.update(
            {
                "units": ds_in["btemps"].attrs.get("units", "K"),
                "long_name": f"ATMS channel {channel} brightness temperature",
            }
        )

    t = pd.to_datetime(
        ds_in.attrs.get("temporal_coverage_start")
        or ds_in.attrs.get("time_coverage_start")
        or extract_time_flex(Path(nc_file).stem)
    ).to_pydatetime()
    return ds, t.replace(tzinfo=None)


def iasi_metop_to_xarray(
    eps_file: str,
    area,
    variables: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    radius_of_influence: float = 30_000.0,
) -> tuple[xr.Dataset, datetime.datetime]:
    """
    Read IASI L1C EPS, select requested spectral channels,
    and optionally regrid to a target area.
    """
    import coda

    coda_file = coda.open(eps_file)
    lats = []
    lons = []
    scan_times = []
    n_mdr = len(coda.fetch(coda_file, "/MDR"))
    if n_mdr == 0:
        raise ValueError(f"No MDR records found in {eps_file}")
    first_spectrum = np.asarray(coda.fetch(coda_file, "/MDR[0]/MDR/GS1cSpect"))
    n_chan = first_spectrum.shape[-1]
    wn_start = 645.0
    wn_step = 0.25
    wn = wn_start + np.arange(n_chan) * wn_step
    t_eps = np.median(np.asarray(coda.fetch(coda_file, "/MDR[0]/MDR/OnboardUTC")))
    epoch = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
    times = epoch + datetime.timedelta(seconds=t_eps)
    channel_indices = {
        name: int(name) - 1
        for name in variables
        if isinstance(name, str) and name.isdecimal()
    }
    invalid_channels = [
        name for name, index in channel_indices.items() if index < 0 or index >= n_chan
    ]
    if invalid_channels:
        raise ValueError(
            f"IASI channels outside available 1..{n_chan}: "
            + ", ".join(invalid_channels)
        )
    if not channel_indices:
        raise ValueError(f"No supported IASI variables requested: {variables}")
    channels_out = {name: [] for name in channel_indices}
    for i in range(n_mdr):
        base = f"/MDR[{i}]/MDR"
        rad = np.asarray(coda.fetch(coda_file, base + "/GS1cSpect"))
        loc = np.asarray(coda.fetch(coda_file, base + "/GGeoSondLoc"))
        onboard_utc = np.asarray(coda.fetch(coda_file, base + "/OnboardUTC"))
        lon, lat = loc[..., 0], loc[..., 1]
        rad = rad.reshape(-1, rad.shape[-1])
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        flat_time_count = rad.shape[0]
        if onboard_utc.size == flat_time_count:
            onboard_utc = onboard_utc.reshape(-1)
        elif onboard_utc.size > 0 and flat_time_count % onboard_utc.size == 0:
            onboard_utc = np.repeat(
                onboard_utc.reshape(-1),
                flat_time_count // onboard_utc.size,
            )
        else:
            onboard_utc = np.full(flat_time_count, np.nanmedian(onboard_utc))

        for name, index in channel_indices.items():
            channels_out[name].append(rad[:, index])

        lats.append(lat)
        lons.append(lon)
        scan_times.append(onboard_utc)

    if hasattr(coda, "close"):
        coda.close(coda_file)

    lats = np.concatenate(lats)
    lons = np.concatenate(lons)
    scan_times = np.concatenate(scan_times)
    channels_out = {k: np.concatenate(v) for k, v in channels_out.items()}

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        inside = (
            (lons >= min_lon)
            & (lons <= max_lon)
            & (lats >= min_lat)
            & (lats <= max_lat)
        )
        if not np.any(inside):
            raise ValueError(f"No IASI samples in bbox {bbox} for {eps_file}")
        lats = lats[inside]
        lons = lons[inside]
        scan_times = scan_times[inside]
        channels_out = {name: values[inside] for name, values in channels_out.items()}

    selected_values = channels_out
    if area is None:
        epoch_naive = datetime.datetime(2000, 1, 1)
        data_vars = {
            name: ("cell", values.astype(np.float32))
            for name, values in selected_values.items()
        }
        data_vars.update(
            {
                "latitude": ("cell", lats.astype(np.float32)),
                "longitude": ("cell", lons.astype(np.float32)),
                "scan_time": (
                    "cell",
                    pd.to_datetime(epoch_naive) + pd.to_timedelta(scan_times, unit="s"),
                ),
            }
        )
        ds = xr.Dataset(
            data_vars,
            coords={"cell": np.arange(lats.size, dtype=np.int64)},
            attrs={
                "source": "IASI L1C EPS",
                "platform": "Metop",
            },
        )
    else:
        from pyresample.geometry import SwathDefinition
        from pyresample.kd_tree import resample_nearest

        swath = SwathDefinition(lons=lons, lats=lats)

        out = {}
        for name, values in selected_values.items():
            grid = resample_nearest(
                swath,
                values,
                area,
                radius_of_influence=radius_of_influence,
                fill_value=np.nan,
            )
            out[name] = grid
        tgt_lons, tgt_lats = area.get_lonlats()
        ds = xr.Dataset(
            {name: (("y", "x"), data) for name, data in out.items()},
            coords={
                "longitude": (("y", "x"), np.asarray(tgt_lons)),
                "latitude": (("y", "x"), np.asarray(tgt_lats)),
            },
            attrs={
                "source": "IASI L1C EPS",
                "platform": "Metop",
            },
        )
    for name, index in channel_indices.items():
        ds[name].attrs.update(
            {
                "long_name": f"IASI channel {name} radiance",
                "channel": int(name),
                "wavenumber_cm-1": float(wn[index]),
            }
        )
    return ds, times.replace(tzinfo=None)


def _format_half_window(window: str | None) -> str | None:
    if window is None:
        return None
    half_window = pd.to_timedelta(window) / 2
    minutes = int(half_window / pd.Timedelta(minutes=1))
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def plot_polar(
    ds,
    t,
    var,
    title: str | None = None,
    aggregation_window: str | None = None,
):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from pyproj import Transformer

    t = pd.to_datetime(t)
    if "time" in ds.dims or "time" in ds.coords:
        ds = ds.sel(time=t)
    if "longitude" in ds and "latitude" in ds:
        lon = ds["longitude"].values.ravel()
        lat = ds["latitude"].values.ravel()
    elif "x" in ds.coords and "y" in ds.coords:
        lon2d, lat2d = np.meshgrid(ds["x"].values, ds["y"].values)
        lon = lon2d.ravel()
        lat = lat2d.ravel()
    else:
        raise KeyError(
            "Dataset must contain either longitude/latitude or x/y coordinates."
        )
    val = ds[var].values.ravel()
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = lon[valid]
    lat = lat[valid]
    val = val[valid]
    # Project lon/lat -> polar stereographic meters
    central_lon = 15
    if np.nanmedian(lat) < 0:
        proj = ccrs.SouthPolarStereo(central_longitude=central_lon)
        latitude_locator = np.arange(-90, -49, 10)
    else:
        proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        latitude_locator = np.arange(50, 91, 10)
    transformer = Transformer.from_crs("EPSG:4326", proj.proj4_init, always_xy=True)
    x, y = transformer.transform(lon, lat)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
    x = x[valid]
    y = y[valid]
    val = val[valid]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = (xmax - xmin) * 0.05  # 5% buffer
    dy = (ymax - ymin) * 0.05
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=proj)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))  # meridians
    gl.ylocator = mticker.FixedLocator(latitude_locator)
    ax.coastlines(linewidth=0.8)
    # ax.set_extent([xmin, xmax, ymin, ymax], crs=proj)
    ax.set_extent([xmin - dx, xmax + dx, ymin - dy, ymax + dy], crs=proj)
    pm = ax.scatter(x, y, c=val, transform=proj, cmap="viridis", s=10)
    plt.colorbar(pm, ax=ax, shrink=0.7)
    plot_title = title or ds[var].attrs.get("long_name") or str(var)
    time_label = f"{pd.to_datetime(t):%Y-%m-%d %H:%M}"
    half_window = _format_half_window(aggregation_window)
    if half_window is not None:
        time_label = f"{time_label} +/-{half_window}"
    plt.title(f"{plot_title} ({time_label})")
    fig.tight_layout(pad=0.3)
    plt.savefig(
        f"{var}_{pd.to_datetime(t).strftime('%Y%m%d_%H%M%S')}.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.3,
    )
