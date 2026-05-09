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
from weathermart.utils import NORDIC_DOMAIN
from weathermart.utils import get_nrows_ncols_from_domain_size_and_reskm

max_workers = min(os.cpu_count() or 4, 8)
logger = logging.getLogger(__name__)
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
                "round_time": "15min",
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
                "round_time": "15min",
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
                "round_time": "360min",
                "reader": "avhrr_l1b_eps",
                "description": (
                    "High-resolution polar imager. "
                    "Surface temperature, cloud mask, snow/ice discrimination, "
                    "surface characterization for short-range forecasting."
                ),
            },
            "iasi_radiances": {
                "code": "EO:EUM:DAT:METOP:IASIL1C-ALL",
                "variables": ["temp_15um", "swir_36um"],
                "valid_times": slice("09:00:00", "21:00:00"),
                "round_time": "770min",
                "native_res": "12km",
                "reader": "coda",
                "format": ".nat",
                "description": ("Hyperspectral IR radiances. All spectral channels."),
            },
            "ascat_coastal_winds": {
                "code": "EO:EUM:DAT:METOP:OSI-104",
                "native_res": "12.5km",
                "valid_times": slice("09:00:00", "21:00:00"),
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
                "round_time": "770min",
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
                ],
                "valid_times": slice("09:00:00", "21:00:00"),
                "native_res": "16km",
                "reader": "atms_l1b_nc",
                "format": ".nc",
                "description": ("MW radiances. 22 spectral channels."),
            },
            "mhs_radiances": {
                "code": "EO:EUM:DAT:METOP:MHSL1",
                "variables": ["1", "2", "3", "4", "5"],
                "valid_times": slice("09:00:00", "21:00:00"),
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
                "round_time": "2min",
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
                "round_time": "2min",
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
                "round_time": "10min",
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
                "round_time": "360min",
                "reader": "goes_imager_nc",
                "format": ".nc",
                "description": (
                    "Polar-orbiting NOAA radiance product accessed via EUMETSAT. Provides radiance measurements from polar imagers/sounders."
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


def extract_time_flex(name: str) -> datetime.datetime | None:
    m = RE_YYYYMMDD_HHMMSS_FLEX.search(name)
    if not m:
        return None
    return datetime.datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")


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


def round_to_nearest_minutes(dt: datetime.datetime, freq=15) -> datetime.datetime:
    return pd.Timestamp(dt).floor(f"{freq}min").to_pydatetime()


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
    if isinstance(coordinates, (list, tuple)):
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
        if isinstance(coordinates, (list, tuple)):
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
    if any(getattr(ds[var_name].data, "chunks", None) is not None for var_name in ds.data_vars):
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
        bbox: tuple[float, float, float, float] = NORDIC_DOMAIN,
        product: str = "auto",
        resolution: str | float | None = None,
        eumdac_credentials_path: str | None = None,
        test: bool = False,
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

        dates, _ = checktype(dates, variables)
        if product == "auto":
            product_names = list(EUMETSAT_SOURCES[source]["products"].keys())
        else:
            product_names = [product]
        metadata = {
            k: v for k, v in EUMETSAT_SOURCES[source].items() if k != "products"
        }
        resolution = resolution or EUMETSAT_SOURCES[source].get("native_res", None)
        if resolution is None:
            raise RuntimeError(
                "No resolution specified and no consistent native_res known for this source."
            )
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
                tuple(format) if isinstance(format, (list, tuple)) else (format,)
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
                                if target_name in satpy_vars and source_name in ds.data_vars
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
                                scn = scn.resample(area, resampler="nearest")
                                if reader.startswith("seviri"):
                                    ds = scn.to_xarray().persist()
                                else:
                                    ds = scn.to_xarray_dataset()
                                    ds = ds.astype({v: np.float32 for v in ds.data_vars})
                            t = scn.start_time or extract_time_flex(Path(f).stem)
                        elif reader == "coda":
                            ds, t = iasi_metop_to_xarray(
                                f,
                                area,
                            )
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
                                ds = regrid_swath_to_area(
                                    ds,
                                    area,
                                )
                            t = extract_time_flex(Path(f).stem)
                        if round_time:
                            t = round_to_nearest_minutes(t, freq=round_time)
                        ds = _sanitize_dataset_for_zarr(ds)
                        ds = ds.expand_dims(time=[t])
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
                set(product_cfg["variables"]).intersection(
                    set(variables)
                )
            )
            if len(satpy_vars) == 0:
                logger.info(
                    f"No requested variables found in product {prod_name}, skipping."
                )
                continue
            round_time = product_cfg.get("round_time", None)
            round_time = int(round_time.replace("min", "")) if round_time else None
            valid_times = product_cfg.get("valid_times", None)
            collection = datastore.get_collection(collection_id)
            products_to_process = []

            for date in dates:
                start_t = (
                    pd.to_timedelta(valid_times.start)
                    if isinstance(valid_times, slice)
                    else pd.to_timedelta("00:00:00")
                )
                end_t = (
                    pd.to_timedelta(valid_times.stop)
                    if isinstance(valid_times, slice)
                    else pd.to_timedelta("23:59:00")
                )
                start = (date + start_t).to_pydatetime()
                end = (date + end_t).to_pydatetime()
                if test:
                    end = start + datetime.timedelta(minutes=30)

                products = list(collection.search(dtstart=start, dtend=end))
                if len(products) == 0:
                    continue

                def ptime(p):
                    t = getattr(p, "sensing_start", None) or getattr(
                        p, "sensing_end", None
                    )
                    if t is None:
                        return None
                    if round_time is not None and round_time < 30:
                        # otherwise too long
                        t = round_to_nearest_minutes(t, freq=round_time)
                    return t

                products.sort(key=lambda p: (ptime(p) or datetime.datetime.max, str(p)))
                chosen = {}
                for p in products:
                    t = ptime(p)
                    if t is None or t < start or t > end:
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
                    all_ds.append(xr.concat(ds_list, dim="time"))

            if not all_ds:
                return xr.Dataset()

        ds = xr.concat(all_ds, dim="time").sortby("time")
        for t in ds.time.values:
            plot_polar(ds, t=t, var=satpy_vars[0])
        ds = ds.assign_attrs(metadata).groupby("time").mean(skipna=True)
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


def iasi_metop_to_xarray(
    eps_file: str,
    area,
    radius_of_influence: float = 30_000.0,
) -> tuple[xr.Dataset, datetime.datetime]:
    """
    Read IASI L1C EPS, select physically meaningful spectral bands,
    and regrid to a target AREA.
    """
    import coda
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest

    f = coda.open(eps_file)
    lats = []
    lons = []
    n_mdr = len(coda.fetch(f, "/MDR"))
    n_chan = 8700
    wn_start = 645.0
    wn_step = 0.25
    wn = wn_start + np.arange(n_chan) * wn_step
    t_eps = np.median(np.asarray(coda.fetch(f, "/MDR[0]/MDR/OnboardUTC")))
    epoch = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
    times = epoch + datetime.timedelta(seconds=t_eps)
    IASI_BANDS = {
        "temp_15um": (650, 770),  # ≈ 13–15.5 µm
        "swir_36um": (2500, 2760),  # ≈ 3.6–4.0 µm
    }
    band_masks = {}
    for name, (wn_min, wn_max) in IASI_BANDS.items():
        mask = (wn >= wn_min) & (wn <= wn_max)

        if not np.any(mask):
            raise ValueError(f"No channels in band {name}")

        band_masks[name] = mask
    bands_out = {name: [] for name in band_masks}
    for i in range(n_mdr):
        base = f"/MDR[{i}]/MDR"
        rad = np.asarray(coda.fetch(f, base + "/GS1cSpect"))
        loc = np.asarray(coda.fetch(f, base + "/GGeoSondLoc"))
        lon, lat = loc[..., 0], loc[..., 1]
        rad = rad.reshape(-1, rad.shape[-1])
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)

        for name, mask in band_masks.items():
            bands_out[name].append(rad[:, mask].mean(axis=1))

        lats.append(lat)
        lons.append(lon)

    # concatenate everything
    lats = np.concatenate(lats)
    lons = np.concatenate(lons)
    bands_out = {k: np.concatenate(v) for k, v in bands_out.items()}
    swath = SwathDefinition(lons=lons, lats=lats)

    out = {}
    for name, values in bands_out.items():
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
            "lon": (("y", "x"), np.asarray(tgt_lons)),
            "lat": (("y", "x"), np.asarray(tgt_lats)),
        },
        attrs={
            "source": "IASI L1C EPS",
            "platform": "Metop",
        },
    )
    return ds, times.replace(tzinfo=None)


def plot_polar(ds, t, var):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from pyproj import Transformer

    if "longitude" in ds and "latitude" in ds:
        lon = ds["longitude"].values.ravel()
        lat = ds["latitude"].values.ravel()
    elif "x" in ds.coords and "y" in ds.coords:
        lon2d, lat2d = np.meshgrid(ds["x"].values, ds["y"].values)
        lon = lon2d.ravel()
        lat = lat2d.ravel()
    else:
        raise KeyError("Dataset must contain either longitude/latitude or x/y coordinates.")
    val = ds.sel(time=t)[var].values.ravel()
    # Project lon/lat -> polar stereographic meters
    central_lon = 15
    proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    transformer = Transformer.from_crs("EPSG:4326", proj.proj4_init, always_xy=True)
    x, y = transformer.transform(lon, lat)
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
    gl.ylocator = mticker.FixedLocator(np.arange(60, 91, 10))
    ax.coastlines(linewidth=0.8)
    # ax.set_extent([xmin, xmax, ymin, ymax], crs=proj)
    ax.set_extent([xmin - dx, xmax + dx, ymin - dy, ymax + dy], crs=proj)
    pm = ax.scatter(x, y, c=val, transform=proj, cmap="viridis", s=10)
    plt.colorbar(pm, ax=ax, shrink=0.7)
    plt.title("Gridded polar stereographic")
    fig.tight_layout(pad=0.3)
    plt.savefig(
        f"{var}_{pd.to_datetime(t).strftime('%Y%m%d_%H%M%S')}.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.3,
    )
