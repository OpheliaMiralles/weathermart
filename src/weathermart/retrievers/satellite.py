import datetime
import glob
import json
import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from typing import Any
import functools
import threading
import time
from pathlib import Path
import pandas as pd
import urllib3
import xarray as xr
import re
import numpy as np
from weathermart.base import BaseRetriever, checktype
from weathermart.utils import ICON_DOMAIN, get_nrows_ncols_from_domain_size_and_reskm

max_workers = min(os.cpu_count() or 4, 8)
logger = logging.getLogger(__name__)
EUMETSAT_SOURCES = {
    "MSG_SEVIRI": {
        "platform": "MSG (SEVIRI)",
        "type": "geostationary",
        "freq": "15min",
        "res": "3km",
        "products": {
            "channels": {
                "code": "EO:EUM:DAT:MSG:HRSEVIRI",
                "variables": ['HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'],
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
                "variables": ['cloud_top_height', 'cloud_top_quality'],
                "description": (
                    "NWC SAF cloud top height/pressure/temperature. "
                    "Key for diagnosing deep convection and storm intensity."
                ),
            },
            "cloud_mask": {
                "code": "EO:EUM:DAT:MSG:CLM",
                "format": ".grb",
                "reader": "seviri_l2_grib",
                "variables": ['cloud_mask'],
                "description": (
                    "Each pixel is classified as one"
                    " of the following four types: clear sky over water, "
                    "clear sky over land, cloud, or not processed "
                    "(off Earth disc).")
            },
        }
    },

    "METOP_AVHRR": {
        "platform": "Metop-A/B/C (AVHRR)",
        "type": "polar",
        "freq": "2–4 passes/day",
        "res": "1km",
        "reader": "avhrr_l1b_eps",
        "products": {
            "avhrr_l1": {
                "code": "EO:EUM:DAT:METOP:AVHRRL1",
                "variables": ["visible", "near_ir", "thermal_ir"],
                "format": ".zip",
                "round_time": "360min", 
                "description": (
                    "High-resolution polar imager. "
                    "Surface temperature, cloud mask, snow/ice discrimination, "
                    "surface characterization for short-range forecasting."
                ),
            },
        },
    },

    "METOP_IASI": {
        "platform": "Metop-A/B/C (IASI)",
        "type": "polar",
        "freq": "2–4 passes/day",
        "res": "12km",
        "reader": "iasi_l1c_eps",
        "products": {
            "iasi_radiances": {
                "code": "EO:EUM:DAT:METOP:IASI_L1C",
                "variables": ["ir_radiances"],
                "round_time": "360min", 
                "format": ".zip",
                "description": (
                    "Hyperspectral IR radiances. "
                    "Temperature and humidity profile information, "
                    "valuable for nowcasting via assimilation or ML encoders."
                ),
            },
            "iasi_cloud_products": {
                "code": "EO:EUM:DAT:METOP:IASIL2CLP",
                "variables": ["cloud_fraction", "cloud_top_pressure"],
                "format": ".zip",
                "description": (
                    "IASI cloud properties complementing SEVIRI, "
                    "especially useful at high latitudes and during polar night."
                ),
            },
        },
    },

    "METOP_MHS": {
        "platform": "Metop-A/B/C (MHS)",
        "type": "polar",
        "freq": "2–4 passes/day",
        "res": "50km",
        "reader": "mhs_l1c_eps",
        "products": {
            "mhs_radiances": {
                "code": "EO:EUM:DAT:METOP:MHSL1",
                "variables": ["brightness_temperature"],
                "round_time": "360min",
                "variables": ["mw_radiances"],
                "format": ".zip",
                "description": (
                    "Microwave humidity sounder radiances. "
                    "Sensitive to upper-tropospheric humidity and precipitation, "
                    "works under cloud cover where IR fails."
                ),
            },
        },
    },

    "METOP_ASCAT": {
        "platform": "Metop-B/C (ASCAT)",
        "type": "polar",
        "freq": "2-4 passes/day", # but one uniform product per 24h (upward+downward swaths)
        "res": "12.5km",
        "reader": "xarray_ascat_winds",
        "products": {
            "ascat_coastal_winds": {
                "code": "EO:EUM:DAT:METOP:OSI-104",
                "variables": ['wvc_index', 'model_speed', 'model_dir', 'ice_prob', 'ice_age', 'wvc_quality_flag', 'wind_speed', 'wind_dir', 'bs_distance'],
                "round_time": "1440min", 
                "format": ".nc",
                "description": (
                    "High-resolution coastal ASCAT winds (Metop-B). "
                    "Improves near-shore wind and precipitation forecasts."
                ),
            },
        },
    },

    "MTG_LI": {
        "platform": "MTG-I (Lightning Imager)",
        "type": "geostationary",
        "freq": "2–10 min",
        "res": "~10 km",
        "products": {
            "li_flashes": {
                "code": "EO:EUM:DAT:0686",
                "variables": ["flash_count"],
                "format": ".nc",
                "description": (
                    "Accumulated lightning flashes. "
                    "Strong indicator of convective intensity and storm lifecycle."
                ),
            },
            "li_flash_area": {
                "code": "EO:EUM:DAT:0687",
                "variables": ["flash_area"],
                "format": ".nc",
                "description": (
                    "Lightning flash area. "
                    "Helps identify organized convection and severe storm evolution."
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
    r'(?<!\d)(?P<date>\d{8})(?P<sep>[-_]?)(?P<time>\d{6})(?!\d)'
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
        for attempt in range(5):     # max retries
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
    return pd.Timestamp(dt).round(f"{freq}min").to_pydatetime()

class EumetsatRetriever(BaseRetriever):
    """
    Generic EUMETSAT retriever for geostationary + polar satellites.
    """
    crs = "epsg:4326"
    sources = tuple(EUMETSAT_SOURCES.keys())
    variables = {
        k: [k] for k in extract_all_variables(EUMETSAT_SOURCES)
    }

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        *,
        bbox: tuple[float, float, float, float] = ICON_DOMAIN,
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
            Default is ICON_DOMAIN.
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
            from satpy.scene import Scene
            from satpy.readers.core.config import available_readers
        except ImportError as exc:
            raise ImportError("Requires eumdac, satpy, pyresample") from exc

        dates, _ = checktype(dates, variables)
        if product == "auto":
            product_names = list(EUMETSAT_SOURCES[source]["products"].keys())
        else:
            product_names = [product]
        metadata = {k: v for k, v in EUMETSAT_SOURCES[source].items() if k != "products"}
        resolution = resolution or EUMETSAT_SOURCES[source].get("res", "1km")
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
            token = get_cached_eumdac_token(
                eumdac,
                key,
                secret
            )
        else:
            try:
                with open(eumdac_credentials_path, encoding="utf-8") as f:
                    cred = json.load(f)
                token = get_cached_eumdac_token(
                    eumdac,
                    cred["consumer_key"],
                    cred["consumer_secret"]
                )
            except KeyError as exc:
                raise RuntimeError(
                    "Please provide a path to a .eumdac_credentials file in kwargs for authentification. "
                    "See https://api.eumetsat.int/api-key/ for instructions and "
                    ".eumdac_credentials_dummy for an example."
                ) from exc


        datastore = eumdac.DataStore(token)
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

            with TemporaryDirectory(dir="/lustre/storeB/users/opmir9231/tmp") as tmpdir:
                @retry_download
                def _download(prod):
                    fname = next(e for e in prod.entries if e.endswith(format))
                    with prod.open(entry=fname) as src, open(
                            f"{tmpdir}/{os.path.basename(fname)}", "wb"
                        ) as dst:
                            shutil.copyfileobj(src, dst)

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    list(ex.map(_download, products))

                files = glob.glob(f"{tmpdir}/*{format}")

                for f in files:
                    try:
                        if reader in available_readers():
                            scn = Scene(reader=reader, filenames=[f])
                            logger.debug(f"Available variables are: {scn.available_dataset_names()}")
                            logger.debug(f"Loading variables {satpy_vars} from file {f}")
                            scn.load(satpy_vars)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                scn = scn.resample(area, resampler="nearest")

                            ds = scn.to_xarray().persist()
                            t = scn.start_time or extract_time_flex(Path(f).stem)
                        else:
                            # assumes it is in xarray compatible format
                            ds = xr.open_dataset(f)
                            if "time" in ds.coords or "time" in ds.data_vars:
                                ds = ds.drop_vars("time")
                            if ds.lon.max() > 180:
                                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
                            if reader == "xarray_ascat_winds":
                                ds = regrid_swath_to_area(
                                    ds,
                                    area,
                                    radius_of_influence=1000*res_km,
                                )
                            t = extract_time_flex(Path(f).stem)
                        if round_time:
                            t = round_to_nearest_minutes(t, freq=int(round_time.replace("min", "")))
                        ds = ds.expand_dims(time=[t])
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
            reader = EUMETSAT_SOURCES[source].get("reader", product_cfg.get("reader", None))
            format = product_cfg.get("format", None)
            if reader is None and format is None:
                raise RuntimeError("No reader or format specified for this product. Available readers are: " + str(available_readers()))
            satpy_vars = product_cfg["variables"]
            round_time = product_cfg.get("round_time", None)
            collection = datastore.get_collection(collection_id)

            ds_list = []

            for date in dates:
                start = datetime.datetime.combine(date, datetime.time.min)
                end = start + datetime.timedelta(days=1)- datetime.timedelta(minutes=10)
                if test:
                    end = start + datetime.timedelta(minutes=30)

                products = list(collection.search(dtstart=start, dtend=end))
                if len(products) == 0:
                    continue
                def ptime(p):
                    t = getattr(p, "sensing_start", None) or getattr(p, "sensing_end", None)
                    if t is None:
                        return None
                    return t
                products.sort(key=lambda p: (ptime(p) or datetime.datetime.max, str(p)))
                ds_list = process_products(products)

                if ds_list:
                    all_ds.append(xr.concat(ds_list, dim="time"))

            if not all_ds:
                return xr.Dataset()

        ds = xr.concat(all_ds, dim="time").sortby("time")
        ds = ds.assign_attrs(metadata).groupby("time").mean(skipna=True)
        ds["time"] = pd.to_datetime(ds["time"].values).tz_localize(None)
        return ds


def regrid_swath_to_area(
    ds: xr.Dataset,
    area,
    *,
    vars_to_grid: list[str] | None = None,
    radius_of_influence: float = 30_000,  # meters; tune per instrument/res
) -> xr.Dataset:
    """
    Swath -> grid regridding using the provided pyresample AreaDefinition.

    - ds must contain lon/lat per sample (1D or 2D)
    - returns dataset on (y, x) with lon/lat coords for the target grid
    """
    from pyresample.geometry import SwathDefinition
    from pyresample.kd_tree import resample_nearest
    lons, lats = ds["lon"].values, ds["lat"].values
    swath = SwathDefinition(lons=lons, lats=lats)
    tgt_lons, tgt_lats = area.get_lonlats()  
    out = {}
    vars_to_grid = vars_to_grid or list(ds.data_vars.keys())
    for v in vars_to_grid:
        da = ds[v]
        a = np.asarray(da)
        a = np.squeeze(a)
        gridded = resample_nearest(
            swath,
            a,
            area,
            radius_of_influence=radius_of_influence,
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

def plot_polar(ds, t):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from pyproj import Transformer
    lon = ds["lon"].values.ravel()
    lat = ds["lat"].values.ravel()
    val = ds.sel(time=t)["wind_speed"].values.ravel() 
    # Project lon/lat -> polar stereographic meters
    proj = ccrs.NorthPolarStereo()
    transformer = Transformer.from_crs("EPSG:4326", proj.proj4_init, always_xy=True)
    x, y = transformer.transform(lon, lat)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=proj)
    ax.coastlines(linewidth=0.8)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=proj)
    pm = ax.scatter(x, y, c=val, transform=proj, cmap='viridis', s=10)
    plt.colorbar(pm, ax=ax, shrink=0.7, label="wind_speed")
    plt.title(f"Gridded polar stereographic")
    plt.savefig(f"polar_{pd.to_datetime(t).strftime('%Y%m%d_%H%M%S')}.png", dpi=150)
