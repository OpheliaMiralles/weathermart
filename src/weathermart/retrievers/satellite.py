import datetime
import glob
import json
import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from tempfile import TemporaryDirectory
from typing import Any

import dask
import pandas as pd
import urllib3
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.utils import ICON_DOMAIN
from weathermart.utils import get_nrows_ncols_from_domain_size_and_reskm

max_workers = os.cpu_count()


def round_to_nearest_15_minutes(dt: datetime.datetime) -> datetime.datetime:
    """Round a datetime object to the nearest 15 minutes."""
    rounded_minutes = 15 * round(dt.minute / 15)
    delta_minutes = rounded_minutes - dt.minute
    return (dt + datetime.timedelta(minutes=delta_minutes)).replace(
        second=0, microsecond=0
    )


class EumetsatRetriever(BaseRetriever):
    crs = "epsg:4326"
    sources = ("SATELLITE",)
    variables = {
        k: [k]
        for k in [
            "VIS006",
            "IR_039",
            "IR_108",
            "HRV",
            "IR_097",
            "WV_062",
            "IR_087",
            "IR_016",
            "VIS008",
            "IR_134",
            "WV_073",
            "IR_120",
        ]
    }

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        bbox: tuple[float, float, float, float] | None = ICON_DOMAIN,
        resolution: str | float = "1km",
        eumdac_credentials_path: str | None = None,
        test: bool = False,  # download only first 30mins per day, for speed
    ) -> xr.Dataset:
        """
        Retrieve OPERA radar data for specified dates and variables.

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
            from eumdac.collection import SearchResults
            from eumdac.product import Product
            from pyresample.geometry import AreaDefinition
            from satpy import Scene
        except ImportError as exc:
            raise ImportError(
                "The 'eumdac', 'pyresample', and 'satpy' packages are required for EUMETSAT data retrieval."
            ) from exc
        dates, variables = checktype(dates, variables)
        if isinstance(resolution, str) and "km" in resolution:
            self.res_km = float(resolution.replace("km", ""))
        else:
            self.res_km = resolution

        # load credentials from environment variables or file
        if eumdac_credentials_path is None:
            eumdac_key = os.environ.get("EUMDAC_KEY")
            eumdac_secret = os.environ.get("EUMDAC_SECRET")
            if eumdac_key and eumdac_secret:
                logging.warning(
                    "Using EUMDAC_KEY and EUMDAC_SECRET environment variable."
                )
                token = eumdac.AccessToken((eumdac_key, eumdac_secret))
            else:
                raise RuntimeError(
                    "Please provide either EUMDAC_KEY and EUMDAC_SECRET environment variables or a path to a .eumdac_credentials file."
                )
        else:
            try:
                with open(eumdac_credentials_path, encoding="utf-8") as json_file:
                    credentials = json.load(json_file)
                    token = eumdac.AccessToken(
                        (credentials["consumer_key"], credentials["consumer_secret"])
                    )
            except KeyError as exc:
                raise RuntimeError(
                    "Please provide a path to a .eumdac_credentials file in kwargs for authentification. "
                    "See https://api.eumetsat.int/api-key/ for instructions and "
                    ".eumdac_credentials_dummy for an example."
                ) from exc

        # connect to the data store
        datastore = eumdac.DataStore(token)
        selected_collection = datastore.get_collection("EO:EUM:DAT:MSG:HRSEVIRI")

        def download_and_resample(
            products: SearchResults, variables: list[tuple[str, dict]]
        ) -> xr.Dataset:
            def download(product: Product, tmpdir: str) -> None:
                start = time.time()
                filename = next(
                    entry for entry in product.entries if entry.endswith(".nat")
                )
                try:
                    with (
                        product.open(entry=filename) as fsrc,
                        open(f"{tmpdir}/{fsrc.name}", "wb") as fdst,
                    ):
                        shutil.copyfileobj(fsrc, fdst)
                except urllib3.exceptions.ProtocolError as e:
                    logging.error("%s with error: %s, skipping file", fsrc.name, e)
                end = time.time()
                logging.debug("Downloading %s took %.2f seconds", filename, end - start)

            def open_and_resample(
                filenames: list[str], variables: list[tuple[str, dict]]
            ) -> list[xr.Dataset]:
                width, height = get_nrows_ncols_from_domain_size_and_reskm(
                    bbox, self.res_km
                )
                area = AreaDefinition(
                    area_id="custom_bbox",
                    proj_id="custom_bbox",
                    description="Custom bounding box area",
                    projection="EPSG:4326",
                    width=width,
                    height=height,
                    area_extent=bbox,
                )
                ds_list = []
                for filename in filenames:
                    start = time.time()
                    try:
                        scn = Scene(reader="seviri_l1b_native", filenames=[filename])
                        varnames = [self.variables[var[0]][0] for var in variables]
                        scn.load(varnames)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="Upgrade 'pyresample' for a more accurate default 'radius_of_influence'.",
                            )
                            try:
                                scn = scn.resample(area, resampler="nearest")
                            except Exception as e:
                                logging.error(
                                    "%s with error: %s, skipping file", filename, e
                                )
                                continue

                        data_arrays = {}
                        for varname in varnames:
                            data = scn[varname]
                            data.attrs = {}
                            data = data.drop_vars("crs")
                            data_arrays[varname] = data

                        # add time dimension
                        timestamp = round_to_nearest_15_minutes(
                            datetime.datetime.strptime(
                                filename.split("-")[5].split(".")[0], "%Y%m%d%H%M%S"
                            )
                        )
                        ds = xr.Dataset(data_arrays).expand_dims(time=[timestamp])
                        ds_list.append(ds)
                        end = time.time()
                        logging.debug(
                            "Resampling %s took %.2f seconds", filename, end - start
                        )
                    except ValueError as e:
                        logging.error("%s with error: %s, skipping file", filename, e)
                return ds_list

            with TemporaryDirectory() as tmpdir:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(download, product, tmpdir)
                        for product in products
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logging.error("Error during download: %s", e)

                filenames = glob.glob(f"{tmpdir}/*.nat")
                ds_list = open_and_resample(filenames, variables)
                start = time.time()
                ds: xr.Dataset = xr.concat(ds_list, dim="time").sortby("time").load()
                end = time.time()
                logging.debug("Loading took %.2f seconds", end - start)
                return ds

        ds_list = []
        for date in dates:
            start = datetime.datetime.combine(
                date, datetime.time(0, 0, 0)
            ) - datetime.timedelta(minutes=10)
            end = datetime.datetime.combine(
                date, datetime.time(23, 59, 59)
            ) - datetime.timedelta(minutes=10)
            if test:
                end = datetime.datetime.combine(date, datetime.time(0, 30, 0))
            # Retrieve datasets that match our filter
            products = selected_collection.search(dtstart=start, dtend=end)
            # count number of images per satellite
            counts = {i: 0 for i in range(1, 5)}
            for product in products:
                # product.satellite contains the number (e.g. MSG3)
                counts[int(product.satellite[3])] += 1
            # pick the maximum msg (higher has priority)
            max_msg = max(counts, key=lambda k: (counts[k], k))
            logging.debug("Counts per satellite: %s", counts)
            logging.debug("Maximum MSG selected: %d", max_msg)
            products = [p for p in products if int(p.satellite[3]) == max_msg]
            # download the data
            with dask.config.set(num_workers=max_workers):
                ds = download_and_resample(products, variables)
            ds_list.append(ds)

        # merge data for all dates
        ds = xr.concat(ds_list, dim="time")
        # rename all variables back to "global" variable names
        ds = ds.rename(
            {k[0]: j for j, k in self.variables.items() if k[0] in ds.data_vars}
        )
        return ds
