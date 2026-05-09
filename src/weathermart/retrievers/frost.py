import datetime
import json
import logging
import os
from typing import Any

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.base import variables_metadata
from weathermart.utils import NORDIC_DOMAIN
from weathermart.utils import batched

frost_vars = list(variables_metadata[variables_metadata.source == "FROST"].short_name.unique())

def frost_observations_to_dataframe(
    data: list[dict]
) -> pd.DataFrame:

    rows = []
    for item in data:
        raw_source_id = item.get("sourceId")
        base_source_id = str(raw_source_id).split(":")[0]
        base = {
            "sourceId": raw_source_id,
            "id": base_source_id,
            "time": item.get("referenceTime"),
        }

        for obs in item.get("observations", []):
            row = base.copy()
            row.update(obs)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    long = pd.DataFrame(rows)

    long["time"] = pd.to_datetime(long["time"], utc=True)
    index_cols = [
        "id",
        "time",
    ]
    index_cols = [col for col in index_cols if col in long.columns]
    values = (
        long.pivot_table(
            index=index_cols,
            columns="elementId",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    return values

def frost_locations_to_gdf(data):
    features = []

    for item in data:
        geometry = item.get("geometry")

        if geometry is None:
            continue

        geometry = geometry.copy()

        if "@type" in geometry:
            geometry["type"] = geometry.pop("@type")

        properties = {
            key: value
            for key, value in item.items()
            if key != "geometry"
        }

        if "nearest" in geometry:
            properties["nearest"] = geometry.pop("nearest")

        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

class FrostRetriever(BaseRetriever):
    """
    Class for retrieving data from the frost service.

    Attributes
    ----------
    URL : str
        Base URL for jretrieve API.
    sources : tuple of str
        Collection of supported source names.
    crs : str
        Coordinate reference system used for the retrieved data.
    variables : dict
        Mapping of variables to their frost short names.
    """
    FROST_URL = "https://frost.met.no"
    sources = ("OBSERVATIONS", "LIGHTNING")
    crs = "epsg:4326"
    variables = frost_vars

    @staticmethod
    def request_from_frost(
        endpoint: str,
        client_id: str,
        client_secret: str,
        args: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """
        Send a GET request to a specified jretrieve endpoint with optional query arguments.

        Parameters
        ----------
        endpoint : str
            The jretrieve API endpoint (appended to the base URL).
        client_id : str
            Client ID for jretrieve authentication.
        client_secret : str
            Client secret for jretrieve authentication.
        args : dict, optional
            Dictionary of query parameters to include in the request.

        Returns
        -------
        httpx.Response
            The response object from the jretrieve request.

        Raises
        ------
        httpx.HTTPStatusError
            If the response status code indicates an error.
        """
        url = f"{FrostRetriever.FROST_URL}/{endpoint}/v0.jsonld"
        CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE", True) or os.environ.get(
            "CA_BUNDLE", True
        )
        with httpx.Client(
            verify=CA_BUNDLE,
            timeout=None,
            auth=(client_id, client_secret),
        ) as client:
            resp = client.get(url, params=args)

        resp.raise_for_status()
        return resp

    @staticmethod
    def get_stations(
        client_id: str,
        client_secret: str,
        longitude_range: tuple[float, float] = NORDIC_DOMAIN[:2],
        latitude_range: tuple[float, float] = NORDIC_DOMAIN[2:],
        stations: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        resp = FrostRetriever.request_from_frost(
            endpoint="sources",
            client_id=client_id,
            client_secret=client_secret,
            args={"ids": ",".join(stations) if stations is not None else None},
        ).json()
        geodf = frost_locations_to_gdf(resp.get("data", []))
        geodf = (
            geodf.rename(columns={"shortName": "stationName"})
            .assign(x=lambda x: x.geometry.x)
            .assign(y=lambda x: x.geometry.y)
        )
        geodf = geodf[
            (geodf.y <= latitude_range[1])
            & (geodf.y >= latitude_range[0])
            & (geodf.x >= longitude_range[0])
            & (geodf.x <= longitude_range[1])
        ]
        return geodf.drop_duplicates("id")[["id", "stationName", "x", "y", "geometry"]]


    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        credentials_path: str | None = None,
        temporal_resolution: str = "T",
        stations: list[str] | None = None,
        longitude_range: tuple[float, float] = NORDIC_DOMAIN[:2],
        latitude_range: tuple[float, float] = NORDIC_DOMAIN[2:],
    ) -> xr.Dataset:
        """
        Retrieve data from the frost service for specified parameters.

        This method constructs a time range based on the provided dates, prepares
        query parameters, batches station requests, retrieves data via jretrieve,
        processes the results into a GeoDataFrame, and finally converts to an xarray.Dataset.

        Parameters
        ----------
        source : str
            Source identifier (e.g., 'OBSERVATIONS').
        variables : list of tuple of (str, dict)
            List of variable tuples. Each tuple comprises the variable name and associated parameters.
        dates : list of datetime.date or datetime.date
            A date or list of dates for which to retrieve the data.
        credentials_path : str, optional
            Path to the jretrieve credentials file. If not provided, the environment variables
            CLIENT_ID and CLIENT_SECRET must be set.
        temporal_resolution : str, optional
            Temporal resolution for the data. Default is 'T'. Can be 'H'.
        stations : list[str], optional
            stations id list. Default is None and retrieves SwissMetNet stations.
        longitude_range : tuple of float, optional
            Longitude range for filtering stations. Default is (0.5, 16.5).
        latitude_range : tuple of float, optional
            Latitude range for filtering stations. Default is (43.0, 50.0).

        Returns
        -------
        xarray.Dataset
            Dataset containing the retrieved data with coordinates 'date' and 'station',
            and associated x, y attributes.

        Raises
        --------
        ValueError
            If the use_limitation is set to a value greater than 40.
        RuntimeError
            If the credentials are not provided and the environment variables are not set.
        FileNotFoundError
            If the specified jretrieve credentials file is not found.
        KeyError
            If the credentials file does not contain the required keys.
        """
        dates, variables = checktype(dates, variables)
        if credentials_path is None:
            try:
                logging.warning(
                    "Using FROST_ID and FROST_SECRET from environment variables."
                )
                client_id = os.environ["FROST_ID"]
                client_secret = os.environ["FROST_SECRET"]
            except KeyError as e:
                raise RuntimeError(
                    "FROST_ID and FORCE_SECRET environment variable or credentials_path"
                    " must be set to retrieve data from the frost."
                ) from e
        else:
            try:
                with open(credentials_path, encoding="utf-8") as f:
                    credentials = json.load(f)

            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Frost credentials file not found at {credentials_path}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Error reading credentials token file: {e}") from e
            try:
                client_id = credentials["client_id"]
                client_secret = credentials["client_secret"]
            except KeyError as e:
                raise KeyError(
                    "client_id and client_secret must be provided in the credentials file."
                ) from e

        if len(dates) == 1:
            start = pd.to_datetime(dates).strftime("%Y-%m-%dT%H:%M:%S")[0]
            stop = (pd.to_datetime(dates) + pd.to_timedelta("23h55m")).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )[0]
        else:
            isodates = [pd.to_datetime(d).strftime("%Y-%m-%dT%H:%M:%S") for d in dates]
            start, stop = isodates[0], isodates[-1]
        timerange = "/".join([start, stop])
        coords_station_df = FrostRetriever.get_stations(
                client_id=client_id,
                client_secret=client_secret,
                longitude_range=longitude_range,
                latitude_range=latitude_range,
                stations=stations,
            )
        stations = coords_station_df["id"].tolist()
        avail_for_date = [v["sourceId"].replace(":0", "") for v in FrostRetriever.request_from_frost(
                endpoint="observations/availableTimeSeries",
                client_id=client_id,
                client_secret=client_secret,
                args={
                    "elements": ",".join(variables),
                    "referencetime": timerange,
                }).json()["data"]]
        to_concat = []
        i = 0
        batches = batched(stations, 500)
        nr_batches = len(list(batched(stations, 500)))
        for s_slice in batches:
            logging.info(
                "Retrieving data for batch %s/%s of 500 stations", i, nr_batches
            )
            
            s_slice = [s for s in s_slice if s in avail_for_date]
            query_args = {
                "sources": ",".join(s_slice),
                "elements": ",".join(variables),
                "referencetime": timerange,
            }
            df = FrostRetriever.request_from_frost(
                endpoint=source.lower(),
                client_id=client_id,
                client_secret=client_secret,
                args=query_args,
            ).json()["data"]
            geodf = frost_observations_to_dataframe(df)
            if len(geodf) == 0:
                logging.warning("No data available for batch %s of 500 stations", i)
            else:
                geodf = (
                    geodf.merge(coords_station_df, on="id", how="left")
                    .set_geometry("geometry")
                )
                # weird xarray issue when converting dataframe pandas with timezone info
                # it is currently being resolved, this is a temporary fix
                geodf["time"] = pd.to_datetime(geodf["time"])
                geodf = geodf.set_index("time")
                geodf.index = geodf.index.tz_convert("UTC").tz_localize(None)
                to_concat.append(geodf)
            i += 1
        geodf = pd.concat(to_concat).set_geometry("geometry").reset_index().rename(columns = {"id": "station"})
        coords_station_df = coords_station_df.rename(columns={"id": "station"})
        # fill variables with nans for non-retrieved stations
        coords_station_xa = (
            xr.Dataset(
                data_vars={
                    var: (
                        ("station",),
                        np.full(len(coords_station_df["station"]), np.nan),
                    )
                    for var in variables
                },
                coords={
                    "station": coords_station_df["station"],
                    "x": ("station", coords_station_df["x"]),
                    "y": ("station", coords_station_df["y"]),
                    "stationName": ("station", coords_station_df["stationName"]),
                    "longitude": ("station", coords_station_df["x"]),
                    "latitude": ("station", coords_station_df["y"]),
                },
            )
            .expand_dims(time=geodf.time.unique())
            .drop_duplicates("station")
        )
        retrieved_stations = geodf.drop_duplicates("station")
        array = (
            geodf.set_index(["time", "station"])
            .to_xarray()
            .drop_vars(["geometry", "x", "y", "stationName"])
            .assign_coords(
                {
                    "x": ("station", retrieved_stations["x"]),
                    "y": ("station", retrieved_stations["y"]),
                    "stationName": ("station", retrieved_stations["stationName"]),
                    "longitude": ("station", retrieved_stations["x"]),
                    "latitude": ("station", retrieved_stations["y"]),
                }
            )
        )
        missing_stations = set(stations).difference(set(geodf["station"]))
        array = xr.concat(
            [array, coords_station_xa.sel(station=list(missing_stations))],
            dim="station",
        )
        array.attrs.update(
            {
                "source": source,
                "temporal_resolution": temporal_resolution,
                "retriever": "FrostRetriever",
            }
        )
        array.attrs.update({"crs": self.crs})
        array["station"] = array.station.astype("str")
        return array[variables]