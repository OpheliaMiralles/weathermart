import datetime
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.base import variables_metadata
from weathermart.utils import NORDIC_DOMAIN
from weathermart.utils import batched

frost_vars = list(
    variables_metadata[variables_metadata.source == "FROST"].short_name.unique()
)
lightning_vars = [
    "lightning_count",
    "lightning_cloud_to_ground",
    "lightning_intracloud",
]
frost_vars = list(dict.fromkeys(frost_vars + lightning_vars))

LIGHTNING_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "nanoseconds",
    "lat",
    "lon",
    "peak_current",
    "multi",
    "nsens",
    "dof",
    "angle",
    "major",
    "minor",
    "chi2",
    "rt",
    "ptz",
    "mrr",
    "cloud",
    "aI",
    "sI",
    "tI",
]
LIGHTNING_TEMPLATE_PATH = Path(
    os.environ.get(
        "LIGHTNING_TEMPLATE_PATH",
        "/lustre/storeB/users/opmir9231/nordic_radar/20250101",
    )
)


def _wkt_polygon_from_bounds(
    longitude_range: tuple[float, float], latitude_range: tuple[float, float]
) -> str:
    lon_min, lon_max = longitude_range
    lat_min, lat_max = latitude_range
    return (
        "POLYGON(("
        f"{lon_min} {lat_min},"
        f"{lon_max} {lat_min},"
        f"{lon_max} {lat_max},"
        f"{lon_min} {lat_max},"
        f"{lon_min} {lat_min}"
        "))"
    )


def _parse_lightning_ualf(text: str) -> pd.DataFrame:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return pd.DataFrame()

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=r"\s+",
        names=LIGHTNING_COLUMNS,
        header=None,
        engine="python",
    )
    for col in ["year", "month", "day", "hour", "minute", "second", "cloud"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=["year", "month", "day", "hour", "minute", "second", "lat", "lon"]
    )
    if df.empty:
        return df

    df["time"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]].astype(int),
        utc=True,
    ).dt.tz_convert(None)
    return df


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Grid coordinates must be 1D.")
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    deltas = np.diff(values)
    edges = np.empty(len(values) + 1, dtype=np.float64)
    edges[1:-1] = values[:-1] + deltas / 2.0
    edges[0] = values[0] - deltas[0] / 2.0
    edges[-1] = values[-1] + deltas[-1] / 2.0
    return edges


def _open_template_grid(
    template_path: os.PathLike[str] | str | None,
    template_crs: str | None = None,
) -> tuple[xr.Dataset, str, str, np.ndarray, np.ndarray, str]:
    path = Path(template_path) if template_path is not None else LIGHTNING_TEMPLATE_PATH
    if path.is_dir():
        ds = xr.open_zarr(path)
    else:
        ds = xr.open_dataset(path)
    if "time" in ds.dims:
        ds = ds.isel(time=0)

    x_name = next((name for name in ("Xc", "x") if name in ds.coords), None)
    y_name = next((name for name in ("Yc", "y") if name in ds.coords), None)
    if x_name is None or y_name is None:
        raise ValueError(
            "Template grid must expose 1D x/y coordinates via Xc/Yc or x/y."
        )

    x = np.asarray(ds[x_name].values, dtype=np.float64)
    y = np.asarray(ds[y_name].values, dtype=np.float64)

    inferred_crs = (
        template_crs
        or ds.attrs.get("crs")
        or ds.attrs.get("grid_mapping")
        or ds.attrs.get("proj4")
    )
    if not inferred_crs:
        from weathermart.retrievers.radar import NordicRadarRetriever

        inferred_crs = NordicRadarRetriever.crs["laea"]

    return ds, x_name, y_name, x, y, str(inferred_crs)


def frost_observations_to_dataframe(data: list[dict]) -> pd.DataFrame:
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
    values = long.pivot_table(
        index=index_cols,
        columns="elementId",
        values="value",
        aggfunc="first",
    ).reset_index()
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

        properties = {key: value for key, value in item.items() if key != "geometry"}

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
        fmt: str = "jsonld",
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
        url = f"{FrostRetriever.FROST_URL}/{endpoint}/v0.{fmt}"
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
    def _load_credentials(
        credentials_path: str | None,
        env_client_id: str = "CLIENT_ID",
        env_client_secret: str = "CLIENT_SECRET",
    ) -> tuple[str, str]:
        if credentials_path is None:
            client_id = os.getenv(env_client_id) or os.getenv("FROST_ID")
            client_secret = os.getenv(env_client_secret) or os.getenv("FROST_SECRET")
            if client_id is None or client_secret is None:
                raise RuntimeError(
                    "Frost credentials are missing. Set CLIENT_ID and CLIENT_SECRET or provide credentials_path."
                )
            return client_id, client_secret

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
            return credentials["client_id"], credentials["client_secret"]
        except KeyError as e:
            raise KeyError(
                "client_id and client_secret must be provided in the credentials file."
            ) from e

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

    def retrieve_lightning(
        self,
        source: str,
        variables: list[str] | str,
        dates: datetime.date | str | pd.Timestamp | list[Any],
        credentials_path: str | None = None,
        template_path: os.PathLike[str] | str | None = None,
        template_crs: str | None = None,
        longitude_range: tuple[float, float] = NORDIC_DOMAIN[:2],
        latitude_range: tuple[float, float] = NORDIC_DOMAIN[2:],
        geometry: str | None = None,
    ) -> xr.Dataset:
        client_id, client_secret = self._load_credentials(credentials_path)
        dates, variables = checktype(dates, variables)
        requested = [var for var in variables if var in self.variables]
        if not requested:
            raise ValueError(
                f"No lightning variables requested. Available variables are {self.variables}."
            )

        template, x_name, y_name, x, y, template_crs = _open_template_grid(
            template_path,
            template_crs=template_crs,
        )
        if geometry is None:
            if "lon" in template.coords and "lat" in template.coords:
                lon = template["lon"]
                lat = template["lat"]
                geometry = _wkt_polygon_from_bounds(
                    (float(np.nanmin(lon.values)), float(np.nanmax(lon.values))),
                    (float(np.nanmin(lat.values)), float(np.nanmax(lat.values))),
                )
            elif "longitude" in template.coords and "latitude" in template.coords:
                lon = template["longitude"]
                lat = template["latitude"]
                geometry = _wkt_polygon_from_bounds(
                    (float(np.nanmin(lon.values)), float(np.nanmax(lon.values))),
                    (float(np.nanmin(lat.values)), float(np.nanmax(lat.values))),
                )
            else:
                geometry = _wkt_polygon_from_bounds(longitude_range, latitude_range)

        transformer = Transformer.from_crs("EPSG:4326", template_crs, always_xy=True)
        x_edges = _centers_to_edges(x)
        y_edges = _centers_to_edges(y)

        day_datasets: list[xr.Dataset] = []
        for date in dates:
            day_start = pd.Timestamp(date).tz_localize("UTC").floor("D")
            day_stop = day_start + pd.Timedelta(days=1)
            response = self.request_from_frost(
                endpoint="lightning",
                client_id=client_id,
                client_secret=client_secret,
                args={
                    "referencetime": f"{day_start:%Y-%m-%dT%H:%M:%S}/{day_stop:%Y-%m-%dT%H:%M:%S}",
                    "geometry": geometry,
                },
                fmt="ualf",
            )
            df = _parse_lightning_ualf(response.text)
            full_times = pd.date_range(
                day_start,
                day_stop,
                freq="5min",
                inclusive="left",
                tz="UTC",
            ).tz_convert(None)
            if df.empty:
                empty = np.zeros((len(full_times), len(y), len(x)), dtype=np.float32)
                ds = xr.Dataset(
                    {
                        "lightning_count": (("time", y_name, x_name), empty),
                        "lightning_cloud_to_ground": (("time", y_name, x_name), empty),
                        "lightning_intracloud": (("time", y_name, x_name), empty),
                    },
                    coords={"time": full_times, y_name: y, x_name: x},
                )
                ds = ds.assign_coords(
                    longitude=(
                        template["longitude"]
                        if "longitude" in template.coords
                        else template["lon"]
                    ),
                    latitude=(
                        template["latitude"]
                        if "latitude" in template.coords
                        else template["lat"]
                    ),
                )
                ds.attrs.update(
                    {
                        "source": source,
                        "retriever": "FrostRetriever",
                        "crs": template_crs,
                        "temporal_resolution": "5min",
                    }
                )
                day_datasets.append(ds[requested])
                continue

            lon_vals = df["lon"].to_numpy(dtype=np.float64)
            lat_vals = df["lat"].to_numpy(dtype=np.float64)
            x_vals, y_vals = transformer.transform(lon_vals, lat_vals)
            x_idx = np.digitize(x_vals, x_edges) - 1
            y_idx = np.digitize(y_vals, y_edges) - 1
            bucket = df["time"].dt.floor("5min")
            time_index = pd.Index(full_times)
            bucket_idx = time_index.get_indexer(bucket)
            cloud = (
                df["cloud"].fillna(1).to_numpy()
                if "cloud" in df
                else np.ones(len(df), dtype=np.float32)
            )
            valid = (
                (bucket_idx >= 0)
                & (x_idx >= 0)
                & (x_idx < len(x))
                & (y_idx >= 0)
                & (y_idx < len(y))
            )
            bucket_idx = bucket_idx[valid]
            x_idx = x_idx[valid]
            y_idx = y_idx[valid]
            cloud = cloud[valid]
            total = np.zeros((len(full_times), len(y), len(x)), dtype=np.float32)
            cloud_to_ground = np.zeros_like(total)
            intracloud = np.zeros_like(total)
            np.add.at(total, (bucket_idx, y_idx, x_idx), 1.0)
            cg_mask = cloud == 0
            if np.any(cg_mask):
                np.add.at(
                    cloud_to_ground,
                    (bucket_idx[cg_mask], y_idx[cg_mask], x_idx[cg_mask]),
                    1.0,
                )
            ic_mask = ~cg_mask
            if np.any(ic_mask):
                np.add.at(
                    intracloud,
                    (bucket_idx[ic_mask], y_idx[ic_mask], x_idx[ic_mask]),
                    1.0,
                )
            ds = xr.Dataset(
                {
                    "lightning_count": (("time", y_name, x_name), total),
                    "lightning_cloud_to_ground": (
                        ("time", y_name, x_name),
                        cloud_to_ground,
                    ),
                    "lightning_intracloud": (("time", y_name, x_name), intracloud),
                },
                coords={"time": full_times, y_name: y, x_name: x},
            )
            ds = ds.assign_coords(
                longitude=(
                    template["longitude"]
                    if "longitude" in template.coords
                    else template["lon"]
                ),
                latitude=(
                    template["latitude"]
                    if "latitude" in template.coords
                    else template["lat"]
                ),
            )
            ds.attrs.update(
                {
                    "source": source,
                    "retriever": "FrostRetriever",
                    "crs": template_crs,
                    "temporal_resolution": "5min",
                }
            )
            day_datasets.append(ds[requested])

        if not day_datasets:
            return xr.Dataset()
        if len(day_datasets) == 1:
            return day_datasets[0]
        return xr.concat(day_datasets, dim="time")

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
        template_path: os.PathLike[str] | str | None = None,
        template_crs: str | None = None,
        geometry: str | None = None,
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
        if source.upper() == "LIGHTNING":
            return self.retrieve_lightning(
                source=source,
                variables=variables,
                dates=dates,
                credentials_path=credentials_path,
                template_path=template_path,
                template_crs=template_crs,
                longitude_range=longitude_range,
                latitude_range=latitude_range,
                geometry=geometry,
            )

        client_id, client_secret = self._load_credentials(credentials_path)

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
        avail_for_date = [
            v["sourceId"].replace(":0", "")
            for v in FrostRetriever.request_from_frost(
                endpoint="observations/availableTimeSeries",
                client_id=client_id,
                client_secret=client_secret,
                args={
                    "elements": ",".join(variables),
                    "referencetime": timerange,
                },
            ).json()["data"]
        ]
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
                geodf = geodf.merge(
                    coords_station_df, on="id", how="left"
                ).set_geometry("geometry")
                # weird xarray issue when converting dataframe pandas with timezone info
                # it is currently being resolved, this is a temporary fix
                geodf["time"] = pd.to_datetime(geodf["time"])
                geodf = geodf.set_index("time")
                geodf.index = geodf.index.tz_convert("UTC").tz_localize(None)
                to_concat.append(geodf)
            i += 1
        geodf = (
            pd.concat(to_concat)
            .set_geometry("geometry")
            .reset_index()
            .rename(columns={"id": "station"})
        )
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
