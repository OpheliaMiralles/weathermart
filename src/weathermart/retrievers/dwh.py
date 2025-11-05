import base64
import datetime
import json
import logging
import os
import urllib
import urllib.request
from typing import Any

import geopandas
import httpx
import numpy as np
import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.base import variables_metadata
from weathermart.utils import batched

dwh_dic = {
    k: [k]
    for k in variables_metadata[variables_metadata.source == "DWH"].short_name.unique()
}
suffixes = {
    "instantaneous": "s0",
    "10min_mean": "z0",
    "hourly_mean": "h0",
    "hourly_max": "h1",
}
temp_res = {"T": ["instantaneous", "10min_mean"], "H": ["hourly_mean", "hourly_max"]}
dwh_dic = {k: v for k, v in dwh_dic.items() if v[0] not in ["", "-"]}
# adds hourly mean if no hourly variable is defined in the data model
for var, dwh_var in dwh_dic.items():
    avail_suf = dwh_var[0][-2:]
    possible_hourly_suffixes = [s for i, s in suffixes.items() if i in temp_res["H"]]
    if avail_suf not in possible_hourly_suffixes:
        dwh_dic[var].append(dwh_var[0][:-2] + possible_hourly_suffixes[0])

MCH_STATIONS = [
    "ARO",
    "RAG",
    "HAI",
    "HLL",
    "DEM",
    "EBK",
    "ELM",
    "EIN",
    "ANT",
    "MER",
    "CHD",
    "GRA",
    "CHM",
    "LAG",
    "KOP",
    "BLA",
    "GRO",
    "BEH",
    "SIA",
    "SMM",
    "DAV",
    "CHU",
    "ROB",
    "SAM",
    "SCU",
    "DOL",
    "PAY",
    "JUN",
    "WYN",
    "SAE",
    "VAD",
    "AIG",
    "MLS",
    "FAH",
    "MVE",
    "ZER",
    "CHA",
    "PIL",
    "ALT",
    "ULR",
    "PIO",
    "LUG",
    "NAP",
    "SIO",
    "MAG",
    "NEU",
    "SBO",
    "INT",
    "DIS",
    "STG",
    "GLA",
    "GVE",
    "KLO",
    "GUE",
    "PUY",
    "GSB",
    "ABO",
    "VIS",
    "CDF",
    "RUE",
    "BUS",
    "LUZ",
    "ENG",
    "SHA",
    "SMA",
    "SBE",
    "WFJ",
    "COV",
    "BAS",
    "CGI",
    "FRE",
    "BER",
    "GUT",
    "GOE",
    "WAE",
    "TAE",
    "REH",
    "OTL",
    "BEZ",
    "MUB",
    "CIM",
    "EVO",
    "LEI",
    "GRH",
    "COM",
    "LAE",
    "HOE",
    "PLF",
    "ROE",
    "BIN",
    "MAR",
    "SAG",
    "CHZ",
    "COY",
    "VEV",
    "BOU",
    "TIT",
    "FRU",
    "MOE",
    "MAH",
    "CHB",
    "MTE",
    "PRE",
    "VLS",
    "ARH",
    "GOS",
    "AND",
    "BIV",
    "MTR",
    "BIA",
    "SCM",
    "AEG",
    "LAT",
    "MOB",
    "CDM",
    "SIM",
    "BUF",
    "BIZ",
    "PFA",
    "FLU",
    "BOL",
    "THU",
    "MAS",
    "VIT",
    "GRE",
    "MOA",
    "CEV",
    "CRM",
    "CMA",
    "GEN",
    "BIE",
    "ORO",
    "BRL",
    "EGO",
    "STK",
    "DIA",
    "BRZ",
    "SPF",
    "QUI",
    "VAB",
    "PMA",
    "NAS",
    "ATT",
    "EVI",
    "GOR",
    "EGH",
    "GIH",
    "GES",
    "SRS",
    "VIO",
    "OBR",
]


class DWHRetriever(BaseRetriever):
    """
    Class for retrieving data from the DWH (Data Warehouse) service via jretrieve.

    Attributes
    ----------
    JRETRIEVE_URL : str
        Base URL for jretrieve API.
    JRETRIEVE_TOKEN_ENDPOINT : str
        Endpoint URL for token retrieval.
    sources : tuple of str
        Collection of supported source names.
    crs : str
        Coordinate reference system used for the retrieved data.
    variables : dict
        Mapping of variables to their DWH short names.
    """

    # jretrieve-related setup
    JRETRIEVE_URL = "https://service.meteoswiss.ch/jretrieve/api/v1/"
    JRETRIEVE_TOKEN_ENDPOINT = "https://service.meteoswiss.ch/auth/realms/meteoswiss.ch/protocol/openid-connect/token"
    # supported jretrieve endpoint in uppercase
    sources = ("SURFACE",)
    crs = "epsg:4326"
    variables = dwh_dic
    # metadata - everything that is not an image dimension

    @staticmethod
    def retrieve_token(JRETRIEVE_CLIENT_ID: str, JRETRIEVE_CLIENT_SECRET: str) -> str:
        """
        Retrieve an authentication token from the DWH token endpoint.

        Parameters
        ----------
        JRETRIEVE_CLIENT_ID : str
            Client ID for jretrieve authentication.
        JRETRIEVE_CLIENT_SECRET : str
            Client secret for jretrieve authentication.

        Returns
        -------
        str
            Authorization header string in the format 'Bearer <access_token>'.
        """
        url = DWHRetriever.JRETRIEVE_TOKEN_ENDPOINT
        with urllib.request.urlopen(
            urllib.request.Request(
                method="POST",
                url=url,
                data=b"grant_type=client_credentials",
                headers={
                    "Authorization": "Basic "
                    + base64.b64encode(
                        f"{JRETRIEVE_CLIENT_ID}:{JRETRIEVE_CLIENT_SECRET}".encode()
                    ).decode()
                },
            )
        ) as f:
            auth_header = (
                "Bearer " + json.loads(f.read().decode("utf-8"))["access_token"]
            )

        return auth_header

    @staticmethod
    def request_from_jretrieve(
        endpoint: str,
        jretrieve_client_id: str,
        jretrieve_client_secret: str,
        args: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """
        Send a GET request to a specified jretrieve endpoint with optional query arguments.

        Parameters
        ----------
        endpoint : str
            The jretrieve API endpoint (appended to the base URL).
        jretrieve_client_id : str
            Client ID for jretrieve authentication.
        jretrieve_client_secret : str
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
        url = DWHRetriever.JRETRIEVE_URL + endpoint
        auth_header = DWHRetriever.retrieve_token(
            jretrieve_client_id, jretrieve_client_secret
        )
        CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE", True) or os.environ.get(
            "CA_BUNDLE", True
        )
        with httpx.Client(
            headers={"Authorization": auth_header}, verify=CA_BUNDLE, timeout=None
        ) as client:
            resp: httpx.Response = client.get(url, params=args)
        resp.raise_for_status()
        return resp

    @staticmethod
    def get_all_jretrieve_stations(
        jretrieve_client_id: str,
        jretrieve_client_secret: str,
        longitude_range: tuple[float, float] = (0.5, 16.5),
        latitude_range: tuple[float, float] = (43.0, 50.0),
    ) -> geopandas.GeoDataFrame:
        """Retrieve all stations from jretrieve within a specified geographical bounding box."""
        resp = DWHRetriever.request_from_jretrieve(
            endpoint="meta_info?infoOptions=nat_abbr,lat,lon,name&format=json-GeoJSON&headerDisabled=true",
            jretrieve_client_id=jretrieve_client_id,
            jretrieve_client_secret=jretrieve_client_secret,
        ).json()
        geodf = geopandas.GeoDataFrame.from_features(resp)
        geodf = (
            geodf.rename(columns={"location": "station"})
            .assign(x=lambda x: x.geometry.x)
            .assign(y=lambda x: x.geometry.y)
        )
        geodf = geodf[
            (geodf.y <= latitude_range[1])
            & (geodf.y >= latitude_range[0])
            & (geodf.x >= longitude_range[0])
            & (geodf.x <= longitude_range[1])
        ]
        return geodf.drop_duplicates("station")

    @staticmethod
    def get_jretrieve_stations(
        jretrieve_client_id: str,
        jretrieve_client_secret: str,
        parameters: list[str],
        use_limitation: int = 20,
        longitude_range: tuple[float, float] = (0.5, 16.5),
        latitude_range: tuple[float, float] = (43.0, 50.0),
    ) -> geopandas.GeoDataFrame:
        """Retrieve stations available for specified parameters within a geographical bounding box and for a given user limitation."""
        resp = DWHRetriever.request_from_jretrieve(
            endpoint=f"meta_info?parameterShortNames={','.join(parameters)}&infoOptions=nat_abbr,use_limitation,lat,lon,name,data_owner&format=json-GeoJSON&headerDisabled=true",
            jretrieve_client_id=jretrieve_client_id,
            jretrieve_client_secret=jretrieve_client_secret,
        ).json()
        geodf = geopandas.GeoDataFrame.from_features(resp)
        geodf["useLimitation"] = geodf["useLimitation"].astype(int)
        geodf = (
            geodf.rename(columns={"location": "station"})
            .assign(x=lambda x: x.geometry.x)
            .assign(y=lambda x: x.geometry.y)
        )
        # filter out all stations with too high useLimitation and outside domain of interest
        geodf = geodf[
            (geodf.y <= latitude_range[1])
            & (geodf.y >= latitude_range[0])
            & (geodf.x >= longitude_range[0])
            & (geodf.x <= longitude_range[1])
            & (geodf.useLimitation <= use_limitation)
        ]
        return geodf.drop_duplicates("station")

    @staticmethod
    def get_coords_for_stations(
        stations: str | list[str],
        jretrieve_client_id: str,
        jretrieve_client_secret: str,
    ) -> geopandas.GeoDataFrame:
        """Retrieve coordinates for a list of station IDs."""
        if isinstance(stations, list):
            stations = ",".join(stations)
        stations = stations.replace("_", "").replace(" ", "")
        resp = DWHRetriever.request_from_jretrieve(
            endpoint=f"meta_info?locationIds={stations}&infoOptions=nat_abbr,lat,lon,name,elev&format=json-GeoJSON&headerDisabled=true",
            jretrieve_client_id=jretrieve_client_id,
            jretrieve_client_secret=jretrieve_client_secret,
        ).json()
        geodf = geopandas.GeoDataFrame.from_features(resp)
        geodf = (
            geodf.rename(columns={"location": "station"})
            .assign(x=lambda x: x.geometry.x)
            .assign(y=lambda x: x.geometry.y)
        )
        return geodf.drop_duplicates("station")

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        jretrieve_credentials_path: str | None = None,
        temporal_resolution: str = "T",
        use_limitation: int | None = None,
        longitude_range: tuple[float, float] = (0.5, 16.5),
        latitude_range: tuple[float, float] = (43.0, 50.0),
    ) -> xr.Dataset:
        """
        Retrieve data from the DWH service for specified parameters.

        This method constructs a time range based on the provided dates, prepares
        query parameters, batches station requests, retrieves data via jretrieve,
        processes the results into a GeoDataFrame, and finally converts to an xarray.Dataset.

        Parameters
        ----------
        source : str
            Source identifier (e.g., 'SURFACE').
        variables : list of tuple of (str, dict)
            List of variable tuples. Each tuple comprises the variable name and associated parameters.
        dates : list of datetime.date or datetime.date
            A date or list of dates for which to retrieve the data.
        jretrieve_credentials_path : str, optional
            Path to the jretrieve credentials file. If not provided, the environment variables
            JRETRIEVE_CLIENT_ID and JRETRIEVE_CLIENT_SECRET must be set.
        temporal_resolution : str, optional
            Temporal resolution for the data. Default is 'T'. Can be 'H'.
        use_limitation : int, optional
            useLimitation parameter for jretrieve. Default is None and retrieves SwissMetNet stations.
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
        possible_suffixes = [
            s for i, s in suffixes.items() if i in temp_res[temporal_resolution]
        ]
        if jretrieve_credentials_path is None:
            try:
                logging.warning(
                    "Using JRETRIEVE_CLIENT_ID and JRETRIEVE_CLIENT_SECRET from environment variables."
                )
                jretrieve_client_id = os.environ["JRETRIEVE_CLIENT_ID"]
                jretrieve_client_secret = os.environ["JRETRIEVE_CLIENT_SECRET"]
            except KeyError as e:
                raise RuntimeError(
                    "JRETRIEVE_CLIENT_ID and JRETRIEVE_CLIENT_SECRET environment variable or jretrieve_credentials_path"
                    " must be set to retrieve data from the DWH."
                ) from e
        else:
            try:
                with open(jretrieve_credentials_path, encoding="utf-8") as f:
                    credentials = json.load(f)

            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"JRetrieve credentials file not found at {jretrieve_credentials_path}"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Error reading credentials token file: {e}") from e
            try:
                jretrieve_client_id = credentials["client_id"]
                jretrieve_client_secret = credentials["client_secret"]
            except KeyError as e:
                raise KeyError(
                    "JRETRIEVE_CLIENT_ID and JRETRIEVE_CLIENT_SECRET must be provided in the credentials file."
                ) from e

        if len(dates) == 1:
            start = pd.to_datetime(dates).strftime("%Y%m%d%H%M%S")[0]
            stop = (pd.to_datetime(dates) + pd.to_timedelta("23h50m")).strftime(
                "%Y%m%d%H%M%S"
            )[0]
        else:
            isodates = [pd.to_datetime(d).strftime("%Y%m%d%H%M%S") for d in dates]
            start, stop = isodates[0], isodates[-1]
        timerange = "-".join([start, stop])
        jretrieve_variables = sum((self.variables[v] for v, _ in variables), [])
        jretrieve_variables = [
            v for v in jretrieve_variables if v[-2:] in possible_suffixes
        ]
        if use_limitation is None:
            stations = MCH_STATIONS
            coords_station_df = DWHRetriever.get_coords_for_stations(
                stations, jretrieve_client_id, jretrieve_client_secret
            )
        else:
            coords_station_df = DWHRetriever.get_jretrieve_stations(
                jretrieve_client_id,
                jretrieve_client_secret,
                jretrieve_variables,
                use_limitation,
                longitude_range,
                latitude_range,
            )
            stations = coords_station_df["station"].tolist()

        if use_limitation is not None and use_limitation > 20:
            logging.warning(
                "use_limitation in jretrieve is set to %s, which is higher than the default of 20.",
                use_limitation,
            )
        elif use_limitation is not None and use_limitation > 40:
            raise ValueError(
                f"use_limitation is set to {use_limitation}, which is higher than 40."
            )

        to_concat = []
        i = 0
        batches = batched(stations, 500)
        nr_batches = len(list(batched(stations, 500)))
        for s_slice in batches:
            logging.info(
                "Retrieving data for batch %s/%s of 500 stations", i, nr_batches
            )
            query_args = {
                "locationIds": ",".join(s_slice),
                "parameterShortNames": ",".join(jretrieve_variables),
                "date": timerange,
                "infoOptions": "nat_abbr,lat,lon,name,data_owner",
                "format": "json-GeoJSON",
                "placeholder": pd.NA,
                "headerDisabled": True,
                "measCatNr": 1,
            }
            if use_limitation is not None:
                query_args["useLimitation"] = use_limitation
            df = DWHRetriever.request_from_jretrieve(
                endpoint=source.lower(),
                jretrieve_client_id=jretrieve_client_id,
                jretrieve_client_secret=jretrieve_client_secret,
                args=query_args,
            ).json()
            geodf = geopandas.GeoDataFrame.from_features(df)
            if len(geodf) == 0:
                logging.warning("No data available for batch %s of 500 stations", i)
            else:
                for col in jretrieve_variables:
                    if col in geodf.columns:
                        geodf[col] = geodf[col].replace("", np.nan).astype(float)
                geodf = (
                    geodf.rename(columns={"location": "station"})
                    .assign(x=lambda x: x.geometry.x)
                    .assign(y=lambda x: x.geometry.y)
                )
                # weird xarray issue when converting dataframe pandas with timezone info
                # it is currently being resolved, this is a temporary fix
                geodf["time"] = pd.to_datetime(geodf["time"])
                geodf = geodf.set_index("time")
                geodf.index = geodf.index.tz_convert("UTC").tz_localize(None)
                to_concat.append(geodf)
            i += 1
        geodf = pd.concat(to_concat).set_geometry("geometry").reset_index()
        # fill variables with nans for non-retrieved stations
        coords_station_xa = (
            xr.Dataset(
                data_vars={
                    var: (
                        ("station",),
                        np.full(len(coords_station_df["station"]), np.nan),
                    )
                    for var in jretrieve_variables
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
        if "dateOwner" in coords_station_df.columns:
            coords_station_xa = coords_station_xa.assign_coords(
                {"dataOwner": ("station", coords_station_df["dataOwner"])}
            )
            geodf = (
                geodf.assign_coords(
                    {"dataOwner": ("station", retrieved_stations["dataOwner"])}
                ),
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
                "retriever": "DWHRetriever",
            }
        )
        array.attrs.update({"crs": self.crs})
        array["station"] = array.station.astype("str")
        return array[jretrieve_variables]
