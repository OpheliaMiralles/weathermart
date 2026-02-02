import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype

TITAN_VARS = [
    "precipitation_amount_ens-raw_ens_mean",
    "precipitation_amount_ens-force_range_control",
    "precipitation_amount_ens-force_range_mean",
    "precipitation_amount_ens-oi_control",
    "precipitation_amount_ens-oi_mean",
    "precipitation_amount-raw_control",
    "precipitation_amount-raw_ens_mean",
    "precipitation_amount-consensus",
    "precipitation_amount-downscaling",
    "precipitation_amount",
]


class TitanRetriever(BaseRetriever):
    """
    Retriever for TITAN Nordic analysis diagnostics (hourly).
    """

    sources = ("TITAN",)
    crs = "epsg:4326"
    variables = list(TITAN_VARS)

    def __init__(
        self,
        root: str
        | Path = "/lustre/storeB/project/metkl/klinogrid/archive/met_nordic_analysis/v4_work",
    ):
        self.root = Path(root)

    def retrieve(
        self,
        source: str,
        variables: list[str] | str,
        dates: datetime.date | str | pd.Timestamp | list[Any],
    ) -> xr.Dataset:
        dates, variables = checktype(dates, variables)
        varnames = [v[0] for v in variables]

        files = []
        for d in dates:
            day = pd.to_datetime(d)
            y, m, dd = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
            folder = self.root / y / m / dd
            files.extend(sorted(folder.glob("met_analysis_diagnostic_nordic_*.nc")))

        if not files:
            raise RuntimeError("No TITAN files found.")
        try:
            ds = xr.open_mfdataset(
                files,
                combine="by_coords",
                engine="netcdf4",
                parallel=True,
            )
        except Exception:
            ds = xr.open_mfdataset(
                files,
                combine="by_coords",
                engine="h5netcdf",
                parallel=True,
            )
        keep = [v for v in varnames if v in ds.data_vars]
        ds = ds[keep].set_coords(
            [
                "latitude",
                "longitude",
                "altitude",
                "land_area_fraction",
                "forecast_reference_time",
            ]
        )
        ds.attrs.update(source="TITAN", provider="MET Norway")
        return ds
