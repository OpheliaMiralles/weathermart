import datetime
import os
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.base import variables_metadata

nwp_dic = {
    k: [k]
    for k in variables_metadata[
        variables_metadata.source == "ECCODES_COSMO"
    ].short_name.unique()
}
type_mapping = {
    "analysis": {"type": "det", "stream": "enda"},
    "forecast": {"type": "ememb", "stream": "enfo"},
}


def initialize_fdb() -> None:
    """
    Initialize the FDB environment for remote retrieval.

    This function sets the necessary environment variables for FDB to operate,
    including FDB5_HOME, FDB5_CONFIG, and FDB_REMOTE_RETRIEVE_QUEUE_LENGTH.
    It assumes that FDB is installed via spack (see fdb_install.md).

    Returns
    -------
    None
    """
    view = Path(os.environ["SCRATCH"]) / "spack-view"
    os.environ["FDB5_HOME"] = str(view)
    os.environ["FDB5_CONFIG"] = """
    ---
    type: remote
    engine: remote
    host: balfrin-ln003.cscs.ch
    port: 30005
    store: remote
    """
    os.environ["FDB_REMOTE_RETRIEVE_QUEUE_LENGTH"] = "100"


class FDBRetriever(BaseRetriever):
    """
    Retriever for extracting data from FDB.

    This class handles the retrieval of meteorological data from FDB by preparing
    requests using the meteodatalab package and concatenating the results.
    """

    def __init__(self) -> None:
        """
        Initialize the NWPRetrieverFDB instance.

        Configures origin coordinates and CRS, sets metadata fields, sources, and variables.
        """
        self.origin_lon = 10
        self.origin_lat = 47
        self.crs = (
            f"+proj=ob_tran +o_proj=longlat +lon_0={-180 + self.origin_lon} +o_lon_p=-180"
            f" +o_lat_p={90 + self.origin_lat}"
        )  # rotated lat-lon
        self.sources = (
            "COSMO-1E",
            "COSMO-2E",
            "KENDA-1",
            "ICON-CH1-EPS",
            "ICON-CH2-EPS",
        )
        self.variables = nwp_dic

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        datatype: str = "analysis",
        ensemble_members: list[int] | int | None = 0,
    ) -> xr.Dataset:
        """
        Retrieve data from FDB for a given source, variables, and dates.

        This method initializes the FDB environment, processes the given dates and variables,
        constructs a request using the meteodatalab package, and concatenates the retrieved data
        into a single xarray.Dataset.

        Parameters
        ----------
        source : str
            The source identifier to be used in the model request.
        variables : dict of str keys and list[Any] values
            List of variable definitions. Each tuple contains variable name and extra parameters.
        dates : list of datetime.date or datetime.date
            Single or list of dates for which data is required.
        datatype : str, optional
            The type of data retrieval. Defaults to 'analysis'. If set to 'forecast',
            the frequency changes accordingly.
        ensemble_members : int, optional
            The ensemble member number for the forecast retrieval. Defaults to 0.

        Returns
        -------
        xr.Dataset
            The concatenated dataset containing the retrieved data.

        Notes
        -----
        For a single date, the retrieval is performed over a date range with a frequency
        of '1H' for analysis or '3H' for forecast.
        """
        from meteodatalab import mars
        from meteodatalab.mch_model_data import get_from_fdb

        initialize_fdb()
        if not isinstance(ensemble_members, tuple):
            if isinstance(ensemble_members, list):
                ensembles = tuple(ensemble_members)
            elif isinstance(ensemble_members, int):
                ensembles = (ensemble_members,)
            else:
                raise TypeError(
                    f"ensemble_members must be int, list or tuple, got {type(ensemble_members)}"
                )
        else:
            ensembles = ensemble_members
        dates, variables = checktype(dates, variables)
        if len(dates) == 1:
            start = pd.to_datetime(dates)[0]
            stop = (pd.to_datetime(dates) + pd.to_timedelta("23h50m"))[0]
            freq = "3H" if datatype == "forecast" else "1H"
            dates = [d.date() for d in pd.date_range(start, stop, freq=freq)]
        var_names = tuple(v[0] for v in variables)
        dic = type_mapping[datatype]
        datasets = []
        for date in dates:
            d = date.strftime("%Y%m%d")
            h = date.strftime("%H:%M")
            request = mars.Request(
                param=var_names,
                date=d,
                time=h,
                stream=dic["stream"],
                type=dic["type"],
                number=ensembles,
                step=tuple(i * 3 * 60 for i in range(4)),  # minutes, e.g. every 3 hours
                levtype=mars.LevType.SURFACE,
                model=source,
            )
            mapping_short_names_to_da = get_from_fdb(request)
            df = xr.merge(mapping_short_names_to_da.values())
            datasets.append(df)
        return xr.concat(datasets, dim="forecast_reference_time")
