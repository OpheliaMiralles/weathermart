import copy
import datetime
import logging
import os
import pathlib
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.retrieve import DataRetriever
from weathermart.utils import read_file

pd.set_option("future.no_silent_downcasting", True)


def chunk_data(data: xr.DataArray | xr.Dataset) -> Any:
    """
    Chunk data with specific case for station indexes.
    """
    if "station" in data.dims:
        data["station"] = data["station"].astype(str)
        data["station"].attrs.update({"dtype": "str"})
        chunks = {dim: 1000 for dim in data.dims}
        return data.chunk(chunks)
    return data


class CacheRetriever:
    """
    Provides caching functionality for retrieved data.
    """

    def __init__(self, path: os.PathLike) -> None:
        """
        Initialize the CacheProvider.

        Parameters
        ----------
        path : pathlib.Path or str
            Path to the cache directory.
        """
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Cache directory {path} does not exist.")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Cache path {path} is not a directory.")
        if not os.access(self.path, os.R_OK):
            raise PermissionError(f"Cache directory {path} is not readable.")
        try:
            # create a dummy file to check if we have write access
            test_file = tempfile.TemporaryFile(dir=self.path)
            test_file.close()
            self.__read_only = False
        except PermissionError:
            self.__read_only = True
            logging.warning(
                "Cache directory %s is not writable. Writing to it will fail.", path
            )

    def provide(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        is_static: bool = False,
        kwargs_str: str = "",
    ) -> tuple[xr.Dataset, list[Any]]:
        """
        Load cached data if available, or return missing variables/dates if not.

        Parameters
        ----------
        source : str
            Identifier of the data source.
        variables : list of tuple (str, dict)
            List of variable definitions (variable name and parameters).
        dates : list of datetime.date or datetime.date
            The date or dates for which to load cached data.
        is_static : bool, optional
            Flag indicating if the variables are static. Default is False.
        kwargs_str : str, optional
            String representation of the keyword arguments used for retrieval.

        Returns
        -------
        tuple
            A tuple containing:
            - xr.Dataset: Merged dataset of cached data.
            - list or dict: Missing variables if any, per date.
        """
        dates, variables = checktype(dates, variables)
        data = []
        missing_vars: dict[pd.Timestamp, list[tuple[str, dict[Any, Any]]]] = {
            date: [] for date in dates
        }
        for date in dates:
            date_str = date.strftime("%Y%m%d")
            to_merge = []
            for variable in variables:
                if not is_static:
                    p = self.path / f"{source}{kwargs_str}/{date_str}/{variable[0]}"
                else:
                    # date irrelevant for static variables
                    p = self.path / f"{source}{kwargs_str}/{variable[0]}"
                if p.exists() and any(p.iterdir()):
                    to_merge.append(xr.open_zarr(p.parent)[variable[0]])
                else:
                    missing_vars[date].append(variable)
            if len(to_merge) > 0:
                data.append(xr.merge(to_merge))
        if len(data) > 0:
            if "forecast_reference_time" in data[0].dims:
                ds = xr.concat(data, dim="forecast_reference_time")
            elif "time" in data[0].dims:
                ds = xr.concat(data, dim="time")
            elif "date" in data[0].dims:
                ds = xr.concat(data, dim="date").rename(
                    {"date": "forecast_reference_time"}
                )
            elif is_static:
                ds = xr.merge(data)
            else:
                raise ValueError(
                    "No time dimension found. Must be one of 'time' or 'forecast_reference_time'."
                )
        else:
            ds = xr.Dataset()
        # nothing missing from cache
        if all(len(v) == 0 for _, v in missing_vars.items()):
            return ds, []

        # variables missing when we retrieve date by date
        if len(dates) == 1:
            return ds, list(missing_vars.values())[0]

        return ds, list((k, missing_vars[k]) for k in set(missing_vars.keys()))

    def save(
        self,
        data: xr.Dataset,
        source: str,
        variables: list[tuple[str, dict]],
        dates: list[datetime.date] | datetime.date,
        is_static: bool = False,
        kwargs_str: str = "",
    ) -> None:
        """
        Save the data to cache.

        Parameters
        ----------
        data : xarray.Dataset
            The dataset to be saved.
        source : str
            Identifier of the data source.
        variables : list of tuple (str, dict)
            List of variable definitions.
        dates : list of datetime.date or datetime.date
            Date or dates associated with the data.
        is_static : bool, optional
            Flag indicating if the variables are static, in which case date is irrelevant.
        kwargs_str : str, optional
            String representation of the keyword arguments used for retrieval.

        Returns
        -------
        None

        Raises
        ------
        PermissionError
            If the cache directory is read-only.
        """
        if self.__read_only:
            raise PermissionError(f"Cache directory {str(self.path)} is read-only.")
        dates, variables = checktype(dates, variables)
        for variable in data.data_vars:
            data[variable].encoding.clear()
            source_path = self.path / f"{source.lower()}{kwargs_str}"
            if not source_path.exists():
                source_path.mkdir()
                source_path.chmod(0o2777)
            if is_static:
                # date irrelevant
                p = self.path / f"{source.lower()}{kwargs_str}/{variable}"
                p.mkdir(parents=True, exist_ok=True, mode=0o2775)
                logging.info(
                    "Saving cache for %s %s (static) at %s", source, variable, p
                )
                data[variable].to_zarr(str(p.parent), mode="a")
                os.system(f"chmod -R 2777 {p.parent}")
            else:
                for date in dates:
                    day_start = pd.to_datetime(date)
                    day_end = pd.to_datetime(day_start + pd.to_timedelta("23h50m"))
                    date_str = date.strftime("%Y%m%d")
                    p = (
                        self.path
                        / f"{source.lower()}{kwargs_str}/{date_str}/{variable}"
                    )
                    p.parent.mkdir(parents=True, exist_ok=True, mode=0o2775)
                    logging.info(
                        "Saving cache for %s %s on %s at %s",
                        source,
                        variable,
                        date,
                        p.parent,
                    )
                    xa = data[variable]
                    if "metadata" in xa.attrs:
                        logging.warning(
                            "WrappedMetaData object from earthkit found in xarray attributes for %s %s on %s. Removing it to save in zarr format.",
                            source,
                            variable,
                            date,
                        )
                        xa.attrs.pop("metadata")  # weird earthkit metadata object

                    unique_time_dim = (
                        "time" if "time" in data.dims else "forecast_reference_time"
                    )
                    # time should always exist, either observation time or validity time of a forecast
                    if len(set(data[unique_time_dim].data.flatten())) != len(
                        data[unique_time_dim].data.flatten()
                    ):
                        logging.info("Saving time with duplicated values")
                    valid_datetimes = sorted(
                        [
                            datetime
                            for datetime in set(data[unique_time_dim].data)
                            if day_start <= datetime <= day_end
                        ]
                    )
                    xa = xa.sel({unique_time_dim: valid_datetimes})
                    xa = chunk_data(xa)
                    xa.to_zarr(str(p.parent), mode="a")
                    os.system(f"chmod -R 2777 {p.parent}")


class DataProvider:
    """
    Provides input data by retrieving from sources and utilizing caching.

    This class integrates data retrieval with caching and exposes methods to obtain
    metadata fields, variable mappings, and coordinate reference systems (CRS) for a given source.
    """

    _ignored_kwargs = [
        "eumdac_credentials_path",
        "jretrieve_credentials_path",
        "meteofranceapi_token_path",
        "path_to_grib",
    ]  # kwargs that are ignored in the kwargs_str
    _override_kwargs = [
        "storage_key"
    ]  # kwargs that override all other kwargs and are not passed to the retriever
    _warned_about = False  # flag to avoid multiple warnings about storage_key

    def __init__(
        self, cache: CacheRetriever | None, retrievers: Sequence[BaseRetriever]
    ) -> None:
        """
        Initialize the DataProvider.

        Parameters
        ----------
        cache : CacheProvider
            Cache provider instance for managing cached data.
        retrievers : sequence of classes that inherit from BaseRetriever
            Retriever instances.
        """
        self.retriever = DataRetriever(retrievers)
        self.cache = cache

    def get_variable_mapping(self, source: str) -> dict[Any, Any]:
        """
        Get variable mapping for the specified source.
        """
        retrievers = self.retriever.subretrievers
        mapping_vars = {}
        for r in retrievers:
            if source.upper() in r.sources:
                mapping_vars = r.variables
        return mapping_vars

    def get_crs(self, source: str) -> str | dict[str, str] | None:
        """
        Get the coordinate reference system (CRS) for the given source.
        """
        retrievers = self.retriever.subretrievers
        for r in retrievers:
            if source.upper() in r.sources:
                return r.crs
        return None

    def get_static_flag(self, source: str) -> bool:
        """
        Determine if the source has static variables.
        """
        retrievers = self.retriever.subretrievers
        for r in retrievers:
            if source.upper() in r.sources:
                if getattr(r, "is_static", False):
                    return True
        return False

    def get_kwargs_str(self, kwargs: dict[str, Any]) -> str:
        """
        Generate a string representation of keyword arguments.

        This method processes a dictionary of keyword arguments and constructs a string
        by concatenating processed representations of each key-value pair, ignoring those
        in DataProvider._ignored_kwargs. Specific keys have custom formatting rules.

        Parameters
        ----------
        kwargs : dict
            A dictionary containing keyword arguments to be processed.

        Returns
        -------
        str
            A concatenated string representation of the keyword arguments, or an empty string
            if kwargs is empty.
        """
        kwargs = copy.deepcopy(kwargs)
        for ignored_kwarg in DataProvider._ignored_kwargs:
            kwargs.pop(ignored_kwarg, None)
        if not kwargs:
            return ""
        if "storage_key" in kwargs:
            if not self._warned_about:
                logging.warning(
                    "Kwarg 'storage_key' used. Ignoring all other kwargs and just using this one: %s",
                    kwargs["storage_key"],
                )
                self._warned_about = True
            return f"_{kwargs['storage_key']}"

        parts: list[str] = []
        for k, v in kwargs.items():
            try:
                formatted_kwarg = _format_kwarg(k, v)
                if formatted_kwarg:
                    parts.append(formatted_kwarg)
            except Exception as e:
                raise ValueError(
                    f"Error creating kwarg string key {k} with value {v}: {e}"
                ) from e
        if parts:
            return "_" + "_".join(parts)
        return ""

    def update_retriever_with_kwargs(
        self, source: str, **kwargs: Any
    ) -> "DataProvider":
        """
        Update the retriever for the specified source with new keyword arguments.

        Parameters
        ----------
        source : str
            The source identifier.
        **kwargs : dict
            Keyword arguments to reinitialize the retriever.

        Returns
        -------
        DataProvider
            The updated DataProvider instance.
        """
        retrievers = list(self.retriever.subretrievers)  # Convert tuple to list
        for i, r in enumerate(retrievers):
            if source.upper() in r.sources:
                r_updated = r.__class__(**kwargs)
                retrievers[i] = r_updated
        self.retriever.subretrievers = tuple(retrievers)  # Convert back to tuple
        return self

    def provide(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Retrieve data from cache and source as needed.

        This method checks the cache for the requested data. If data for any date or variable
        is missing in the cache, it retrieves the missing parts from the source, saves them to the cache,
        and then concatenates all datasets.

        Parameters
        ----------
        source : str
            The source identifier.
        variables : list of tuple (str, dict)
            List of variable definitions.
        dates : list of datetime.date or datetime.date
            Date or dates for which data is requested.
        **kwargs : dict
            Additional keyword arguments to pass to the retriever.

        Returns
        -------
        xarray.Dataset
            The merged dataset containing the requested data.
        """
        dates, variables = checktype(dates, variables)
        is_static = self.get_static_flag(source)
        all_cached_data = []

        for date in pd.date_range(dates[0], dates[-1], freq="D"):
            dates_to_retrieve = sorted(
                [d for d in dates if pd.to_datetime(d).date() == date.date()]
            )
            logging.info(
                "Reading %s %s data from cache for %s",
                source,
                ",".join([v[0] for v in variables]),
                date,
            )
            if self.cache is not None:
                cached, missing_vars = self.cache.provide(
                    source.lower(),
                    variables,
                    date,
                    is_static=is_static,
                    kwargs_str=self.get_kwargs_str(kwargs),
                )
            else:
                cached = xr.Dataset()
                missing_vars = variables
            if len(cached) > 0:
                time_dim = (
                    "time"
                    if "time" in cached.dims
                    else (
                        "forecast_reference_time"
                        if "forecast_reference_time" in cached.dims
                        else ("date" if "date" in cached.dims else None)
                    )
                )
                cached = chunk_data(cached)
                if len(
                    pd.date_range(dates_to_retrieve[0], dates_to_retrieve[-1], freq="D")
                ) != len(dates_to_retrieve):
                    # the cache returns data for the whole day, but we only want specific datetimes
                    all_cached_data.append(
                        cached.sel({time_dim: dates_to_retrieve}, method="nearest")
                    )
                else:
                    all_cached_data.append(cached)

            if missing_vars:
                logging.info(
                    "Couldn't find %s in cache for %s",
                    ",".join([m[0] for m in missing_vars]),
                    date,
                )
                logging.info(
                    "Retrieving %s from source %s",
                    ",".join([m[0] for m in missing_vars]),
                    source,
                )
                kwargs_copy = copy.deepcopy(kwargs)
                for k in self._override_kwargs:
                    kwargs_copy.pop(k, None)
                new_data = self.retriever.retrieve(
                    source, missing_vars, dates_to_retrieve, **kwargs_copy
                )
                new_data = chunk_data(new_data)
                all_cached_data.append(new_data)
                if self.cache is not None:
                    try:
                        self.cache.save(
                            new_data,
                            source,
                            missing_vars,
                            date,
                            is_static=is_static,
                            kwargs_str=self.get_kwargs_str(kwargs),
                        )
                    except PermissionError:
                        logging.warning(
                            "Cache directory is read-only. Data will not be saved to cache."
                        )

        # Merge all datasets
        time_dim = (
            "time"
            if "time" in all_cached_data[0].dims
            else (
                "forecast_reference_time"
                if "forecast_reference_time" in all_cached_data[0].dims
                else None
            )
        )
        if time_dim is not None:
            data = xr.concat(
                all_cached_data, dim=time_dim, coords="minimal", compat="override"
            )
        elif is_static:
            data = xr.merge(all_cached_data)
        else:
            raise ValueError("No time dimension found.")
        return data

    def provide_from_config(self, config: dict[str, Any], **kwargs: Any) -> None:
        """
        Provide data based on a configuration dictionary.

        The configuration dictionary should contain a 'dates' key and keys for each source
        with corresponding variable names. Additional keyword arguments can be specified in the
        'kwargs' key of the configuration.

        Parameters
        ----------
        config : dict of {str: list of str}
            Configuration dictionary with key 'dates' and source-to-variable mappings.
        **kwargs : dict
            Additional keyword arguments to pass to the provide method.

        Returns
        -------
        None
        """
        config = copy.deepcopy(config)
        dates_conf = config.pop("dates")
        config_kwargs: dict[str, Any] = config.pop("kwargs", {})
        kwargs.update(config_kwargs)
        for source, variables in config.items():
            dates, params = checktype(dates_conf, variables)
            expanded_kwargs = kwargs.copy()
            for k, v in kwargs.items():
                if isinstance(v, str) and v.startswith("$file:"):
                    expanded_kwargs[k] = read_file(v.removeprefix("$file:"))
            self.provide(source, params, dates, **expanded_kwargs)


def _is_sorted(lst: list) -> bool:
    """
    Check if a list is sorted in ascending order.
    """
    return all(a <= b for a, b in zip(lst, lst[1:]))


def _format_kwarg(k: str, v: Any) -> str | None:
    """
    Format a single keyword argument.

    Parameters
    ----------
    k : str
        The key of the argument.
    v : Any
        The value of the argument.

    Returns
    -------
    str or None
        Formatted string for the given key/value pair or None.
    Raises
    ------
    ValueError
        If the value contains an invalid character ("/") or if the value is invalid for the key.
    """
    if isinstance(v, np.ndarray):
        v = v.tolist()
    elif isinstance(v, tuple):
        v = list(v)
    if not isinstance(v, list):
        v = [v]

    # validate that v does not contain duplicates
    if len(v) != len(set(v)):
        raise ValueError(f"Duplicate values in {k}: {v}")

    # Validate that v does not contain an invalid character.
    for item in v:
        if isinstance(item, str) and "/" in item:
            raise ValueError(f"Invalid character '/' in value {item} for key {k}")

    # Process according to the key.
    if k == "bbox":
        v = [int(x) for x in v]
        if len(v) != 4:
            raise ValueError(f"Invalid bbox length for {k}: {v}")
        return f"{v[0]}_{v[1]}_{v[2]}_{v[3]}"
    if k in ["levels", "step_hours", "ensemble_members", "use_limitation"]:
        # shorten ensemble_members to ens
        if k == "ensemble_members":
            k = "ens"
        if k == "step_hours":
            k = "step"
        if k == "levels":
            k = "level"
        # shorten use_limitation to limitation and skip if value is 20
        if k == "use_limitation":
            if v[0] == 20:
                return None
            k = "limitation"
        if len(v) == 0:
            raise ValueError(f"Empty values for {k}: {v}")
        if len(list(v)) == 1:
            return f"{k}{v[0]}"
        # check if v is sorted
        if not _is_sorted(v):
            raise ValueError(f"Values for {k} are not sorted: {v}")
        return f"{k}{v[0]}to{v[-1]}"
    if k in ["unique_valid_datetimes", "instant_precip"]:
        return k
    else:
        values = [str(item).replace(".", "") for item in v]
        return "_".join(values)
