import datetime
from typing import Any

import pandas as pd
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.base import variables_metadata


class DataRetriever(BaseRetriever):
    """
    Retrieve data from available sub-retrievers.

    This class iterates through provided sub-retrievers and attempts to retrieve
    data for a given source, variables, and dates. It checks if a retriever supports
    the source and then delegates the retrieval. The resulting dataset is renamed
    based on the retriever's variable mappings.

    Parameters
    ----------
    subretrievers : list of classes that inherit from BaseRetriever
        A list of retriever instances used to fetch data.
    """

    def __init__(self, subretrievers: tuple[BaseRetriever, ...]) -> None:
        self.subretrievers = subretrievers

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        rename: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Retrieve data for a given source, set of variables, and dates.

        The method iterates over the sub-retrievers to find one that supports the
        specified source. For the matching retriever, it filters and validates
        requested variables, calls the retriever's retrieve method, renames the
        variables in the returned dataset, and returns the dataset with only the
        successfully retrieved data variables.

        Parameters
        ----------
        source : str
            The source identifier for which data should be retrieved.
        variables : list of tuple of (str, dict)
            List of variable definitions, where each tuple contains the variable name
            and associated properties.
        dates : list of datetime.date or datetime.date
            A single date or list of dates for data retrieval.
        rename : bool, optional
            If True, the retrieved dataset's variables will be renamed according to
            the retriever's variable mappings. Defaults to True.
        **kwargs : dict
            Additional keyword arguments to be passed to each subretriever's retrieve method.

        Returns
        -------
        xarray.Dataset
            Dataset containing the retrieved data, with variable names renamed and filtered.

        Raises
        ------
        ValueError
            If a variable is not defined for the source or if no retriever supports the given source.
        RuntimeError
            If the retrieved dataset's time coordinate is not sorted.
        """
        dates, variables = checktype(dates, variables)
        # check if all kwargs are valid
        self.validate_kwargs(list(kwargs.keys()))
        for r in self.subretrievers:
            variables_to_retrieve = []
            if source.upper() in r.sources:
                for vname, props in variables:
                    if vname not in r.variables:
                        raise ValueError(
                            f"Variables {vname} not defined for source {source}"
                        )
                    variables_to_retrieve.append((vname, props))
                retriever_kwargs = r.get_kwargs()
                relevant_kwargs = {
                    k: kwargs[k] for k in retriever_kwargs if k in kwargs
                }
                ds = r.retrieve(source, variables_to_retrieve, dates, **relevant_kwargs)
                # check if dataset is sorted by time
                time_dim = (
                    "forecast_reference_time"
                    if "forecast_reference_time" in ds.dims
                    else "time"
                )
                if not ds[time_dim].to_index().is_monotonic_increasing:
                    raise RuntimeError(
                        "Time coordinate for retriever {r}, source {source} and date {dates} is not sorted."
                    )
                if not rename:
                    return ds
                var_names = [m[0] for m in variables_to_retrieve]
                ds = ds.rename(
                    {
                        k[next(i for i, v in enumerate(k) if v in ds)]: j
                        for j, k in r.variables.items()
                        if j in var_names and any(v in ds for v in k)
                    }
                )
                vars_retrieved = [v for v in var_names if v in ds.data_vars]
                ds = ds[vars_retrieved]
                for variable in vars_retrieved:
                    if variable in variables_metadata:
                        ds[variable].attrs.update(variables_metadata[variable])
                return ds
        raise ValueError(f"No retriever defined for source {source}")
