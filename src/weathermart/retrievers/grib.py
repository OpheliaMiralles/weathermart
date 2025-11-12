import datetime
import logging
import pathlib
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from meteodatalab import data_source
from meteodatalab import grib_decoder
from meteodatalab.ogd_api import _geo_coords

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.base import variables_metadata

nwp_dic = {
    k: [k]
    for k in variables_metadata[
        variables_metadata.source == "ECCODES_COSMO"
    ].short_name.unique()
}


class GribRetriever(BaseRetriever):
    """
    Retriever for grib data stored on disk.
    """

    def __init__(self) -> None:
        self.type_mapping = {"forecast": "FCST", "analysis": "ANA", "first_guess": "FG"}
        self.prefix_mapping = {"ICON-CH1-EPS": "i1", "COSMO-1E": "c1", "COSMO-2E": "c2"}
        self.ensemble_mapping: dict[str | int | None, str | None] = {
            None: None,
            "all": "*",
            "median": "median",
            "mean": "mean",
            "det": "det",
        }
        for i in range(10):
            self.ensemble_mapping[str(i)] = f"{i:03}"
            self.ensemble_mapping[i] = f"{i:03}"

        self.sources = ("KENDA-CH1", "ICON-CH1-EPS", "COSMO-1E", "COSMO_2E")
        self.variables = nwp_dic
        self.crs = "epsg:4326"

    @staticmethod
    def handle_metadata(ds: xr.Dataset) -> xr.Dataset:
        """Flatten WrappedMetaData objects from earthkit in the attributes to simple dicts to save in zarr format."""
        for variable in ds.data_vars:
            xa = ds[variable]
            if "metadata" in xa.attrs:
                logging.warning(
                    "WrappedMetaData object from earthkit found in xarray attributes for %s. Transforming to dict to save in zarr format.",
                    variable,
                )
                meta = xa.attrs["metadata"].metadata
                xa.attrs["metadata"] = dict(meta)
                for k, v in dict(meta).items():
                    if isinstance(v, np.ndarray):
                        xa.attrs["metadata"][k] = v.tolist()
            ds[variable] = xa

        for c in ds.coords:
            if "metadata" in ds[c].attrs:
                logging.warning(
                    "WrappedMetaData object from earthkit found in xarray attributes for coord %s. Transforming to dict to save in zarr format.",
                    c,
                )
                meta = ds[c].attrs["metadata"].metadata
                ds[c].attrs["metadata"] = dict(meta)
                for k, v in dict(meta).items():
                    if isinstance(v, np.ndarray):
                        ds[c].attrs["metadata"][k] = v.tolist()
        return ds

    def _convert_tot_precip_to_instant(
        self,
        ds: xr.Dataset,
        step_hours: list[int],
        step_hours_sec: list[np.timedelta64],
    ) -> xr.Dataset:
        """
        Convert accumulated TOT_PREC to instant precipitation using available forecast_reference_time
        and lead_time. This encapsulates the logic previously inline in open_and_process_data.
        """
        valid_time = np.add.outer(
            ds.forecast_reference_time.data.flatten(),
            ds.lead_time.data.flatten(),
        ).flatten()
        # create a continuous hourly time series based on step_hour between each updated forecast_reference_time
        first, last = (
            ds.forecast_reference_time.data[0],
            ds.forecast_reference_time.data[-1],
        )
        daterange = list(pd.date_range(first, last, freq="1h")[:: len(step_hours)])
        ds = ds.sel(forecast_reference_time=daterange, lead_time=step_hours_sec)

        valid_time = np.add.outer(
            ds.forecast_reference_time.data.flatten(),
            ds.lead_time.data.flatten(),
        ).flatten()
        unique_valid_times = set(valid_time)
        if len(valid_time) != len(unique_valid_times):
            raise ValueError(
                "Cannot compute instant precipitation when validity time has duplicate values."
            )

        ds_start = ds.isel(forecast_reference_time=[0])
        valid_time = (
            ds.forecast_reference_time.data[0] + ds_start.lead_time.data.flatten()
        )
        ds_start = (
            ds_start.stack(z=("forecast_reference_time", "lead_time"))
            .assign_coords(time=("z", valid_time))
            .swap_dims(z="time")
            .drop("valid_time")
        )
        conc = [ds_start]

        # cumsum over several days
        for t_val in sorted(list(set(ds.forecast_reference_time.data)))[1:]:
            ds_accum = ds.sel(forecast_reference_time=[t_val]).isel(
                lead_time=slice(0, None)
            ) + conc[-1].isel(time=-1)
            valid_time = t_val + ds_accum.lead_time.data.flatten()
            ds_accum = (
                ds_accum.stack(z=("forecast_reference_time", "lead_time"))
                .assign_coords(time=("z", valid_time))
                .swap_dims(z="time")
                .drop("valid_time")
            )
            conc.append(ds_accum)
        # then we diff to get the instant precip
        ds = xr.concat(conc, dim="time").diff("time")
        valid_dates = [
            t
            for t in ds.time.data
            if pd.to_datetime(t).date() > pd.to_datetime(first).date()
            and pd.to_datetime(t).date() < pd.to_datetime(last).date()
        ]
        ds = (
            ds.sel(time=valid_dates)
            .swap_dims(time="z")
            .set_index(z=["forecast_reference_time", "lead_time"])
            .unstack()
        )
        return ds

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict[str, Any]]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        path_to_grib: Path | str | None = None,
        datatype: list[str] = ["analysis"],
        levels: int | list[int] = [80],
        step_hours: int | list[int] = 0,
        ensemble_members: int | list[int | None] | None = None,
        instant_precip: bool = False,
    ) -> xr.Dataset:
        """
        Retrieve and process local grib data for given source, variables, and dates using meteodata-lab and earthkit data.

        Parameters
        ----------
        source : str
            Identifier of the data source.
        variables : list of tuple of (str, dict)
            List of variable definitions as tuples containing variable names and parameters.
        dates : list of datetime.date or datetime.date
            One or multiple dates for which data is to be retrieved.
        path_to_grib : str or Path, optional
            Path to the directory containing the local grib files.
        datatype : list of str, optional
            List of data types to retrieve, can be "forecast" or "analysis".
            Default is ["analysis"].
        levels : int or list of int, optional
            Model level(s) to retrieve. Default is [80].
        step_hours : int or list of int, optional
            Step hour(s) to retrieve. Default is 0.
        ensemble_members : int or list of int, optional
            Ensemble member(s) to retrieve. Default is None.
        instant_precip: bool, optional
            If True and "TOT_PREC" is requested, convert accumulated precipitation to
            instant precipitation. Default is False (default accumulation period).

        Returns
        -------
        xarray.Dataset
            Merged dataset containing the processed local SEN data.
        """
        # check if eccodes is installed.
        # there are two things that can go wrong here:
        # 1. eccodes is not installed
        # 2. eccodes is installed but the library is not found
        if path_to_grib is None:
            self.path = None
        elif isinstance(path_to_grib, str):
            self.path = Path(path_to_grib)
        else:
            self.path = path_to_grib

        if self.path is None:
            raise ValueError("Path to local GRIB files must be provided.")
        dates, variables = checktype(dates, variables)
        self.requested_variables = [v[0] for v in variables]
        step_hours = [step_hours] if isinstance(step_hours, int) else step_hours
        step_hours_sec = [
            pd.to_timedelta(s, unit="h").to_timedelta64() for s in step_hours
        ]

        datatypes = [
            self.type_mapping[typ]
            for typ in (datatype if isinstance(datatype, list) else [datatype])
        ]
        levels = [levels] if isinstance(levels, int) else levels
        self.levels = levels

        if not all(75 <= lev <= 80 for lev in levels) and not all(
            t in ["ANA", "FG"] for t in datatypes
        ):
            logging.warning(
                "Only vertical levels 75 to 80 are supported for forecast data. Levels input: %s",
                levels,
            )
        ensemble_members = (
            [ensemble_members]
            if isinstance(ensemble_members, str | int | type(None))
            else ensemble_members
        )
        self.ensembles = [self.ensemble_mapping[e] for e in ensemble_members]

        def get_all_files_from_date(
            date: datetime.date,
            t: str,
        ) -> list[pathlib.Path]:
            """
            Get list of files given the date and type of data requested. Can be adapted to different filesystems.
            """
            file = f"i{t[0].lower()}f"
            ensembles = [
                e if e is not None else ("000" if t == "FCST" else "det")
                for e in self.ensembles
            ]
            y = str(pd.to_datetime(date).year)[-2:]
            prefix = (
                f"{self.prefix_mapping[source]}eff00"
                if source in self.prefix_mapping
                else ""
            )
            specific_path = (
                self.path / source / f"{t}{y}" / ensembles[0]
                if t != "FCST"
                else self.path / source / f"{t}{y}"
            )
            step_hours_str = [
                f"{int(s / np.timedelta64(1, 'h')):02}" for s in step_hours_sec
            ]
            if t == "FCST":
                patterns_list = [
                    f"{y}{date.strftime('%m%d')}*/grib/{prefix}{s}0000_{e}"
                    for s in step_hours_str
                    for e in ensembles
                ]
            else:
                patterns_list = [f"{file}{date.strftime('%Y%m%d')}*"]
            patterns = set(patterns_list)
            all_files = sorted(
                chain.from_iterable(specific_path.glob(pattern) for pattern in patterns)
            )
            return all_files

        def open_and_process_data(all_paths: list[pathlib.Path]) -> xr.Dataset:
            try:
                fds = data_source.FileDataSource(datafiles=[str(p) for p in all_paths])
                vars_on_vertical_levels = [
                    "U",
                    "V",
                    "T",
                    "QV",
                    "P",
                    "W_SO",
                    "T_SO",
                    "WSHEAR_DIFF",
                ]
                non_vert_params = list(
                    set(self.requested_variables).difference(
                        set(vars_on_vertical_levels)
                    )
                )
                dic = {}
                if len(non_vert_params):
                    dic = grib_decoder.load(
                        fds, {"param": non_vert_params}, geo_coords=_geo_coords
                    )
                requested_vars_on_vertical_levels = list(
                    set(self.requested_variables).intersection(
                        set(vars_on_vertical_levels)
                    )
                )
                if len(requested_vars_on_vertical_levels) > 0:
                    dic.update(
                        grib_decoder.load(
                            fds,
                            {
                                "param": requested_vars_on_vertical_levels,
                                "levelist": self.levels,
                            },
                            geo_coords=_geo_coords,
                        )
                    )
                for var, da in dic.items():
                    arr = da
                    if "z" in arr.dims and arr.sizes["z"] == 1:
                        arr = arr.isel(z=0, drop=True)
                        if "z" in arr.coords:
                            arr = arr.drop_vars("z")
                    elif "z" in arr.dims and arr.sizes["z"] > 1:
                        vdim = arr.attrs.get("vcoord_type", "z")
                        arr = arr.rename({"z": vdim})
                    if "eps" in arr.dims and arr.sizes["eps"] == 1:
                        arr = arr.isel(eps=0, drop=True)
                        if "eps" in arr.coords:
                            arr = arr.drop_vars("eps", errors="ignore")
                    dic[var] = arr
                ds = xr.merge([dic[p].rename(p) for p in dic])
                ds = ds.squeeze(drop=True)
            except ValueError as exc:
                raise ValueError("Could not retrieve data.") from exc
            name_mapping = {
                "ref_time": "forecast_reference_time",
                "lon": "longitude",
                "lat": "latitude",
            }
            ds = ds.rename(
                {k: name_mapping[str(k)] for k in ds.coords if k in name_mapping}
            )
            ds = self.handle_metadata(ds)
            if "TOT_PREC" in self.requested_variables and instant_precip:
                ds_accum = self._convert_tot_precip_to_instant(
                    ds[["TOT_PREC"]], step_hours, step_hours_sec
                )
                ds = ds.sel(
                    forecast_reference_time=ds_accum.forecast_reference_time,
                    lead_time=ds_accum.lead_time,
                )
                ds_accum = ds_accum.transpose(*ds.dims)
                ds = ds.drop_vars("TOT_PREC").assign(TOT_PREC=ds_accum["TOT_PREC"])
            return ds.unify_chunks()

        def get_data_for_type(t: str) -> list[xr.Dataset]:
            """
            Loop through relevant configuration, retrieve requested variables.
            If precipitation is requested, retrieve the first guess file if the analysis data is requested
            instantaneous rain rate.
            For forecast data, if instant_precip is set to True in the kwargs of the retriever, this function also
            retrieves the day before to get the last available forecast reference time and lead times for instantaneous
            rain rate computation.

            Parameters
            ----------
            t: str
                The type of data to retrieve, can be "FCST" for forecast, "FG" for first guess or "AN" for analysis.

            Returns
            -------
            list of xr.Dataset
                List of retrieved datasets.
            """
            concat = []
            for date in dates:
                all_paths = get_all_files_from_date(date, t)
                if "TOT_PREC" in self.requested_variables and instant_precip:
                    # need to retrieve last available forecast reference time and lead times to compute
                    # instantaneous rain rate, which is a derivative of the accumulated precipitation
                    all_paths += get_all_files_from_date(
                        date + pd.to_timedelta("1d"), t
                    )
                    all_paths += get_all_files_from_date(
                        date - pd.to_timedelta("1d"), t
                    )
                if all_paths:
                    new_data = open_and_process_data(all_paths)
                    if new_data is not None:
                        concat.append(new_data.drop_duplicates(...))
            return concat

        results = []
        for t in datatypes:
            results.extend(get_data_for_type(t))

        if results and len(results) > 1:
            ds = xr.merge(results, compat="override").drop_duplicates(...)
        elif results and len(results) == 1:
            ds = results[0]
            return ds
        raise RuntimeError("No results found for the given parameters.")
