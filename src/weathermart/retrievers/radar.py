import datetime
import logging
import os
import pathlib
import tarfile
from tempfile import TemporaryDirectory
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Proj
from pyproj import Transformer
from glob import glob 
from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.utils import NORDIC_DOMAIN
from weathermart.utils import assign_latlon_coords

def read_radar_file_or_raise(file: os.PathLike) -> Any:
    """
    Read a file using wradlib and raise an error if the file could not be read.

    Parameters
    ----------
    file : os.PathLike
        Path to the file to be read.

    Returns
    -------
    Any
        The read METRANET data file.

    Raises
    ------
    ImportError
        If the 'wradlib' package is not installed.
    RuntimeError
        If the file read by radlib is empty.
    """
    try:
        import wradlib as wrl
    except ImportError:
        raise ImportError(
            "The 'wradlib' package is required to use the RadarRetriever."
        )
    try:
        radar_data = wrl.io.hdf.read_opera_hdf5(file)
        return radar_data
    except Exception as e:
        raise RuntimeError(f"File {file} does not exist.", e)


class OperaRetriever(BaseRetriever):
    """
    Class for retrieving and processing Opera radar data from the MeteoFrance API.
    """

    sources = ("OPERA",)
    variables = {
        "TOT_PREC": ["TOT_PREC"],
    }
    url = "https://partner-api.meteofrance.fr/partner/radar/opera/1.0/"
    # the corners of the OPERA radar data are given in lat/lon
    LL_lon, LL_lat = (np.float64(-10.434576838640398), np.float64(31.746215319325056))
    LR_lon, LR_lat = (np.float64(29.421038635578032), np.float64(31.98765027794496))
    UL_lon, UL_lat = (np.float64(-39.5357864125034), np.float64(67.02283275830867))
    UR_lon, UR_lat = (np.float64(57.81196475014995), np.float64(67.62103710275053))
    # the projection definition of the OPERA radar data
    crs = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
    src_proj = Proj(crs)
    # Transform the lat/lon corners to the OPERA radar projection
    lonlat_transformer = Transformer.from_proj(
        Proj("EPSG:4326"), src_proj, always_xy=True
    )
    LL_x, LL_y = lonlat_transformer.transform(LL_lon, LL_lat)
    LR_x, LR_y = lonlat_transformer.transform(LR_lon, LR_lat)
    UL_x, UL_y = lonlat_transformer.transform(UL_lon, UL_lat)
    UR_x, UR_y = lonlat_transformer.transform(UR_lon, UR_lat)
    # The grid is regular in the radar projection (at least we assume so)
    x = np.linspace(LL_x, LR_x, 1900)
    y = np.linspace(LL_y, UL_y, 2200)

    # target grid: Regular lat lon grid
    custom_target_grid = xr.Dataset(
        {
            "x": np.arange(NORDIC_DOMAIN[0], NORDIC_DOMAIN[2], 0.01),
            "y": np.arange(NORDIC_DOMAIN[1], NORDIC_DOMAIN[3], 0.01),
        },
        attrs={
            "grid_mapping": {
                "epsg_code": "4326",
            },
            "source": "Custom grid",
        },
    )

    def process_radar_file(
        self, radar_file: pathlib.Path, remove_clutter: bool = True
    ) -> xr.Dataset:
        """
        Process a single OPERA radar file and return an xarray.Dataset.

        Parameters
        ----------
        radar_file : pathlib.Path
            Path to the radar file.
        remove_clutter : bool, optional
            If True, remove clutter (nan values, values over 150 mm/h) from the radar data. Default is True.

        Returns
        -------
        xr.Dataset
            Dataset containing the radar data with dimensions ('time', 'y', 'x').
        """
        radar_data = read_radar_file_or_raise(radar_file)
        header = radar_data["dataset1/what"]
        timestamp = datetime.datetime.strptime(
            (radar_data["what"]["date"] + radar_data["what"]["time"]).decode(),
            "%Y%m%d%H%M%S",
        )
        nodata = header["nodata"]
        undetect = header["undetect"]
        values = radar_data["dataset1/data1/data"]
        # TODO: agree on solution here
        # https://www.mdpi.com/2073-4433/10/6/320 (paper about OPERA)
        # there is both "nodata" (i.e. it is not in range of any radar)
        if remove_clutter:
            values[values == nodata] = np.nan
            # and "undetect" (i.e. it is in range of a radar but no rain was detected, this is the normal no-rain case)
            values[values == undetect] = 0
            # also, sometimes there are very high values (around 300).
            values[values > 150] = 0
        else:
            logging.warning(
                "Warning: Clutter is not removed from the OPERA radar data."
            )

        ds = xr.Dataset(
            {"TOT_PREC": (("time", "y", "x"), values[None, ::-1, :])},
            coords={
                "y": self.y,
                "x": self.x,
                "time": [timestamp],
            },
        )

        if not remove_clutter:
            ds["TOT_PREC"].attrs["nodata"] = nodata
            ds["TOT_PREC"].attrs["undetect"] = undetect
        else:
            ds["TOT_PREC"].attrs["nodata"] = "nan"
            ds["TOT_PREC"].attrs["undetect"] = 0

        return ds

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        meteofranceapi_token_path: os.PathLike | None = None,
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
        meteofranceapi_token_path : str, optional
            Path to the file containing the MeteoFrance API token. Default is None.

        Returns
        -------
        xr.Dataset
            Merged dataset containing the radar data for all specified dates and variables.
        Raises
        ------
        RuntimeError
            If the meteofranceapi_token_path is not set or if the token file cannot be read.
        FileNotFoundError
            If the token file is not found.

        """
        logging.warning(
            "The MeteoFrance API Retriever is not fully implemented. Be careful when using data from this retriever."
        )
        if meteofranceapi_token_path is None:
            token = os.getenv("OPERA_API_TOKEN")
            if token is None:
                raise RuntimeError(
                    "The meteofranceapi_token_path is not set. Please provide a path to the token file as arg or set the OPERA_API_TOKEN environment variable."
                )
        else:
            try:
                with open(meteofranceapi_token_path) as f:
                    token = f.read().strip()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Token file not found: {meteofranceapi_token_path}"
                )
            except Exception as e:
                raise RuntimeError(f"Error reading token file: {e}") from e

        headers = {"apikey": token}

        dates, variables = checktype(dates, variables)

        for date in dates:
            with TemporaryDirectory() as tmpdirname:
                url = f"{self.url}archive/odyssey/composite/RAINFALL_RATE/{date.strftime('%Y-%m-%d')}?format=HDF5"
                request = requests.get(url, headers=headers)
                if request.status_code != 200:
                    raise RuntimeError(
                        f"Failed to retrieve data: {request.status_code} - {request.text}"
                    )
                with open(tmpdirname + "file.tar", "wb") as f:
                    f.write(request.content)
                # untar
                with tarfile.open(tmpdirname + "file.tar", "r") as tar:
                    tar.extractall(tmpdirname)
                # process the files
                files = sorted(pathlib.Path(tmpdirname).glob("*.h5"))
                radar_dataarrays = []
                for file in files:
                    radar_dataarrays.append(self.process_radar_file(file))
                radar_dataset = xr.concat(radar_dataarrays, dim="time")
                # todo: regrid to desired grid, metadata, etc.
                return radar_dataset
        raise RuntimeError("No data found for the given dates.")
    
# MAPPING TO ALIGN OLD RADAR GRIDS TO NEW TEMPLATE
csv_path = Path("/home/opmir9231/weathermart/nordic_radar_old_to_new_idx.csv")
R = 6371000.0  # Earth radius (m)
def align_to_template(other, tol_m=1000.0):
    grid_dims = [d for d in other.dims if d not in ["time", "number"]]
    other_stacked = other.stack(cell = grid_dims).drop(grid_dims)
    if other_stacked.sizes["cell"]==3614996: #norm for all files from Nov 2020 onwards
        return other
    td = xr.open_zarr("/lustre/storeB/users/opmir9231/nordic_radar/20250101").isel(time=0).coords.to_dataset()
    td = td.stack(cell = td.dims)
    ok, idx, (tlon, tlat) = get_old_to_new_idx(other=other_stacked, tol_m=tol_m)
    out = other_stacked.isel(cell=idx)
    out["lwe_precipitation_rate"] = out["lwe_precipitation_rate"].where(xr.DataArray(ok.astype(bool), dims=["cell"]), np.nan)
    out = out.reindex_like(td).assign_coords(lat=("cell", tlat), lon=("cell", tlon)).unstack("cell")
    return out

def ll_to_xyz(
    lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z]).astype("float32")

def get_old_to_new_idx(other=None, tol_m=1000.0):
    csv_path = Path("/home/opmir9231/weathermart/nordic_radar_old_to_new_idx.csv")
    td = xr.open_zarr("/lustre/storeB/users/opmir9231/nordic_radar/20250101").isel(time=0).coords.to_dataset()
    tlat = td["lat"].values.ravel()
    tlon = td["lon"].values.ravel()
    if csv_path.exists():
        dataframe = pd.read_csv(csv_path)
        ok = dataframe["mask"].values
        idx = dataframe["selected_index"].values
        return ok, idx, (tlon, tlat)
    other = other or xr.open_zarr("/lustre/storeB/users/opmir9231/nordic_radar/20200415").isel(time=0).coords.to_dataset()
    grid_dims = [d for d in other.dims if d not in ["time", "number"]]
    slat = other["lat"].values.ravel()
    slon = other["lon"].values.ravel()
    tree = cKDTree(ll_to_xyz(slat, slon))
    d_chord, idx = tree.query(ll_to_xyz(tlat, tlon), k=1)
    arc = 2.0 * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))
    d_m = arc * R
    ok = d_m <= tol_m
    dataframe = pd.DataFrame({
        "selected_index": idx,
        "distance": d_m,
        "mask": ok
    })
    dataframe.to_csv(csv_path, index=False)
    return ok, idx, (tlon, tlat)


class NordicRadarRetriever(BaseRetriever):
    """
    Retriever for Nordic radar reflectivity (YR WMS MOS PCAPPI).
    """
    sources = ("NORDIC_RADAR",)
    flags = [
        "is_nodata",
        "is_blocked",
        "is_lowele",
        "is_highele",
        "is_seaclutter",
        "is_groundclutter",
        "is_otherclutter",
        "is_convective",
    ]
    variables = {
        k: [k]
        for k in [
            "lwe_precipitation_rate",
            "block_percent",
            "projection_lambert",
            "mosaic_info",
            "equivalent_reflectivity_factor",
            "clutter_probability",
        ]
        + flags
        + ["qc_flags"]
    }
    mapping_flags = {
                0: "is_nodata",
                1: "is_blocked",
                2: "is_lowele",
                3: "is_highele",
                4: "is_seaclutter",
                5: "is_groundclutter",
                6: "is_otherclutter",
                7: "is_convective",
                8: "block_gt50",
                9: "extreme_20_50",
                10: "invalid_high_gt50",
            }
    ROOT = "/lustre/storeB/project/remotesensing/radar/reflectivity-nordic"
    FILE_TEMPLATE = [
        "yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{date}*.nc",
        "yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.laea-yrwms-1000.{date}*.nc",
    ]
    crs = {"lcc":"+proj=lcc +lat_0=63.3 +lon_0=15 \
            +lat_1=63.3 +lat_2=63.3 \
            +a=6371000 +b=6371000 \
            +x_0=0 +y_0=0 +units=m +no_defs",
          "laea": "+proj=laea +lat_0=63.3 +lon_0=15 +a=6371000 +b=6371000 +x_0=0 +y_0=0 +units=m +no_defs",
            }

    def build_qc_flags(self, ds: xr.Dataset) -> xr.Dataset:
        qc = xr.zeros_like(ds[self.flags[0]], dtype="uint16")
        rr = ds["lwe_precipitation_rate"]

        for i, flag in enumerate(self.flags):
            qc = qc | (ds[flag].astype("uint16") << np.uint16(i))

        extreme_flag = (rr > 20.0) & (
            rr <= 50.0
        )
        invalid_high_flag = (rr > 50.0)
        block_flag = (ds["is_blocked"]) & (ds["block_percent"] > 50.0)
        qc = qc | (block_flag.astype("uint16") << np.uint16(len(self.flags)))  # bit 8
        qc = qc | (
            extreme_flag.astype("uint16") << np.uint16(len(self.flags) + 1)
        )  # bit 9
        qc = qc | (
            invalid_high_flag.astype("uint16") << np.uint16(len(self.flags) + 2)
        )  # bit 10
        hard_nan = (
            ds["is_nodata"].astype(bool)
            | ds["is_seaclutter"].astype(bool)
            | ds["is_groundclutter"].astype(bool)
            | ds["is_otherclutter"].astype(bool)
            | invalid_high_flag.astype(bool)
            | block_flag.astype(bool)
        )
        mx = rr.isel(time=0).max(skipna=True).compute()
        cnt = (rr.isel(time=0) > 50).sum().compute()
        rr = rr.where(~hard_nan, np.nan)
        ds["lwe_precipitation_rate"] = rr
        ds = ds.drop_vars(self.flags, errors="ignore")
        ds = ds.assign(qc_flags=qc)
        ds["qc_flags"].attrs["qc_flags_description"] = (
            "Bit 0: is_nodata, Bit 1: is_blocked, Bit 2: is_lowele, Bit 3: is_highele, "
            "Bit 4: is_seaclutter, Bit 5: is_groundclutter, Bit 6: is_otherclutter, Bit 7: is_convective, "
            "Bit 8: Blocked at > 50%, Bit 9: is_extreme (20mm/h < rate <= 50mm/h), Bit 10: is_invalid_high (rate > 50mm/h)"
        )
        return ds

    def decode_qc_flags(
        self,
        ds: xr.Dataset,
        *,
        flag_var: str = "qc_flags",
    ) -> xr.Dataset:
        """
        Decode packed qc_flags into individual 0/1 masks.
        """
        if flag_var not in ds:
            raise KeyError(f"{flag_var} not in dataset")
        out = {}
        for bit, name in self.mapping_flags.items():
            out[name] = ((ds[flag_var] >> np.uint64(bit)) & np.uint64(1)).astype("uint8")
        return xr.Dataset(out, coords=ds.coords, attrs={"decoded_from": flag_var})


    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        dense_qc: bool = True,
        test: bool = False,
    ) -> xr.Dataset:
        """
        Read 5-min radar files and return daily xarray datasets.
        """
        dates, variables = checktype(dates, variables)
        unique_days = sorted(set(pd.to_datetime(d).date() for d in dates))
        var_list = [v[0] for v in variables]

        datasets: list[xr.Dataset] = []
        for day in unique_days:
            day = pd.Timestamp(day, tz="UTC")

            year = day.strftime("%Y")
            month = day.strftime("%m")
            ymd = day.strftime("%Y%m%d")
            day_dir = pathlib.Path(self.ROOT) / year / month
            candidates = [day_dir / t.format(date=ymd) for t in self.FILE_TEMPLATE]
            fpath = next((glob(str(p)) for p in candidates if len(glob(str(p))) > 0), None)

            if fpath is None:
                logging.warning("Missing radar file: %s", fpath)
                continue

            logging.info("Reading radar file %s", fpath)
            ds_list = [xr.open_dataset(f, chunks={"time": 1},
                    engine="netcdf4") for f in fpath]
            ds = xr.concat(ds_list, dim="time")
            if test:
                ds = ds.isel(time=slice(0, 3))
            if ds.time.dtype == np.int32:
                ds["time"] = pd.to_datetime(ds.time, unit="s", utc=True).tz_convert(
                    None
                )
            ds = ds.sortby("time")
            if "lat" not in ds.coords or "lon" not in ds.coords:
                if "lon" in ds.data_vars and "lat" in ds.data_vars:
                    ds = ds.set_coords(["lon", "lat"])
                else:
                    if "laea" in fpath.name:
                        ds = assign_latlon_coords(ds, crs=self.crs["laea"])
                    else:
                        ds = assign_latlon_coords(ds, crs=self.crs["lcc"])
            ds["lwe_precipitation_rate"] = ds["lwe_precipitation_rate"].where(ds["lwe_precipitation_rate"] < 1e6, np.nan)
            ds = align_to_template(ds)
            if dense_qc:
                ds = self.build_qc_flags(ds)
                var_list.append("qc_flags")
                var_list = list(set(var_list))
            ds.attrs["source"] = source
            ds = ds.chunk({"time": 24, "Yc": 748, "Xc": 689})
            print(np.nanmax(ds.lwe_precipitation_rate.isel(time=0).values))
            datasets.append(ds[[v for v in var_list if v in ds.variables]])

        if not datasets:
            return xr.Dataset()

        if len(datasets) == 1:
            return datasets[0]

        return xr.concat(
            datasets,
            dim="time",
            join="outer",
            compat="no_conflicts",
            coords="minimal",
        )


def plot_rr(ds: xr.Dataset, time_index: int = 0, output_file: str = "rr.png"):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from pyproj import Transformer

    lon = ds["lon"].values.ravel()
    lat = ds["lat"].values.ravel()
    proj = ccrs.NorthPolarStereo()
    rr = ds.isel(time=time_index)["lwe_precipitation_rate"].values.ravel()
    transformer = Transformer.from_crs("EPSG:4326", proj.proj4_init, always_xy=True)
    x, y = transformer.transform(lon, lat)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    ax = plt.axes(projection=proj)
    pm = ax.scatter(
        x,
        y,
        c=rr,
        s=6,
        cmap="Blues",
        linewidths=0,
        transform=proj,
    )
    ax.coastlines(linewidth=0.8)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=proj)
    plt.colorbar(pm, ax=ax, shrink=0.7, label="LWE Precipitation Rate (mm/h)")
    plt.title(f"Gridded polar stereographic")
    plt.savefig(output_file, dpi=150)


def plot_qc_flags(ds: xr.Dataset, time_index: int = 0, output_file: str = "qc.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from pyproj import Transformer
    from matplotlib.patches import Patch

    BIT_COLORS = {
        0: (0.0, 0.0, 0.0),  # nodata
        1: (1.0, 0.0, 0.0),  # blocked
        4: (0.0, 1.0, 1.0),  # sea clutter
        5: (0.6, 0.4, 0.2),  # ground clutter
        6: (0.6, 0.6, 0.6),  # other clutter
        7: (1.0, 0.0, 1.0),  # convective
        8: (0.0, 1.0, 0.0),  # blocked >50%
        9: (0.2, 0.2, 1.0),  # extreme
        10: (0.5, 0.0, 0.8),  # invalid
    }

    BIT_LABELS = {
        0: "nodata",
        1: "blocked (any)",
        2: "low elevation",
        3: "high elevation",
        4: "sea clutter",
        5: "ground clutter",
        6: "other clutter",
        7: "convective",
        8: "blocked >50%",
        9: "extreme (20–50 mm/h)",
        10: "invalid high (>50 mm/h)",
    }
    lon = ds["lon"].values.ravel()
    lat = ds["lat"].values.ravel()
    flags = ds.isel(time=time_index)["qc_flags"].values.ravel()
    m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(flags)
    lon, lat, flags = lon[m], lat[m], flags[m]
    proj = ccrs.NorthPolarStereo()
    transformer = Transformer.from_crs("EPSG:4326", proj.proj4_init, always_xy=True)
    x, y = transformer.transform(lon, lat)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    rgba = np.zeros((flags.size, 4), dtype=np.float32)
    anybit = np.zeros(flags.size, dtype=bool)

    for bit, rgb in BIT_COLORS.items():
        mask = ((flags >> bit) & 1).astype(bool)
        if not mask.any():
            continue
        anybit |= mask
        rgba[mask, 0] = np.maximum(rgba[mask, 0], rgb[0])
        rgba[mask, 1] = np.maximum(rgba[mask, 1], rgb[1])
        rgba[mask, 2] = np.maximum(rgba[mask, 2], rgb[2])

    rgba[anybit, 3] = 0.55
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=proj)

    ax.scatter(
        x,
        y,
        c=rgba,
        s=4,
        linewidths=0,
        rasterized=True,
        transform=proj,
    )
    ax.coastlines(linewidth=0.8)
    ax.set_extent([xmin, xmax, ymin, ymax], crs=proj)
    ax.set_title("QC flags (bitmask overlay)")
    present_bits = sorted(
        bit for bit in BIT_COLORS if np.any(((flags >> bit) & 1) == 1)
    )
    handles = [
        Patch(
            facecolor=(*BIT_COLORS[bit], 0.55),
            edgecolor="none",
            label=BIT_LABELS.get(bit, f"bit {bit}"),
        )
        for bit in present_bits
    ]
    ax.legend(
        handles=handles,
        title="QC flags",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
        ncol=2,
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
