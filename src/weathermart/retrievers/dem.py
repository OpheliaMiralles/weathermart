import os
import pathlib
import tempfile
import zipfile
from typing import Any
from typing import cast

import numpy as np
import requests
import xarray as xr
from pyproj import CRS
from pyproj import Transformer

from weathermart.base import BaseRetriever


def _get_tmpdir() -> str | None:
    tmpdir = os.environ.get("WEATHERMART_TMPDIR")
    if tmpdir is not None:
        path = pathlib.Path(tmpdir)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    base = pathlib.Path("/lustre/storeB/users")
    user = os.environ.get("USER")
    if user is None or not base.exists():
        return None

    path = base / user / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class DEMRetriever(BaseRetriever):
    """
    Base class for Digital Elevation Model (DEM) retrieval.

    This class provides utility methods for DEM retrieval, such as getting a projected bounding
    box and loading a file from a URL.
    """

    is_static = True

    @staticmethod
    def get_projected_bbox(
        bounds: list[float] | tuple[float, float, float, float],
        source_crs: str | CRS,
        target_crs: str | CRS,
    ) -> dict[str, slice]:
        """
        Project bounding box from target CRS to source CRS.

        Parameters
        ----------
        bounds : list or tuple
            The bounding box in the format [xmin, ymin, xmax, ymax].
        source_crs : str or pyproj.CRS
            The source coordinate reference system.
        target_crs : str or pyproj.CRS
            The target coordinate reference system.

        Returns
        -------
        dict
            A dictionary with keys 'x' and 'y' containing slice objects for the projected bounds.
        """
        bounds = list(bounds)
        xmin, ymin, xmax, ymax = bounds

        src_crs = CRS.from_user_input(source_crs)
        dest_crs = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(dest_crs, src_crs, always_xy=True)
        x_min, y_min, x_max, y_max = transformer.transform_bounds(
            xmin, ymin, xmax, ymax
        )
        bbox = {"x": slice(x_min, x_max), "y": slice(y_min, y_max)}
        return bbox

    @staticmethod
    def load_from_url(url: str) -> str:
        """
        Load content from a given URL and save it to a temporary file.

        Parameters
        ----------
        url : str
            The URL to download.

        Returns
        -------
        str
            The filename of the temporary file where the content was saved.
        """
        response = requests.get(url, stream=True)
        if response.ok:
            block_size = 1024 * 10
            tmpdir = _get_tmpdir()
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, dir=tmpdir) as temp_file:
                temp_filename = temp_file.name
                # Write the response content to the temporary file
                for data in response.iter_content(block_size):
                    temp_file.write(data)
            return temp_filename
        raise RuntimeError(f"Failed to download content from {url}.")

    def retrieve(
        self,
        source: str,
        variables: Any,
        dates: Any,
    ) -> xr.Dataset:
        """
        Retrieve DEM data.

        This method should be overridden by subclasses to implement specific retrieval logic.

        Parameters
        ----------
        source : str
            The source identifier.
        variables : Any
            Variables to retrieve (not used in this implementation).
        dates : Any
            Dates for which data is requested (not used in this implementation).

        Returns
        -------
        xr.Dataset
            Dataset containing the DEM data.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class DHM25Retriever(DEMRetriever):
    """
    Retriever for DHM25 DEM data.
    """

    sources = ("DHM25",)
    variables = {s.lower(): [s.lower()] for s in sources}
    crs = "epsg:3035"

    @staticmethod
    def txt_to_xarray(
        elevation: np.ndarray, header: dict[str, float | int]
    ) -> xr.DataArray:
        """
        Convert ASCII grid text data to an xarray.DataArray.

        Parameters
        ----------
        elevation : numpy.ndarray
            Array containing the elevation data.
        header : dict
            Dictionary containing header information such as cellsize, xllcorner, yllcorner,
            number of columns and rows, and NODATA value.

        Returns
        -------
        xarray.DataArray
            DataArray containing the processed elevation data.
        """
        dx = header["cellsize"]
        xmin = header["xllcorner"]
        ymin = header["yllcorner"]
        e = (2000000 + xmin + dx / 2) + np.arange(
            dx * header["ncols"], dtype=np.float32
        )[::1] * dx
        n = (1000000 + ymin + dx / 2) + np.arange(
            dx * header["nrows"], dtype=np.float32
        )[::1] * dx
        da = xr.DataArray(
            elevation[::-1], dims=("y", "x"), coords={"x": ("x", e), "y": ("y", n)}
        )
        da = da.where(da != header["NODATA_value"]) / 10
        da = da.chunk("auto").persist()
        return da

    @staticmethod
    def load_dhm(url: str) -> xr.DataArray:
        """
        Load and process DHM25 DEM data from a given URL.
        This method downloads a zip archive from the URL, extracts the ASCII grid file,
        reads the header and elevation data, converts it to an xarray.DataArray, and cleans up.
        """
        # saves in tempfile to save time
        temp_filename = DEMRetriever.load_from_url(url)
        with tempfile.TemporaryDirectory(dir=_get_tmpdir()) as temp_dir:
            with zipfile.ZipFile(temp_filename, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
                extracted_path = (
                    pathlib.Path(temp_dir) / "ASCII_GRID_1part/dhm25_grid_raster.asc"
                )
                with open(extracted_path) as ascii_file:
                    header: dict[str, int | float] = {}
                    for _ in range(6):
                        line = ascii_file.readline().rstrip("\n").split()
                        if line[0] in ("ncols", "nrows"):
                            header[line[0]] = int(line[1])
                        else:
                            header[line[0]] = float(line[1])
                    # Read the elevation data
                    elevation = np.loadtxt(ascii_file, dtype=np.float32)
        pathlib.Path(temp_filename).unlink()
        dem = DHM25Retriever.txt_to_xarray(elevation, header)
        return dem

    def retrieve(self, source: str, variables: Any, dates: Any) -> xr.Dataset:
        """Retrieve DHM25 DEM data."""
        dem = self.load_dhm(
            "https://cms.geo.admin.ch/ogd/topography/DHM25_MM_ASCII_GRID.zip"
        )
        dem_dataset = dem.to_dataset(name=source.lower())
        dem_dataset.attrs["source"] = source.upper()
        return dem_dataset


class CEDTMRetriever(DEMRetriever):
    """
    Retriever for CEDTM DEM data.
    """

    sources = ("CEDTM",)
    variables = {s.lower(): [s.lower()] for s in sources}
    crs = "epsg:3035"

    @classmethod
    def load_dtm(
        cls,
        url: str,
        bounds: list[float] | tuple[float, float, float, float],
        source_crs: str | CRS,
        target_crs: str | CRS,
        resolution: str | float | None = None,
    ) -> xr.DataArray:
        """
        Load and process CEDTM DEM data from a given URL within specified bounds.

        Parameters
        ----------
        url : str
            URL to download the CEDTM DEM data.
        bounds : list or tuple
            Bounding box in the format [xmin, ymin, xmax, ymax].
        source_crs : str or pyproj.CRS
            The source coordinate reference system.
        target_crs : str or pyproj.CRS
            The target coordinate reference system.
        resolution : str or float, optional
            Requested DEM resolution. Strings may use ``m`` or ``km`` suffixes.
            Resolutions coarser than the native 30 m grid are approximated by taking
            every Nth grid cell.

        Returns
        -------
        xarray.DataArray
            Processed DEM data within the defined bounding box.
        """
        try:
            from rioxarray import open_rasterio
        except ImportError as exc:
            raise ImportError(
                "The 'rioxarray' package is required for the CEDTMRetriever."
            ) from exc
        bbox = cls.get_projected_bbox(bounds, source_crs, target_crs)
        temp_filename = cls.load_from_url(url)
        dem = cast(xr.DataArray, open_rasterio(temp_filename))
        dem = dem.sel(band=1, drop=True).sortby("y", ascending=True).sel(bbox)

        if resolution is not None:
            if isinstance(resolution, str):
                if resolution.endswith("km"):
                    resolution_m = float(resolution[:-2]) * 1000
                elif resolution.endswith("m"):
                    resolution_m = float(resolution[:-1])
                else:
                    resolution_m = float(resolution)
            else:
                resolution_m = float(resolution)
            native_resolution_m = min(
                abs(float(dem["x"][1] - dem["x"][0])),
                abs(float(dem["y"][1] - dem["y"][0])),
            )
            if resolution_m < native_resolution_m:
                raise ValueError(
                    f"Argument resolution must be >= {native_resolution_m:g}m for {cls.__name__}."
                )
            stride = max(1, int(np.ceil(resolution_m / native_resolution_m)))
            if stride > 1:
                dem = dem.isel(
                    y=slice(None, None, stride),
                    x=slice(None, None, stride),
                )

        dem = dem.astype(np.float32)
        pathlib.Path(temp_filename).unlink()
        dem = dem.where(dem != -99999.0) / 10
        dem = dem.chunk("auto")
        return dem

    def retrieve(
        self,
        source: str,
        variables: Any,
        dates: Any,
        bounds: list[float] | tuple[float, float, float, float] | None = None,
        target_crs: str | CRS | None = None,
        resolution: str | float | None = None,
    ) -> xr.Dataset:
        """
        Retrieve CEDTM DEM data for a given bounding box.

        Parameters
        ----------
        source : str
            The source identifier.
        variables : Any
            Variables to retrieve (not used in this implementation).
        dates : Any
            Dates for which data is requested (not used in this implementation).
        bounds : list or tuple, optional
            Bounding box in the format [xmin, ymin, xmax, ymax]. Must be provided.
        target_crs : str or pyproj.CRS, optional
            The target coordinate reference system. Must be provided.
        resolution : str or float, optional
            Requested DEM resolution. Strings may use ``m`` or ``km`` suffixes.
            For broad domains, a coarser setting such as ``500m`` can reduce memory use.

        Returns
        -------
        xr.Dataset
            Dataset containing the DEM data with the source metadata.

        Raises
        ------
        ValueError
            If either 'bounds' or 'target_crs' is not provided.
        """
        if bounds is None or target_crs is None:
            raise ValueError(
                f"Argument bounds and target_crs need to be set for {self.__class__}"
            )
        dem = self.load_dtm(
            (
                "https://s3.eu-central-1.wasabisys.com/eumap/dtm/"
                "dtm_elev.lowestmode_gedi.eml_mf_30m_0..0cm_2000..2018_eumap_epsg3035_v0.3.tif"
            ),
            bounds,
            self.crs,
            target_crs,
            resolution,
        )
        dem_dataset = dem.to_dataset(name=source.lower())
        dem_dataset.attrs["source"] = source.upper()
        return dem_dataset


class NASADEMRetriever(DEMRetriever):
    """
    Retriever for NASADEM DEM data.
    """

    sources = ("NASADEM",)
    variables = {s.lower(): [s.lower()] for s in sources}
    crs = "epsg:4326"

    @classmethod
    def load_nasadem(
        cls,
        url: str,
        bounds: list[float] | tuple[float, float, float, float],
        source_crs: str | CRS,
        target_crs: str | CRS,
    ) -> xr.DataArray:
        """
        Load NASADEM DEM data by querying a STAC catalog and merge the tiles.

        Parameters
        ----------
        url : str
            URL of the STAC catalog.
        bounds : list or tuple
            Bounding box in the format [xmin, ymin, xmax, ymax].
        source_crs : str or pyproj.CRS
            The source coordinate reference system.
        target_crs : str or pyproj.CRS
            The target coordinate reference system.

        Returns
        -------
        xarray.DataArray
            Merged and processed DEM data.
        """
        bbox = cls.get_projected_bbox(bounds, source_crs, target_crs)
        xmin = bbox["x"].start
        xmax = bbox["x"].stop
        ymin = bbox["y"].start
        ymax = bbox["y"].stop
        try:
            import planetary_computer
            import pystac_client
            from rioxarray import open_rasterio
            from rioxarray.merge import merge_arrays
        except ImportError:
            raise ImportError(
                "The 'planetary_computer', 'pystac_client', 'rioxarray' and 'rasterio'"
                " packages are required for the NASADEMRetriever."
            )
        # catalog query
        catalog = pystac_client.Client.open(
            url, modifier=planetary_computer.sign_inplace
        )
        search = catalog.search(collections=["nasadem"], bbox=[xmin, ymin, xmax, ymax])
        items = search.get_all_items()
        # download and merge tiles
        tiles = []
        for item in items:
            tiles.append(
                cast(xr.DataArray, open_rasterio(item.assets["elevation"].href))
            )
        dem = merge_arrays(tiles)
        dem = dem.sel(band=1, drop=True)
        dem = dem.where(dem != dem.attrs["_FillValue"])
        return dem

    def retrieve(
        self,
        source: str,
        variables: Any,
        dates: Any,
        bounds: list[float] | tuple[float, float, float, float] | None = None,
        source_crs: str | CRS | None = None,
    ) -> xr.Dataset:
        """
        Retrieve NASADEM DEM data for a specified bounding box.

        Parameters
        ----------
        source : str
            The source identifier.
        variables : Any
            Variables to retrieve (not used in this implementation).
        dates : Any
            Dates for which data is requested (not used in this implementation).
        bounds : list or tuple, optional
            Bounding box in the format [xmin, ymin, xmax, ymax]. Must be provided.
        source_crs : str or pyproj.CRS, optional
            The source coordinate reference system. Defaults to the class CRS if not provided.

        Returns
        -------
        xr.Dataset
            Dataset containing the DEM data with the source metadata.

        Raises
        ------
        ValueError
            If 'bounds' is not provided.
        """
        if bounds is None:
            raise ValueError(f"Argument bounds need to be set for {self.__class__}")
        if source_crs is None:
            source_crs = self.crs
        dem = self.load_nasadem(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            bounds,
            source_crs,
            self.crs,
        )
        dem_dataset = dem.to_dataset(name=source.lower()).transpose()
        dem_dataset.attrs["source"] = source.upper()
        return dem_dataset
