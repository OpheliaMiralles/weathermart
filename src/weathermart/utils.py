import json
from collections.abc import Iterable
from collections.abc import Iterator
from itertools import islice
from math import cos
from math import radians
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS, Transformer

ICON_DOMAIN = (0.5, 43, 16.5, 50)
NORDIC_DOMAIN = (-8.08, 53.14, 40.73, 73.06)
SWISS_EPSG = "epsg:2056"
EARTH_CIRCUMFERENCE_KM = 40075
KM_PER_DEGREE = EARTH_CIRCUMFERENCE_KM / 360.0
EARTH_RADIUS_KM = EARTH_CIRCUMFERENCE_KM / 2 * np.pi

def get_nrows_ncols_from_domain_size_and_reskm(
    domain: tuple[float, float, float, float],
    res_km: float,
) -> tuple[int, int]:

    min_lon, min_lat, max_lon, max_lat = domain
    lat_km = (max_lat - min_lat) * KM_PER_DEGREE
    avg_lat = (min_lat + max_lat) / 2.0
    lon_km = (max_lon - min_lon) * KM_PER_DEGREE * cos(radians(avg_lat))
    return int(lat_km // res_km), int(lon_km // res_km)


def read_file(path: str | Path) -> list[str]:
    """
    Read a file and return its content based on the file extension.

    This function currently supports JSON files. If a JSON file is provided,
    it returns the list of keys from the parsed JSON object.
    """
    path = Path(path)
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as fd:
            return list(json.load(fd).keys())
    raise ValueError(f"Unable to handle {path.suffix}")


def batched(iterable: Iterable[Any], n: int) -> Iterator[tuple[Any, ...]]:
    """
    Batch data into tuples of length n.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def resolution_degrees_to_km(res_lon_deg: float, res_lat_deg: float) -> float:
    """
    Convert resolution from degrees to kilometers.
    """
    distance_km_yaxis = distance_from_coordinates((0, 0), (0, res_lat_deg))
    distance_km_xaxis = distance_from_coordinates((0, 0), (0, res_lon_deg))
    resolution_km = distance_km_yaxis / distance_km_xaxis * distance_km_yaxis
    return resolution_km


def distance_from_coordinates(
    z1: tuple[float, float], z2: tuple[float, float]
) -> float:
    """
    Calculate the distance between two geographical coordinates.
    The Haversine formula is used to compute the distance between two points on the Earth's surface.

    Parameters
    ----------
    z1 : tuple
        A tuple (lon, lat) representing the longitude and latitude of the first location.
    z2 : tuple
        A tuple (lon, lat) representing the longitude and latitude of the second location.

    Returns
    -------
    float
        The distance between the two locations in kilometers.
    """
    lon1, lat1 = z1
    lon2, lat2 = z2
    r = EARTH_RADIUS_KM  # radius of Earth in kilometers
    p = np.pi / 180  # factor to convert degrees to radians
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    d = 2 * r * np.arcsin(np.sqrt(a))
    return d

def reproject(
    x_coords: np.ndarray | list | tuple,
    y_coords: np.ndarray | list | tuple,
    src_crs: CRS | str,
    dst_crs: CRS | str,
):
    # Local copy to avoid circular import with interp2grid/destaggering
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x_coords, y_coords)

def assign_latlon_coords(
    array: Any,
    crs: str = SWISS_EPSG,
) -> Any:
    geodims = [d for d in array.dims if d in ("x", "y", "station", "cell")]
    if len(geodims) < 2 and crs == "epsg:4326" and "x" in array.coords:
        return array.assign_coords(longitude=array.x.values, latitude=array.y.values)
    if geodims == ["y", "x"]:
        xv, yv = np.meshgrid(array.x.values, array.y.values)
        lon, lat = (
            (xv, yv)
            if crs == "epsg:4326"
            else reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
        )
        return array.assign_coords(
            longitude=(("y", "x"), lon), latitude=(("y", "x"), lat)
        )
    xv, yv = np.meshgrid(array.x.values, array.y.values, indexing="ij")
    lon, lat = (
        reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
        if crs != "epsg:4326"
        else (xv, yv)
    )
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))
