import datetime
import sys
import types

import numpy as np
import pandas as pd
import xarray as xr

from weathermart.retrievers.eumetsat import _centered_time_window
from weathermart.retrievers.eumetsat import _prepare_eumetsat_dataset_time
from weathermart.retrievers.eumetsat import iasi_metop_to_xarray


def test_centered_time_window_uses_half_aggregation_window() -> None:
    start, end = _centered_time_window(
        pd.Timestamp("2025-01-01T09:00:00"),
        "3h",
    )

    assert start == datetime.datetime(2025, 1, 1, 7, 30)
    assert end == datetime.datetime(2025, 1, 1, 10, 30)


def test_granule_time_aggregates_with_scan_time_metadata() -> None:
    t = datetime.datetime(2025, 1, 1, 10)
    scan_time = pd.date_range(t, periods=2, freq="5min")
    first = xr.Dataset(
        {
            "6": (("scan", "fov"), np.array([[1.0, 2.0], [10.0, 20.0]])),
            "scan_time": ("scan", scan_time),
        },
        coords={"scan": [0, 1], "fov": [0, 1]},
    )
    second = xr.Dataset(
        {
            "6": (("scan", "fov"), np.array([[3.0, 4.0], [30.0, 40.0]])),
            "scan_time": ("scan", scan_time),
        },
        coords={"scan": [0, 1], "fov": [0, 1]},
    )

    prepared = [
        _prepare_eumetsat_dataset_time(
            ds,
            t,
            reader="generic_swath_reader",
            time_index="granule",
        )
        for ds in (first, second)
    ]
    aggregated = xr.concat(prepared, dim="time").groupby("time").mean(skipna=True)

    assert aggregated.sizes["time"] == 1
    assert aggregated.sizes["scan"] == 2
    assert aggregated.sizes["fov"] == 2
    np.testing.assert_allclose(
        aggregated["6"].isel(time=0).values,
        np.array([[2.0, 3.0], [20.0, 30.0]]),
    )


def test_non_granule_time_reader_can_promote_scan_time_to_time() -> None:
    scan_time = pd.date_range("2025-01-01T10:00:00", periods=2, freq="5min")
    ds = xr.Dataset(
        {
            "var": ("scan", np.array([1.0, 2.0])),
            "scan_time": ("scan", scan_time),
        }
    )

    prepared = _prepare_eumetsat_dataset_time(
        ds,
        datetime.datetime(2025, 1, 1, 10),
        reader="generic_scan_reader",
        time_index="scan",
    )

    assert "scan" not in prepared.dims
    assert prepared.sizes["time"] == 2
    np.testing.assert_array_equal(prepared["time"].values, scan_time.values)


def test_iasi_raw_reader_returns_native_cells(monkeypatch) -> None:
    class FakeCoda:
        @staticmethod
        def open(path):
            return object()

        @staticmethod
        def close(handle):
            return None

        @staticmethod
        def fetch(handle, path):
            if path == "/MDR":
                return [0, 1]
            if path.endswith("/GS1cSpect"):
                mdr_index = int(path.split("[", 1)[1].split("]", 1)[0])
                return np.arange(16, dtype=np.float32).reshape(2, 2, 4) + (
                    100 * mdr_index
                )
            if path.endswith("/GGeoSondLoc"):
                mdr_index = int(path.split("[", 1)[1].split("]", 1)[0])
                lons = np.array([[10, 11], [12, 13]], dtype=np.float32) + mdr_index
                lats = np.array([[50, 51], [52, 53]], dtype=np.float32) + mdr_index
                return np.stack([lons, lats], axis=-1)
            if path.endswith("/OnboardUTC"):
                mdr_index = int(path.split("[", 1)[1].split("]", 1)[0])
                return np.array([[0, 1], [2, 3]], dtype=np.float64) + (10 * mdr_index)
            raise KeyError(path)

    monkeypatch.setitem(
        sys.modules,
        "coda",
        types.SimpleNamespace(
            open=FakeCoda.open,
            close=FakeCoda.close,
            fetch=FakeCoda.fetch,
        ),
    )

    ds, t = iasi_metop_to_xarray(
        "fake.nat",
        area=None,
        variables=["1", "3"],
        bbox=(0, 20, 180, 90),
    )

    assert t == datetime.datetime(2000, 1, 1, 0, 0, 1, 500000)
    assert ds.sizes["cell"] == 8
    np.testing.assert_allclose(
        ds["1"].values,
        np.array([0, 4, 8, 12, 100, 104, 108, 112], dtype=np.float32),
    )
    np.testing.assert_allclose(
        ds["3"].values,
        np.array([2, 6, 10, 14, 102, 106, 110, 114], dtype=np.float32),
    )
    np.testing.assert_allclose(ds["latitude"].values[:2], np.array([50, 51]))
    assert "scan_time" in ds
    assert ds["3"].attrs["channel"] == 3
