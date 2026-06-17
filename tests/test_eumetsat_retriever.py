import datetime

import numpy as np
import pandas as pd
import xarray as xr

from weathermart.retrievers.eumetsat import _centered_time_window
from weathermart.retrievers.eumetsat import _concat_cell_observations_by_time
from weathermart.retrievers.eumetsat import _prepare_eumetsat_dataset_time


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


def test_concat_cell_observations_handles_unique_time_label() -> None:
    ds = xr.Dataset(
        {
            "6": (("time", "cell"), np.array([[1.0, 2.0]])),
            "longitude": (("time", "cell"), np.array([[10.0, 11.0]])),
            "latitude": (("time", "cell"), np.array([[60.0, 61.0]])),
        },
        coords={
            "time": [np.datetime64("2025-01-01T09:00:00")],
            "cell": [0, 1],
        },
        attrs={"radiance_layout": "mars_odb_like"},
    )

    out = _concat_cell_observations_by_time(ds)

    assert out.sizes == {"time": 1, "cell": 2}
    np.testing.assert_allclose(out["6"].isel(time=0).values, [1.0, 2.0])
