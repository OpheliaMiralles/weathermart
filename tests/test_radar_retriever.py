import numpy as np
import pytest
import xarray as xr

from weathermart.retrievers.radar import NordicRadarRetriever
from weathermart.retrievers.radar import read_radar_file_or_raise


def test_read_radar_file_or_raise():
    with pytest.raises(RuntimeError):
        read_radar_file_or_raise("fake_file")


def test_rainbow_template_alignment_uses_fixed_radar_blocks(
    tmp_path, monkeypatch
):
    template_path = tmp_path / "rainbow-template.zarr"
    template_radars = np.repeat(["A", "B", "C"], 3)
    template = xr.Dataset(
        coords={
            "cell": (
                "cell",
                np.array([f"rainbow_{i}" for i in range(9)], dtype=object),
            ),
            "radar": ("cell", template_radars),
            "azimuth": ("cell", np.tile([0.0, 0.0, 1.0], 3)),
            "range": ("cell", np.tile([1.0, 2.0, 1.0], 3)),
        }
    )
    template.to_zarr(template_path)
    monkeypatch.setenv("RAINBOW_COLUMN_PRODUCT_TEMPLATE", str(template_path))

    daily = xr.Dataset(
        {
            "ezc20": (
                ("time", "cell"),
                np.array([[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]]),
            )
        },
        coords={
            "time": [np.datetime64("2025-01-01T00:00:00")],
            "cell": np.arange(6),
            "radar": ("cell", np.repeat(["A", "C"], 3)),
            "azimuth": ("cell", np.tile([0.0, 0.0, 1.0], 2)),
            "range": ("cell", np.tile([1.0, 2.0, 1.0], 2)),
        },
    )

    aligned = NordicRadarRetriever()._align_rainbow_cell_template(daily)

    assert aligned.sizes["cell"] == 9
    np.testing.assert_allclose(
        aligned["ezc20"].values,
        [[1.0, 2.0, 3.0, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0]],
        equal_nan=True,
    )
    np.testing.assert_array_equal(aligned["cell"].values, template["cell"].values)
    np.testing.assert_array_equal(aligned["radar"].values, template_radars)


def test_retrieve_nowcast_inputs_keeps_native_grids_separate(monkeypatch):
    times = np.array(
        ["2025-01-01T12:00:00", "2025-01-01T12:05:00"],
        dtype="datetime64[ns]",
    )
    radar_source = xr.Dataset(
        {
            "lwe_precipitation_rate": (
                ("time", "Yc", "Xc"),
                np.ones((2, 2, 3)),
            )
        },
        coords={"time": times},
    )
    rainbow_source = xr.Dataset(
        {"ezc20": (("time", "cell"), np.ones((2, 4)))},
        coords={"time": times, "cell": np.arange(4)},
    )
    retriever = NordicRadarRetriever()
    monkeypatch.setattr(
        retriever,
        "_open_postprocessed_radar",
        lambda day, variables, test: radar_source,
    )
    monkeypatch.setattr(
        retriever,
        "_open_rainbow_column_products",
        lambda day, variables, test: rainbow_source,
    )

    radar, rainbow = retriever.retrieve_nowcast_inputs(
        [np.datetime64("2025-01-01T12:00:00")],
        radar_variables=["lwe_precipitation_rate"],
        rainbow_variables=["ezc20"],
    )

    assert radar.sizes == {"time": 1, "Yc": 2, "Xc": 3}
    assert rainbow.sizes == {"time": 1, "cell": 4}
