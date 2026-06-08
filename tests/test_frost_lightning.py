from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from weathermart.retrievers.frost import FrostRetriever


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


def _make_lightning_line(
    *,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    lat: float,
    lon: float,
    cloud: int,
) -> str:
    values = [
        year,
        month,
        day,
        hour,
        minute,
        second,
        0,
        lat,
        lon,
        100,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        cloud,
        0,
        0,
        0,
    ]
    return " ".join(str(v) for v in values)


def test_lightning_retriever_grids_5min_counts(tmp_path: Path, monkeypatch) -> None:
    template = xr.Dataset(
        data_vars={"dummy": (("y", "x"), np.zeros((2, 2), dtype=np.float32))},
        coords={
            "x": ("x", np.array([0.0, 1.0])),
            "y": ("y", np.array([60.0, 61.0])),
        }
    )
    lon, lat = np.meshgrid(template["x"].values, template["y"].values)
    template = template.assign_coords(
        lon=(("y", "x"), lon),
        lat=(("y", "x"), lat),
    )
    template_path = tmp_path / "template.zarr"
    template.to_zarr(template_path)

    payload = "\n".join(
        [
            _make_lightning_line(
                year=2024,
                month=1,
                day=10,
                hour=0,
                minute=0,
                second=0,
                lat=60.1,
                lon=0.1,
                cloud=0,
            ),
            _make_lightning_line(
                year=2024,
                month=1,
                day=10,
                hour=0,
                minute=4,
                second=59,
                lat=60.1,
                lon=0.1,
                cloud=1,
            ),
            _make_lightning_line(
                year=2024,
                month=1,
                day=10,
                hour=0,
                minute=5,
                second=0,
                lat=61.1,
                lon=1.1,
                cloud=0,
            ),
        ]
    )

    retriever = FrostRetriever()
    monkeypatch.setattr(
        FrostRetriever,
        "_load_credentials",
        staticmethod(lambda credentials_path=None: ("id", "secret")),
    )
    monkeypatch.setattr(
        FrostRetriever,
        "request_from_frost",
        staticmethod(
            lambda **kwargs: _Resp(payload)
            if kwargs["endpoint"] == "lightning"
            else _Resp("")
        ),
    )

    ds = retriever.retrieve(
        "LIGHTNING",
        ["lightning_count", "lightning_cloud_to_ground", "lightning_intracloud"],
        ["2024-01-10"],
        template_path=template_path,
        template_crs="epsg:4326",
    )

    assert ds.sizes["time"] == 288
    assert (
        ds["lightning_count"].sel(time=pd.Timestamp("2024-01-10 00:00:00"))[0, 0].item()
        == 2
    )
    assert (
        ds["lightning_cloud_to_ground"].sel(time=pd.Timestamp("2024-01-10 00:00:00"))[0, 0]
        .item()
        == 1
    )
    assert (
        ds["lightning_intracloud"].sel(time=pd.Timestamp("2024-01-10 00:00:00"))[0, 0]
        .item()
        == 1
    )
    assert (
        ds["lightning_count"].sel(time=pd.Timestamp("2024-01-10 00:05:00"))[1, 1].item()
        == 1
    )
