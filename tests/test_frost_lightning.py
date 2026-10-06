from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import xarray as xr

from weathermart.retrievers.frost import FROST_BASE_OBSERVATION_VARIABLES
from weathermart.retrievers.frost import FrostRetriever


def test_frost_canonical_surface_pressure_and_wind_are_available() -> None:
    assert "surface_air_pressure" in FROST_BASE_OBSERVATION_VARIABLES
    assert "wind_speed" in FROST_BASE_OBSERVATION_VARIABLES
    assert "surface_air_pressure" in FrostRetriever.variables
    assert "wind_speed" in FrostRetriever.variables


def test_frost_request_retries_transport_errors(monkeypatch) -> None:
    attempts = []
    client_options = {}

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

    class Client:
        def __init__(self, **kwargs) -> None:
            client_options.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url, params):
            attempts.append((url, params))
            if len(attempts) < 3:
                raise httpx.ReadTimeout("timed out")
            return Response()

    delays = []
    monkeypatch.setattr("weathermart.retrievers.frost.httpx.Client", Client)
    monkeypatch.setattr("weathermart.retrievers.frost.time.sleep", delays.append)

    response = FrostRetriever.request_from_frost(
        "observations", "id", "secret", {"sources": "SN1"}
    )

    assert isinstance(response, Response)
    assert len(attempts) == 3
    assert delays == [1, 2]
    assert client_options["timeout"].connect == 30.0
    assert client_options["timeout"].read == 120.0


def test_frost_observation_batch_splits_http_412(monkeypatch) -> None:
    requested_sources = []

    class Response:
        status_code = 200

        def __init__(self, stations) -> None:
            self._stations = stations

        def json(self):
            return {"data": [{"sourceId": station} for station in self._stations]}

    def request_from_frost(*, args, **kwargs):
        stations = args["sources"].split(",")
        requested_sources.append(stations)
        if len(stations) > 2:
            request = httpx.Request("GET", "https://frost.test/observations")
            response = httpx.Response(412, request=request)
            raise httpx.HTTPStatusError(
                "batch too large", request=request, response=response
            )
        return Response(stations)

    monkeypatch.setattr(FrostRetriever, "request_from_frost", request_from_frost)

    data = FrostRetriever._request_observation_batch(
        endpoint="observations",
        client_id="id",
        client_secret="secret",
        stations=["SN1", "SN2", "SN3", "SN4", "SN5"],
        query_args={"elements": "air_temperature"},
    )

    assert [item["sourceId"] for item in data] == [
        "SN1",
        "SN2",
        "SN3",
        "SN4",
        "SN5",
    ]
    assert requested_sources == [
        ["SN1", "SN2", "SN3", "SN4", "SN5"],
        ["SN1", "SN2"],
        ["SN3", "SN4", "SN5"],
        ["SN3"],
        ["SN4", "SN5"],
    ]


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
