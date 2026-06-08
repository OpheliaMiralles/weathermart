import sys
import types
from pathlib import Path

import pandas as pd

from weathermart.retrievers.mars import MINIMAL_RADIANCE_ODB_COLUMNS
from weathermart.retrievers.mars import NORTH_LATITUDE_20_DOMAIN_FILTER
from weathermart.retrievers.mars import POLAR_DOMAIN_FILTER
from weathermart.retrievers.mars import SCRIPT_ODB_COLUMNS
from weathermart.retrievers.mars import MarsODBRetriever
from weathermart.retrievers.mars import odb_dataframe_to_xarray
from weathermart.retrievers.mars import read_odb_to_xarray


def test_build_request_text_selects_instrument_by_reportype_not_channel(
    tmp_path: Path,
) -> None:
    retriever = MarsODBRetriever()
    target = tmp_path / "atms_20200101.odb"
    text = retriever.build_request_text(
        date=pd.Timestamp("2020-01-01"),
        target=target,
        channel_group="ATMS",
        reportypes=["21009", "21010"],
        domain_filter="lat > 50",
    )

    assert "REPORTYPE=21009/21010" in text
    assert "STREAM=lwda" in text
    assert "TYPE=OFB" in text
    assert "FORMAT=odb" in text
    assert f'TARGET="{target}"' in text
    assert "vertco_reference_1 >=" not in text
    assert "channel >=" not in text
    assert "zenith" in text
    assert "azimuth" in text
    assert "scanline" in text
    assert "scanpos" in text
    assert "an_depar" in text
    assert "snow_density" not in text


def test_mars_atms_default_reportypes_are_standard_radiances(
    tmp_path: Path,
) -> None:
    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["ATMS"],
        dates=["2024-02-06"],
        output_dir=tmp_path,
        submit=False,
    )

    request_text = Path(ds["request_file"].isel(request=0).item()).read_text(
        encoding="utf-8"
    )
    assert "REPORTYPE=34001/34002/34003" in request_text
    assert "REPORTYPE=49001/49002/49003" not in request_text


def test_mars_retriever_uses_requested_datetime_hour_by_default(
    tmp_path: Path,
) -> None:
    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["ATMS"],
        dates=["2024-02-06T15:00:00"],
        output_dir=tmp_path,
        submit=False,
    )

    request_file = Path(ds["request_file"].isel(request=0).item())
    request_text = request_file.read_text(encoding="utf-8")
    target_file = Path(ds["target"].isel(request=0).item())
    assert "TIME=15" in request_text
    assert "TIME=00/12" not in request_text
    assert request_file.name.startswith("atms_2024020615")
    assert target_file.name.startswith("atms_2024020615")


def test_mars_atms_allsky_selector_uses_allsky_reportypes(
    tmp_path: Path,
) -> None:
    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["ATMS_ALLSKY"],
        dates=["2024-02-06"],
        output_dir=tmp_path,
        submit=False,
    )

    request_text = Path(ds["request_file"].isel(request=0).item()).read_text(
        encoding="utf-8"
    )
    assert "REPORTYPE=49001/49002/49003" in request_text


def test_build_request_text_can_use_script_columns(tmp_path: Path) -> None:
    retriever = MarsODBRetriever()
    target = tmp_path / "amsua_20200101.odb"
    text = retriever.build_request_text(
        date=pd.Timestamp("2020-01-01"),
        target=target,
        channel_group="AMSU-A",
        reportypes=["21009"],
        domain_filter=POLAR_DOMAIN_FILTER,
        odb_columns=SCRIPT_ODB_COLUMNS,
    )

    assert "snow_density" in text
    assert "datum_rdbflag" in text
    assert "tausfc_cld" in text
    assert "where (lat > 20)" in text
    assert "lsm < 0.2" not in text


def test_build_request_text_can_use_minimal_north_lat20_request(
    tmp_path: Path,
) -> None:
    retriever = MarsODBRetriever()
    target = tmp_path / "minimal_20240206.odb"
    text = retriever.build_request_text(
        date=pd.Timestamp("2024-02-06"),
        target=target,
        channel_group="AMSU-A",
        reportypes=["21009"],
        domain_filter=NORTH_LATITUDE_20_DOMAIN_FILTER,
        odb_columns=MINIMAL_RADIANCE_ODB_COLUMNS,
    )

    assert "select distinct reportype, date, lsm, seaice, time, lat, lon" in text
    assert "vertco_reference_1, obsvalue, datum_status" in text
    assert ", ,obsvalue" not in text
    assert "fg_depar where" in text
    assert "where (lat > 20)" in text
    assert "lat<-20" not in text
    assert "lsm <" not in text


def test_mars_retriever_uses_default_reportypes_for_instrument(tmp_path: Path) -> None:
    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["brightness_temperature"],
        instruments=["AMSU-A"],
        dates=["2020-01-01"],
        output_dir=tmp_path,
        submit=False,
    )

    request_file = Path(ds["request_file"].isel(request=0).item())
    request_text = request_file.read_text(encoding="utf-8")
    assert "REPORTYPE=21001/21002/21003/21004/21005/21007/21008/21009/21010" in request_text
    assert "where (lat > 20)" in request_text
    assert "vertco_reference_1 >=" not in request_text
    assert "lsm < 0.2" not in request_text


def test_mars_retriever_writes_request_files(tmp_path: Path) -> None:
    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["AMSU-A"],
        dates=["2020-01-01", "2020-01-02"],
        output_dir=tmp_path,
        reportypes=["21009"],
        submit=False,
    )

    assert ds.sizes["request"] == 2
    assert int(ds["submitted"].sum()) == 0
    request_file = Path(ds["request_file"].isel(request=0).item())
    target_file = Path(ds["target"].isel(request=0).item())
    assert request_file.exists()
    assert request_file.read_text(encoding="utf-8").startswith("RETRIEVE,")
    assert target_file.name.startswith("amsu-a_20200101")


def test_mars_retriever_passes_rc_credential_path(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(cmd, check, env, timeout):
        calls.append((cmd, check, env, timeout))

    monkeypatch.setattr("weathermart.retrievers.mars.subprocess.run", fake_run)

    rc_file = tmp_path / ".ecmwfapirc"
    rc_file.write_text('{"url":"https://api.ecmwf.int/v1","key":"x","email":"x@y"}')

    retriever = MarsODBRetriever()
    retriever.retrieve(
        source="MARS_ODB",
        variables=["AMSU-A"],
        dates=["2020-01-01"],
        output_dir=tmp_path,
        reportypes=["21009"],
        submit=True,
        mars_executable="/tmp/mars",
        rc_credential_path=rc_file,
        read_odb=False,
    )

    assert len(calls) == 1
    cmd, check, env, timeout = calls[0]
    assert cmd[0] == "/tmp/mars"
    assert check is True
    assert timeout is None
    assert env["ECMWF_API_RC_FILE"] == str(rc_file)


def test_odb_dataframe_to_xarray_uses_time_and_cell_dims() -> None:
    frame = pd.DataFrame(
        {
            "date@hdr": [20200101, 20200101, 20200101],
            "time@hdr": [126, 134, 500],
            "lat@hdr": [70.0, 71.0, 72.0],
            "lon@hdr": [10.0, 11.0, 12.0],
            "obsvalue@body": [251.0, 252.0, 253.0],
            "vertco_reference_1@body": [5.0, 6.0, 5.0],
            "reportype": [21009, 21009, 21010],
            "satellite_identifier@sat": [3, 3, 4],
            "satellite_instrument@sat": [570, 570, 570],
            "andate": [20200101, 20200101, 20200101],
            "antime": [0, 0, 0],
            "expver": ["0001", "0001", "0001"],
            "zenith@sat": [52.0, 53.0, 54.0],
            "azimuth@sat": [10.0, 11.0, 12.0],
            "scanline@sat": [1, 1, 2],
            "scanpos@sat": [15, 16, 17],
            "an_depar@body": [0.1, 0.2, 0.3],
        }
    )

    ds = odb_dataframe_to_xarray(frame, aggregation_window="5min")

    assert ds.sizes["time"] == 2
    assert ds.sizes["cell"] == 2
    assert "brightness_temperature" in ds
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert float(ds["brightness_temperature"].isel(time=0, cell=0)) == 251.0
    assert float(ds["brightness_temperature"].isel(time=1, cell=0)) == 253.0
    assert pd.isna(ds["brightness_temperature"].isel(time=1, cell=1).item())
    assert int(ds["reportype"].isel(time=0, cell=0)) == 21009
    assert float(ds["zenith"].isel(time=0, cell=0)) == 52.0
    assert int(ds["scanpos"].isel(time=1, cell=0)) == 17
    assert float(ds["an_depar"].isel(time=0, cell=1)) == 0.2


def test_odb_dataframe_to_xarray_filters_to_requested_analysis_window() -> None:
    frame = pd.DataFrame(
        {
            "date@hdr": [20200101, 20200101, 20200101, 20200101],
            "time@hdr": [0, 12900, 13000, 30000],
            "lat@hdr": [70.0, 71.0, 72.0, 73.0],
            "lon@hdr": [10.0, 11.0, 12.0, 13.0],
            "obsvalue@body": [251.0, 252.0, 253.0, 254.0],
            "vertco_reference_1@body": [5.0, 6.0, 7.0, 8.0],
        }
    )

    ds = odb_dataframe_to_xarray(
        frame,
        ["brightness_temperature", "channel"],
        aggregation_window="3h",
        analysis_times=[pd.Timestamp("2020-01-01T00:00:00")],
    )

    assert ds.sizes["time"] == 1
    assert pd.Timestamp(ds["time"].item()) == pd.Timestamp("2020-01-01T00:00:00")
    assert ds.sizes["cell"] == 2
    assert ds["brightness_temperature"].isel(time=0).count().item() == 2
    assert set(ds["channel"].isel(time=0).values.tolist()) == {5.0, 6.0}


def test_read_odb_to_xarray_skips_empty_files(tmp_path: Path, monkeypatch) -> None:
    valid_file = tmp_path / "valid.odb"
    empty_file = tmp_path / "empty.odb"
    valid_file.write_bytes(b"odb")
    empty_file.write_bytes(b"")

    def fake_read_odb(path, single):
        assert Path(path) == valid_file
        assert single is True
        return pd.DataFrame(
            {
                "date@hdr": [20240206],
                "time@hdr": [0],
                "lat@hdr": [21.0],
                "lon@hdr": [10.0],
                "obsvalue@body": [251.0],
            }
        )

    monkeypatch.setitem(
        sys.modules,
        "pyodc",
        types.SimpleNamespace(read_odb=fake_read_odb),
    )

    ds = read_odb_to_xarray(
        [valid_file, empty_file],
        ["brightness_temperature"],
    )

    assert ds.sizes["time"] == 1
    assert float(ds["brightness_temperature"].isel(time=0, cell=0)) == 251.0
    assert str(empty_file) in ds.attrs["empty_odb_files"]


def test_read_odb_to_xarray_filters_each_file_to_its_analysis_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    valid_file = tmp_path / "valid.odb"
    valid_file.write_bytes(b"odb")

    def fake_read_odb(path, single):
        assert Path(path) == valid_file
        assert single is True
        return pd.DataFrame(
            {
                "date@hdr": [20240206, 20240206],
                "time@hdr": [0, 30000],
                "lat@hdr": [21.0, 22.0],
                "lon@hdr": [10.0, 11.0],
                "obsvalue@body": [251.0, 252.0],
            }
        )

    monkeypatch.setitem(
        sys.modules,
        "pyodc",
        types.SimpleNamespace(read_odb=fake_read_odb),
    )

    ds = read_odb_to_xarray(
        [valid_file],
        ["brightness_temperature"],
        aggregation_window="3h",
        analysis_times=[pd.Timestamp("2024-02-06T00:00:00")],
    )

    assert ds.sizes["time"] == 1
    assert pd.Timestamp(ds["time"].item()) == pd.Timestamp("2024-02-06T00:00:00")
    assert ds.sizes["cell"] == 1
    assert float(ds["brightness_temperature"].isel(time=0, cell=0)) == 251.0
