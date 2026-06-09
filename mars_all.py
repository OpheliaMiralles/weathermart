import datetime
import os
import sys
from pathlib import Path

import pandas as pd

from weathermart.default_provider import default_provider
from weathermart.retrievers.mars import MINIMAL_RADIANCE_ODB_COLUMNS
from weathermart.retrievers.mars import NORTH_LATITUDE_20_DOMAIN_FILTER

MARS_INSTRUMENTS = ["AMSU-A", "AMSU-B/MHS", "ATMS", "MWHS-2"]
MARS_RADIANCE_VARIABLES = [
    "brightness_temperature",
    "channel",
    "reportype",
    "lsm",
    "seaice",
    "satellite_identifier",
    "satellite_instrument",
    "zenith",
    "azimuth",
    "scanline",
    "scanpos",
    "datum_status",
    "biascorr_fg",
    "fg_depar",
]
ANALYSIS_HOURS = range(0, 24, 3)
AGGREGATION_WINDOW = "3h"
STORAGE_KEY = "mars_all_radiances_v2"


def _env_timestamp(
    name: str, default: str | datetime.datetime, *, end: bool
) -> pd.Timestamp:
    raw_value = os.environ.get(name)
    timestamp = pd.Timestamp(raw_value or default).tz_localize(None)
    if end and raw_value is not None and len(raw_value) == 10:
        timestamp = timestamp + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return timestamp


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"", "0", "none", "null", "false", "off"}:
        return None
    return float(raw_value)


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"", "0", "none", "null", "false", "off"}:
        return None
    return int(raw_value)


DEFAULT_START_TIME = _env_timestamp("MARS_START_DATE", "1979-01-01", end=False)
DEFAULT_END_TIME = _env_timestamp(
    "MARS_END_DATE",
    datetime.datetime.now(datetime.UTC),
    end=True,
)


def month_days(year: int, month: int) -> list[pd.Timestamp]:
    first = pd.Timestamp(year=year, month=month, day=1)
    last = first + pd.offsets.MonthEnd(0)
    days = pd.date_range(first, last, freq="D")
    return [
        day
        for day in days
        if DEFAULT_START_TIME.normalize() <= day <= DEFAULT_END_TIME.normalize()
    ]


def analysis_times(day: pd.Timestamp) -> list[pd.Timestamp]:
    return [
        analysis_time
        for analysis_time in (day + pd.Timedelta(hours=hour) for hour in ANALYSIS_HOURS)
        if DEFAULT_START_TIME <= analysis_time <= DEFAULT_END_TIME
    ]


def retrieve_month(year: int, month: int) -> None:
    provider = default_provider()
    days = month_days(year, month)
    print(
        f"[INFO] retrieving MARS radiances for {year}-{month:02d}, {len(days)} days",
        flush=True,
    )
    if not days:
        return

    rc_credential_path = Path(os.environ.get("ECMWF_API_RC_FILE", ".ecmwfapirc"))
    output_root = Path(
        os.environ.get(
            "MARS_ODB_OUTPUT_DIR",
            "/lustre/storeB/users/opmir9231/tmp/mars_odb_requests",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    mars_executable = os.environ.get("MARS_EXECUTABLE", "mars")
    mars_timeout = _env_optional_float("MARS_TIMEOUT")
    mars_max_queued_requests = _env_optional_int("MARS_MAX_QUEUED_REQUESTS", 19)
    mars_queue_poll_seconds = _env_optional_float("MARS_QUEUE_POLL_SECONDS", 300) or 300
    mars_queue_wait_timeout = _env_optional_float("MARS_QUEUE_WAIT_TIMEOUT")
    domain_filter = os.environ.get(
        "MARS_DOMAIN_FILTER",
        NORTH_LATITUDE_20_DOMAIN_FILTER,
    )

    for day in days:
        times = analysis_times(day)
        if not times:
            continue
        print(f"[INFO] retrieving MARS {day.date()}", flush=True)
        start = datetime.datetime.now(datetime.UTC)
        try:
            ds = provider.provide(
                source="MARS_ODB",
                variables=MARS_RADIANCE_VARIABLES,
                instruments=MARS_INSTRUMENTS,
                dates=times,
                output_dir=output_root / f"{day:%Y%m}",
                rc_credential_path=rc_credential_path,
                submit=True,
                mars_executable=mars_executable,
                mars_timeout=mars_timeout,
                mars_max_queued_requests=mars_max_queued_requests,
                mars_queue_poll_seconds=mars_queue_poll_seconds,
                mars_queue_wait_timeout=mars_queue_wait_timeout,
                domain_filter=domain_filter,
                odb_columns=MINIMAL_RADIANCE_ODB_COLUMNS,
                aggregation_window=AGGREGATION_WINDOW,
                storage_key=STORAGE_KEY,
            )
            end = datetime.datetime.now(datetime.UTC)
            print(
                f"[INFO] finished MARS {day.date()}, "
                f"dims={dict(ds.sizes)}, duration={end - start}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] failed retrieving MARS {day.date()}: {exc}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: mars_all.py YEAR MONTH")
        sys.exit(1)

    retrieve_month(int(sys.argv[1]), int(sys.argv[2]))
