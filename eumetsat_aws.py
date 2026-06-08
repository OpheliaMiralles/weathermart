import datetime
import os
import sys

import pandas as pd

from weathermart.default_provider import default_provider

AWS_CHANNELS = [str(channel) for channel in [*range(4, 9), *range(11, 16)]]
ANALYSIS_HOURS = range(0, 24, 3)
AGGREGATION_WINDOW = "3h"
STORAGE_KEY = "eumetsat_aws_dop"
DEFAULT_START_DATE = pd.Timestamp(os.environ.get("EUMETSAT_AWS_START_DATE", "2025-01-01"))
DEFAULT_END_DATE = pd.Timestamp(os.environ.get("EUMETSAT_AWS_END_DATE", "2026-06-08"))


def month_days(year: int, month: int) -> list[pd.Timestamp]:
    first = pd.Timestamp(year=year, month=month, day=1)
    last = first + pd.offsets.MonthEnd(0)
    days = pd.date_range(first, last, freq="D")
    return [
        day
        for day in days
        if DEFAULT_START_DATE.normalize() <= day <= DEFAULT_END_DATE.normalize()
    ]


def analysis_times(day: pd.Timestamp) -> list[pd.Timestamp]:
    return [day + pd.Timedelta(hours=hour) for hour in ANALYSIS_HOURS]


def retrieve_month(year: int, month: int) -> None:
    provider = default_provider()
    days = month_days(year, month)
    print(
        f"[INFO] retrieving EUMETSAT AWS MWR for {year}-{month:02d}, "
        f"{len(days)} days",
        flush=True,
    )
    if not days:
        return

    credentials_path = os.environ.get(
        "EUMDAC_CREDENTIALS_PATH",
        ".eumdac_credentials.json",
    )

    for day in days:
        print(f"[INFO] retrieving EUMETSAT AWS {day.date()}", flush=True)
        start = datetime.datetime.now(datetime.UTC)
        try:
            ds = provider.provide(
                source="AWS",
                variables=AWS_CHANNELS,
                product="mwr_l1b",
                dates=analysis_times(day),
                eumdac_credentials_path=credentials_path,
                resample=False,
                aggregate_time=True,
                aggregation_window=AGGREGATION_WINDOW,
                storage_key=STORAGE_KEY,
            )
            end = datetime.datetime.now(datetime.UTC)
            print(
                f"[INFO] finished EUMETSAT AWS {day.date()}, "
                f"dims={dict(ds.sizes)}, duration={end - start}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARN] failed retrieving EUMETSAT AWS {day.date()}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: eumetsat_aws.py YEAR MONTH")
        sys.exit(1)

    retrieve_month(int(sys.argv[1]), int(sys.argv[2]))
