import datetime

import pandas as pd

from weathermart.default_provider import default_provider

xmin, xmax = -8.08, 40.73
ymin, ymax = 53.14, 73.06
bbox = (xmin, ymin, xmax, ymax)
p = default_provider()
dates_missing = list(pd.date_range("2025-02-01", "2025-02-01", freq="D"))


def retrieve():
    for date in dates_missing:
        try:
            print(f"Retrieving date {date.date()}", flush=True)
            start = datetime.datetime.now(datetime.UTC)
            data = p.provide(
                source="MTG",
                variables=["flash_count"],
                product="li_flashes",
                bbox=bbox,
                dates=[pd.to_datetime(date)],
                storage_key="li_test",
                eumdac_credentials_path=".eumdac_credentials.json",
                resolution="3km",
                test=True,
            )
            print(data.sizes, flush=True)
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue


if __name__ == "__main__":
    retrieve()
