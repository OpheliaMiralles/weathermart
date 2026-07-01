import datetime
import os

import pandas as pd

from weathermart.default_provider import default_provider

dates = [
    "2020-05-10",
]
all_dates = pd.to_datetime(dates)


def retrieve():
    provider = default_provider(cache_location=os.environ.get("WEATHERMART_CACHE"))
    for date in pd.to_datetime(dates):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.now(datetime.UTC)
            data = provider.provide(
                source="NORDIC_RADAR",
                variables=["lwe_precipitation_rate", "qc_flags"],
                dates=[pd.to_datetime(date)],
                dense_qc=True,
                endpoint="archive",
                test=True,
                storage_key="start_2020_example",
            )
            print(data)
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue


if __name__ == "__main__":
    retrieve()
