import datetime
import os

import pandas as pd

from weathermart.default_provider import default_provider

template_path = os.environ.get(
    "LIGHTNING_TEMPLATE_PATH",
    "/lustre/storeB/users/opmir9231/nordic_radar/20250101",
)
credentials_path = os.environ.get("FROST_CREDENTIALS_PATH", ".frost_credentials.json")


def retrieve() -> None:
    provider = default_provider(cache_location=os.environ.get("WEATHERMART_CACHE"))
    for date in pd.date_range("2024-01-10", "2024-01-10", freq="D"):
        try:
            print(f"Retrieving lightning for {date.date()}")
            start = datetime.datetime.now(datetime.UTC)
            data = provider.provide(
                source="LIGHTNING",
                credentials_path=credentials_path,
                storage_key="lightning_example",
                variables=["lightning_count"],
                dates=[pd.to_datetime(date)],
                template_path=template_path,
            )
            print(data)
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue


if __name__ == "__main__":
    retrieve()
