import datetime
import os

import pandas as pd

from weathermart.default_provider import default_provider

p = default_provider()
template_path = os.environ.get(
    "LIGHTNING_TEMPLATE_PATH",
    "/lustre/storeB/users/opmir9231/nordic_radar/20250101",
)


def retrieve() -> None:
    for date in pd.date_range("2024-01-10", "2024-01-10", freq="D"):
        try:
            print(f"Retrieving lightning for {date.date()}")
            start = datetime.datetime.utcnow()
            data = p.provide(
                source="LIGHTNING",
                credentials_path=".frost_credentials.json",
                storage_key="lightning_example",
                variables=["lightning_count"],
                dates=[pd.to_datetime(date)],
                template_path=template_path,
            )
            print(data)
            end = datetime.datetime.utcnow()
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue


if __name__ == "__main__":
    retrieve()
