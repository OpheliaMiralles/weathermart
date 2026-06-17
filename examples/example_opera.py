import datetime
import os

import pandas as pd

from weathermart.default_provider import default_provider

dates = [
    "20200110",
]

TOKEN_PATH = os.environ.get(
    "METEOFRANCE_API_TOKEN_PATH",
    ".meteofranceapi_token.json",
)


def retrieve():
    provider = default_provider(cache_location=os.environ.get("WEATHERMART_CACHE"))
    for date in pd.to_datetime(dates):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.now(datetime.UTC)
            data = provider.provide(
                source="OPERA",
                variables=["RAINFALL_RATE", "qc_flags"],
                dates=[pd.to_datetime(date)],
                meteofranceapi_token_path=TOKEN_PATH,
                storage_key="",
            )
            print(data)
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue


if __name__ == "__main__":
    retrieve()
