import datetime

import pandas as pd

from weathermart.default_provider import default_provider

dates = [
    "20200110",
]

p = default_provider()


def retrieve():
    for date in pd.to_datetime(dates):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.utcnow()
            data = p.provide(
                source="OPERA",
                variables=["RAINFALL_RATE", "qc_flags"],
                dates=[pd.to_datetime(date)],
                meteofranceapi_token_path="/home/opmir9231/weathermart/.meteofranceapi_token.json",
                storage_key="",
            )
            end = datetime.datetime.utcnow()
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue
        
if __name__ == "__main__":
    retrieve()
