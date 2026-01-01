import datetime

import pandas as pd

from weathermart.default_provider import default_provider

p = default_provider()


def retrieve():
    for date in pd.date_range("2020-01-10", "2020-01-10", freq="D"):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.utcnow()
            data = p.provide(
            source='NETATMO',
            variables=["ta", "rr", "ff"],
            dates=[pd.to_datetime(date)],
            storage_key="netatmo_example_fast")
            print(data)
            end = datetime.datetime.utcnow()
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue
        
if __name__ == "__main__":
    retrieve()
