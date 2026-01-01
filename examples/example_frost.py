import datetime

import pandas as pd

from weathermart.default_provider import default_provider

p = default_provider()


def retrieve():
    for date in pd.date_range("2024-01-10", "2024-01-10", freq="D"):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.utcnow()
            data = p.provide(
                source="OBSERVATIONS",
                credentials_path=".frost_credentials.json",
                storage_key="frost_example",
                variables=[
                    "over_time(time_of_maximum_wind_speed PT1H)",
                    "mean(wind_speed PT1H)",
                    "sum(precipitation_amount PT6H)",
                ],
                dates=[pd.to_datetime(date)],
                stations=[
                    "SN50540",
                    "SN63705",
                    "SN27780",
                    "SN30650",
                    "SN30255",
                    "SN28380",
                    "SN10380",
                    "SN13420",
                    "SN71550",
                ],
            )
            print(data)
            end = datetime.datetime.utcnow()
            print(f"Retrieval took {end - start}")
        except Exception as e:
            print(f"Failed retrieving date {date.date()}: {e}")
            continue
        
if __name__ == "__main__":
    retrieve()
