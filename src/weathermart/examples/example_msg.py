import datetime
import os

import pandas as pd

from weathermart.default_provider import default_provider

xmin, xmax = -8.08, 40.73
ymin, ymax = 53.14, 73.06
bbox = (xmin, ymin, xmax, ymax)
dates_missing = list(pd.date_range("2023-09-29", "2023-09-29", freq="D"))
CREDENTIALS_PATH = os.environ.get("EUMDAC_CREDENTIALS_PATH", ".eumdac_credentials.json")


def retrieve():
    provider = default_provider(cache_location=os.environ.get("WEATHERMART_CACHE"))
    for date in dates_missing:
        try:
            print(f"Retrieving date {date.date()}", flush=True)
            start = datetime.datetime.now(datetime.UTC)
            data = provider.provide(
                source="MSG_SEVIRI",
                variables=[
                    "VIS006",
                    "VIS008",
                    "IR_016",
                    "IR_039",
                    "WV_062",
                    "WV_073",
                    "IR_087",
                    "IR_097",
                    "IR_108",
                    "HRV",
                    "IR_120",
                    "IR_134",
                    "cloud_mask",
                    "cloud_top_height",
                    "cloud_top_quality",
                ],
                bbox=bbox,
                dates=[pd.to_datetime(date)],
                storage_key="msg_benchmark",
                eumdac_credentials_path=CREDENTIALS_PATH,
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
