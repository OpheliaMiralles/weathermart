import datetime
import os
import traceback

import pandas as pd

xmin, xmax = -180, 180
ymin, ymax = 50, 90
bbox = (xmin, ymin, xmax, ymax)
from weathermart.default_provider import default_provider  # noqa: E402
from weathermart.retrievers.eumetsat import plot_polar  # noqa: E402

vars_iasi = ["38", "49"]


def retrieve():
    provider = default_provider(cache_location=os.environ.get("WEATHERMART_CACHE"))
    for date in pd.date_range("2021-01-10", "2021-01-10", freq="D"):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.now(datetime.UTC)
            data = provider.provide(
                source="METOP",
                variables=vars_iasi,
                product="iasi_radiances",
                bbox=bbox,
                test=True,
                dates=[pd.to_datetime(date)],
                storage_key="iasi_test",
                eumdac_credentials_path=".eumdac_credentials.json",
                resolution="10km",
            )
            print(data)
            plot_polar(data, t=data.time.values[0], var=vars_iasi[0])
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            traceback.print_exc()
            print(f"Failed retrieving date {date.date()}: {e}")
            continue


if __name__ == "__main__":
    retrieve()
