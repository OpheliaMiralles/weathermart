import datetime
import traceback

import pandas as pd

xmin, xmax = -180, 180
ymin, ymax = 20, 90
bbox = (xmin, ymin, xmax, ymax)
from weathermart.default_provider import default_provider  # noqa: E402
from weathermart.retrievers.eumetsat import plot_polar  # noqa: E402

p = default_provider()
vars_ascat = ["wvc_index",
            "model_speed",
            "model_dir",
            "ice_prob",
            "ice_age",
            "wvc_quality_flag",
            "wind_speed",
            "wind_dir",
            "bs_distance",
]


def retrieve():
    for date in pd.date_range('2021-01-10', '2021-01-10', freq='D'):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.now(datetime.UTC)
            data = p.provide(
            source='METOP',
            variables=vars_ascat,
            product='ascat_coastal_winds',
            bbox=bbox,
            test=True,
            dates=  [pd.to_datetime("2025-04-10") + pd.Timedelta(hours=hour) for hour in range(0, 24, 3)],
            storage_key="ascat_test",
            eumdac_credentials_path=".eumdac_credentials.json",
            resample=True,
            resolution="12km",
            aggregation_window="3h",
            aggregate_time=True)
            print(data)
            print(data.time.values)
            plot_polar(data.rename({"lon": "longitude", "lat": "latitude"}), t=data.time.values[0], var=vars_ascat[0])
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            traceback.print_exc()
            print(f"Failed retrieving date {date.date()}: {e}")
            continue
        
if __name__ == "__main__":
    retrieve()
