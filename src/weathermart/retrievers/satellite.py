import datetime
import traceback

import pandas as pd

xmin, xmax = -180, 180
ymin, ymax = 50, 90
bbox = (xmin, ymin, xmax, ymax)
from weathermart.default_provider import default_provider  # noqa: E402
from weathermart.retrievers.satellite import plot_polar  # noqa: E402

p = default_provider()
vars_iasi = ["temp_15um", "swir_36um"] # TODO: add all interesting channels in supported variables
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
vars_avhrr = ['1', '2', '3a', '4', '5', 'satellite_zenith_angle', 'solar_zenith_angle']


def retrieve():
    for date in pd.date_range('2021-01-10', '2021-01-10', freq='D'):
        try:
            print(f"Retrieving date {date.date()}")
            start = datetime.datetime.now(datetime.UTC)
            data = p.provide(
            source='METOP',
            variables=vars_avhrr,
            product='avhrr_l1',
            bbox=bbox,
            test=True,
            dates=[pd.to_datetime(date)],
            storage_key="avhrr_test",
            eumdac_credentials_path=".eumdac_credentials.json",
            resolution='10km')
            print(data)
            plot_polar(data, t=data.time.values[0], var=vars_avhrr[0])
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as e:
            traceback.print_exc()
            print(f"Failed retrieving date {date.date()}: {e}")
            continue
        
if __name__ == "__main__":
    retrieve()
