import datetime
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weathermart.default_provider import default_provider
from weathermart.retrievers.eumetsat import plot_polar
from weathermart.utils import NORTH_LATITUDE_20_BBOX

ATMS_CHANNELS = [str(channel) for channel in [*range(6, 16), *range(18, 23)]]
AWS_CHANNELS = [str(channel) for channel in [*range(4, 9), *range(11, 16)]]
PLOT_METADATA_VARIABLES = ["latitude", "longitude"]

CREDENTIALS_PATH = ".eumdac_credentials.json"
PLOT_DIR = Path("plots/radiance_instruments")
TEST_MODE = False
AGGREGATION_WINDOW = "3h"

REQUESTS = [
    {
        "source": "AWS",
        "product": "mwr_l1b",
        "variables": [*AWS_CHANNELS, *PLOT_METADATA_VARIABLES],
        "bbox": NORTH_LATITUDE_20_BBOX,
        # AWS MWR L1B collection EO:EUM:DAT:0905 starts in 2025, not 2024.
        "date": [pd.to_datetime("2025-04-11") + pd.Timedelta(hours=hour) for hour in range(0, 24, 3)],
        "storage_key": "eumetsat_aws_channels_example",
        "plot_var": "4",
        "output": "eumetsat_aws_channel_4_lat_gt_20_polar.png",
    },
]


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def print_dataset_summary(label: str, ds) -> None:
    time_steps = ds.sizes.get("time", 0)
    variables = list(ds.data_vars)
    scan_times = ""
    if "scan_time" in ds:
        scan_time_values = np.asarray(ds["scan_time"].values).ravel()
        finite_scan_times = scan_time_values[~pd.isna(scan_time_values)]
        scan_times = f", scan_time_values={len(finite_scan_times)}"
    print(
        f"{label}: time_steps={time_steps}, dims={dict(ds.sizes)}, "
        f"variables={len(variables)} {variables}{scan_times}"
    )


def mask_northern_domain(ds, min_latitude: float = 20):
    return ds.where(ds["latitude"] > min_latitude)


def first_time_with_points(ds, var: str):
    valid = (
        np.isfinite(ds["longitude"])
        & np.isfinite(ds["latitude"])
        & np.isfinite(ds[var])
    )
    valid_times = valid.any([dim for dim in valid.dims if dim != "time"])
    valid_times = valid_times.compute()
    if not bool(valid_times.any()):
        return None
    return ds["time"].where(valid_times, drop=True).isel(time=0).values


def count_plot_points(ds, t, var: str) -> int:
    ds_time = ds.sel(time=t) if "time" in ds.dims or "time" in ds.coords else ds
    valid = (
        np.isfinite(ds_time["longitude"])
        & np.isfinite(ds_time["latitude"])
        & np.isfinite(ds_time[var])
    )
    return int(valid.sum().compute().item())


def plot_request(request: dict, ds) -> None:
    plot_var = request["plot_var"]
    if not ds.sizes.get("time", 0):
        print(f"No EUMETSAT data available for {request['source']}")
        return

    print_dataset_summary(
        f"EUMETSAT {request['source']} {request['product']}",
        ds,
    )
    ds = mask_northern_domain(ds, min_latitude=20)
    plot_time = first_time_with_points(ds, plot_var)
    if plot_time is None:
        print(f"No EUMETSAT lat > 20 points for {request['source']}")
        return

    plotted_points = count_plot_points(ds, plot_time, plot_var)
    print(
        f"EUMETSAT {request['source']} {plot_var}: "
        f"time={pd.to_datetime(plot_time)}, lat > 20, "
        f"plotted_points={plotted_points}"
    )
    if not plotted_points:
        print(f"No EUMETSAT lat > 20 points for {request['source']}")
        return

    plot_polar(
        ds,
        plot_time,
        plot_var,
        title=f"EUMETSAT {request['source']} {request['product']} channel {plot_var}",
        aggregation_window=AGGREGATION_WINDOW,
    )
    save_current_figure(PLOT_DIR / request["output"])


def retrieve() -> None:
    provider = default_provider()
    for request in REQUESTS:
        try:
            start = datetime.datetime.now(datetime.UTC)
            data = provider.provide(
                source=request["source"],
                variables=request["variables"],
                product=request["product"],
                dates=request["date"],
                storage_key=request["storage_key"],
                eumdac_credentials_path=CREDENTIALS_PATH,
                resample=False,
                aggregation_window=AGGREGATION_WINDOW,
                aggregate_time=True,
                test=TEST_MODE,
            )
            print(data)
            plot_request(request, data)
            end = datetime.datetime.now(datetime.UTC)
            print(f"Retrieval took {end - start}")
        except Exception as exc:
            traceback.print_exc()
            print(
                f"Failed retrieving {request['source']} "
                f"{request['product']} for {request['date']:%Y-%m-%d %H:%M}: {exc}"
            )


if __name__ == "__main__":
    retrieve()
