import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weathermart.retrievers.eumetsat import EumetsatRetriever
from weathermart.retrievers.eumetsat import plot_polar
from weathermart.retrievers.mars import MARS_INSTRUMENT_REPORTYPES
from weathermart.retrievers.mars import read_odb_to_xarray

PLOT_DIR = Path("plots/radiance_instruments")
MARS_ODB_DIR = Path("mars_odb_requests")
MARS_ODB_DATE = pd.Timestamp("2024-02-06T00:00:00")
EUMETSAT_TEST_MODE = True
AGGREGATION_WINDOW = "3h"
CREDENTIALS_PATH = os.environ.get("EUMDAC_CREDENTIALS_PATH", ".eumdac_credentials.json")

ODB_INSTRUMENTS = ["AMSU-A", "AMSU-B/MHS", "ATMS", "MWHS-2", "AWS"]
ATMS_CHANNELS = [str(channel) for channel in [*range(6, 16), *range(18, 23)]]


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


def count_plot_points(ds, t, var: str) -> int:
    ds_time = ds.sel(time=t) if "time" in ds.dims or "time" in ds.coords else ds
    valid = (
        np.isfinite(ds_time["longitude"])
        & np.isfinite(ds_time["latitude"])
        & np.isfinite(ds_time[var])
    )
    return int(valid.sum().compute().item())


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


def plot_mars_odb() -> None:
    targets = sorted(MARS_ODB_DIR.glob(f"*_{MARS_ODB_DATE:%Y%m%d}.odb"))
    if not targets:
        print(f"No ODB files found in {MARS_ODB_DIR}")
        return

    ds = read_odb_to_xarray(
        targets,
        variables=["brightness_temperature", "reportype", "channel"],
        source="MARS_ODB",
        aggregation_window=AGGREGATION_WINDOW,
    )
    if not ds.sizes.get("time", 0):
        print("No non-empty ODB files available for plotting.")
        return
    print_dataset_summary("MARS ODB combined", ds)

    reportype_values = ds["reportype"].values
    for instrument in ODB_INSTRUMENTS:
        reportypes = list(MARS_INSTRUMENT_REPORTYPES[instrument])
        if not np.isin(reportype_values, reportypes).any():
            print(f"No ODB data available for {instrument}")
            continue
        mask = ds["reportype"].isin(reportypes) & np.isfinite(
            ds["brightness_temperature"]
        )
        valid_times = mask.any("cell")
        if not bool(valid_times.any()):
            print(f"No finite ODB brightness temperatures available for {instrument}")
            continue
        counts = mask.groupby(ds["channel"]).sum()
        counts = counts.where(counts > 0, drop=True)
        if not counts.sizes.get("channel", 0):
            print(f"No finite ODB channel values available for {instrument}")
            continue
        channel = int(counts.idxmax("channel").item())
        channel_mask = mask & (ds["channel"] == channel)
        valid_times = channel_mask.any("cell")
        plot_time = ds["time"].where(valid_times, drop=True).isel(time=0).values
        if not bool(channel_mask.sel(time=plot_time).any()):
            print(f"No finite ODB channel values available for {instrument}")
            continue
        plotted_points = int(channel_mask.sel(time=plot_time).sum().item())
        print(
            f"MARS ODB {instrument}: time={pd.to_datetime(plot_time)}, "
            f"channel={channel}, plotted_points={plotted_points}"
        )
        instrument_ds = ds.where(
            ds["reportype"].isin(reportypes) & (ds["channel"] == channel)
        )
        plot_polar(
            instrument_ds,
            plot_time,
            "brightness_temperature",
            title=f"MARS ODB {instrument} channel {channel}",
            aggregation_window=AGGREGATION_WINDOW,
        )
        safe_name = instrument.lower().replace("/", "_").replace(" ", "_")
        save_current_figure(PLOT_DIR / f"mars_odb_{safe_name}_channel_{channel}.png")


def plot_eumetsat() -> None:
    retriever = EumetsatRetriever()
    requests = [
        {
            "source": "METOP",
            "product": "atms_radiances",
            "variables": ATMS_CHANNELS,
            "date": pd.Timestamp("2015-01-01T09:00:00"),
            "plot_var": "6",
            "output": "eumetsat_atms_channel_6_lat_gt_20_polar.png",
        },
    ]
    for request in requests:
        ds = retriever.retrieve(
            source=request["source"],
            variables=request["variables"],
            product=request["product"],
            dates=[request["date"]],
            eumdac_credentials_path=CREDENTIALS_PATH,
            resolution="16km",
            resample=False,
            aggregation_window=AGGREGATION_WINDOW,
            aggregate_time=True,
            test=EUMETSAT_TEST_MODE,
        )
        if not ds.sizes.get("time", 0):
            print(f"No EUMETSAT data available for {request['source']}")
            continue
        print_dataset_summary(
            f"EUMETSAT {request['source']} {request['product']}",
            ds,
        )
        ds = mask_northern_domain(ds, min_latitude=20)
        plot_time = first_time_with_points(ds, request["plot_var"])
        if plot_time is None:
            print(f"No EUMETSAT lat > 20 points for {request['source']}")
            continue
        plotted_points = count_plot_points(ds, plot_time, request["plot_var"])
        print(
            f"EUMETSAT {request['source']} {request['plot_var']}: "
            f"time={pd.to_datetime(plot_time)}, lat > 20, "
            f"plotted_points={plotted_points}"
        )
        if not plotted_points:
            print(f"No EUMETSAT lat > 20 points for {request['source']}")
            continue
        plot_polar(
            ds,
            plot_time,
            request["plot_var"],
            title=(
                f"EUMETSAT {request['source']} {request['product']} "
                f"channel {request['plot_var']}"
            ),
            aggregation_window=AGGREGATION_WINDOW,
        )
        save_current_figure(PLOT_DIR / request["output"])


def main() -> None:
    plot_mars_odb()
    plot_eumetsat()


if __name__ == "__main__":
    main()
