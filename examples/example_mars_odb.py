import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weathermart.retrievers.eumetsat import plot_polar
from weathermart.retrievers.mars import MARS_INSTRUMENT_REPORTYPES
from weathermart.retrievers.mars import MINIMAL_RADIANCE_ODB_COLUMNS
from weathermart.retrievers.mars import MarsODBRetriever
from weathermart.utils import NORTH_LATITUDE_20_DOMAIN_FILTER

retriever = MarsODBRetriever()
rc_credential_path = Path(".ecmwfapirc")
output_dir = Path("mars_odb_requests")
plot_dir = Path("plots/radiance_instruments")
plot_output = True
aggregation_window = "3h"
request_date = pd.Timestamp("2024-02-02T00:00:00")
instruments = ["AMSU-A", "AMSU-B/MHS", "ATMS", "MWHS-2"]
variables = [
    "brightness_temperature",
    "channel",
    "reportype",
    "lsm",
    "seaice",
    "satellite_identifier",
    "satellite_instrument",
    "zenith",
    "azimuth",
    "scanline",
    "scanpos",
    "datum_status",
    "biascorr_fg",
    "fg_depar",
]


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_instrument(data, instrument: str) -> None:
    reportypes = list(MARS_INSTRUMENT_REPORTYPES[instrument])
    instrument_mask = data["reportype"].isin(reportypes) & np.isfinite(
        data["brightness_temperature"]
    )
    if not bool(instrument_mask.any()):
        print(f"No finite ODB brightness temperatures available for {instrument}")
        return

    counts = instrument_mask.groupby(data["channel"]).sum()
    counts = counts.where(counts > 0, drop=True)
    if not counts.sizes.get("channel", 0):
        print(f"No finite ODB channel values available for {instrument}")
        return

    channel = int(counts.idxmax("channel").item())
    channel_mask = instrument_mask & (data["channel"] == channel)
    valid_times = channel_mask.any("cell")
    if not bool(valid_times.any()):
        print(f"No finite ODB channel values available for {instrument}")
        return

    plot_time = data["time"].where(valid_times, drop=True).isel(time=0).values
    plotted_points = int(channel_mask.sel(time=plot_time).sum().item())
    print(
        f"MARS ODB {instrument}: time={pd.to_datetime(plot_time)}, "
        f"channel={channel}, plotted_points={plotted_points}"
    )
    instrument_data = data.where(
        data["reportype"].isin(reportypes) & (data["channel"] == channel)
    )
    plot_polar(
        instrument_data,
        plot_time,
        "brightness_temperature",
        title=f"MARS ODB {instrument} channel {channel}",
        aggregation_window=aggregation_window,
    )
    safe_name = instrument.lower().replace("/", "_").replace(" ", "_")
    save_current_figure(plot_dir / f"mars_odb_{safe_name}_channel_{channel}.png")


def retrieve() -> None:
    start = datetime.datetime.now(datetime.UTC)
    data = retriever.retrieve(
        source="MARS_ODB",
        variables=variables,
        instruments=instruments,
        dates=[request_date],
        output_dir=output_dir,
        rc_credential_path=rc_credential_path,
        submit=True,
        domain_filter=NORTH_LATITUDE_20_DOMAIN_FILTER,
        odb_columns=MINIMAL_RADIANCE_ODB_COLUMNS,
        aggregation_window=aggregation_window,
    )
    print(data)
    print("Requested instruments:", ", ".join(instruments))
    print("Missing ODB variables:", data.attrs.get("missing_odb_variables", ""))
    if plot_output and data.sizes.get("time", 0):
        for instrument in instruments:
            plot_instrument(data, instrument)
    end = datetime.datetime.now(datetime.UTC)
    print(f"Retrieval took {end - start}")


if __name__ == "__main__":
    retrieve()
