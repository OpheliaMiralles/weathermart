"""Fetch model-ready radar inputs without mixing their spatial geometries."""

import pandas as pd
import xarray as xr

from weathermart.retrievers.radar import NordicRadarRetriever

RAINBOW_VARIABLES = [
    "beam_height",
    "czc",
    "ezc20",
    "ezc45",
    "hzc",
    "lzc",
    "rzc",
]
RADAR_VARIABLES = [
    "lwe_precipitation_rate",
    "lwe_precipitation_rate_netatmo",
]


def retrieve_nowcast_inputs(
    dates: list[pd.Timestamp],
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return native-grid radar/Netatmo and Anemoi-aligned Rainbow inputs."""
    return NordicRadarRetriever().retrieve_nowcast_inputs(
        dates=dates,
        radar_variables=RADAR_VARIABLES,
        rainbow_variables=RAINBOW_VARIABLES,
    )


if __name__ == "__main__":
    radar_input, rainbow_input = retrieve_nowcast_inputs(
        [pd.Timestamp("2025-01-01T12:00:00")]
    )
    print(radar_input)
    print(rainbow_input)
