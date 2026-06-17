from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weathermart.default_provider import default_provider

SOURCE = "CEDTM"
VARIABLES = [SOURCE.lower()]
RESOLUTION = "500m"
OUTPUT_FILE = Path("plots") / f"example_dem_{RESOLUTION}.png"

# Continental Nordic domain in lon/lat. This covers Denmark, mainland Norway,
# Sweden and Finland, but not Iceland or Svalbard.
xmin, xmax = 4.0, 32.0
ymin, ymax = 54.0, 72.0
bounds = (xmin, ymin, xmax, ymax)

# CEDTM is the best fit here:
# - DHM25 only covers Switzerland.
# - NASADEM only reaches 60N, so it misses large parts of the Nordics.
#
# CEDTM is native 30 m. For a domain this large, request a coarser grid to
# keep memory use manageable.
#
# DEM sources are static, but the provider interface still expects a date.
dates = [pd.Timestamp("2025-01-01")]

p = default_provider()


def retrieve():
    data = p.provide(
        source=SOURCE,
        variables=VARIABLES,
        dates=dates,
        bounds=bounds,
        target_crs="epsg:4326",
        resolution=RESOLUTION,
    )
    print(data)
    print(data.sizes)
    return data


def plot(data, output_file: Path = OUTPUT_FILE) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    da = data[VARIABLES[0]]
    values = da.values
    vmin, vmax = np.nanpercentile(values, [2, 98])

    fig, ax = plt.subplots(figsize=(9, 8))
    mesh = ax.pcolormesh(
        da["x"].values / 1000,
        da["y"].values / 1000,
        values,
        cmap="terrain",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label="Elevation [m]")
    ax.set_title(f"{SOURCE} DEM over the Nordic domain ({RESOLUTION})")
    ax.set_xlabel("Projected x [km]")
    ax.set_ylabel("Projected y [km]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_file}")
    return output_file


if __name__ == "__main__":
    plot(retrieve())
