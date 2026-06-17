# weathermart: the weather data market

https://weathermart.readthedocs.io/en/latest/

`weathermart` provides a shared interface for retrieving weather, remote
sensing, terrain, and observation datasets. Retrievers can be called directly or
through `DataProvider`, which checks a local cache before retrieving missing
data.

## Requirements

This project uses `uv`, which you can install with:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the package and development tools:

```shell
uv sync --all-extras --dev
```

## Optional dependency groups

Some retrievers need optional dependencies:

| Source | Extra | Main dependencies |
| --- | --- | --- |
| EUMETSAT API | `eumetsat` | `eumdac`, `satpy`, `pyresample` |
| OPERA radar API | `radar` | `wradlib` |
| GRIB files | `grib` | `earthkit`, `meteodata-lab` |
| DEM products | `dem` | `rioxarray`, `pystac-client`, `planetary-computer` |

The FDB retriever also requires a working ECMWF FDB installation. See
[`src/weathermart/retrievers/fdb_install.md`](src/weathermart/retrievers/fdb_install.md).

## Available retrievers

The currently registered retrievers are:

| Retriever | Sources | Description |
| --- | --- | --- |
| `GribRetriever` | configured by file archive | GRIB data from local model archives. |
| `FDBRetriever` | configured by FDB request | GRIB data from an ECMWF FDB installation. |
| `OperaRetriever` | `OPERA` | OPERA radar composites from the MeteoFrance API. |
| `EumetsatRetriever` | `MSG_SEVIRI`, `METOP`, `MTG`, `NOAA`, `AWS` | Geostationary and polar-orbiting EUMETSAT products. |
| `MarsODBRetriever` | `MARS_ODB`, `ECMWF_ODB` | ECMWF MARS ODB radiance request files and optional submission. |
| `MarsRetriever` | `MARS`, `ECMWF_MARS`, `MARS_GRIB` | ECMWF MARS requests. |
| `CEDTMRetriever` | `CEDTM` | Copernicus DEM tiles. |
| `NASADEMRetriever` | `NASADEM` | NASADEM tiles. |
| `DHM25Retriever` | `DHM25` | Swiss DHM25 terrain data. |

Available variables are exposed on each retriever through its `variables`
attribute.

## Direct retrieval

Call a retriever directly when you know the source and the retriever-specific
arguments:

```python
import pandas as pd

from weathermart.retrievers.radar import OperaRetriever

retriever = OperaRetriever()
ds = retriever.retrieve(
    source="OPERA",
    variables=["TOT_PREC"],
    dates=[pd.Timestamp("2024-01-01T12:00:00")],
    meteofranceapi_token_path=".meteofranceapi_token.json",
)
```

## Retrieval through a provider

`DataProvider` can combine a cache with all registered retrievers:

```python
import pandas as pd

from weathermart.default_provider import default_provider

provider = default_provider(cache_location="/path/to/cache")
config = {
    "dates": [pd.Timestamp("2024-01-01T12:00:00")],
    "OPERA": ["TOT_PREC"],
    "MSG_SEVIRI": ["IR_039"],
    "NASADEM": ["nasadem"],
}
ds = provider.provide_from_config(
    config,
    bounds=(4.0, 54.0, 32.0, 72.0),
    target_crs="epsg:4326",
)
```

Pass `cache_location=None` to skip caching.

Runnable examples live in [`examples/`](examples/), and Sphinx documentation is
under [`docs/`](docs/).
