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

### Extras

Some of the dependencies are split up into different extras. The extra dependency groups list as follows:

| Source  | `<XYZ>` |
| -------- | ------- |
| EUMETSAT API | eumetsat |
| Radar data (OPERA) | radar |
| GRIB | eccodes |
| DEM (Digital elevation model) | dem |

## Implemented retrievers

The data-provider package allows to retrieve data from:

- **nwp forecasts** via gridefix and grib files on balfrin
- **station observations** via jretrieve
- **radar data** via MeteoFrance API
- **EUMETSAT data** via EUMETSAT API
- **dem raw products** from various sources via url queries.

The retrievers can be accessed _individually_ or via the InputProvider, where data is _read from a cache_ before trying to retrieve. Available variables and their mapping to their original name are listed in the "variables" attribute of each retriever. To retrieve data (e.g. "U_10M") from a particular source (e.g. COSMO-1E) for given dates (e.g. 2021-01-01) you can call:
```python
ds = GribRetriever().retrieve("COSMO-1E", "U_10M", pd.to_datetime("2021-01-01"))
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

Finally, describe your request in a dict-like config:
```python
config = {
    "dates": dates,
    "OPERA": ["TOT_PREC"],
    "MSG_SEVIRI": ["IR_039"],
    "ICON-CH1-EPS": ["U_10M", "V_10M"],
    "SYNOP": "tde200s0",
    "NASADEM": "nasadem",
}
```

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
