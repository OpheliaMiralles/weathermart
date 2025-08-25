# Weathermart
https://weathermart.readthedocs.io/en/latest/

### Requirements

This project uses `uv`, which you can install by running the following command:
``
curl -LsSf https://astral.sh/uv/install.sh | sh``

With uv ready, run `uv sync --all-extras`. This will install the correct python version and dependencies, the package **with all extras** and `pre-commit`.

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
- **satellite data** via EUMETSAT API
- **dem raw products** from various sources via url queries.

The retrievers can be accessed _individually_ or via the InputProvider, where data is _read from a cache_ before trying to retrieve. Available variables and their mapping to their original name are listed in the "variables" attribute of each retriever. To retrieve data (e.g. "U_10M") from a particular source (e.g. COSMO-1E) for given dates (e.g. 2021-01-01) you can call:
```python
ds=GribRetriever().retrieve("COSMO-1E", "U_10M", pd.to_datetime("2021-01-01"))
```

If you want to retrieve data from several data sources (e.g. wind from COSMO1E, temperature from station observations) for the same time period, you might as well create an DataProvider and perform the following steps.

First, setup the desired time period:
```python
dates = pd.date_range("20240705", "20240706")
```
and specific arguments to pass some of the retrievers (NASADEMRetriever requires bounds for example):
```python
bbox = (2459987.5, 1059987.5, 2850012.5, 1313012.5)
```

then define the location of the cache:
```python
cache = CacheProvider(cache_path)
```

and initialise the provider.

```python
provider = DataProvider(cache)
```
You can also use directly the default data provider. It will use by default all of the retrievers to request the desired data.
```python
from weathermart.default_provider import default_provider
provider = default_provider()
```

Finally, describe your request in a dict-like config:
```python
config = {
    "dates": dates,
    "OPERA": ["TOT_PREC"],
    "SATELLITE": ["IR_039"],
    "ICON-CH1-EPS": ["U_10M", "V_10M"],
    "SYNOP": "tde200s0",
    "NASADEM": "nasadem",
    }
```

and call the provider from config:
```python
provider.provide_from_config(config, bounds=bbox, target_crs=crs)
```

An example of a full script retrieving ICON forecasts can be found in the [example.py](example.py) file:
```python
import pandas as pd
import numpy as np
from weathermart.default_provider import default_provider

provider = default_provider()
config = {"ICON-CH1-EPS": ["CLCT", "TOT_PREC", "U_10M", "V_10M", "QV_2M", "T_2M", "P", "SP"], "dates": pd.date_range("2023-08-01", "2024-09-09")}
provider.provide_from_config(config, datatype="forecast", ensemble_members=0, step_hours=np.arange(1,13))
```
The provider will loop through the cache and the retrievers' available sources to get data. It also should save every missing data field in the cache.
