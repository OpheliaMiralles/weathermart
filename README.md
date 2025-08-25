# Weathermart

## Installation

This is written for usage on Balfrin.

### Requirements

* pyenv: https://github.com/pyenv/pyenv
* poetry: https://python-poetry.org/docs/

If you didn't have these tools before, quickly restart your shell to assure poetry can find pyenv.

With poetry and pyenv ready, run `./install.sh`. This will install the correct python version including the requirements for `sqlite3` and `lzma`, the poetry package **with all extras** and `pre-commit`.
In theory, you can also use the retriever with a python version without `sqlite3` and `lzma` if you don't plan to use the satellite retriever.

### Extras

Some of the dependencies are split up into different extras.
By default, `install.sh` will install all extras.

| Source  | `<XYZ>` |
| -------- | ------- |
| EUMETSAT API | eumetsat |
| Radar data (MCH, OPERA) | radar |
| GRIB | eccodes |
| DEM (Digital elevation model) | dem |

## Implemented retrievers

The data-provider package allows to retrieve data from:

- **nwp forecasts** via gridefix and grib files on balfrin
- **nowcasting analyses** via gridefix
- **station observations** via jretrieve
- **radar data** via msrad archive on balfrin / OPERA data on balfrin / MeteoFrance API
- **satellite data** on balfrin and EUMETSAT API
- **dem raw products** from various sources via url queries.

The retrievers can be accessed _individually_ or via the InputProvider, where data is _read from a cache on balfrin_ before trying to retrieve. Available variables and their mapping to their original name are listed in the "variables" attribute of each retriever. To retrieve data (e.g. "U_10M" -originally called "northward_wind") from a particular source (e.g. COSMO-1E) for given dates (e.g. 2021-01-01) you can call:
```python
ds=NWPRetriever().retrieve("COSMO-1E", "U_10M", pd.to_datetime("2021-01-01"))
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

then define the location of the cache see [confluence page for nowcasting data cache](https://meteoswiss.atlassian.net/wiki/spaces/Nowcasting/pages/322143175/Data+cache)
```python
cache = CacheProvider(cache_path)
```

and initialise the provider.

```python
provider = DataProvider(cache)
```
You can also use directly the default data provider, which will use the default cache path. It will use by default all of the retrievers to request the desired data.
```python
from weathermart.default_provider import default_provider
provider = default_provider()
```

Finally, describe your request in a dict-like config:
```python
config = {
    "dates": dates,
    "CMSAF": "CLCT",
    "ICON-CH1-EPS": ["U_10M", "V_10M"],
    "SURFACE": "T_2M",
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
provider.provide_from_config(config, through="balfrin", type="forecast", ensemble_members=0, step_hour=np.arange(1,13))
```
The provider will loop through the cache and the retrievers' available sources to get data. It also should save every missing data field in the cache.

The provide_from_config method can also be called using a yaml config file, you just need to add the following lines to read and resolve the config. The config needs to be in a format resembling anemoi config yaml as described [here](https://anemoi-datasets.readthedocs.io/en/latest/building/introduction.html#concepts).
```python
dict_config = get_retrieve_config("config.yaml")
provider.provide_from_config(dict_config)
```
