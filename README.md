# weathermart: the weather data market

Documentation: <https://weathermart.readthedocs.io/en/latest/>

`weathermart` provides a common provider/retriever interface over several weather data sources, including EUMETSAT satellite products, MeteoFrance OPERA radar products, GRIB-based forecast data, and terrain datasets.

## Installation

Run the setup script from the repository root:

```bash
chmod +x install.sh
./install.sh
source .venv/bin/activate
```

`install.sh` installs `uv`, creates the local `.venv`, installs the package with all extras, and sets up the CODA and HARP dependencies needed for EUMETSAT products such as METOP and MTG.

## Credentials

Some retrievers require external accounts and API credentials.

### EUMETSAT

Create an account in the EUMETSAT portal and generate API credentials:

- Portal: <https://api.eumetsat.int/api-key/>

Store the credentials in `.eumdac_credentials.json` in the repository root:

```json
{
  "consumer_key": "YOUR_EUMETSAT_CONSUMER_KEY",
  "consumer_secret": "YOUR_EUMETSAT_CONSUMER_SECRET"
}
```

The satellite examples in `src/weathermart/examples/` use this file by default.

### MeteoFrance OPERA Radar

Create an account at the MeteoFrance API portal and generate an API token:

- Portal: <https://portail-api.meteofrance.fr>

Store the token in `.meteofranceapi_token.json`:

```json
{
  "OPERA_API_TOKEN": "YOUR_METEOFRANCE_API_TOKEN"
}
```

Alternatively, you can export the token directly:

```bash
export OPERA_API_TOKEN=YOUR_METEOFRANCE_API_TOKEN
```

## Implemented Retrievers

The package can retrieve data from:

- NWP forecasts via GRIB files
- Radar data via the MeteoFrance OPERA API
- Satellite data via the EUMETSAT API
- DEM raw products via URL-based downloads

Retrievers can be used directly or through the provider, which checks the cache first and only retrieves missing data.

## Provider Example

```python
import numpy as np
import pandas as pd

from weathermart.default_provider import default_provider

provider = default_provider()
config = {
    "ICON-CH1-EPS": ["CLCT", "TOT_PREC", "U_10M", "V_10M", "QV_2M", "T_2M", "P", "SP"],
    "dates": pd.date_range("2023-08-01", "2024-09-09"),
}

provider.provide_from_config(
    config, datatype="forecast", ensemble_members=0, step_hours=np.arange(1, 13)
)
```

## Examples

Example scripts are available in `src/weathermart/examples/`.

For example:

```bash
python src/weathermart/examples/example_msg.py
python src/weathermart/examples/example_mtg.py
python src/weathermart/examples/example_metop.py
```
