weathermart: the weather data market
====================================

``weathermart`` provides a shared interface for retrieving weather, remote
sensing, terrain, and observation datasets. Retrievers can be called directly or
through ``DataProvider``, which checks a local cache before retrieving missing
data.

Available retrievers
--------------------

.. list-table::
   :header-rows: 1

   * - Retriever
     - Sources
     - Description
   * - ``GribRetriever``
     - configured by file archive
     - GRIB data from local model archives.
   * - ``FDBRetriever``
     - configured by FDB request
     - GRIB data from an ECMWF FDB installation.
   * - ``OperaRetriever``
     - ``OPERA``
     - OPERA radar composites from the MeteoFrance API.
   * - ``EumetsatRetriever``
     - ``MSG_SEVIRI``, ``METOP``, ``MTG``, ``NOAA``, ``AWS``
     - Geostationary and polar-orbiting EUMETSAT products.
   * - ``MarsRetriever``
     - ``MARS``, ``ECMWF_MARS``, ``MARS_GRIB``
     - ECMWF MARS requests.
   * - ``MarsODBRetriever``
     - ``MARS_ODB``, ``ECMWF_ODB``
     - ECMWF MARS ODB radiance request files and optional submission.
   * - ``CEDTMRetriever``
     - ``CEDTM``
     - Copernicus DEM tiles.
   * - ``NASADEMRetriever``
     - ``NASADEM``
     - NASADEM tiles.
   * - ``DHM25Retriever``
     - ``DHM25``
     - Swiss DHM25 terrain data.

Examples
--------

Direct retriever use:

.. code:: python

   import pandas as pd

   from weathermart.retrievers.radar import OperaRetriever

   retriever = OperaRetriever()
   ds = retriever.retrieve(
       source="OPERA",
       variables=["TOT_PREC"],
       dates=[pd.Timestamp("2024-01-01T12:00:00")],
       meteofranceapi_token_path=".meteofranceapi_token.json",
   )

Provider use:

.. code:: python

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

Use ``cache_location=None`` to disable caching. More examples are available in
the ``examples/`` directory.
