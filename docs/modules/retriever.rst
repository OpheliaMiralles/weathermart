Retriever
=========

The ``weathermart.retrievers`` package contains concrete ``BaseRetriever``
implementations for API-backed, MARS/FDB-backed, and local-file-backed data
sources. New retrievers should expose:

* a ``retrieve`` method;
* a ``sources`` attribute listing the source identifiers it can serve;
* a ``variables`` attribute listing the public variable names it accepts;
* a ``crs`` attribute when the returned data uses a known coordinate reference
  system.

The ``DataRetriever`` iterates through its configured retrievers and calls the
first retriever that supports the requested source and variables.

Current built-in retrievers
---------------------------

.. list-table::
   :header-rows: 1

   * - Retriever
     - Sources
     - Notes
   * - ``GribRetriever``
     - configured by file archive
     - GRIB model archives.
   * - ``FDBRetriever``
     - configured by FDB request
     - ECMWF FDB-backed GRIB retrieval.
   * - ``OperaRetriever``
     - ``OPERA``
     - OPERA radar composites from the MeteoFrance API.
   * - ``EumetsatRetriever``
     - ``MSG_SEVIRI``, ``METOP``, ``MTG``, ``NOAA``, ``AWS``
     - EUMETSAT geostationary and polar-orbiting products.
   * - ``MarsRetriever``
     - ``MARS``, ``ECMWF_MARS``, ``MARS_GRIB``
     - ECMWF MARS requests.
   * - ``MarsODBRetriever``
     - ``MARS_ODB``, ``ECMWF_ODB``
     - MARS ODB request files and optional submission.
   * - ``TitanRetriever``
     - ``TITAN``
     - Local TITAN Nordic analysis diagnostics.
   * - ``CEDTMRetriever``
     - ``CEDTM``
     - Copernicus DEM tiles.
   * - ``NASADEMRetriever``
     - ``NASADEM``
     - NASADEM tiles.
   * - ``DHM25Retriever``
     - ``DHM25``
     - Swiss DHM25 terrain data.

.. automodule:: weathermart.retrieve
   :members:
   :noindex:
