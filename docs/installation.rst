.. highlight:: shell

============
Requirements
============

This project uses ``uv``, which you can install by running the following command:

.. code-block:: shell

    curl -LsSf https://astral.sh/uv/install.sh | sh

With uv ready, run ``uv sync --all-extras``. This will install the correct python version and dependencies, the package **with all extras**.


======
Extras
======

Some of the dependencies are split up into different extras.

+-------------------------------+------------+--------------------------------------------+
| Source                        | Group name | Dependencies                               |
+===============================+============+============================================+
| EUMETSAT API                  | eumetsat   | eumdac satpy pyresample                    |
+-------------------------------+------------+--------------------------------------------+
| OPERA API                     | radar      | wradlib                                    |
+-------------------------------+------------+--------------------------------------------+
| GRIB                          | grib       | earthkit meteodata-lab                     |
+-------------------------------+------------+--------------------------------------------+
| DEM (Digital elevation model) | dem        | rioxarray pystac-client planetary-computer |
+-------------------------------+------------+--------------------------------------------+

If you want to download and read grib files using ECMWF's `FDB <https://github.com/ecmwf/fdb>`_, you will need to follow specific instructions detailed in ``weathermart/retrievers/fdb-install.md``.
