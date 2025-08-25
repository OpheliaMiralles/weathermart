.. highlight:: shell

============
Requirements
============
### Requirements

This project uses `uv`, which you can install by running the following command:
``
curl -LsSf https://astral.sh/uv/install.sh | sh``

With uv ready, run `uv sync --all-extras`. This will install the correct python version and dependencies, the package **with all extras** and `pre-commit`.


============
Extras
============

Some of the dependencies are split up into different extras.

+---------------------------+---------+
| Source                    | ``<XYZ>`` |
+===========================+=========+
| EUMETSAT API              | eumetsat|
+---------------------------+---------+
| OPERA API                 | radar   |
+---------------------------+---------+
| GRIB                      | eccodes |
+---------------------------+---------+
| DEM (Digital elevation model) | dem |
+---------------------------+---------+