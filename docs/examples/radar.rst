OPERA Radar
===========

You can also retrieve OPERA radar data from the MeteoFrance API.
This requires an API key, which you can obtain by registering on the MeteoFrance website.

.. code-block:: python

    import pandas as pd

    from weathermart.retrievers.radar import OperaRetriever

    retriever = OperaRetriever()
    ds = retriever.retrieve(
        source="OPERA",
        variables=["TOT_PREC"],
        dates=[pd.Timestamp("2024-01-01T12:00:00")],
        meteofranceapi_token_path=".meteofranceapi_token.json",
    )

.. image:: ../_static/opera_20231020.png
    :width: 800
    :align: center

Live Rainbow volumes
====================

Run MET's ``zmq-datacopier`` subscribers as persistent services and point both
publishers at one local spool directory. Subscribe to ``dBZ vol``. Weathermart
does not own the long-running subscription; it reads the completed flat files
from that spool and derives the same Rainbow column products as the archive
extractor.

.. code-block:: python

    import pandas as pd

    from weathermart.retrievers.radar import NordicRadarRetriever

    times = pd.date_range(
        "2026-10-07T08:35:00Z",
        "2026-10-07T09:00:00Z",
        freq="5min",
    )
    rainbow = NordicRadarRetriever().open_live_rainbow_volumes(
        "/data/prorad/rainbow-incoming",
        times,
        ["czc", "ezc20", "ezc45", "hzc"],
    )

The default completeness check requires every requested time for the 13 current
operational radars. The fixed template still retains the historical GRM block,
which remains missing rather than changing the model grid.
