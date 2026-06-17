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
