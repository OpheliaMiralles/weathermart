EUMETSAT API
============

You can retrieve data directly from the EUMETSAT API using the ``EumetsatRetriever`` class.
This is generally the preferred way to access EUMETSAT data, since you get the data from a bigger domain.
You need to get an API key from EUMETSAT to use this retriever, see `the EUMETSAT website <https://api.eumetsat.int/api-key/>`_ for instructions.

.. code-block:: python

    import pandas as pd

    from weathermart.retrievers.eumetsat import EumetsatRetriever

    retriever = EumetsatRetriever()
    ds = retriever.retrieve(
        source="MSG_SEVIRI",
        variables=["IR_039"],
        dates=[pd.Timestamp("2024-01-01T12:00:00")],
        eumdac_credentials_path=".eumdac_credentials.json",
    )

.. image:: ../_static/satellite_eumetsat_20231020.png
    :width: 800
    :align: center
