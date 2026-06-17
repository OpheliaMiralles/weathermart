MARS ODB
========

The MARS retriever writes ODB request files for the requested channel groups and
optionally submits them with the local ``mars`` executable.

.. code-block:: python

    from weathermart.retrievers.mars import MarsODBRetriever

    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["brightness_temperature", "channel", "reportype"],
        instruments=["AMSU-A"],
        dates=["2020-01-01"],
        output_dir="mars_odb_requests",
        submit=False,
    )
