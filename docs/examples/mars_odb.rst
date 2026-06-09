MARS ODB
========

The MARS retriever writes ODB request files for the requested channel groups and
optionally submits them with the local ``mars`` executable.

.. code-block:: python

    from weathermart.retrievers.mars import MarsODBRetriever

    retriever = MarsODBRetriever()
    ds = retriever.retrieve(
        source="MARS_ODB",
        variables=["AMSU-A"],
        dates=["2020-01-01"],
        output_dir="mars_odb_requests",
        reportypes=["21009", "21010"],
        submit=False,
    )

