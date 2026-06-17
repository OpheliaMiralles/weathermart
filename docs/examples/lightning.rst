Lightning
=========

You can retrieve lightning strikes from Frost and project them onto the cached
Nordic radar template grid at 5 minute resolution.

.. code-block:: python

    from weathermart.default_provider import default_provider

    provider = default_provider()
    ds = provider.provide(
        source="LIGHTNING",
        variables=["lightning_count"],
        dates=["2024-01-10"],
        credentials_path=".frost_credentials.json",
        template_path="/path/to/nordic_radar_template.zarr",
    )
