import logging
import os
from collections.abc import Sequence
from importlib.metadata import entry_points
from pathlib import Path

from weathermart.base import BaseRetriever
from weathermart.provide import CacheRetriever
from weathermart.provide import DataProvider

DEFAULT_CACHE = Path("") / os.environ["USER"]


def available_retrievers() -> Sequence[BaseRetriever]:
    """
    Get all available retriever instances.

    Returns
    -------
    list
        List containing instances of available data retrievers.
    """
    retrievers: list[BaseRetriever] = []

    plugins = entry_points(group="weathermart.retriever")
    for plugin in plugins:
        try:
            target = plugin.load()
            if callable(target):
                instance = target()
            if isinstance(instance, BaseRetriever):
                retrievers.append(instance)
        except Exception as e:
            logging.exception("Failed loading retriever plugin %s: %s", plugin.name, e)

    retrievers.sort(key=lambda r: (r.priority, r.__class__.__name__), reverse=True)

    return retrievers


def default_provider(cache_location: Path | None = DEFAULT_CACHE) -> DataProvider:
    """
    Create the default DataProvider with caching.

    This function creates a CacheProvider using the given location,
    prints a message indicating the cache location, and returns an DataProvider
    configured with all available retrievers.

    Parameters
    ----------
    cache_location : Path, optional
        Path to the cache directory. If None, no caching is used. Default is None.

    Returns
    -------
    DataProvider
        An input data provider instance configured with caching and all retrievers.
    """
    if cache_location:
        cache = CacheRetriever(cache_location)
        logging.warning("Using the cache located at %s.", cache_location)
    else:
        cache = None
        logging.warning("No cache is being used.")
    return DataProvider(cache, retrievers=available_retrievers())
