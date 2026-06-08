import logging
import os
import pathlib
import tempfile

import pytest
import yaml

from weathermart.default_provider import available_retrievers
from weathermart.provide import CacheRetriever
from weathermart.provide import DataProvider


def load_yaml(yaml_file: str = None) -> dict:
    # Open and load the YAML file
    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    # `data` is now a dictionary
    assert isinstance(data, dict)
    return data


@pytest.fixture
def test_provider() -> DataProvider:
    """
    Create a temporary DataProvider for testing.

    This function creates a CacheProvider using a temporary directory,
    prints a message indicating the temporary cache location, and returns an DataProvider
    configured with all available retrievers.

    Returns
    -------
    DataProvider
        An input data provider instance configured with a temporary cache and all retrievers.
    """
    with tempfile.TemporaryDirectory() as temp_cache:
        cache_path = pathlib.Path(temp_cache)
        cache = CacheRetriever(cache_path)
        logging.info(f"Using temporary cache located at {str(cache_path)}.")
        yield DataProvider(cache, retrievers=available_retrievers())


@pytest.mark.skipif(  # comment this if you want to run the test locally
    not os.getenv("RUN_E2E_TEST", False),
    reason="e2e test takes too long, run separately",
)
@pytest.mark.parametrize(
    "source",
    [
        "surface",
        "eumetsat-seviri",
        "opera",
        "kenda-ch1-before-filename-change",
        "kenda-ch1-after-filename-change",
    ],
)
def test_end2end(
    test_provider: DataProvider,
    source: str,
) -> None:
    """
    End-to-end test for data retrieval from all sources.

    This test validates the functionality of the data provider by loading
    configuration parameters from a YAML file, retrieving data for the specified
    source, and performing basic assertions to ensure data is returned.

    Parameters
    Parameters
    ----------
    test_provider : DataProvider
        An instance of the DataProvider class, configured with all retrievers
        and a temporary cache for testing purposes.
    source : str
        The source identifier to test (e.g., "surface", "radar", "gridefix-model-data").

    Notes
    -----
    - The configuration for each source is loaded from `tests/config_end2end.yaml`.
    - Some sources are commented out due to known issues (e.g., missing credentials,
      no data available, or runtime errors).
    - Additional tests for data validation (e.g., variable checks, date checks,
      non-zero values) should be implemented in the future.

    Raises
    ------
    AssertionError
        If no data is returned for the specified source.
    """
    configdict = load_yaml("tests/config_end2end.yaml")
    kwargs = configdict[source]
    logging.info(f"Testing source: {source} with kwargs: {kwargs}")
    data = test_provider.provide(**kwargs)

    # basic data checks
    varname = list(data.keys())[0]
    time_dim = "time" if "time" in data[varname].dims else "forecast_reference_time"
    assert data, "Data is None or empty"
    assert kwargs["variables"] == varname, "Dataset is missing requested variable."
    assert (
        data[time_dim].values[0].astype("datetime64[D]").astype(object)
        == kwargs["dates"]
    ), "wrong date in dataset"
    assert not data.to_array().isnull().all(), "All values are NaN."
