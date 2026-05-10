import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weathermart.base import BaseRetriever
from weathermart.base import checktype
from weathermart.provide import CacheRetriever
from weathermart.provide import DataProvider
from weathermart.retrieve import DataRetriever


class DummyRetriever(BaseRetriever):
    """
    Retriever that does nothing and does not implement retrieve() and therefore cannot be initialized.
    """

    pass


class MockRetriever(BaseRetriever):
    """
    MockRetriever is a class that mocks the behavior of a retriever for testing purposes.
    The returned data does not correspond to any real data and is constant for all dates.
    """

    fake_stations: list[str] = ["ZRH", "GVA"]
    variables = {
        "TOT_PREC": "Total precipitation",
        "T2M": "2m temperature",
    }
    sources: list[str] = ("FAKE",)

    def retrieve(self, source, variables, dates):
        dates, variables = checktype(dates, variables)
        # create fake dataset
        ds = xr.Dataset()
        if source not in self.sources:
            raise ValueError(f"Unknown source {source}.")
        for date in dates:
            time = pd.date_range(
                start=f"{date} 00:00:00", end=f"{date} 23:50:50", freq="10min"
            )
            temperature = 15 * np.ones((len(time), len(self.fake_stations)))
            precipitation = 10 * np.ones((len(time), len(self.fake_stations)))
            var_map = {
                "TOT_PREC": precipitation,
                "T2M": temperature,
            }
            for variable in variables:
                if variable not in self.variables.keys():
                    raise ValueError(f"Unknown variable {variable}.")
                ds_var = xr.Dataset(
                    data_vars={
                        variable: (["time", "station"], var_map[variable]),
                    },
                    coords=dict(
                        time=time,
                        station=self.fake_stations,
                    ),
                    attrs=dict(description="Fake weather data."),
                )
                ds = xr.merge([ds_var, ds])
        return ds


class MockRetrieverWronglySorted(MockRetriever):
    """
    Like MockRetriever, but the time coordinate is not sorted.
    """

    def retrieve(self, source, variables, dates):
        ds = super().retrieve(source, variables, dates)
        ds = ds.isel(time=slice(None, None, -1))
        return ds


class MockRetrieverWithKwargs(MockRetriever):
    def retrieve(self, source, variables, dates, special_kwarg=None):
        """
        The same as MockRetriever, but with a special kwarg (that is written as attr).
        """
        dates, variables = checktype(dates, variables)
        # create fake dataset
        ds = super().retrieve(source, variables, dates)
        ds.attrs["special_kwarg"] = special_kwarg
        return ds


class MockRetrieverWithOtherKwargs(MockRetriever):
    def retrieve(self, source, variables, dates, other_kwarg=None):
        """
        The same as MockRetriever, but with a special kwarg (that is written as attr).
        """
        dates, variables = checktype(dates, variables)
        # create fake dataset
        ds = super().retrieve(source, variables, dates)
        ds.attrs["other_kwarg"] = other_kwarg
        return ds


class MockBatchRetriever(MockRetriever):
    sources: list[str] = ("BATCH",)
    batch_dates = True

    def __init__(self):
        self.calls = []

    def retrieve(self, source, variables, dates):
        dates, variables = checktype(dates, variables)
        self.calls.append((source, list(variables), list(dates)))
        return super().retrieve(source, variables, dates)


def test_mockretriever():
    m = MockRetriever()
    assert isinstance(m, BaseRetriever)
    date = ["2024-01-01", "2024-01-02"]
    ds = m.retrieve("FAKE", ["TOT_PREC", "T2M"], date)
    assert isinstance(ds, xr.Dataset)
    assert ds["T2M"].shape == (288, 2)
    assert ds["TOT_PREC"].shape == (288, 2)
    assert ds["T2M"].mean().values == 15
    assert ds["TOT_PREC"].mean().values == 10
    for var in ["TOT_PREC", "T2M"]:
        assert var in ds.data_vars
    assert ds.attrs == {"description": "Fake weather data."}


def test_dataretriever_wrongly_sorted():
    with tempfile.TemporaryDirectory() as tmpdir:
        m = MockRetrieverWronglySorted()
        date = ["2024-01-01", "2024-01-02"]
        cache = CacheRetriever(tmpdir)
        provider = DataProvider(cache, [m])
        with pytest.raises(RuntimeError):
            provider.provide_from_config({"FAKE": ["TOT_PREC", "T2M"], "dates": date})


def test_baseretriever():
    with pytest.raises(TypeError):
        BaseRetriever()


def test_dummyretriever():
    with pytest.raises(TypeError):
        DummyRetriever()


def test_args():
    m = MockRetriever()
    assert m.get_kwargs() == []
    with pytest.raises(ValueError):
        m.validate_kwargs(["fake_kwarg"])

    m_with_kwargs = MockRetrieverWithKwargs()
    assert m_with_kwargs.get_kwargs() == ["special_kwarg"]
    m_with_kwargs.validate_kwargs(["special_kwarg"])
    with pytest.raises(ValueError):
        m_with_kwargs.validate_kwargs(["fake_kwarg"])


def test_dataretriever_kwargs():
    retriever = DataRetriever(
        [MockRetrieverWithKwargs(), MockRetrieverWithOtherKwargs()]
    )

    assert retriever.get_kwargs() == [
        "special_kwarg",
        "other_kwarg",
    ] or retriever.get_kwargs() == [
        "other_kwarg",
        "special_kwarg",
    ]
    retriever.validate_kwargs(["special_kwarg", "other_kwarg"])
    retriever.validate_kwargs(["other_kwarg", "special_kwarg"])  # order does not matter
    retriever.validate_kwargs(["special_kwarg"])  # only one kwarg
    retriever.validate_kwargs(["other_kwarg"])  # only one kwarg
    retriever.validate_kwargs([])  # no kwarg
    with pytest.raises(ValueError):
        retriever.validate_kwargs(["fake_kwarg"])  # no kwarg is valid
    with pytest.raises(ValueError):
        retriever.validate_kwargs(
            ["special_kwarg", "fake_kwarg"]
        )  # one kwarg is not valid


def test_provider_batches_dates_for_batch_capable_retriever():
    with tempfile.TemporaryDirectory() as tmpdir:
        retriever = MockBatchRetriever()
        provider = DataProvider(CacheRetriever(tmpdir), [retriever])

        provider.provide("BATCH", ["TOT_PREC"], ["2024-01-01", "2024-01-02"])
        assert len(retriever.calls) == 1
        assert retriever.calls[0][0] == "BATCH"
        assert retriever.calls[0][1] == ["TOT_PREC"]
        assert retriever.calls[0][2] == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
        ]

        provider.provide("BATCH", ["TOT_PREC"], ["2024-01-01", "2024-01-02"])
        assert len(retriever.calls) == 1
