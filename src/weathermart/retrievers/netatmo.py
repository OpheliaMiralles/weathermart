from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Any
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import numpy as np
import json
from pathlib import Path
import yrlib
import yrlib.netatmo
from weathermart.base import BaseRetriever, checktype

CHUNK_SIZE = 24        # 24 × 5 min = 2 hours per worker
MAX_WORKERS = min(12, os.cpu_count() or 4)
OUT_ROOT = Path("/lustre/storeB/users/opmir9231/netatmo_daily")
VAR_META = {
    "ta": dict(use="Temperature", units="C", name="Temperature", deacc=False),
    "pr": dict(use="Pressure", units="hPa", name="Pressure", deacc=False),
    "uu": dict(use="Humidity", units="%", name="Humidity", deacc=False),
    "rr": dict(use="Rain", units="mm/h", name="Precipitation rate", deacc=True),
    "rr_hourly": dict(use="sum_rain_1", units="kg/m^2", name="Hourly precipitation", deacc=False),
    "ff": dict(use="wind", units="m/s", name="Wind speed", deacc=False),
    "dd": dict(use="wind", units="degrees", name="Wind direction", deacc=False),
    "fg": dict(use="wind_gust", units="m/s", name="Wind gust", deacc=False),
}
TITAN_VARIABLES = list(VAR_META.keys())

def read_json_all(filename, latrange=None, lonrange=None):
    data = {}

    if filename is None or not os.path.exists(filename):
        return data

    check_latlon = latrange is not None and lonrange is not None

    with open(filename) as f:
        text = f.read()
        if not text:
            return data

        if text[0] == "{":
            text = "[%s" % text
        text = text.replace("}]{", "}{").replace("}][{", "},{").replace("}{", "},{")
        text = '{"body": %s}' % text

        try:
            body = json.loads(text)["body"]
        except json.JSONDecodeError:
            return data

    for entry in body:
        if "location" not in entry or "data" not in entry:
            continue

        lat = float(entry["location"][1])
        lon = float(entry["location"][0])
        if check_latlon:
            if not (latrange[0] <= lat <= latrange[1] and lonrange[0] <= lon <= lonrange[1]):
                continue

        elev = entry.get("altitude", np.nan)
        loc = yrlib.Location(lat, lon, elev)

        if "time_utc" not in entry["data"]:
            continue

        t = entry["data"]["time_utc"]
        if loc not in data:
            data[loc] = {"unixtime": t}

        for k, v in entry["data"].items():
            if k in ["wind", "wind_gust"]:
                for _, w in v.items():
                    data[loc][k] = (float(w[0]) / 3.6, float(w[1]))
                    break
            else:
                try:
                    data[loc][k] = float(v)
                except Exception:
                    pass

    return data

def get_multi(
    unixtimes,
    variables,
    interpolation="linear",
    aggregation=3600,
    dt=1800,
    datahall="/lustre/storeB",
    latrange=None,
    lonrange=None,
):
    required_unixtimes = set()
    for t in unixtimes:
        required_unixtimes |= {t, t - 600, t + 600}
        if "rr" in variables:
            required_unixtimes.add(t - aggregation)

    required_unixtimes = sorted(required_unixtimes)
    filenames = {}
    for t in required_unixtimes:
        filenames[t] = yrlib.netatmo.get_json_filename(t, datahall)

    raw = {}
    for t, fname in filenames.items():
        if fname is None:
            continue
        data = read_json_all(fname, latrange, lonrange)
        for loc, payload in data.items():
            if loc not in raw:
                raw[loc] = {}
            raw[loc][t] = payload
    locations = list(raw.keys())
    L = len(locations)
    T = len(unixtimes)

    results = {}

    for var in variables:
        meta = VAR_META[var]
        values = np.full((T, L), np.nan, dtype=np.float32)

        for i, loc in enumerate(locations):
            series_t = []
            series_v = []

            for t_req, payload in raw[loc].items():
                if meta["use"] not in payload:
                    continue

                if var == "dd":
                    series_v.append(payload[meta["use"]][1])
                elif var in ["ff", "fg"]:
                    series_v.append(payload[meta["use"]][0])
                else:
                    series_v.append(payload[meta["use"]])

                series_t.append(payload["unixtime"])

            if not series_t:
                continue

            series_t = np.array(series_t)
            series_v = np.array(series_v)

            values[:, i] = yrlib.util.interpolate(
                unixtimes,
                series_t,
                series_v,
                interpolation,
                dt,
                meta["deacc"],
            )

            if meta["deacc"]:
                prev = yrlib.util.interpolate(
                    [t - aggregation for t in unixtimes],
                    series_t,
                    series_v,
                    interpolation,
                    dt,
                    True,
                )
                values[:, i] -= prev
                values[values < 0] = np.nan

        results[var] = yrlib.dataset.Point(
            unixtimes,
            locations,
            values,
            variable=meta["name"],
            units=meta["units"],
        )
    return results


def netatmo_to_xarray(datetimes):
    unixtime = [(pd.to_datetime(d)-pd.to_datetime("1970-01-01T00:00:00Z")).total_seconds() for d in pd.to_datetime(datetimes)]
    def point_to_xarray(p, varname):
        time = pd.to_datetime(p.times, unit="s", utc=True)

        locs = p.locations
        loc_id = np.array([hash(loc) for loc in locs])
        lat = np.array([loc.lat for loc in locs])
        lon = np.array([loc.lon for loc in locs])
        alt = np.array([loc.elev for loc in locs])

        ds = xr.Dataset(
            data_vars={
                varname: (("time", "location"), p.values)
            },
            coords=dict(
                time=time,
                location=("location", loc_id),
                latitude=("location", lat),
                longitude=("location", lon),
                altitude=("location", alt),
            ),
        )

        ds[varname].attrs["units"] = p.units
        ds[varname].attrs["long_name"] = p.variable

        return ds

    datasets = []
    print("Fetching data for ", unixtime, flush=True)
    results = get_multi(
        unixtimes=unixtime,
        variables=TITAN_VARIABLES,
    )
    for v in TITAN_VARIABLES:
        p = results[v]
        if len(p.locations) == 0:
            continue
        ds_v = point_to_xarray(p, v)
        datasets.append(ds_v)
    ds = xr.merge(
        datasets,
        join="outer",        
        compat="no_conflicts",
    )
    return ds

def netatmo_to_xarray_parallel(datetimes):
    datetimes = pd.to_datetime(datetimes)
    datetimes = datetimes.sort_values()
    blocks = [
        datetimes[i:i + CHUNK_SIZE]
        for i in range(0, len(datetimes), CHUNK_SIZE)
    ]
    print("Total blocks:", len(blocks), flush=True)

    datasets = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_worker_netatmo_block, block): block
            for block in blocks
        }

        for fut in as_completed(futures):
            ds = fut.result()
            if ds is not None and len(ds.data_vars) > 0:
                datasets.append(ds)

    if not datasets:
        return xr.Dataset()
    print(f"Combining {len(datasets)} datasets", flush=True)

    ds = xr.concat(
        datasets,
        dim="time",
        join="outer",
        compat="no_conflicts",
        coords="minimal",
    ).sortby("time")
    ds["time"] = pd.to_datetime(ds.time.values).tz_localize(None)
    return ds

def _worker_netatmo_block(datetimes_block):
    try:
        print("Processing block:", datetimes_block[0], "to", datetimes_block[-1], flush=True)
        return netatmo_to_xarray(datetimes_block)
    except Exception as e:
        print(f"[WARN] block failed: {e}", flush=True)
        return None


class NetAtmoRetriever(BaseRetriever):
    """
    Retriever for Netatmo station data (5-minute resolution).
    """

    sources = ("NETATMO",)
    variables = {k: [k] for k in VAR_META.keys()}
    crs = "epsg:4326"

    def retrieve(
        self,
        source: str,
        variables: list[tuple[str, dict]],
        dates: datetime.date | str | pd.Timestamp | list[Any],
        datahall: str = "/lustre/storeB",
        latrange: list[float] | None = None,
        lonrange: list[float] | None = None,
        freq: str = "5min",
    ) -> xr.Dataset:

        dates, variables = checktype(dates, variables)
        unique_dates = list(set(pd.to_datetime(d).date() for d in dates))
        datasets = []
        for d in pd.to_datetime(unique_dates).tz_localize("UTC"):
            times = pd.date_range(
                d,
                d + pd.Timedelta(days=1)-pd.Timedelta(minutes=5),
                freq=freq,
                inclusive="left",
                tz="UTC",
            )
            ds = netatmo_to_xarray_parallel(times)
            if ds is None or len(ds.data_vars) == 0:
                print(f"{d:%Y-%m-%d} → no data", flush=True)
                continue
            for v in ds.variables:
                ds[v].encoding.clear()
            ds = ds.chunk(dict(time=-1, location=5000))
            datasets.append(ds)

        varnames = [vname for vname, _ in variables]
        if not datasets:
            return xr.Dataset()
        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = xr.concat(
                datasets,
                dim="time",
                join="outer",
                compat="no_conflicts",
                coords="minimal",
            ).sortby("time")
        ds = ds[[v for v in varnames if v in ds.data_vars]]
        ds.attrs["source"] = "Netatmo"
        ds.attrs["provider"] = "YR / MET Norway"
        ds["time"] = pd.to_datetime(ds.time.values).tz_localize(None)
        return ds

