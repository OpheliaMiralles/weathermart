import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yrlib
import yrlib.netatmo

from weathermart.base import BaseRetriever
from weathermart.base import checktype

CHUNK_SIZE = 24  # 24 × 5 min = 2 hours per worker
MAX_WORKERS = min(12, os.cpu_count() or 4)
OUT_ROOT = Path("/lustre/storeB/users/" + os.environ["USER"] + "/netatmo_daily")
VAR_META = {
    "ta": dict(use="Temperature", units="C", name="Temperature", deacc=False),
    "pr": dict(use="Pressure", units="hPa", name="Pressure", deacc=False),
    "uu": dict(use="Humidity", units="%", name="Humidity", deacc=False),
    "rr": dict(use="Rain", units="mm/h", name="Precipitation rate", deacc=True),
    "rr_hourly": dict(
        use="sum_rain_1", units="kg/m^2", name="Hourly precipitation", deacc=False
    ),
    "ff": dict(use="wind", units="m/s", name="Wind speed", deacc=False),
    "dd": dict(use="wind", units="degrees", name="Wind direction", deacc=False),
    "fg": dict(use="wind_gust", units="m/s", name="Wind gust", deacc=False),
}
TITAN_VARIABLES = list(VAR_META.keys())
QC_LIMITS = {
    "ta": dict(min=-50.0, max=50.0),  # °C
    "ff": dict(min=0.0, max=80.0),  # m/s
    "fg": dict(min=0.0, max=120.0),  # m/s (gusts)
    "rr": dict(min=0.0, max=1000.0),  # mm/h, rate
    "pr": dict(min=850.0, max=1100.0),  # hPa
    "uu": dict(min=0.0, max=100.0),  # %
    "dd": dict(min=0.0, max=360.0),  # degrees
    "rr_hourly": dict(min=0.0, max=500.0),  # mm/h, hourly total
    "altitude": dict(min=-50.0, max=5000.0),  # m
}


def qc(
    da: xr.DataArray,
    varname: str,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """
    Apply physical range QC and return:
      - QC'ed DataArray (out-of-range → NaN)
      - Boolean flag per sample indicating presence of NaNs in original data
      - Filled DataArray (forward/backward fill to replace NaNs in time and buddy fill to replace NaNs in space if densify=True)
    """
    if varname not in QC_LIMITS:
        da_qc = da
    else:
        vmin = QC_LIMITS[varname]["min"]
        vmax = QC_LIMITS[varname]["max"]
        da_qc = da.where((da >= vmin) & (da <= vmax))
    has_nan = da_qc.isnull()
    da_qc.attrs.update(
        qc_method="physical_range",
        qc_min=QC_LIMITS.get(varname, {}).get("min"),
        qc_max=QC_LIMITS.get(varname, {}).get("max"),
    )
    da = da_qc
    qc_flag = has_nan.astype("uint8")
    qc_flag.name = f"{varname}_is_nan"
    return da_qc, qc_flag


def pack_isnan_flags(
    ds: xr.Dataset, varnames: list[str], name: str = "has_nan"
) -> xr.Dataset:
    qc = xr.zeros_like(ds[varnames[0]], dtype="uint64")

    for k, v in enumerate(varnames):
        bit = ds[v].astype("uint64")
        qc = qc | (bit << np.uint64(k))

    qc = qc.rename(name)
    qc.attrs["qc_bit_layout"] = "bit k: 1 means flag=true for var k"
    qc.attrs["qc_vars"] = ",".join(varnames)
    return ds.drop_vars(varnames).assign({name: qc})


def unpack_isnan_flag(qc_flag: xr.DataArray, k: int) -> xr.DataArray:
    return ((qc_flag >> np.uint64(k)) & np.uint64(1)).astype("uint8")


def decode_qc_flags(ds: xr.Dataset, flag_var: str = "has_nan") -> xr.Dataset:
    """
    Decode individual variable flags from a packed uint64 flag variable.
    """
    mapping_flags = {i: f"{v}_is_nan" for i, v in enumerate(ds.variables)}
    if flag_var not in ds:
        raise KeyError(f"{flag_var} not in dataset")
    out = {}
    for bit, name in mapping_flags.items():
        out[name] = ((ds[flag_var] >> np.uint64(bit)) & np.uint64(1)).astype("uint8")
    return xr.Dataset(out, coords=ds.coords, attrs={"decoded_from": flag_var})


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
            text = f"[{text}"
        text = text.replace("}]{", "}{").replace("}][{", "},{").replace("}{", "},{")
        text = f'{{"body": {text}}}'

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
            if not (
                latrange[0] <= lat <= latrange[1] and lonrange[0] <= lon <= lonrange[1]
            ):
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
    unixtime = [
        (pd.to_datetime(d) - pd.to_datetime("1970-01-01T00:00:00Z")).total_seconds()
        for d in pd.to_datetime(datetimes)
    ]

    def point_to_xarray(p, varname):
        time = pd.to_datetime(p.times, unit="s", utc=True)

        locs = p.locations
        loc_id = np.array([hash(loc) for loc in locs])
        lat = np.array([loc.lat for loc in locs])
        lon = np.array([loc.lon for loc in locs])
        alt = np.array([loc.elev for loc in locs])

        ds = xr.Dataset(
            data_vars={varname: (("time", "location"), p.values)},
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
        datetimes[i : i + CHUNK_SIZE] for i in range(0, len(datetimes), CHUNK_SIZE)
    ]
    print("Total blocks:", len(blocks), flush=True)

    datasets = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_worker_netatmo_block, block): block for block in blocks}

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
        print(
            "Processing block:",
            datetimes_block[0],
            "to",
            datetimes_block[-1],
            flush=True,
        )
        return netatmo_to_xarray(datetimes_block)
    except Exception as e:
        print(f"[WARN] block failed: {e}", flush=True)
        return None


class NetAtmoRetriever(BaseRetriever):
    """
    Retriever for Netatmo station data (5-minute resolution).
    """

    sources = ("NETATMO",)
    variables = list(VAR_META.keys())
    crs = "epsg:4326"

    def retrieve(
        self,
        source: str,
        variables: list[str] | str,
        dates: datetime.date | str | pd.Timestamp | list[Any],
        freq: str = "5min",
        *,
        do_qc: bool = True,
        pack_nan_flags: bool = True,
        round_decimals: int = 4,
    ) -> xr.Dataset:
        dates, variables = checktype(dates, variables)
        varnames_req = [vname for vname, _ in variables]
        unique_days = sorted(set(pd.to_datetime(d).date() for d in dates))

        day_datasets: list[xr.Dataset] = []
        for d in pd.to_datetime(unique_days).tz_localize("UTC"):
            times = pd.date_range(
                d,
                d + pd.Timedelta(days=1) - pd.Timedelta(minutes=5),
                freq=freq,
                inclusive="left",
                tz="UTC",
            )

            ds = netatmo_to_xarray_parallel(times)
            if ds is None or len(ds.data_vars) == 0:
                print(f"{d:%Y-%m-%d} → no data", flush=True)
                continue

            for v in ds.data_vars:
                ds[v].encoding.clear()

            lon = np.round(ds["longitude"].values.astype(np.float64), round_decimals)
            lat = np.round(ds["latitude"].values.astype(np.float64), round_decimals)
            lon[lon == -0.0] = 0.0
            lat[lat == -0.0] = 0.0
            loc_id = np.char.add(
                np.char.mod(f"%.{round_decimals}f", lon),
                np.char.add("_", np.char.mod(f"%.{round_decimals}f", lat)),
            )

            ds = ds.assign_coords(id=("location", loc_id)).swap_dims({"location": "id"})
            ds = ds.drop_vars("location")
            idx = pd.Index(ds["id"].values)
            keep = ~idx.duplicated(keep="first")
            ds = ds.isel(id=np.where(keep)[0])
            ds["time"] = pd.to_datetime(ds.time.values).tz_localize(None)
            keep_vars = [v for v in varnames_req if v in ds.data_vars]
            ds = ds[keep_vars]
            if do_qc:
                out_vars: dict[str, xr.DataArray] = {}
                nan_flag_names: list[str] = []

                for v in keep_vars:
                    da_qc, qc_flag = qc(ds[v], v)
                    out_vars[v] = da_qc
                    out_vars[qc_flag.name] = qc_flag
                    nan_flag_names.append(qc_flag.name)

                ds = xr.Dataset(out_vars, coords=ds.coords, attrs=ds.attrs)
                if pack_nan_flags and nan_flag_names:
                    ds = pack_isnan_flags(ds, nan_flag_names, name="has_nan")

            chunk_spec = {}
            if "time" in ds.dims:
                chunk_spec["time"] = -1
            if "id" in ds.dims:
                chunk_spec["id"] = 5000
            ds = ds.chunk(chunk_spec)

            ds.attrs["source"] = "Netatmo"
            ds.attrs["provider"] = "YR / MET Norway"
            day_datasets.append(ds)

        if not day_datasets:
            return xr.Dataset()

        if len(day_datasets) == 1:
            return day_datasets[0]

        ds_all = xr.concat(
            day_datasets,
            dim="time",
            join="outer",
            compat="no_conflicts",
            coords="minimal",
        ).sortby("time")

        ds_all.attrs["source"] = "Netatmo"
        ds_all.attrs["provider"] = "YR / MET Norway"
        return ds_all
