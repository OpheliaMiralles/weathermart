import os
from pathlib import Path

from weathermart.retrievers.mars import MarsRetriever

output_dir = Path(os.environ.get("MARS_OUTPUT_DIR", "mars_requests_t_700"))
rc_credential_path = Path(os.environ.get("ECMWF_API_RC_PATH", ".ecmwfapirc"))

OPER_BASE_REQUEST = {
    "class": "od",
    "expver": "1",
    "stream": "oper",
    "type": "an",
}

STRUCTURE_TIMES_BY_DATE = {
    "2025-10-03": ["18:00:00"],
    "2025-10-04": ["00:00:00"],
}


def structure_requests() -> list[tuple[str, dict, str]]:
    requests = []
    for date, times in STRUCTURE_TIMES_BY_DATE.items():
        date_suffix = date.replace("-", "")
        requests.extend(
            [
                (
                    f"ifs_850_structure_{date_suffix}",
                    {
                        **OPER_BASE_REQUEST,
                        "date": date,
                        "time": times,
                        "levelist": "850",
                        "levtype": "pl",
                        "param": ["131", "132"],
                    },
                    f"ifs_850_structure_{date_suffix}.grib",
                ),
                (
                    f"ifs_surface_structure_{date_suffix}",
                    {
                        **OPER_BASE_REQUEST,
                        "date": date,
                        "time": times,
                        "levtype": "sfc",
                        "param": "151",
                    },
                    f"ifs_surface_structure_{date_suffix}.grib",
                ),
            ]
        )
    return requests


def retrieve() -> None:
    retriever = MarsRetriever()
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = structure_requests()

    for _, _, part_target in parts:
        (output_dir / part_target).unlink(missing_ok=True)

    for request_name, request, part_target in parts:
        ds = retriever.retrieve(
            source="MARS",
            request=request,
            target=part_target,
            output_dir=output_dir,
            rc_credential_path=rc_credential_path,
            submit=True,
            request_name=request_name,
            mars_max_queued_requests=20,
            mars_queue_poll_seconds=60,
        )
        print(ds.attrs["request_text"])


if __name__ == "__main__":
    retrieve()
