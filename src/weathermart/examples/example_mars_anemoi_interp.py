import os
from pathlib import Path

from weathermart.retrievers.mars import MarsRetriever

output_dir = Path(
    os.environ.get("MARS_OUTPUT_DIR", "mars_requests_rd_enfo_anemoi_20251003T00")
)
rc_credential_path = Path(
    os.environ.get("ECMWF_API_RC_PATH", "/home/opmir9231/weathermart/.ecmwfapirc")
)

BASE_REQUEST = {
    "class": "rd",
    "date": "2025-10-03",
    "expver": "j4s4",
    "step": "1/to/96/by/1",
    "stream": "enfo",
    "number": "1/to/10",
    "time": "00:00:00",
    "type": "pf",
}

REQUESTS = [
    (
        "rd_enfo_uv850_pf_20251003T00_members_1_to_10_steps_1_to_96",
        {
            **BASE_REQUEST,
            "levelist": "850",
            "levtype": "pl",
            "param": ["131", "132"],
        },
        "rd_enfo_uv850_pf_20251003T00_members_1_to_10_steps_1_to_96.grib",
    ),
    (
        "rd_enfo_tp_mslp_pf_20251003T00_members_1_to_10_steps_1_to_96",
        {
            **BASE_REQUEST,
            "levtype": "sfc",
            "param": ["tp", "151"],
        },
        "rd_enfo_tp_mslp_pf_20251003T00_members_1_to_10_steps_1_to_96.grib",
    ),
]


def retrieve() -> None:
    retriever = MarsRetriever()
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, _, part_target in REQUESTS:
        (output_dir / part_target).unlink(missing_ok=True)

    for request_name, request, part_target in REQUESTS:
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
