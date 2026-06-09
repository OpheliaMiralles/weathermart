#!/usr/bin/env python
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CANCEL_STATUSES = {"queued", "active", "submitted"}


def _credentials() -> tuple[str, str, str]:
    rc_path = Path(os.environ.get("ECMWF_API_RC_FILE", ".ecmwfapirc")).expanduser()
    with rc_path.open(encoding="utf-8") as file:
        rc = json.load(file)
    return rc["key"], rc["url"].rstrip("/"), rc["email"]


def _request(url: str, headers: dict[str, str], method: str = "GET"):
    request = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(request, timeout=30)


def main() -> int:
    statuses = set(sys.argv[1:]) or DEFAULT_CANCEL_STATUSES
    key, base_url, email = _credentials()
    headers = {
        "Accept": "application/json",
        "From": email,
        "X-ECMWF-KEY": key,
    }

    with _request(f"{base_url}/services/mars/requests", headers) as response:
        payload = json.loads(response.read().decode("utf-8"))

    requests = [
        item
        for item in payload.get("mars", [])
        if item.get("status") in statuses and item.get("href")
    ]
    print(f"ECMWF MARS cleanup: cancelling {len(requests)} request(s)")

    failures = 0
    for item in requests:
        status = item.get("status")
        name = item.get("name")
        try:
            with _request(item["href"], headers, method="DELETE") as response:
                print(f"cancelled {status} {name}: HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            failures += 1
            body = exc.read().decode(errors="replace").replace("\n", " ")[:300]
            print(f"failed {status} {name}: HTTP {exc.code} {body}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
