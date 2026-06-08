#!/usr/bin/env bash
set -euo pipefail

MARS_HOME="${MARS_HOME:-$HOME/mars}"
MARS_BIN="$MARS_HOME/bin"
MARS_LIB="$MARS_HOME/lib"
MARS_EXE="$MARS_BIN/mars"

mkdir -p "$MARS_BIN" "$MARS_LIB"

if python -m pip --version >/dev/null 2>&1; then
    python -m pip install ecmwf-api-client
else
    uv pip install ecmwf-api-client
fi

cat > "$MARS_EXE" <<'PY'
#!/usr/bin/env python
import argparse
import os
import re

import ecmwfapi


parser = argparse.ArgumentParser(description="Run MARS request.")
parser.add_argument(
    "infile",
    nargs="?",
    default="-",
    type=argparse.FileType("r"),
    help="file containing a MARS request or STDIN otherwise",
)

args = parser.parse_args()
req = args.infile.read()

if "WEBMARS_TARGET" in os.environ:
    target = os.environ["WEBMARS_TARGET"]
else:
    match = re.search(
        r"\btar(g(e(t)?)?)?\s*=\s*([^'\",\s]+|\"[^\"]*\"|'[^']*')",
        req,
        re.I | re.M,
    )
    if match is None:
        raise Exception("Cannot extract target")

    target = match.group(4)
    if target is None:
        raise Exception("Cannot extract target")

if target[0] == target[-1] and target[0] in ['"', "'"]:
    target = target[1:-1]

client = ecmwfapi.ECMWFService("mars")
client.execute(req, target)
PY

chmod +x "$MARS_EXE"

echo "Installed MARS Web API launcher at $MARS_EXE"
echo "Add this to PATH when needed:"
echo "  export PATH=\"$MARS_BIN:\$PATH\""
echo
echo "Credential lookup order used by ecmwf-api-client:"
echo "  1. ECMWF_API_KEY, ECMWF_API_URL, ECMWF_API_EMAIL"
echo "  2. ECMWF_API_RC_FILE"
echo "  3. ~/.ecmwfapirc"
