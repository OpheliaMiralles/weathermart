import datetime
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from weathermart.retrievers.eumetsat import EumetsatRetriever
from weathermart.retrievers.eumetsat import plot_polar
from weathermart.utils import NORTH_LATITUDE_20_BBOX

CREDENTIALS_PATH = ".eumdac_credentials.json"
PLOT_DIR = Path("plots/radiance_instruments")

DATE = pd.Timestamp("2021-01-10T12:00:00")
AGGREGATION_WINDOW = "3h"
TEST_MODE = True

IASI_CHANNEL_SPEC = """
38, 49, 51, 55, 57, 61, 63, 83, 85, 87, 104, 109, 111, 116, 122, 128,
135, 141, 146, 148, 154, 159, 161, 167, 173, 179-180, 185, 187, 193,
199, 205, 207, 210, 212, 214, 217, 219, 222, 224, 226, 230, 232, 236,
239, 242-243, 246, 249, 252, 254, 256, 258, 260, 262, 265, 267, 269,
275, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 299, 306, 308,
310, 312, 314, 316, 318, 320, 323, 325, 327, 329, 331, 333, 335, 337,
341, 345, 347, 350, 352, 354, 356, 358, 360, 362, 364, 366, 369, 371,
373, 375, 377, 379, 381, 383, 386, 389, 398, 401, 404, 407, 410, 414,
416, 426, 428, 432, 434, 439, 445, 457, 515, 546, 552, 559, 566, 571,
573, 646, 662, 668, 756, 867, 921, 1027, 1133, 1191, 1194, 1271, 1805,
1884, 1946, 1991, 2094, 2239, 2701, 2819, 2910, 2919, 2991, 2993,
3002, 3008, 3014, 3098, 3207, 3228, 3281, 3309, 3322, 3438, 3442,
3484, 3491, 3499, 3506, 3575, 3582, 3658, 4032
"""


def expand_channel_spec(spec: str) -> list[str]:
    channels = []
    for token in spec.replace("\n", " ").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = [int(part.strip()) for part in token.split("-", 1)]
            channels.extend(str(channel) for channel in range(start, end + 1))
        else:
            channels.append(str(int(token)))
    return channels


IASI_CHANNELS = expand_channel_spec(IASI_CHANNEL_SPEC)
PLOT_CHANNEL = IASI_CHANNELS[0]


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def retrieve() -> None:
    retriever = EumetsatRetriever()
    try:
        print(
            f"Retrieving {len(IASI_CHANNELS)} IASI channels for "
            f"{DATE:%Y-%m-%d %H:%M}"
        )
        start = datetime.datetime.now(datetime.UTC)
        data = retriever.retrieve(
            source="METOP",
            variables=IASI_CHANNELS,
            product="iasi_radiances",
            dates=[DATE],
            bbox=NORTH_LATITUDE_20_BBOX,
            eumdac_credentials_path=CREDENTIALS_PATH,
            resample=False,
            aggregate_time=False,
            aggregation_window=AGGREGATION_WINDOW,
            test=TEST_MODE,
        )
        print(data)
        if data.sizes.get("time", 0) and PLOT_CHANNEL in data:
            plot_time = data.time.values[0]
            plot_polar(
                data,
                t=plot_time,
                var=PLOT_CHANNEL,
                title=f"EUMETSAT METOP IASI channel {PLOT_CHANNEL}",
                aggregation_window=AGGREGATION_WINDOW,
            )
            save_current_figure(
                PLOT_DIR / f"eumetsat_iasi_channel_{PLOT_CHANNEL}_polar.png"
            )
        end = datetime.datetime.now(datetime.UTC)
        print(f"Retrieval took {end - start}")
    except Exception as exc:
        traceback.print_exc()
        print(f"Failed retrieving IASI channels for {DATE:%Y-%m-%d %H:%M}: {exc}")


if __name__ == "__main__":
    retrieve()
