from pathlib import Path
from typing import Any

import pandas as pd


def get_variables() -> dict[str, Any]:
    """
    Retrieve variable definitions from the local csv file.

    Returns
    -------
    dict
        Dictionary containing variable definitions, with source names as keys
        and variable definitions as values.
    """
    return pd.read_csv(Path(__file__).parent / "variable_metadata.csv")
