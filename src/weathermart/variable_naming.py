from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_frost_variables() -> dict[str, list[str]]:
    import json

    import requests

    client_id = json.load(
        open(Path(__file__).parent.parent.parent / ".frost_credentials.json")
    )["client_id"]
    r = requests.get(
        "https://frost.met.no/elements/v0.jsonld?fields=id,oldElementCodes,category,name,description,unit,sensorLevelType,sensorLevelUnit,sensorLevelDefaultValue,sensorLevelValues,cmMethod,cmMethodDescription,cmInnerMethod,cmInnerMethodDescription,status&lang=en-US",
        auth=(client_id, ""),
    )
    variable_metadata = pd.DataFrame(r.json()["data"]).rename(
        columns={
            "id": "short_name",
            "category": "group",
            "name": "cf_name",
            "oldConvention": "id",
        }
    )
    groups = [
        "Snow and state of the ground",
        "Precipitation",
        "Atmospheric pressure",
        "Temperature",
        "Metadata",
        "Operation",
        "Wind",
        "Clouds",
        "Visibility",
        "Vegetation",
        "Humidity",
        "Sea ice",
        "Sea water",
        "Icing",
        "Water level",
        "Radiation",
        "Ocean waves",
        "Sunshine",
        "Weather",
        "Evaporation",
    ]
    corresponding_metadata_group = [
        "snow",
        "precipitation",
        "pressure",
        "temperature",
        "static",
        "static",
        "wind",
        "cloud",
        "cloud",
        "static",
        "humidity",
        "sea_ice",
        "sea_water",
        "icing",
        "height",
        "radiation",
        "ocean_wave",
        "sunshine",
        "weather_type",
        "humidity",
    ]
    variable_metadata["group"] = variable_metadata["group"].apply(
        lambda x: corresponding_metadata_group[groups.index(x)]
        if x in groups
        else "other"
    )
    variable_metadata.id = variable_metadata.id.apply(
        lambda x: x["elementCodes"] if isinstance(x, dict) else x
    )
    variable_metadata = variable_metadata.assign(time_granularity="T")
    variable_metadata.calculationMethod = variable_metadata.calculationMethod.apply(
        lambda x: x["methodDescription"]
        if isinstance(x, dict) and "methodDescription" in x
        else ""
    )
    variable_metadata.description = (
        variable_metadata.description + variable_metadata.calculationMethod
    )

    def simplify_calculation_method(value: str) -> str:
        text = str(value).lower().strip()
        if text == "":
            return "instant"
        if "standard deviation" in text or "sd" in text:
            return "sd"
        if "maximum value" in text or "maximum wind speed" in text:
            return "max"
        if "minimum value" in text:
            return "min"
        if (
            "mean value" in text
            or "mean values" in text
            or "arithmetic mean" in text
            or "køppen formula" in text
        ):
            return "mean"
        if (
            "arithmetic sum" in text
            or "sum of data" in text
            or "sum of datavalues" in text
            or "sum values" in text
            or "sum over" in text
            or "sum should" in text
        ):
            return "sum"

        if (
            "accumulated" in text
            or "count of days" in text
            or "since last observation" in text
            or "since 1. january" in text
            or "deviation above" in text
            or "deviation below" in text
        ):
            return "accum"

        return "instant"

    def extract_levels(sensor_levels):
        if not isinstance(sensor_levels, dict):
            return [{"level_type": "surface", "level": np.nan}]
        level_type = sensor_levels.get("levelType")
        unit = sensor_levels.get("unit")

        if level_type is None or unit is None:
            return [{"level_type": "surface", "level": np.nan}]

        full_level_type = f"{level_type}_{unit}"
        values = sensor_levels.get("values")
        if values is None or len(values) == 0:
            default_value = sensor_levels.get("defaultValue", np.nan)
            values = [default_value]
        return [
            {
                "level_type": full_level_type,
                "level": value,
            }
            for value in values
        ]

    def expand_sensor_levels(df, column="sensorLevels"):
        out = df.copy()
        out["_levels"] = out[column].apply(extract_levels)
        out = out.explode("_levels", ignore_index=True)
        levels = pd.json_normalize(out["_levels"])
        out = out.drop(columns="_levels")
        out["level_type"] = levels["level_type"].to_numpy()
        out["level"] = levels["level"].to_numpy()
        return out

    variable_metadata["step_type"] = variable_metadata["calculationMethod"].apply(
        simplify_calculation_method
    )
    variable_metadata = variable_metadata.assign(source="FROST").assign(
        data_type=lambda x: x.dtypes
    )
    variable_metadata = expand_sensor_levels(variable_metadata)
    variable_metadata = variable_metadata[
        [
            "short_name",
            "id",
            "cf_name",
            "description",
            "group",
            "time_granularity",
            "step_type",
            "data_type",
            "unit",
            "source",
            "level_type",
            "level",
        ]
    ]
    vm = pd.read_csv(Path(__file__).parent / "variable_metadata.csv")
    vm = vm[~(vm.source == "FROST")]
    vm = pd.concat([vm, variable_metadata], ignore_index=True).drop_duplicates(
        subset=["short_name"], keep="first"
    )
    vm.to_csv(Path(__file__).parent / "variable_metadata.csv", index=False)
    return variable_metadata


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


if __name__ == "__main__":
    get_frost_variables()
