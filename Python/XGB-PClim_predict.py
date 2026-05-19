#!/usr/bin/env python3
"""
PPM2 Climate Prediction -- Python Implementation
=================================================
Predict mean annual temperature (MAT) and mean annual precipitation (MAP)
from major-oxide geochemistry of paleosols using pre-trained XGBoost models.

Usage
-----
    python ppm2_predict.py
    python ppm2_predict.py --data path/to/input.csv --output path/to/output.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox


# Required for unpickling the temperature pipeline (custom identity transformer)
def identity(x):
    """Identity function used as a no-op transformer in the temperature pipeline."""
    return x


OXIDE_COLUMNS = [
    "CaO", "MgO", "Na2O", "K2O", "Al2O3",
    "Fe2O3", "SiO2", "MnO", "TiO2", "P2O5",
]

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "retallack_data.csv"
DEFAULT_PRECIP_MODEL = Path(__file__).parent / "models" / "precipitation_model_artifacts.joblib"
DEFAULT_TEMP_MODEL = Path(__file__).parent / "models" / "temperature_model_artifacts.joblib"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "results" / "retallack_climate_predictions.csv"


def load_models(precip_path: Path, temp_path: Path):
    """Load precipitation and temperature model artifacts."""
    precip_art = joblib.load(precip_path)
    temp_art = joblib.load(temp_path)
    return precip_art, temp_art


def predict(data: pd.DataFrame, precip_art: dict, temp_art: dict) -> pd.DataFrame:
    """
    Run temperature and precipitation predictions on input data.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain all columns listed in OXIDE_COLUMNS.
    precip_art : dict
        Precipitation model artifacts (pipeline, lambda, shift, oxide_columns).
    temp_art : dict
        Temperature model artifacts (pipeline).

    Returns
    -------
    pd.DataFrame
        Copy of input data with Predicted_Temperature and Predicted_Precipitation
        columns appended.
    """
    oxide_cols = precip_art["oxide_columns"]
    lam = precip_art["fitted_lambda"]
    shift = precip_art["shift"]

    mask = data[oxide_cols].notnull().all(axis=1)
    X = data.loc[mask, oxide_cols]

    # Precipitation: predict in Box-Cox space, then invert
    precip_bc = precip_art["pipeline"].predict(X)
    precip = inv_boxcox(precip_bc, lam) - shift
    precip = np.maximum(precip, 0.0)

    # Temperature: direct prediction
    temp = temp_art["pipeline"].predict(X)

    result = data.copy()
    result["Predicted_Temperature"] = np.nan
    result["Predicted_Precipitation"] = np.nan
    result.loc[mask, "Predicted_Temperature"] = temp
    result.loc[mask, "Predicted_Precipitation"] = precip

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PPM2: Predict paleoclimate from paleosol geochemistry",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH,
        help="Path to input CSV with oxide columns (default: data/retallack_data.csv)",
    )
    parser.add_argument(
        "--precip-model", type=Path, default=DEFAULT_PRECIP_MODEL,
        help="Path to precipitation model .joblib file",
    )
    parser.add_argument(
        "--temp-model", type=Path, default=DEFAULT_TEMP_MODEL,
        help="Path to temperature model .joblib file",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path for output CSV (default: results/retallack_climate_predictions.csv)",
    )
    args = parser.parse_args()

    # Validate inputs exist
    for label, path in [("Data", args.data), ("Precip model", args.precip_model), ("Temp model", args.temp_model)]:
        if not path.exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Load
    data = pd.read_csv(args.data)
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    precip_art, temp_art = load_models(args.precip_model, args.temp_model)

    # Predict
    result = predict(data, precip_art, temp_art)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")
    print(f"  Rows predicted: {result['Predicted_Temperature'].notna().sum()} / {len(result)}")


if __name__ == "__main__":
    main()
