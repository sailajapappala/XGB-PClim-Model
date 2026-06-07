#!/usr/bin/env python3
"""
XGB-PClim Climate Prediction -- Python Implementation
=====================================================
Predict mean annual temperature (MAT) and mean annual precipitation (MAP)
from major-oxide geochemistry of paleosols using pre-trained XGBoost models.

Usage
-----
    python XGB-PClim_predict.py
    python XGB-PClim_predict.py --data path/to/input.csv --output path/to/output.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox


# ==========================================================
# Required for unpickling the temperature pipeline
# ==========================================================
def identity(x):
    """Identity function used as a no-op transformer."""
    return x


# ==========================================================
# Required oxide columns
# ==========================================================
OXIDE_COLUMNS = [
    "CaO", "MgO", "Na2O", "K2O", "Al2O3",
    "Fe2O3", "SiO2", "MnO", "TiO2", "P2O5",
]


# ==========================================================
# Default paths
# ==========================================================
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "input.csv"

DEFAULT_PRECIP_MODEL = (
    Path(__file__).parent
    / "models"
    / "precipitation_model_artifacts.joblib"
)

DEFAULT_TEMP_MODEL = (
    Path(__file__).parent
    / "models"
    / "temperature_model_artifacts.joblib"
)

DEFAULT_OUTPUT = (
    Path(__file__).parent.parent
    / "results"
    / "output.csv"
)


# ==========================================================
# Model uncertainties
# ==========================================================
MAT_UNCERTAINTY = 4.1      # °C
MAP_UNCERTAINTY = 322    # mm/year


# ==========================================================
# Load models
# ==========================================================
def load_models(precip_path: Path, temp_path: Path):
    """Load precipitation and temperature model artifacts."""

    precip_art = joblib.load(precip_path)
    temp_art = joblib.load(temp_path)

    return precip_art, temp_art


# ==========================================================
# Prediction function
# ==========================================================
def predict(
    data: pd.DataFrame,
    precip_art: dict,
    temp_art: dict
) -> pd.DataFrame:
    """
    Run MAT and MAP predictions on input data.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing oxide columns.

    precip_art : dict
        Precipitation model artifacts.

    temp_art : dict
        Temperature model artifacts.

    Returns
    -------
    pd.DataFrame
        Input dataframe with prediction columns appended.
    """

    oxide_cols = precip_art["oxide_columns"]

    lam = precip_art["fitted_lambda"]
    shift = precip_art["shift"]

    # ------------------------------------------------------
    # Keep rows with complete oxide data
    # ------------------------------------------------------
    mask = data[oxide_cols].notnull().all(axis=1)

    X = data.loc[mask, oxide_cols]

    # ======================================================
    # MAP prediction
    # ======================================================
    precip_bc = precip_art["pipeline"].predict(X)

    precip = inv_boxcox(precip_bc, lam) - shift

    precip = np.maximum(precip, 0.0)

    # ======================================================
    # MAT prediction
    # ======================================================
    temp = temp_art["pipeline"].predict(X)

    # ======================================================
    # Round predictions
    # ======================================================
    temp = np.round(temp, 1)

    precip = np.round(precip, 0)
    precip = precip.astype(int)
    # ======================================================
    # Calculate uncertainty ranges
    # ======================================================
    temp_min = np.round(temp - MAT_UNCERTAINTY, 1)

    temp_max = np.round(temp + MAT_UNCERTAINTY, 1)

    precip_min = np.round(
        np.maximum(precip - MAP_UNCERTAINTY, 0),
        0
    )

    precip_max = np.round(
        precip + MAP_UNCERTAINTY,
        0
    )

    # ======================================================
    # Create result dataframe
    # ======================================================
    result = data.copy()

    # MAT columns
    result["MAT_Best"] = np.nan
    result["MAT_Min"] = np.nan
    result["MAT_Max"] = np.nan

    # MAP columns
    result["MAP_Best"] = np.nan
    result["MAP_Min"] = np.nan
    result["MAP_Max"] = np.nan

    # ======================================================
    # Store predictions
    # ======================================================
    result.loc[mask, "MAT_Best"] = temp
    result.loc[mask, "MAT_Min"] = temp_min
    result.loc[mask, "MAT_Max"] = temp_max

    result.loc[mask, "MAP_Best"] = precip
    result.loc[mask, "MAP_Min"] = precip_min
    result.loc[mask, "MAP_Max"] = precip_max

    return result


# ==========================================================
# Main
# ==========================================================
def main():

    parser = argparse.ArgumentParser(
        description=(
            "XGB-PClim: Predict paleoclimate from "
            "paleosol geochemistry"
        ),
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to input CSV with oxide columns",
    )

    parser.add_argument(
        "--precip-model",
        type=Path,
        default=DEFAULT_PRECIP_MODEL,
        help="Path to precipitation model .joblib file",
    )

    parser.add_argument(
        "--temp-model",
        type=Path,
        default=DEFAULT_TEMP_MODEL,
        help="Path to temperature model .joblib file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for output CSV",
    )

    args = parser.parse_args()

    # ======================================================
    # Validate input files
    # ======================================================
    for label, path in [
        ("Data", args.data),
        ("Precipitation model", args.precip_model),
        ("Temperature model", args.temp_model),
    ]:

        if not path.exists():

            print(
                f"Error: {label} file not found:\n{path}",
                file=sys.stderr
            )

            sys.exit(1)

    # ======================================================
    # Load input data
    # ======================================================
    data = pd.read_csv(args.data)

    # Remove accidental unnamed columns
    data = data.loc[
        :,
        ~data.columns.str.contains("^Unnamed")
    ]

    # ======================================================
    # Load models
    # ======================================================
    precip_art, temp_art = load_models(
        args.precip_model,
        args.temp_model
    )

    # ======================================================
    # Run predictions
    # ======================================================
    result = predict(
        data,
        precip_art,
        temp_art
    )

    # ======================================================
    # Save output
    # ======================================================
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        args.output,
        index=False
    )

    # ======================================================
    # Console output
    # ======================================================
    print(f"\nPredictions saved to:\n{args.output}")

    print(
        f"Rows predicted: "
        f"{result['MAT_Best'].notna().sum()} / {len(result)}"
    )


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    main()
