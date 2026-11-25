#!/usr/bin/env python3
"""
XGB-PClim Climate Prediction -- Python Implementation
=====================================================
Predict mean annual temperature (MAT) and mean annual precipitation (MAP)
from major-oxide geochemistry of paleosols using pre-trained XGBoost models.

XGB-PClim is calibrated on 1,090 modern soil pedons. Predictions for
compositions outside that calibration range are not recommended. The
script reports any sample that falls outside it. See the XGB-PClim
application criteria in table S3 of the supplementary information.

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
from scipy.special import boxcox, inv_boxcox
from scipy.stats import boxcox as boxcox_fit
from sklearn.base import BaseEstimator, TransformerMixin


# ==========================================================
# Required for unpickling the saved models
# ==========================================================
# A pickle stores functions and classes by name, so these definitions
# must exist before the artifacts will load.
def identity(x):
    """Identity function used as a no-op transformer."""
    return x


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Target transform used by the precipitation model."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        _, self.lambda_ = boxcox_fit(np.asarray(X, dtype=float).ravel())
        return self

    def transform(self, X):
        return boxcox(np.asarray(X, dtype=float), self.lambda_)

    def inverse_transform(self, X):
        return inv_boxcox(np.asarray(X, dtype=float), self.lambda_)


# The artifacts were pickled from __main__, so register these there too.
# Without this the models load when the script is run directly but not
# when it is imported as a module.
import __main__ as _main                                   # noqa: E402
for _obj in (identity, BoxCoxTransformer):
    if not hasattr(_main, _obj.__name__):
        setattr(_main, _obj.__name__, _obj)


# ==========================================================
# Required oxide columns
# ==========================================================
OXIDE_COLUMNS = [
    "CaO", "MgO", "Na2O", "K2O", "Al2O3",
    "Fe2O3", "SiO2", "MnO", "TiO2", "P2O5",
]


# ==========================================================
# Calibration domain
# ==========================================================
# Oxide ranges, in wt%, of the 1,090 modern pedons XGB-PClim was
# calibrated on. Predictions for compositions outside these ranges are
# not recommended, because gradient-boosted trees return a constant
# value beyond the outermost split, so the result is a boundary value
# rather than a prediction. See the XGB-PClim application criteria in
# table S3 of the supplementary information.
CALIBRATION_RANGE = {
    "CaO":   (0.00, 38.86),
    "MgO":   (0.01, 19.22),
    "Na2O":  (0.00, 16.35),
    "K2O":   (0.00,  4.42),
    "Al2O3": (0.77, 27.99),
    "Fe2O3": (0.21, 45.82),
    "SiO2":  (3.88, 90.36),
    "MnO":   (0.00,  1.00),
    "TiO2":  (0.04, 12.53),
    "P2O5":  (0.00, 18.06),
}


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

# Used if the default temperature artifact is not present.
ALT_TEMP_MODEL = (
    Path(__file__).parent
    / "models"
    / "temperature_final_pipeline.joblib"
)

DEFAULT_OUTPUT = (
    Path(__file__).parent.parent
    / "results"
    / "output.csv"
)


# ==========================================================
# Model uncertainties
# ==========================================================
MAT_UNCERTAINTY = 4.0      # °C
MAP_UNCERTAINTY = 327      # mm/year


# ==========================================================
# Artifact helpers
# ==========================================================
def get_estimator(art):
    """Return the fitted estimator from an artifact.

    Artifacts key it as 'model' or 'pipeline', or may be the estimator
    itself rather than a dict.
    """
    if not isinstance(art, dict):
        return art
    for key in ("model", "pipeline", "estimator"):
        if key in art:
            return art[key]
    raise KeyError(
        "no estimator found in artifact, keys are %s" % list(art.keys()))


def get_oxide_columns(art):
    """Return the oxide order the model expects."""
    if isinstance(art, dict) and "oxide_columns" in art:
        return list(art["oxide_columns"])
    return list(OXIDE_COLUMNS)


def get_boxcox(art):
    """Return (lambda, shift), or (None, 0.0) if the artifact has none."""
    if not isinstance(art, dict):
        return None, 0.0
    lam = next((art[k] for k in ("fitted_lambda", "boxcox_lambda", "lambda_")
                if k in art), None)
    shift = next((art[k] for k in ("shift", "boxcox_shift") if k in art), 0.0)
    return (None if lam is None else float(lam)), float(shift)


def load_models(precip_path: Path, temp_path: Path):
    """Load precipitation and temperature model artifacts."""
    return joblib.load(precip_path), joblib.load(temp_path)


# ==========================================================
# Calibration check
# ==========================================================
def check_calibration_range(X):
    """Identify samples with any oxide outside the calibration domain.

    Works from the oxide columns alone. No sample-name column is
    required or looked for, so any input file can be checked.

    Returns a Series aligned to X listing the offending oxides for each
    row, empty where the composition is inside the domain.
    """
    offending = []
    for _, row in X.iterrows():
        offending.append(", ".join(
            col for col in X.columns
            if col in CALIBRATION_RANGE
            and not CALIBRATION_RANGE[col][0] <= row[col]
            <= CALIBRATION_RANGE[col][1]
        ))
    flags = pd.Series(offending, index=X.index)

    n_bad = int((flags != "").sum())
    if n_bad:
        print("\nWARNING: %d of %d samples fall outside the XGB-PClim "
              "calibration domain." % (n_bad, len(X)))
        for idx, bad in flags.items():
            if not bad:
                continue
            # reported by input row number, counting the header as row 1,
            # so it can be found in the file without relying on an ID column
            print("  row %d: %s" % (int(idx) + 2, bad))
        print("Applying XGB-PClim to these compositions is not recommended. "
              "See the application criteria in table S3 of the "
              "supplementary information.\n")
    else:
        print("\nAll %d samples fall inside the calibration domain.\n"
              % len(X))
    return flags


# ==========================================================
# Prediction function
# ==========================================================
def predict(
    data: pd.DataFrame,
    precip_art,
    temp_art
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

    # Each model is given its own oxide order, taken from the artifact
    # it was trained with. Columns are selected by name, so the order
    # they appear in the input file never matters, and the two models
    # stay correct even if their training orders differ from each other.
    precip_cols = get_oxide_columns(precip_art)
    temp_cols = get_oxide_columns(temp_art)
    lam, shift = get_boxcox(precip_art)

    required = list(dict.fromkeys(precip_cols + temp_cols))
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError("input data is missing the oxide columns %s" % missing)

    # ------------------------------------------------------
    # Keep rows with complete oxide data
    # ------------------------------------------------------
    mask = data[required].notnull().all(axis=1)

    X_precip = data.loc[mask, precip_cols]
    X_temp = data.loc[mask, temp_cols]

    # ------------------------------------------------------
    # Flag anything outside the calibration domain
    # ------------------------------------------------------
    flags = check_calibration_range(X_precip)

    # ======================================================
    # MAP prediction
    # ======================================================
    precip_model = get_estimator(precip_art)
    raw = np.asarray(precip_model.predict(X_precip), dtype=float)

    if hasattr(precip_model, "regressor_"):
        # A TransformedTargetRegressor inverts the Box-Cox transform
        # itself, so predict already returns mm/year.
        precip = raw
    elif lam is not None:
        precip = inv_boxcox(raw, lam) - shift
    else:
        precip = raw

    precip = np.maximum(precip, 0.0)

    # ======================================================
    # MAT prediction
    # ======================================================
    temp = np.asarray(get_estimator(temp_art).predict(X_temp), dtype=float)

    # ======================================================
    # Round predictions
    # ======================================================
    temp = np.round(temp, 1)

    precip = np.round(precip, 0)

    # A non-finite prediction cast to int becomes a nonsense integer
    # rather than raising, so report it and carry it through as missing.
    bad = ~np.isfinite(precip)
    if bad.any():
        print("Warning: %d precipitation predictions are not finite and "
              "are reported as missing." % int(bad.sum()))
        precip = np.where(bad, np.nan, precip)

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

    # Calibration flag, empty where the composition is inside the domain
    result["Outside_calibration"] = ""

    # ======================================================
    # Store predictions
    # ======================================================
    result.loc[mask, "MAT_Best"] = temp
    result.loc[mask, "MAT_Min"] = temp_min
    result.loc[mask, "MAT_Max"] = temp_max

    result.loc[mask, "MAP_Best"] = precip
    result.loc[mask, "MAP_Min"] = precip_min
    result.loc[mask, "MAP_Max"] = precip_max

    result.loc[mask, "Outside_calibration"] = flags

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
        default=None,
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
    # Resolve the temperature model
    # ======================================================
    if args.temp_model is None:
        if DEFAULT_TEMP_MODEL.exists():
            args.temp_model = DEFAULT_TEMP_MODEL
        elif ALT_TEMP_MODEL.exists():
            args.temp_model = ALT_TEMP_MODEL
            print("Note: using %s" % ALT_TEMP_MODEL.name)
        else:
            args.temp_model = DEFAULT_TEMP_MODEL

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

    # Tolerate stray spaces around column headings, a common result of
    # editing the CSV in a spreadsheet. Order and extra columns are
    # already handled, because oxides are selected by name.
    data.columns = [str(c).strip() for c in data.columns]

    # Remove accidental unnamed columns, which spreadsheets create by
    # saving trailing commas on every line. Only where the column is
    # also empty, so a real column is never removed.
    _unnamed = data.columns.str.match(r"^Unnamed")
    _empty = data.isna().all(axis=0).to_numpy()
    _drop = _unnamed & _empty
    if _drop.any():
        print("Note: dropped %d empty unnamed column(s) from the input "
              "file." % int(_drop.sum()))
        data = data.loc[:, ~_drop]

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

    n_flagged = int((result["Outside_calibration"] != "").sum())
    if n_flagged:
        print(f"Samples outside the calibration domain: {n_flagged} "
              f"(see the Outside_calibration column)")


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    main()
