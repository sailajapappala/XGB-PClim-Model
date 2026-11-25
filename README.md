# XGB-PClim -- eXtreme Gradient Boosting Paleoclimate model

XGBoost-based prediction of **mean annual temperature (MAT)** and **mean annual precipitation (MAP)** from major-oxide geochemistry of paleosols.

Both a **Python** and an **R** implementation are provided. They produce equivalent results from the same trained models and input data.

**New users:** start with [**USAGE\_GUIDE.md**](USAGE_GUIDE.md) for step-by-step instructions.

## Repository Structure

Folder layout (do not move or rename these folders)

```
XGB-PClim-Model-main/
├── Python/
│   ├── XGB-PClim\_predict.py          # Main prediction script
│   ├── requirements.txt         # Python dependencies
│   ├── data/
│   │   └── input.csv   # Input oxide geochemistry
│   └── models/                   # Do not delete 
│       ├── precipitation\_model\_artifacts.joblib
│       └── temperature\_model\_artifacts.joblib
├── R/
│   ├── XGB-PClim\_predict.R           # Main prediction script
│   ├── data/
│   │   └── input.csv   # Input oxide geochemistry
│   └── models/                        # Do not delete 
│       ├── precipitation\_xgb\_model.json
│       ├── temperature\_xgb\_model.json
│       ├── precp\_boxcox\_parameters.csv
│       ├── precp\_scaler\_mean.csv
│       ├── precp\_scaler\_scale.csv
│       ├── temp\_scaler\_mean.csv
│       └── temp\_scaler\_scale.csv
├── results/                     # Generated predictions (gitignored)
├── README.md
├── USAGE\_GUIDE.md
├── LICENSE
└── .gitignore
```

## Quick Start

### Python

```bash
cd Python
pip install -r requirements.txt
python XGB-PClim\_predict.py

# if python fails, try py -3 instead:
py -3 XGB-PClim\_predict.py

# or with custom paths:
python XGB-PClim\_predict.py --data path/to/data.csv --output results.csv
```

### R

```r
source("R/XGB-PClim\_predict.R")
# or from the command line:
Rscript R/XGB-PClim\_predict.R
```

## Input Format

The input CSV must contain the following oxide columns (weight %):

|Column|Description|
|-|-|
|CaO|Calcium oxide|
|MgO|Magnesium oxide|
|Na2O|Sodium oxide|
|K2O|Potassium oxide|
|Al2O3|Aluminium oxide|
|Fe2O3|Iron(III) oxide|
|SiO2|Silicon dioxide|
|MnO|Manganese oxide|
|TiO2|Titanium dioxide|
|P2O5|Phosphorus pentoxide|

Rows with any missing oxide value are skipped during prediction.

## Output

The script appends seven columns to the input data:

* **MAT\_Best** -- best-estimated MAT prediction in °C
* **MAT\_Min** -- lower bound of the MAT prediction range, calculated as MAT\_Best minus the model uncertainty in °C
* **MAT\_Max** -- upper bound of the MAT prediction range, calculated as MAT\_Best plus the model uncertainty in °C
* **MAP\_Best** -- best-estimated MAP prediction in mm yr⁻¹
* **MAP\_Min** -- lower bound of the MAP prediction range, calculated as MAP\_Best minus the model uncertainty in mm yr⁻¹
* **MAP\_Max** -- upper bound of the MAP prediction range, calculated as MAP\_Best plus the model uncertainty in mm yr⁻¹
* **Outside\_calibration** -- blank when the sample lies inside the calibration range, otherwise the oxides that fall outside it

The MAT and MAP ranges are based on the model prediction uncertainties (RMSPE) of ±4.0 °C for MAT and ±327 mm yr⁻¹ for MAP.

## Calibration Range

**XGB-PClim should not be applied to paleosols whose composition falls outside the calibration range.** The model is calibrated on 1,090 modern soil pedons, and gradient-boosted trees do not extrapolate. Beyond the outermost split in a tree the model returns a constant, so a sample outside these ranges receives a boundary value rather than a prediction, and the reported uncertainty does not apply to it.

Both scripts check every sample against the ranges below and report any that fall outside in the **Outside\_calibration** column, with a warning printed to the console. Those rows still receive numbers, but those numbers should not be interpreted as paleoclimate estimates.

|Oxide|Minimum (wt %)|Maximum (wt %)|
|-|-|-|
|CaO|0.00|38.86|
|MgO|0.01|19.22|
|Na2O|0.00|16.35|
|K2O|0.00|4.42|
|Al2O3|0.77|27.99|
|Fe2O3|0.21|45.82|
|SiO2|3.88|90.36|
|MnO|0.00|1.00|
|TiO2|0.04|12.53|
|P2O5|0.00|18.06|

Before applying XGB-PClim to a paleosol, check it against the full XGB-PClim application criteria in table S3 of the supplementary information, which also covers parent material, degree of pedogenic development, burial alteration and the climate range the model can resolve.

## Method Summary

1. **Temperature model**: Standard-scaled oxides fed into an XGBoost regressor.
2. **Precipitation model**: log1p-transformed, standard-scaled oxides fed into an XGBoost regressor. Predictions are inverse Box-Cox transformed to recover MAP in mm/year.

## Requirements

* **Python 3.12 or newer** with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`. The models were trained on Python 3.12.7, and the pinned versions in `requirements.txt` cannot be installed on Python 3.11 or earlier.
* **R >= 4.0** with the `xgboost` package

## Usage Guide

See [**USAGE\_GUIDE.md**](USAGE_GUIDE.md) for detailed instructions on:

* Testing the models with bundled data
* Preparing your own input CSV
* Running predictions on new files
* Interpreting output and troubleshooting

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use XGB-PClim model in your research, please cite:

> \\\*Pappala, V.S., Stinchcomb, G.E., Lukens, W.E., Nordt, L.C. Machine Learning Climate Prediction from Modern Soil Geochemistry: Implications for Paleosols. American Journal of Science.\\\*

