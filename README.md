# XGB-PClim -- eXtreme Gradient Boosting Paleoclimate model

XGBoost-based prediction of **mean annual temperature (MAT)** and **mean annual precipitation (MAP)** from major-oxide geochemistry of paleosols.

Both a **Python** and an **R** implementation are provided. They produce equivalent results from the same trained models and input data.

**New users:** start with [**USAGE_GUIDE.md**](USAGE_GUIDE.md) for step-by-step instructions. 

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
├── USAGE_GUIDE.md
├── LICENSE
└── .gitignore
```

## Quick Start

### Python

```bash
cd Python
pip install -r requirements.txt
python XGB-PClim\_predict.py
If `python` fails, try `py -3` instead

# or with custom paths:
python XGB-PClim\_predict.py --data path/to/data.csv --output results.csv
```

### R

```r
source("R/XGB-PClim\_predict.R")
# or from the command line:
Rscript R/XGB-PClim_predict.R
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

The script appends six columns to the input data:

* **MAT\_Best** -- estimated MAT in degrees Celsius
* **MAT\_Min** -- estimated MAT in degrees Celsius
* **MAT\_Max** -- estimated MAT in degrees Celsius
* **MAP\_Best** -- estimated MAP in mm/year
* **MAP\_Min** -- estimated MAP in mm/year
* **MAP\_Max** -- estimated MAP in mm/year

## Method Summary

1. **Temperature model**: Standard-scaled oxides fed into an XGBoost regressor.
2. **Precipitation model**: log1p-transformed, standard-scaled oxides fed into an XGBoost regressor. Predictions are inverse Box-Cox transformed to recover MAP in mm/year.

## Requirements

* **Python >= 3.9** with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`
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

> \*Pappala, V.S., Stinchcomb, G.E., Lukens, W.E., Nordt, L.C., 2026. Machine Learning Climate Prediction from Modern Soil Geochemistry: Implications for Paleosols. American Journal of Science (in submission).\*

