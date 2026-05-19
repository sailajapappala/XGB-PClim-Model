# PPM2 Usage Guide -- Testing Models and Predicting on New Data

This guide walks through how to **verify that the models work correctly** using the
bundled test data and how to **run predictions on your own new CSV file**.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Test the Models with Bundled Data](#2-test-the-models-with-bundled-data)
3. [Prepare Your Own Input File](#3-prepare-your-own-input-file)
4. [Run Predictions on a New File](#4-run-predictions-on-a-new-file)
5. [Interpret the Output](#5-interpret-the-output)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### Python

```bash
cd Python
pip install -r requirements.txt
```

Requires **Python >= 3.9** and the packages listed in `requirements.txt`
(`pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`).

### R

```bash
# The script auto-installs xgboost if missing, but you can install manually:
Rscript -e 'install.packages("xgboost", repos="https://cloud.r-project.org")'
```

Requires **R >= 4.0** with the `xgboost` package.

---

## 2. Test the Models with Bundled Data

The repository ships with `retallack_data.csv` so you can verify the models
run correctly right away.

### Python

```bash
cd Python
python ppm2_predict.py
```

Expected output:

```
Predictions saved to: ../results/retallack_climate_predictions.csv
  Rows predicted: <N> / <total>
```

### R

```bash
cd R
Rscript ppm2_predict.R
```

Expected output:

```
Predictions saved to: ../results/retallack_climate_predictions_R.csv
  Rows predicted: <N> / <total>
```

### Verify the results

Open the output CSV and confirm:

- Two new columns exist: `Predicted_Temperature` and `Predicted_Precipitation`.
- Temperature values are in a plausible range (roughly -10 to 30 degrees C).
- Precipitation values are non-negative (roughly 0 to 5000 mm/year).
- Rows with missing oxide data have `NA` / blank predictions.

### Quick sanity check (Python)

```python
import pandas as pd

df = pd.read_csv("../results/retallack_climate_predictions.csv")

print(df[["Predicted_Temperature", "Predicted_Precipitation"]].describe())

# Check no negative precipitation
assert (df["Predicted_Precipitation"].dropna() >= 0).all(), "Negative precipitation found!"

# Check temperature is in a reasonable range
temp = df["Predicted_Temperature"].dropna()
assert temp.between(-15, 35).all(), "Temperature out of expected range!"

print("All sanity checks passed.")
```

### Quick sanity check (R)

```r
df <- read.csv("../results/retallack_climate_predictions_R.csv")

summary(df[, c("Predicted_Temperature", "Predicted_Precipitation")])

# Check no negative precipitation
stopifnot(all(df$Predicted_Precipitation[!is.na(df$Predicted_Precipitation)] >= 0))

# Check temperature is in a reasonable range
temp <- df$Predicted_Temperature[!is.na(df$Predicted_Temperature)]
stopifnot(all(temp > -15 & temp < 35))

cat("All sanity checks passed.\n")
```

---

## 3. Prepare Your Own Input File

Create a CSV file with **at minimum** these 10 oxide columns (values in weight %):

```
CaO,MgO,Na2O,K2O,Al2O3,Fe2O3,SiO2,MnO,TiO2,P2O5
```

### Example: `my_samples.csv`

```csv
Sample_ID,CaO,MgO,Na2O,K2O,Al2O3,Fe2O3,SiO2,MnO,TiO2,P2O5
SITE_001,2.65,6.67,0.28,0.13,14.50,11.47,37.04,0.58,2.54,0.96
SITE_002,1.01,2.82,0.01,0.08,20.01,19.00,33.05,0.08,2.93,0.09
SITE_003,4.60,7.76,0.57,0.87,13.09,10.62,38.88,0.17,1.54,2.09
```

### Rules

| Rule | Detail |
|------|--------|
| Column names | Must match **exactly** (case-sensitive): `CaO`, `MgO`, `Na2O`, `K2O`, `Al2O3`, `Fe2O3`, `SiO2`, `MnO`, `TiO2`, `P2O5` |
| Column order | Does **not** matter -- the scripts select columns by name |
| Extra columns | Allowed -- they are passed through to the output unchanged |
| Missing values | Rows where **any** oxide is blank / `NA` will be skipped (predictions set to `NA`) |
| Units | Weight percent (wt %). Do **not** normalize to sum to 100 -- use raw analytical values |
| File encoding | UTF-8 recommended |

---

## 4. Run Predictions on a New File

### Python

```bash
cd Python
python ppm2_predict.py --data /path/to/my_samples.csv --output /path/to/my_predictions.csv
```

All available flags:

| Flag              | Description                                       | Default                                        |
|-------------------|---------------------------------------------------|------------------------------------------------|
| `--data`          | Path to input CSV                                 | `data/retallack_data.csv`                      |
| `--output`        | Path for output CSV                               | `../results/retallack_climate_predictions.csv`  |
| `--precip-model`  | Path to precipitation `.joblib` model              | `models/precipitation_model_artifacts.joblib`   |
| `--temp-model`    | Path to temperature `.joblib` model                | `models/temperature_model_artifacts.joblib`     |

### R

```bash
cd R
Rscript ppm2_predict.R --data /path/to/my_samples.csv --output /path/to/my_predictions.csv
```

All available flags:

| Flag       | Description        | Default                                           |
|------------|--------------------|---------------------------------------------------|
| `--data`   | Path to input CSV  | `data/retallack_data.csv`                         |
| `--output` | Path for output CSV| `../results/retallack_climate_predictions_R.csv`   |

---

## 5. Interpret the Output

The output CSV is a copy of your input with two appended columns:

| Column                     | Unit       | Description                                |
|----------------------------|------------|--------------------------------------------|
| `Predicted_Temperature`    | degrees C  | Estimated mean annual temperature (MAT)    |
| `Predicted_Precipitation`  | mm/year    | Estimated mean annual precipitation (MAP)  |

### Example output

```csv
Sample_ID,CaO,MgO,Na2O,K2O,Al2O3,Fe2O3,SiO2,MnO,TiO2,P2O5,Predicted_Temperature,Predicted_Precipitation
SITE_001,2.65,6.67,0.28,0.13,14.50,11.47,37.04,0.58,2.54,0.96,14.32,1245.67
SITE_002,1.01,2.82,0.01,0.08,20.01,19.00,33.05,0.08,2.93,0.09,22.18,2034.51
SITE_003,4.60,7.76,0.57,0.87,13.09,10.62,38.88,0.17,1.54,2.09,12.05,987.23
```

### Notes on predictions

- **Temperature** can be negative (cold climates) or above 25 C (tropical climates).
- **Precipitation** is clamped to a minimum of 0. Very low values (< 100 mm/year) indicate arid conditions.
- **NA values** in output mean that row had one or more missing oxide inputs.
- Python and R predictions may differ by tiny floating-point amounts (< 0.01) due to implementation differences.

---

## 6. Troubleshooting

### "File not found" error

```
Error: Data file not found: /path/to/file.csv
```

Check that the path you passed to `--data` exists and is readable. Use absolute paths
to avoid ambiguity.

### "Missing oxide columns" error

```
Missing oxide columns: CaO, MgO
```

Your CSV is missing one or more required columns. Check that column headers match
exactly (case-sensitive). Common mistakes:
- `Cao` instead of `CaO`
- `Al2O3 ` (trailing space)
- Using semicolons instead of commas as the CSV delimiter

### All predictions are NA

This means every row had at least one missing oxide value. Check your data for:
- Empty cells or cells containing text like `"N/A"`, `"n.d."`, `"--"` (these are not
  recognized as numeric).
- Columns that are entirely blank.

### Negative or unrealistic predictions

The models are trained on paleosol data within typical geochemical ranges. Predictions
may be unreliable for:
- Samples that are not paleosols (e.g., fresh rock, sediment, volcanic ash).
- Oxide compositions far outside the training range.
- Samples where oxides do not approximately sum to ~90-100%.

### Python: `ModuleNotFoundError`

```bash
pip install -r requirements.txt
```

### R: `xgboost` package not found

```r
install.packages("xgboost", repos = "https://cloud.r-project.org")
```

### Cross-checking Python vs R results

Both implementations should produce very similar results. To compare:

```bash
# Python
cd Python && python ppm2_predict.py --data ../R/data/retallack_data.csv --output /tmp/py_results.csv

# R
cd R && Rscript ppm2_predict.R --data ../Python/data/retallack_data.csv --output /tmp/r_results.csv
```

Then compare:

```python
import pandas as pd

py = pd.read_csv("/tmp/py_results.csv")
r  = pd.read_csv("/tmp/r_results.csv")

print("Max temperature difference:", abs(py.Predicted_Temperature - r.Predicted_Temperature).max())
print("Max precipitation difference:", abs(py.Predicted_Precipitation - r.Predicted_Precipitation).max())
```

Differences should be negligible (< 0.1 for temperature, < 1.0 for precipitation).
