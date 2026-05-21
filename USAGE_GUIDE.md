# XGB-PClim User Guide

## Predicting Paleoclimate from Paleosol Geochemistry
This guide explains how to test the XGB-PClim models, prepare your own data, and run paleoclimate predictions using either Python or R. The instructions are written for users with little or no programming experience.

# 1. Before You Begin
XGB-PClim-Model includes both Python and R versions. **You only need one version:** Python **or** R (not both).

## Python Users
The Python version can be run using:
* JupyterLab
* Jupyter Notebook
* Command Prompt or Terminal
### Installing Python Requirements
You only need to install the required Python packages one time before running the model.
### Step 1 — Open the XGB-PClim-Model-main Project Folder
Example folder structure:
```text
XGB-PClim-Model-main/
│
├── Python/
├── R/
├── data/
├── models/
└── results/
```
### Step 2 — Open a Terminal Inside the Project Folder
1. Open the `XGB-PClim-Model-main` folder
2. Click inside the folder address bar
3. Type: cmd
4. Press **Enter**
This opens Command Prompt in the correct folder.

### Step 3 — Move into the Python Folder
Now type:
```bash
cd Python
```
This moves into the folder containing the Python script and `requirements.txt`.

### Step 4 — Install Required Packages
```bash
pip install -r requirements.txt
```
If pip or python fails, use:
```bash
py -3 -m pip install -r requirements.txt
```
This installs all required Python packages automatically.

### Requirements
* Python 3.9 or newer

## R Users
The R version can be run using:
* RStudio
* R Console
* Terminal using `Rscript`
  
### Install Required R Package
Open R or RStudio and run:
```
install.packages("xgboost", repos = "https://cloud.r-project.org")
```
### Requirements
* R version 4.0 or newer

# 2. Test the Models with the Included Example File
The repository already includes an example input file in the `data` folder so you can test the models immediately before using your own data.

Example input file:
```text
data/input.csv
```
# Running the R Version
You can run the R version in two ways:

## Option 1 — Run Directly in RStudio (Recommended for Beginners)
### Step 1 - 
Open **RStudio** 
### Step 2
Open the file XGB-PClim_predict.R located inside R folder 
### Step 3
Click the **Run** button at the top of the script editor.
### Step 4
The script will automatically:
* read `data/input.csv`
* save prediction results to:
results/output_R.csv

If successful, you should see:
```text
Predictions saved to: results/output_R.csv
Rows predicted: 41 / 41
```
This means the model ran successfully.

## Option 2 — Run from the Terminal or Command Prompt
### Step 1
Open the **XGB-PClim-Model-main** project folder.
### Step 2
Open Command Prompt or Terminal inside the folder.
### Step 3
Run:
```bash
cd R
Rscript XGB-PClim_predict.R
```
Explanation:
* `cd R` moves into the `R` folder
* `Rscript XGB-PClim_predict.R` runs the prediction script

If Rscript is not recongnized, use:
"C:\Program Files\R\R-4.4.1\bin\Rscript.exe" XGB-PClim_predict.R

The script automatically saves predictions to:
results/output_R.csv

# Running the Python Version
## Option 1 — Run in JupyterLab or Jupyter Notebook (Recommended for Beginners)
### Step 1
Open **JupyterLab**, or **Jupyter Notebook**
### Step 2
Navigate to the XGB-PClim-Model-Main folder and open Python folder
### Step 3
Create a new notebook in that folder
### Step 4
In a cell, add:
%run XGB-PClim_predict.py
### Step 5
Run the cell
### Step 6
The prediction results will be saved to:
results/output.csv

If successful, you should see:
Predictions saved to: results/output.csv
Rows predicted: 41 / 41

## Option 2 — Run from the Terminal or Command Prompt
### Step 1
Open XGB-PClim-Model-Main folder
### Step 2
Open a terminal or command prompt and run:
```bash
cd Python
XGB-PClim_predict.py
```
Explanation:
* `cd Python` moves into the `Python` folder
* `XGB-PClim_predict.py` runs the prediction script

The script saves predictions to:
results/output.csv

# 3. Prepare Your Own Data File

Your CSV file must contain the following ten oxide columns:

```text
CaO, MgO, Na2O, K2O, Al2O3, Fe2O3, SiO2, MnO, TiO2, P2O5
```
Example format:

| Sample_ID | CaO  | MgO  | Na2O | K2O  | Al2O3 | Fe2O3 | SiO2  | MnO  | TiO2 | P2O5 |
| --------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ---- | ---- | ---- |
| SITE_001  | 2.65 | 6.67 | 0.28 | 0.13 | 14.50 | 11.47 | 37.04 | 0.58 | 2.54 | 0.96 |
| SITE_002  | 1.01 | 2.82 | 0.01 | 0.08 | 20.01 | 19.00 | 33.05 | 0.08 | 2.93 | 0.09 |
| SITE_003  | 4.60 | 7.76 | 0.57 | 0.87 | 13.09 | 10.62 | 38.88 | 0.17 | 1.54 | 2.09 |

## Important Rules
* Column names must match exactly, including capitalization (case-sensitive)
* The order of columns does not matter
* Additional columns such as Sample ID, location, or notes are allowed
* Rows with missing oxide values will return blank (`NA`) predictions
* Oxide values should be provided in weight percent (wt %) units

# 4. Running Predictions on Your Own Data
## Recommended Method for Beginners
1. Open the provided `input.csv` file inside the `data` folder
2. Replace the example data with your own samples
3. Save the file
4. Run the script using either the R or Python version
5. Open `output.csv` (Python) or `output_R.csv` (R) stored in the `results` folder

This is the easiest method because the scripts are already configured to automatically read `input.csv` and save results to the `results` folder.

## Optional: Use Custom File Names
Advanced users can specify custom input and output file names.
### Python Example
```bash
python XGB-PClim_predict.py --data my_samples.csv --output my_predictions.csv
```
### R Example
```bash
Rscript XGB-PClim_predict.R --data my_samples.csv --output my_predictions.csv
```
# 5. Understanding the Output
Open the output CSV file stored in the `results` folder using Excel or another spreadsheet program.

The file will contain:
* your original data
* predicted temperature values
* predicted precipitation values

| Column                  | Unit    | Meaning                                           |
| ----------------------- | ------- | -----------------------------------------------   |
| MAT_Best                | °C      | Estimated best mean annual temperature (MAT)      |
| MAT_Min                 | °C      | Estimated minimum mean annual temperature (MAT)   |
| MAT_Max                 | °C      | Estimated maximum mean annual temperature (MAT)   |
| MAP_Best                | mm/year | Estimated best mean annual precipitation (MAP)    |
| MAP_Min                 | mm/year | Estimated minimum mean annual precipitation (MAP) |
| MAP_Max                 | mm/year | Estimated maximum mean annual precipitation (MAP) |

### Notes

* Blank values (`NA`) usually indicate missing oxide data
* The Python and R versions should produce nearly identical results

# 6. Common Problems and Solutions
## File Not Found
Check that the CSV file exists and the file path is correct.

## Missing Oxide Columns
Check that all required oxide column names are present and spelled correctly.
Example mistakes:
* `Cao` instead of `CaO`
* `Al2O3 ` with an extra space

## All Predictions Are Blank
Check for:
* empty cells
* text values such as `"N/A"` or `"--"`
This usually means one or more oxide values are missing.

## Python Package Error
Run:
```bash
pip install -r requirements.txt
```

## R `xgboost` Package Missing
Run:
```r
install.packages("xgboost")
```
# Final Notes
* Check your data carefully before running predictions
* The Python and R versions should produce nearly identical results
