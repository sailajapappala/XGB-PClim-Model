#!/usr/bin/env Rscript
# ==========================================================
# XGB-PClim Climate Prediction -- R Implementation
# ==========================================================
# Predict mean annual temperature (MAT) and mean annual
# precipitation (MAP) from major-oxide geochemistry of
# paleosols using pre-trained XGBoost models.
#
# Usage:
#   Rscript XGB-PClim_predict.R
#   Rscript XGB-PClim_predict.R --data path/to/input.csv --output path/to/output.csv
#
# Recommended project structure:
#
# XGB-PClim/
# ├── R/
# │   ├── XGB-PClim_predict.R
# │   ├── models/
# │   └── data/
# ├── results/
# ==========================================================

# --- Dependencies --------------------------------------------------------

required_packages <- c("xgboost")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}
library(xgboost)
# ---------------------------------------------------------
# Set working directory to script location
# ---------------------------------------------------------
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  setwd(dirname(normalizePath(script_path)))
}
cat("Working directory:\n")
print(getwd())

# --- Configuration -------------------------------------------------------

OXIDE_COLUMNS <- c(
  "CaO", "MgO", "Na2O", "K2O", "Al2O3",
  "Fe2O3", "SiO2", "MnO", "TiO2", "P2O5"
)

# --- Default paths -------------------------------------------------------

default_paths <- list(
  data          = "data/input.csv",
  temp_model    = "models/temperature_xgb_model.json",
  temp_mean     = "models/temp_scaler_mean.csv",
  temp_scale    = "models/temp_scaler_scale.csv",
  prec_model    = "models/precipitation_xgb_model.json",
  prec_mean     = "models/precp_scaler_mean.csv",
  prec_scale    = "models/precp_scaler_scale.csv",
  prec_boxcox   = "models/precp_boxcox_parameters.csv",
  output        = "../results/output_R.csv"
)

# --- Parse command-line arguments -----------------------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  paths <- default_paths
  
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--data" && i < length(args)) {
      paths$data <- args[i + 1]; i <- i + 2
    } else if (args[i] == "--output" && i < length(args)) {
      paths$output <- args[i + 1]; i <- i + 2
    } else {
      i <- i + 1
    }
  }
  paths
}

# --- Helper: inverse Box-Cox transform -----------------------------------
inv_boxcox <- function(y, lambda) {
  if (abs(lambda) < 1e-10) {
    exp(y)
  } else {
    (lambda * y + 1)^(1 / lambda)
  }
}

# --- Main -----------------------------------------------------------------
main <- function() {
  paths <- parse_args()
  # Print detected project root
  cat("Working directory:", getwd(), "\n")
  # Validate inputs
  required <- c("data", "temp_model", "temp_mean", "temp_scale",
                "prec_model", "prec_mean", "prec_scale", "prec_boxcox")
  for (key in required) {
    if (!file.exists(paths[[key]])) {
      stop(sprintf("File not found: %s (%s)", paths[[key]], key))
    }
  }
  
  # --- Load data ----------------------------------------------------------
  data <- read.csv(paths$data, stringsAsFactors = FALSE)
  
  # Check oxide columns exist
  missing_cols <- setdiff(OXIDE_COLUMNS, names(data))
  if (length(missing_cols) > 0) {
    stop(paste("Missing oxide columns:", paste(missing_cols, collapse = ", ")))
  }
  
  # --- Load models and parameters -----------------------------------------
  temp_model <- xgb.load(paths$temp_model)
  prec_model <- xgb.load(paths$prec_model)
  
  temp_mean  <- read.csv(paths$temp_mean)$mean
  temp_scale <- read.csv(paths$temp_scale)$scale
  prec_mean  <- read.csv(paths$prec_mean)$mean
  prec_scale <- read.csv(paths$prec_scale)$scale
  
  prec_bc_params <- read.csv(paths$prec_boxcox)
  lambda <- prec_bc_params$boxcox_lambda
  shift  <- prec_bc_params$shift
  
  # --- Identify complete rows ---------------------------------------------
  mask <- complete.cases(data[, OXIDE_COLUMNS, drop = FALSE])
  
  # --- Temperature prediction ---------------------------------------------
  X_temp <- as.matrix(data[mask, OXIDE_COLUMNS])
  X_temp <- sweep(X_temp, 2, temp_mean, "-")
  X_temp <- sweep(X_temp, 2, temp_scale, "/")
  temp_pred <- predict(temp_model, X_temp)
  
  # --- Precipitation prediction -------------------------------------------
  X_prec <- as.matrix(data[mask, OXIDE_COLUMNS])
  X_prec <- sweep(log1p(X_prec), 2, prec_mean, "-")
  X_prec <- sweep(X_prec, 2, prec_scale, "/")
  prec_pred_bc <- predict(prec_model, X_prec)
  prec_pred <- inv_boxcox(prec_pred_bc, lambda) - shift
  prec_pred <- pmax(prec_pred, 0)
  
  # --- Assemble output ---------------------------------------------------- 
  temp_uncertainty <- 4.1   # example: ±1.8 °C
  prec_uncertainty <- 317    # example: ±95 mm/year
  temp_pred <- round(temp_pred, 1)
  prec_pred <- round(prec_pred, 0)
  
  data$MAT_Best <- NA_real_
  data$MAT_Min  <- NA_real_
  data$MAT_Max  <- NA_real_

  data$MAP_Best <- NA_real_
  data$MAP_Min  <- NA_real_
  data$MAP_Max  <- NA_real_

# --- Fill predictions -------------------------------------------------

  data$MAT_Best[mask] <- temp_pred
  data$MAT_Min[mask]  <- round(temp_pred - temp_uncertainty, 1)
  data$MAT_Max[mask]  <- round(temp_pred + temp_uncertainty, 1)

  data$MAP_Best[mask] <- prec_pred
  data$MAP_Min[mask]  <- round(pmax(prec_pred - prec_uncertainty, 0), 0)
  data$MAP_Max[mask]  <- round(prec_pred + prec_uncertainty, 0)
  
  # --- Save ---------------------------------------------------------------
  output_dir <- dirname(paths$output)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  write.csv(data, paths$output, row.names = FALSE)
  
  
  # --- Summary ----------------------------------------------------------
  
  cat(sprintf("Predictions saved to: %s\n", paths$output))
  cat(sprintf("  Rows predicted: %d / %d\n", sum(mask), nrow(data)))
}

main()
