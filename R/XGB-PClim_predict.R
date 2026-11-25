#!/usr/bin/env Rscript
# ==========================================================
# XGB-PClim Climate Prediction -- R Implementation
# ==========================================================
# Predict mean annual temperature (MAT) and mean annual
# precipitation (MAP) from major-oxide geochemistry of
# paleosols using pre-trained XGBoost models.
#
# XGB-PClim is calibrated on 1,090 modern soil pedons.
# Predictions for compositions outside that calibration range
# are not recommended. The script reports any sample that
# falls outside it. See the XGB-PClim application criteria in
# table S3 of the supplementary information.
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

# --- Model uncertainties -------------------------------------------------

TEMP_UNCERTAINTY <- 4.0    # degrees C
PREC_UNCERTAINTY <- 327    # mm/year

# --- Calibration domain --------------------------------------------------
# Oxide ranges, in wt%, of the 1,090 modern pedons XGB-PClim was
# calibrated on. Predictions for compositions outside these ranges are
# not recommended, because gradient-boosted trees return a constant
# value beyond the outermost split, so the result is a boundary value
# rather than a prediction. See the XGB-PClim application criteria in
# table S3 of the supplementary information.
#
# This table is looked up by oxide name, so its row order does not
# matter. The scaler CSVs are read positionally and must stay in
# OXIDE_COLUMNS order.

CALIBRATION_RANGE <- data.frame(
  oxide = c("CaO", "MgO", "Na2O", "K2O", "Al2O3",
            "Fe2O3", "SiO2", "MnO", "TiO2", "P2O5"),
  min   = c(0.00,  0.01,  0.00,  0.00,  0.77,
            0.21,  3.88,  0.00,  0.04,  0.00),
  max   = c(38.86, 19.22, 16.35, 4.42,  27.99,
            45.82, 90.36, 1.00,  12.53, 18.06),
  stringsAsFactors = FALSE
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

# --- Helper: calibration domain check ------------------------------------
# Works from the oxide columns alone. No sample-name column is required
# or looked for, so any input file can be checked.
# row_numbers gives the position of each row in the input file, so the
# warning stays correct when rows with missing oxides have been dropped.
check_calibration_range <- function(X, row_numbers = seq_len(nrow(X))) {
  flags <- character(nrow(X))

  for (i in seq_len(nrow(X))) {
    bad <- character(0)
    for (j in seq_len(nrow(CALIBRATION_RANGE))) {
      ox <- CALIBRATION_RANGE$oxide[j]
      v  <- X[i, ox]
      if (is.na(v) || v < CALIBRATION_RANGE$min[j] ||
          v > CALIBRATION_RANGE$max[j]) {
        bad <- c(bad, ox)
      }
    }
    flags[i] <- paste(bad, collapse = ", ")
  }

  n_bad <- sum(flags != "")
  if (n_bad > 0) {
    cat(sprintf(
      "\nWARNING: %d of %d samples fall outside the XGB-PClim calibration domain.\n",
      n_bad, nrow(X)))
    for (i in which(flags != "")) {
      # reported by input row number, counting the header as row 1, so it
      # can be found in the file without relying on an ID column
      cat(sprintf("  row %d: %s\n", row_numbers[i] + 1L, flags[i]))
    }
    cat(paste0("Applying XGB-PClim to these compositions is not recommended. ",
               "See the application criteria in table S3 of the ",
               "supplementary information.\n\n"))
  } else {
    cat(sprintf("\nAll %d samples fall inside the calibration domain.\n\n",
                nrow(X)))
  }
  flags
}

# --- Main -----------------------------------------------------------------
main <- function() {
  paths <- parse_args()

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

  # Tolerate stray spaces around column headings, a common result of
  # editing the CSV in a spreadsheet. read.csv also turns a heading
  # such as "Al2O3 " into "Al2O3.", so strip that too.
  names(data) <- trimws(gsub("\\.+$", "", names(data)))

  # Spreadsheets often save trailing commas on every line, which read.csv
  # names X, X.1, X.2 and so on. Drop those, but only where the column is
  # also empty, so a real column is never removed.
  auto <- grepl("^X(\\.[0-9]+)?$", names(data))
  empty <- vapply(data,
                  function(col) all(is.na(col) |
                                    trimws(as.character(col)) == ""),
                  logical(1))
  drop_cols <- auto & empty
  if (any(drop_cols)) {
    cat(sprintf(paste0("Note: dropped %d empty unnamed column(s) from the ",
                       "input file.\n"), sum(drop_cols)))
    data <- data[, !drop_cols, drop = FALSE]
  }

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

  # The parameter file has been written with the lambda column named
  # either boxcox_lambda or lambda, depending on which script produced
  # it, so accept both rather than failing on a NULL.
  prec_bc_params <- read.csv(paths$prec_boxcox)

  lambda <- prec_bc_params$boxcox_lambda
  if (is.null(lambda)) lambda <- prec_bc_params$lambda
  if (is.null(lambda)) {
    stop(sprintf("no lambda column in %s. Columns found: %s",
                 paths$prec_boxcox,
                 paste(names(prec_bc_params), collapse = ", ")))
  }

  shift <- prec_bc_params$shift
  if (is.null(shift)) shift <- prec_bc_params$boxcox_shift
  if (is.null(shift)) shift <- 0

  lambda <- as.numeric(lambda)[1]
  shift  <- as.numeric(shift)[1]

  # --- Identify complete rows ---------------------------------------------
  mask <- complete.cases(data[, OXIDE_COLUMNS, drop = FALSE])

  # --- Calibration domain check -------------------------------------------
  flags <- check_calibration_range(data[mask, OXIDE_COLUMNS, drop = FALSE],
                                   which(mask))

  # --- Temperature prediction ---------------------------------------------
  # Columns are selected by name into OXIDE_COLUMNS order, so the order
  # they appear in the input file never matters. That order must match
  # the scaler CSVs, which are read positionally.
  X_temp <- as.matrix(data[mask, OXIDE_COLUMNS])
  X_temp <- sweep(log1p(X_temp), 2, temp_mean, "-")
  X_temp <- sweep(X_temp, 2, temp_scale, "/")
  temp_pred <- predict(temp_model, X_temp)

  # --- Precipitation prediction -------------------------------------------
  X_prec <- as.matrix(data[mask, OXIDE_COLUMNS])
  X_prec <- sweep(log1p(X_prec), 2, prec_mean, "-")
  X_prec <- sweep(X_prec, 2, prec_scale, "/")
  prec_pred_bc <- predict(prec_model, X_prec)
  prec_pred <- inv_boxcox(prec_pred_bc, lambda) - shift

  # The inverse Box-Cox is undefined below -1/lambda, so guard against a
  # non-finite result rather than writing NaN into the output silently.
  bad <- !is.finite(prec_pred)
  if (any(bad)) {
    cat(sprintf(paste0("Warning: %d precipitation predictions are not ",
                       "finite and are reported as missing.\n"), sum(bad)))
    prec_pred[bad] <- NA_real_
  }
  prec_pred <- pmax(prec_pred, 0)

  # --- Assemble output ----------------------------------------------------
  temp_pred <- round(temp_pred, 1)
  prec_pred <- round(prec_pred, 0)

  data$MAT_Best <- NA_real_
  data$MAT_Min  <- NA_real_
  data$MAT_Max  <- NA_real_

  data$MAP_Best <- NA_real_
  data$MAP_Min  <- NA_real_
  data$MAP_Max  <- NA_real_

  data$Outside_calibration <- ""

  # --- Fill predictions ---------------------------------------------------

  data$MAT_Best[mask] <- temp_pred
  data$MAT_Min[mask]  <- round(temp_pred - TEMP_UNCERTAINTY, 1)
  data$MAT_Max[mask]  <- round(temp_pred + TEMP_UNCERTAINTY, 1)

  data$MAP_Best[mask] <- prec_pred
  data$MAP_Min[mask]  <- round(pmax(prec_pred - PREC_UNCERTAINTY, 0), 0)
  data$MAP_Max[mask]  <- round(prec_pred + PREC_UNCERTAINTY, 0)

  data$Outside_calibration[mask] <- flags

  # --- Save ---------------------------------------------------------------
  output_dir <- dirname(paths$output)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  write.csv(data, paths$output, row.names = FALSE)

  # --- Summary ------------------------------------------------------------

  cat(sprintf("Predictions saved to: %s\n", paths$output))
  cat(sprintf("  Rows predicted: %d / %d\n", sum(mask), nrow(data)))

  n_flagged <- sum(data$Outside_calibration != "")
  if (n_flagged > 0) {
    cat(sprintf(paste0("  Samples outside the calibration domain: %d ",
                       "(see the Outside_calibration column)\n"), n_flagged))
  }
}

main()
