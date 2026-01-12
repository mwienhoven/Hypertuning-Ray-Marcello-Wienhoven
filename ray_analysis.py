from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from loguru import logger
from ray.tune import ExperimentAnalysis

# -------------------------------
# SETTINGS
# -------------------------------
BASE_DIR = Path(__file__).parent.resolve()
TUNE_LOGS_DIR = BASE_DIR / "logs" / "ray"
ANALYSIS_LOG_DIR = BASE_DIR / "logs" / "analysis"
IMG_DIR = BASE_DIR / "img"

ANALYSIS_LOG_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# LOGGING CONFIG
# -------------------------------
log_file = ANALYSIS_LOG_DIR / "analysis.log"
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> | <level>{level}</level> | {message}",
)
logger.add(
    log_file,
    rotation="5 MB",
    retention="10 days",
    level="INFO",
    enqueue=True,
)

logger.info(
    "Starting Ray Tune pairwise hyperparameter contour analysis (single plots)..."
)

# -------------------------------
# LOAD LATEST RAY TUNE EXPERIMENT
# -------------------------------
experiment_folders = [d for d in TUNE_LOGS_DIR.iterdir() if d.is_dir()]
if not experiment_folders:
    logger.error("No Ray Tune experiment folders found in %s", TUNE_LOGS_DIR)
    raise SystemExit(1)

latest_experiment = max(experiment_folders, key=lambda d: d.stat().st_mtime)
logger.info(f"Using latest Ray session folder: {latest_experiment}")

analysis = ExperimentAnalysis(str(latest_experiment))
df = analysis.results_df.copy()
logger.info(f"Loaded {len(df)} trial results.")
logger.info(f"Columns in the results: {df.columns.tolist()}")

# -------------------------------
# DETECT METRIC COLUMN
# -------------------------------
possible_metric_cols = [
    c for c in df.columns if "acc" in c.lower() or "accuracy" in c.lower()
]
metric_col = None

if possible_metric_cols:
    metric_col = possible_metric_cols[0]
    logger.info(f"Detected metric column for analysis: {metric_col}")
elif "test_acc" in df.columns:
    metric_col = "test_acc"
    logger.info("Falling back to 'test_acc' metric column")
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        metric_col = numeric_cols[0]
        logger.warning(
            f"No accuracy-like column found. Using first numeric column: {metric_col}"
        )
    else:
        logger.error("No numeric columns available for metric analysis. Exiting.")
        raise SystemExit(1)

# -------------------------------
# SELECT HYPERPARAMETER COLUMNS
# -------------------------------
columns_of_interest = [
    metric_col,
    "config/filters",
    "config/units1",
    "config/units2",
    "config/dropout",
    "config/kernel_size",
    "config/num_layers",
]
columns_of_interest = [c for c in columns_of_interest if c in df.columns]
df_selected = df[columns_of_interest].dropna().reset_index(drop=True)
logger.info(f"Columns selected for contour analysis: {columns_of_interest}")

# -------------------------------
# CREATE TIMESTAMPED IMAGE FOLDER
# -------------------------------
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_img_dir = IMG_DIR / run_timestamp
run_img_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Images will be saved to folder: {run_img_dir}")


# -------------------------------
# CONTOUR PLOT FUNCTION
# -------------------------------
def contour_plot(x_param, y_param, df, metric=metric_col, save_dir=run_img_dir):
    x = df[x_param].values
    y = df[y_param].values
    z = df[metric].values

    # Grid for interpolation
    xi = np.linspace(np.min(x), np.max(x), 200)
    yi = np.linspace(np.min(y), np.max(y), 200)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    # Fallback for NaNs
    if np.isnan(zi).any():
        zi = griddata((x, y), z, (xi, yi), method="linear")
    if np.isnan(zi).any():
        zi = griddata((x, y), z, (xi, yi), method="nearest")

    # Slightly expand min/max for axes so edge dots are visible
    x_margin = (np.max(x) - np.min(x)) * 0.05  # 5% margin
    y_margin = (np.max(y) - np.min(y)) * 0.05

    x_min, x_max = np.min(x) - x_margin, np.max(x) + x_margin
    y_min, y_max = np.min(y) - y_margin, np.max(y) + y_margin

    # Create figure
    plt.figure(figsize=(19.2, 10.8), dpi=100)
    cf = plt.contourf(xi, yi, zi, levels=30, cmap="coolwarm", alpha=0.8)
    plt.scatter(x, y, c="k", s=80, edgecolors="white", linewidths=1.0)  # bigger dots
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"{metric} across {x_param} vs {y_param}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    cbar = plt.colorbar(cf)
    cbar.set_label(metric)

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"contour_{x_param}_vs_{y_param}.png".replace("/", "_")
    img_path = save_dir / filename
    plt.savefig(img_path, dpi=150)
    plt.close()
    logger.info(f"Saved contour plot: {img_path}")


# -------------------------------
# GENERATE ALL PAIRWISE CONTOUR PLOTS
# -------------------------------
params_only = [c for c in columns_of_interest if c != metric_col]

for i, x_param in enumerate(params_only):
    for j, y_param in enumerate(params_only):
        if i >= j:
            continue  # Only unique pairs above the diagonal (skip self-pairs)
        contour_plot(x_param, y_param, df_selected, metric=metric_col)
