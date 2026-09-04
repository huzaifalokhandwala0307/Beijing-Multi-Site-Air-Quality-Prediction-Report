# Beijing Multi-Site Air Quality Prediction

Time-series forecasting of PM2.5 concentration in Beijing using deep recurrent neural networks (RNN, LSTM, GRU, and their bidirectional variants), trained on the UCI/Kaggle Beijing Multi-Site Air Quality dataset.

## Overview

This project builds and compares six recurrent neural network architectures to predict hourly PM2.5 levels from historical air quality and weather measurements at the Aotizhongxin monitoring station. Each model consumes a 24-hour lookback window of features and predicts the PM2.5 value for the next hour.

## Dataset

- **Source:** [Beijing Multi-Site Air Quality Data Set](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set) (downloaded via `kagglehub`)
- **Station used:** Aotizhongxin (`PRSA_Data_Aotizhongxin_20130301-20170228.csv`)
- **Period:** March 2013 – February 2017, hourly readings
- **Features:** PM2.5, PM10, SO2, NO2, CO, O3, temperature, pressure, dew point, rainfall, wind speed, wind direction, plus derived datetime features (month, day, hour)
- **Target:** PM2.5 concentration

## Methodology

1. **Data loading & cleaning** — Combine `year`/`month`/`day`/`hour` into a single `datetime` column, sort chronologically, and set it as the index.
2. **Train/test split** — Chronological 80/20 split (no shuffling, to preserve temporal order).
3. **Missing value handling** — Forward-fill (`ffill`) imputation on both train and test sets; the index is resampled to hourly frequency to fill any gaps.
4. **Exploratory data analysis** — Summary statistics and boxplots of all numeric features to inspect distribution and outliers.
5. **Feature engineering** — A `ColumnTransformer` pipeline:
   - Cyclical features (`month`, `day`, `hour`) → `SplineTransformer`
   - Numeric features (pollutants, weather) → `StandardScaler`
   - Categorical feature (`wd`, wind direction) → `OneHotEncoder`
6. **Sequence generation** — Sliding-window sequences with a 24-hour lookback used as input to the RNNs.
7. **Modeling** — Six stacked recurrent architectures, each trained for 10 epochs (Adam optimizer, MSE loss, MAE metric):
   - Deep Simple RNN
   - Deep LSTM
   - Deep GRU
   - Bidirectional RNN
   - Bidirectional LSTM
   - Bidirectional GRU
8. **Evaluation** — Best validation MAE compared across all six models, with a summary table and bar chart.

## Results

Best validation MAE is computed for each model and ranked from lowest to highest. In this run, the Deep RNN and Bidirectional RNN models achieved the lowest validation MAE (~10.1), outperforming the LSTM-based models (~11.8–15.4) — a discussion of why this can happen (dataset size, sequence length, and RNN vs. LSTM inductive bias) is included in the notebook's final section.

*(Exact numbers depend on the training run — see the notebook output and the `results_df` table for current values.)*

## Requirements

```
tensorflow
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
```

Install with:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn kagglehub
```

A Kaggle account/API token is required for `kagglehub` to download the dataset automatically.

## Usage

1. Open `Beijing_Multi_Site_Air_Quality_Prediction_Report.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — the notebook downloads the dataset, preprocesses it, trains all six models, and generates comparison plots and tables.
3. Review Section 5 (Model Performance Comparison) for the MAE leaderboard, and Section 6 for the RNN vs. LSTM discussion.

## Notebook Structure

| Section | Contents |
|---|---|
| 1. Data Loading and Initial Inspection | Load CSV, create datetime index, train/test split, missing value handling |
| 2. Exploratory Data Analysis | Summary stats, boxplots of numeric features |
| 3. Feature Engineering and Preprocessing | Spline/scaling/one-hot pipeline, sequence generation |
| 4. Model Training and Evaluation | Six RNN-family models trained and validated |
| 5. Model Performance Comparison | MAE leaderboard, bar chart, detailed metrics table |
| 6. Discussion | RNN vs. LSTM performance analysis |

## Notes

- Sequences are built with a lookback window of 24 hours; this can be tuned via the `lookback_window` variable.
- All models are trained for a fixed 10 epochs — extending training or adding early stopping/regularization may change the relative ranking of architectures.
