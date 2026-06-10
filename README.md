# DM2026-Assignment-3

# Human Activity Recognition using Wrist-Worn Accelerometer Data

This project was developed for the Data Mining Assignment 3 Kaggle competition. The objective is to classify human activities from wrist-worn accelerometer recordings. Each recording corresponds to a 5-minute sequence sampled into 300 one-second observations, and each sequence is associated with one activity label from 0 to 5.

The notebook implements a complete machine learning pipeline for Human Activity Recognition. It loads the training and test CSV files, extracts fixed-length features from the accelerometer time series, trains several machine learning models, evaluates them using Macro F1-score, compares different feature engineering strategies, and generates a Kaggle submission file in the required format.

---

## Method Overview

The main idea is to transform each 5-minute accelerometer sequence into a fixed-length feature vector. The notebook progressively compares several feature sets:

1. **Statistical features**: mean, standard deviation, min, max, median, quantiles, interquartile range, and signal energy.
2. **Temporal and frequency features**: temporal chunks and FFT-based descriptors.
3. **Improved HAR features**: acceleration magnitude, jerk, axis correlations, dominant frequency, spectral energy, spectral entropy, and richer temporal summaries.

Several tree-based models are evaluated, including Random Forest, Extra Trees, HistGradientBoosting, and XGBoost. GroupKFold cross-validation is used to avoid placing recordings from the same user in both training and validation sets.

---

## How to Run

1. Clone or download the project repository.

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Place the dataset files in the expected folders:

```text
train/
test/
```

The `train/` and `test/` folders should keep the original user-based structure provided in the Kaggle competition.

4. Open and run the notebook:

```bash
jupyter notebook DM_assignment3_HAR_pipeline_improved.ipynb
```

5. Run all cells from top to bottom. The notebook will:

- load the dataset,
- extract the features,
- train and compare the models,
- display the validation results and analysis figures,
- train the final model,
- generate the Kaggle submission file.

## Evaluation Metric

The models are evaluated using the Macro F1-score, which gives equal importance to all activity classes.
