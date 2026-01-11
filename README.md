[![CircleCI](https://dl.circleci.com/status-badge/img/gh/RY4N-L/DAT5501-002ReportAndCodeProject/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/RY4N-L/DAT5501-002ReportAndCodeProject/tree/main)
# DAT5501 002 Report and Code Project
(Used‑Car Price Analysis & Machine Learning Pipeline)

This repository contains the full data‑cleaning pipeline, modelling workflow, and automated testing framework developed for the **DAT5501 002 Report and Code Project**.

All raw data in ```data/raw``` is sourced from the DVM-CAR Dataset (https://figshare.com/articles/figure/DVM-CAR_Dataset/19586296/2?file=38754867)

---

---

## Highlights

- Built a **comprehensive cleaning pipeline** for ad, sales and price datasets.
- Implemented **unit tests** (30+ tests) validating all cleaning functions and the final processed dataset.
- Configured **CircleCI** for automated testing on every push.
- Developed a full **modelling pipeline** (Decision Tree, Random Forest, Gradient Boosting).
- Performed **hyperparameter tuning** using RandomizedSearchCV.
- Generated **feature‑importance plots**, correlation heatmaps, and brand‑level error analysis.
- Saved **final trained models**, metrics, and figures for reproducibility.

---

## Folder Structure

- **.circleci**  
  - `config.yml` – CircleCI configuration (Python 3.13 Docker image, installs dependencies, runs unit tests).

- **data/**  
  - **raw/**  
    - `Ad_table (extra).csv` – main dataset of UK used‑car advertisements.  
    - `Ad_table.csv`, `Basic_table.csv`, `Image_table.csv`, `Price_table.csv`, `Sales_table.csv`, `Trim_table.csv` – additional raw data sources.  
  - **processed/**  
    - `ad.csv` – cleaned advertisement dataset.  
    - `price.csv` – cleaned price dataset.  
    - `sales.csv` – cleaned sales dataset.  
    - `trim.csv` – cleaned trim dataset.  
    - `final_dataset.csv` – fully merged and validated modelling dataset.

- **figures/**  
  - `raw_price_box_plot.png` – initial price distribution.  
  - `correlation_heatmap.png` – numeric feature correlations.  
  - `final_rf_feature_importance.png`, `final_gb_feature_importance.png`, `final_dtr_feature_importance.png` – model feature importances.  
  - `absolute_error_binned.png`, `relative_error_binned.png` – error analysis.  
  - `brand_level_bias.png` – brand‑specific residuals.  

- **models/**  
  - `final_random_forest_metrics.json`  
  - `final_random_forest_tuned_metrics.json`  
  - `final_decision_tree_regressor_metrics.json`  
  - `final_gradient_boosting_metrics.json`  
  - (All contain MAE, RMSE, R², and hyperparameters.)

- **src/**  
  - **cleaning/**  
    - `clean_ad.py` – cleans advertisement dataset (units, types, outliers, derived features).  
    - `clean_price.py` – cleans price dataset.  
    - `clean_sales.py` – cleans sales dataset.  
    - `clean_trim.py` – cleans trim dataset.  
    - `merge_ad_sales_price.py` – merges cleaned datasets into a unified modelling table.  
  - `model_analysis.py` – full modelling pipeline (train/test split, encoding, model training, tuning, evaluation, saving metrics, figures and models).

- **tests/**  
  - **unit/**  
    - `test_clean_ad.py` – tests for ad cleaning functions.  
    - `test_clean_price.py` – tests for price cleaning.  
    - `test_clean_sales.py` – tests for sales cleaning.  
    - `test_final_dataset.py` – validates final merged dataset (no missing values, numeric types, valid ranges).  
    - `__init__.py` – enables test discovery.

- **requirements.txt**  
  - Contains dependencies: `pytest`, `numpy`, `pandas`, `matplotlib`, `seaborn`.

---

## Requirements

- Python 3.10+  
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `scipy` (for `scipy.stats.randint`)
  - `joblib`

 Install Libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
  ```


## How to Run

Clone the repository:
```bash
git clone https://github.com/RY4N-L/DAT5501-002ReportAndCodeProject.git
cd DAT5501-002ReportAndCodeProject
```
Run unit tests
```
pytest
```
Run the modelling pipeline
```bash
python src/model_analysis.py
```
PLEASE NOTE: the modelling pipeline saves trained models locally as .pkl files within the /models folder (these models are ignored by Git)


---
