# Optimal Station Recommender - Detailed Documentation

## Repository Overview

This repository contains a data science pipeline that recommends optimal locations for new transit stations across multiple cities. It combines data collection, feature engineering using H3 hexagonal grids, a machine learning model, and various visualizations. The top-level directory layout is:

```
- OSM_GUIDE.md              # Guide for optimized OSM data fetching
- README.md                 # Quick start and usage guide
- cache/                    # Caching for API calls
- models/                   # Saved Random Forest models
- results/                  # Evaluation metrics and visualizations
- src/                      # Source code (configuration, ML pipeline, web app)
```

Subdirectories include model artifacts (`models/`), result plots and metrics (`results/`), and implementation code in `src/` (configuration, machine learning, and web interface). The `results/old_training_backup_20250606` folder stores metrics from an earlier experiment with fewer features and a different train/test split.

## Problem Statement

Public transit planners often need to decide where to build new stations. These decisions require analyzing population density, nearby amenities, and existing transit coverage. The **Optimal Station Recommender** addresses this by predicting which H3 hexagonal cells (roughly 175 m across at resolution 9) are suitable for new stations. Each hexagon is labeled `has_station = 1` if an actual station exists within it; otherwise `0`. The main challenge is the extremely imbalanced dataset:

- Class 0 (no station): 469,615 samples (99.13%)
- Class 1 (has station): 4,112 samples (0.87%)
- Imbalance ratio ≈ 114:1

The goal is to learn patterns that distinguish true station locations from areas without stations while dealing with this skewed distribution.

## Data Sources

1. **WorldPop** – 100 m resolution raster data providing population counts for each country (default year 2020). See README lines describing the source:
   - `WorldPop 100m resolution population datasets`【F:README.md†L97-L103】
2. **OpenStreetMap** – boundaries, station locations, and a rich set of amenities. The `OSM_GUIDE.md` file describes optimized fetching methods including spatial chunking and progress bars.

Configured cities are listed in the README (London, Paris, Berlin, ... Sydney)【F:README.md†L69-L86】. File `src/config/config.py` defines how OSM amenities are categorized. For example, amenity groups such as `education`, `healthcare`, `shopping_food`, `transportation`, etc. are specified as dictionaries of OSM tags【F:src/config/config.py†L46-L75】. Population rasters for each city are also configured in this file under `WORLDPOP_CONFIG`【F:src/config/config.py†L96-L125】.

### Data Acquisition

- **Boundaries and Stations** are fetched from OSM. The `OSM_GUIDE.md` details advanced strategies like spatial chunking to speed up queries【F:OSM_GUIDE.md†L64-L89】.
- **Population** rasters are downloaded via `worldpop_downloader.py`, stored under `data/raw/`. The pipeline can resume interrupted downloads and supports multi-threading.

### Feature Engineering

Features are engineered in `src/ml/feature_engineering.py`. Key steps include:

1. **H3 Grid Generation** – `create_h3_grid_for_city` converts each city boundary to a set of H3 cells at resolution 9, storing the hexagon geometry and index【F:src/ml/feature_engineering.py†L20-L88】.
2. **Population Aggregation** – `aggregate_raster_data_to_grid` sums WorldPop pixel values inside each hexagon, producing a `population` column【F:src/ml/feature_engineering.py†L245-L339】.
3. **Amenity Counts** – `count_points_in_polygons` counts OSM amenities of each category within every hexagon, creating columns like `count_amenity_education`, `count_amenity_healthcare`, etc.【F:src/ml/feature_engineering.py†L360-L411】.
4. **Station Presence** – hexagons containing stations are labeled in `create_features_for_city` with `has_station` (target variable)【F:src/ml/feature_engineering.py†L454-L517】.

The final dataset for all cities is saved as a GeoPackage `all_cities_features_h9.gpkg`. An earlier experiment recorded roughly 473k hexagons total【F:results/old_training_backup_20250606/README.md†L10-L11】.

## Model and Training Procedure

`src/ml/model_training.py` loads the combined features, performs a **city-level split** (training on 12 cities, testing on 3 others), and trains a **Random Forest** classifier. The pipeline uses `class_weight='balanced'` to partially address class imbalance and explores many hyperparameters through grid search:

```python
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 15, 25, 35],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__bootstrap': [True, False],
    'classifier__criterion': ['gini', 'entropy']
}
```
【F:src/ml/model_training.py†L178-L207】

Random Forests were chosen for their ability to model non-linear relationships, handle mixed feature scales without normalization, and output feature importances for interpretation. Geographic city-level splitting prevents data leakage across nearby locations.【F:src/ml/model_training.py†L38-L60】

## Evaluation Metrics

The pipeline evaluates the model with metrics suited to imbalanced classification:

- **ROC-AUC** and **PR-AUC** scores are calculated in `evaluate_model`【F:src/ml/model_training.py†L229-L253】.
- Example overall results (resolution 9) are stored in `results/evaluation_metrics_h9.csv`:
  - Accuracy 0.979, Precision 0.183, Recall 0.344, ROC‑AUC 0.922, PR‑AUC 0.217【F:results/evaluation_metrics_h9.csv†L1-L2】.
- City-specific metrics are reported per test city in `results/city_metrics_h9.csv`【F:results/city_metrics_h9.csv†L1-L4】.
- Training configuration and city split are saved in `training_config_h9.txt`【F:results/training_config_h9.txt†L1-L15】.

Visual outputs include feature importance plots and decision tree diagrams stored under `results/`.

## Handling Class Imbalance

The dataset’s heavy imbalance (≈114:1) prompted several strategies:

1. **`class_weight='balanced'`** inside the RandomForest classifier automatically reweights classes during training.
2. **Comprehensive Hyperparameter Grid** explores parameters like `min_samples_leaf` and different `criterion` values to find settings that might better capture minority-class patterns.
3. **City‑level train/test split** ensures evaluation on unseen geographic areas, avoiding optimistic results from spatially correlated neighbors.

## Conclusion

This project demonstrates a full workflow for recommending transit station locations. Data from WorldPop and OSM is aggregated into H3 cells, producing features such as population counts and amenity densities. A Random Forest model, tuned with a large hyperparameter grid and balanced class weights, predicts the presence of stations. Evaluation metrics show moderate precision and recall due to the inherent class imbalance, but ROC-AUC and PR-AUC values (~0.92 and ~0.22) indicate the model captures useful signals. Results and configuration files in the `results/` directory provide transparency and reproducibility.

