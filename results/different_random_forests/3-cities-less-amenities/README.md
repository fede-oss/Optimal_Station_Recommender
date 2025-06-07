# Training Results Backup - June 6, 2025

This folder contains backup results from the previous Random Forest model training conducted on **June 1, 2024**.

## Model Configuration (OLD)
- **Training Date**: June 1, 2024
- **Split Method**: Random 80/20 split across all cities
- **Cities**: All 15 cities mixed together
- **Features**: 7 amenity categories + population
- **Total Hexagons**: ~473k
- **H3 Resolution**: 9

## Feature Set (OLD - 7 Categories)
1. population
2. count_amenity_education
3. count_amenity_healthcare  
4. count_amenity_shopping_food
5. count_amenity_leisure_recreation
6. count_amenity_workplaces
7. count_amenity_public_services

## Model Performance (OLD)
- **Accuracy**: 95.5%
- **Precision**: 30%
- **Recall**: 12.3%
- **F1-Score**: 17.4%

## Files Included
- `station_recommender_h9_OLD.joblib` - Trained Random Forest model
- `evaluation_metrics_h9.csv` - Performance metrics
- `feature_importances_h9.csv` - Feature importance rankings
- `confusion_matrix_h9.png` - Confusion matrix visualization
- `feature_importance_plot_h9.png` - Feature importance bar chart
- `decision_trees_h9/` - Individual decision tree visualizations
- `rankings/` - Station ranking results and visualizations

## Changes Made After This Backup
- Updated to 11 amenity categories (added: transportation, accommodation, entertainment, religious, tourism)
- Changed to city-level train/test split for better generalization testing
- Training cities: Paris, Berlin, Madrid, Stockholm, Rome, Seoul, Singapore, Hong Kong, New York, Mexico City, São Paulo, Buenos Aires
- Testing cities: Warsaw, Toronto, Sydney

This backup preserves the original results for comparison and analysis purposes.
