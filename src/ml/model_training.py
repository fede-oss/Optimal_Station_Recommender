import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import os
import subprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
import graphviz

# Define directories (consistent with other scripts)
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Should point to Optimal_Station_Recommender directory
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# H3 resolution used in feature engineering
H3_RESOLUTION = 9


def load_feature_data():
    """
    Load the combined feature data created by feature_engineering.py
    
    Returns:
        gpd.GeoDataFrame: Combined features from all cities
    """
    feature_file = DATA_PROCESSED_DIR / f"all_cities_features_h{H3_RESOLUTION}.gpkg"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    print(f"Loading feature data from {feature_file}")
    features_gdf = gpd.read_file(feature_file)
    print(f"Loaded {len(features_gdf)} H3 cells with features")
    
    return features_gdf


def prepare_data_for_training(features_gdf):
    """
    Prepare the feature data for model training using city-level split.
    
    Args:
        features_gdf (gpd.GeoDataFrame): The combined feature data
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, train_cities, test_cities
    """
    # Define city splits
    TRAIN_CITIES = [
        'paris', 'berlin', 'madrid', 'stockholm', 'rome', 
        'seoul', 'singapore', 'hong_kong', 'new_york', 
        'mexico_city', 'sao_paulo', 'buenos_aires'
    ]
    
    TEST_CITIES = ['warsaw', 'toronto', 'sydney']
    
    print("\n--- CITY-LEVEL TRAIN/TEST SPLIT ---")
    print(f"Training cities ({len(TRAIN_CITIES)}): {', '.join(TRAIN_CITIES)}")
    print(f"Testing cities ({len(TEST_CITIES)}): {', '.join(TEST_CITIES)}")
    
    # First, check if the population column is numeric and convert if necessary
    print("\n--- POPULATION DATA INVESTIGATION ---")
    print(f"Population column data type before conversion: {features_gdf['population'].dtype}")
    if features_gdf['population'].dtype == 'object':
        print("Population column is non-numeric (strings). Converting to numeric values...")
        features_gdf['population'] = pd.to_numeric(features_gdf['population'], errors='coerce')
        features_gdf['population'] = features_gdf['population'].fillna(0)
        print(f"Conversion complete. New data type: {features_gdf['population'].dtype}")
    
    # Debug: Print information about population data
    print(f"Total cells with population data: {features_gdf['population'].notna().sum()} out of {len(features_gdf)}")
    print(f"Population data statistics:\n{features_gdf['population'].describe()}")
    print(f"Number of cells with zero population: {(features_gdf['population'] == 0).sum()}")
    print(f"Percentage of cells with zero population: {(features_gdf['population'] == 0).mean() * 100:.2f}%")
    
    # Debug: Compare population for cells with and without stations
    has_station_pop = features_gdf.loc[features_gdf['has_station'] == 1, 'population']
    no_station_pop = features_gdf.loc[features_gdf['has_station'] == 0, 'population']
    print(f"Population in cells WITH stations: mean={has_station_pop.mean():.2f}, median={has_station_pop.median():.2f}")
    print(f"Population in cells WITHOUT stations: mean={no_station_pop.mean():.2f}, median={no_station_pop.median():.2f}")
    
    # Check if required columns exist
    required_columns = ['population', 'has_station', 'city']
    # Check for at least one amenity column
    has_amenity_columns = any(col.startswith('count_amenity_') for col in features_gdf.columns)
    if not has_amenity_columns:
        required_columns.append('at least one amenity count column')
    
    missing_columns = [col for col in required_columns if col not in features_gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Verify all specified cities exist in the data
    available_cities = set(features_gdf['city'].unique())
    missing_train_cities = set(TRAIN_CITIES) - available_cities
    missing_test_cities = set(TEST_CITIES) - available_cities
    
    if missing_train_cities:
        print(f"Warning: Missing training cities: {missing_train_cities}")
        TRAIN_CITIES = [city for city in TRAIN_CITIES if city in available_cities]
    
    if missing_test_cities:
        print(f"Warning: Missing test cities: {missing_test_cities}")
        TEST_CITIES = [city for city in TEST_CITIES if city in available_cities]
    
    print(f"Final training cities: {TRAIN_CITIES}")
    print(f"Final test cities: {TEST_CITIES}")
    
    # Split data by cities
    train_data = features_gdf[features_gdf['city'].isin(TRAIN_CITIES)].copy()
    test_data = features_gdf[features_gdf['city'].isin(TEST_CITIES)].copy()
    
    # Drop any rows with NaN values
    train_data = train_data.dropna(subset=required_columns)
    test_data = test_data.dropna(subset=required_columns)
    
    print(f"\n--- CITY-LEVEL DATA DISTRIBUTION ---")
    print(f"Training data: {len(train_data)} cells from {len(TRAIN_CITIES)} cities")
    print(f"Test data: {len(test_data)} cells from {len(TEST_CITIES)} cities")
    
    # Print city-wise statistics
    print("\nTraining cities distribution:")
    train_city_stats = train_data.groupby('city').agg({
        'h3_index': 'count',
        'has_station': 'sum'
    }).rename(columns={'h3_index': 'total_cells', 'has_station': 'stations'})
    train_city_stats['station_density'] = (train_city_stats['stations'] / train_city_stats['total_cells'] * 100).round(2)
    print(train_city_stats)
    
    print("\nTest cities distribution:")
    test_city_stats = test_data.groupby('city').agg({
        'h3_index': 'count',
        'has_station': 'sum'
    }).rename(columns={'h3_index': 'total_cells', 'has_station': 'stations'})
    test_city_stats['station_density'] = (test_city_stats['stations'] / test_city_stats['total_cells'] * 100).round(2)
    print(test_city_stats)
    
    # Select feature columns (excluding geometry, h3_index, city, and target variable)
    feature_columns = [col for col in features_gdf.columns 
                      if col not in ['geometry', 'h3_index', 'has_station', 'city']]
    
    print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")
    
    # Split into features (X) and target (y)
    X_train = train_data[feature_columns]
    y_train = train_data['has_station']
    X_test = test_data[feature_columns]
    y_test = test_data['has_station']
    
    print(f"\nFinal training set: {X_train.shape[0]} samples")
    print(f"Final test set: {X_test.shape[0]} samples")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test, feature_columns, TRAIN_CITIES, TEST_CITIES


def train_model(X_train, y_train, feature_names):
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        feature_names (list): Names of the feature columns
        
    Returns:
        tuple: Best model and grid search results
    """
    print("Training Random Forest model with hyperparameter tuning...")
    
    # Create a pipeline with Random Forest (no scaling needed for tree-based models)
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(
            random_state=42,
            class_weight='balanced',  # Handle class imbalance automatically
            n_jobs=-1,  # Use all CPU cores
            oob_score=True  # Out-of-bag scoring for additional validation
        ))
    ])
    
    # Enhanced hyperparameter grid for Random Forest
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],  # Focus on higher values
        'classifier__max_depth': [None, 15, 25, 35],  # Adjusted range
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],  # Feature randomness
        'classifier__bootstrap': [True, False],  # Sampling strategy
        'classifier__criterion': ['gini', 'entropy']  # Split quality measure
    }
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Random Forest specific information
    rf_model = best_model.named_steps['classifier']
    if hasattr(rf_model, 'oob_score_'):
        print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importances.head(10))
    
    return best_model, grid_search, feature_importances


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    
    # Calculate comprehensive metrics for imbalanced classification
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)  # Precision-Recall AUC
    }
    
    # Print metrics with better formatting for imbalanced data
    print("\n=== MODEL PERFORMANCE METRICS ===")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f} (better for imbalanced data)")
    
    # Additional context for imbalanced classification
    positive_rate = y_test.mean()
    print(f"\nClass distribution in test set:")
    print(f"Positive class rate: {positive_rate:.3f} ({positive_rate*100:.1f}%)")
    print(f"Negative class rate: {1-positive_rate:.3f} ({(1-positive_rate)*100:.1f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred


def save_results(model, feature_importances, metrics, X_test, y_test, y_pred, train_cities=None, test_cities=None, city_metrics=None):
    """
    Save the model, feature importances, and evaluation results.
    
    Args:
        model: Trained model
        feature_importances (pd.DataFrame): Feature importances
        metrics (dict): Evaluation metrics
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        y_pred (np.array): Predicted values
        train_cities (list, optional): List of training cities
        test_cities (list, optional): List of test cities  
        city_metrics (dict, optional): City-specific evaluation metrics
    """
    # Save model
    model_file = MODELS_DIR / f"station_recommender_h{H3_RESOLUTION}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved to {model_file}")
    
    # Save feature importances
    importance_file = RESULTS_DIR / f"feature_importances_h{H3_RESOLUTION}.csv"
    feature_importances.to_csv(importance_file, index=False)
    print(f"Feature importances saved to {importance_file}")
    
    # Save evaluation metrics
    metrics_file = RESULTS_DIR / f"evaluation_metrics_h{H3_RESOLUTION}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"Evaluation metrics saved to {metrics_file}")
    
    # Save city-specific metrics if provided
    if city_metrics and test_cities:
        city_metrics_file = RESULTS_DIR / f"city_metrics_h{H3_RESOLUTION}.csv"
        city_metrics_df = pd.DataFrame.from_dict(city_metrics, orient='index')
        city_metrics_df.index.name = 'city'
        city_metrics_df.to_csv(city_metrics_file)
        print(f"City-specific metrics saved to {city_metrics_file}")
    
    # Save training configuration if provided
    if train_cities and test_cities:
        config_file = RESULTS_DIR / f"training_config_h{H3_RESOLUTION}.txt"
        with open(config_file, 'w') as f:
            f.write(f"Training Configuration - H3 Resolution {H3_RESOLUTION}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Split Method: City-level split\n")
            f.write(f"Training Cities ({len(train_cities)}): {', '.join(train_cities)}\n")
            f.write(f"Test Cities ({len(test_cities)}): {', '.join(test_cities)}\n\n")
            f.write("Overall Model Performance:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        print(f"Training configuration saved to {config_file}")
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importances.sort_values('Importance', ascending=True).tail(10)
    )
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    importance_plot_file = RESULTS_DIR / f"feature_importance_plot_h{H3_RESOLUTION}.png"
    plt.savefig(importance_plot_file, dpi=300)
    print(f"Feature importance plot saved to {importance_plot_file}")
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Station', 'Has Station'],
                yticklabels=['No Station', 'Has Station'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_plot_file = RESULTS_DIR / f"confusion_matrix_h{H3_RESOLUTION}.png"
    plt.savefig(cm_plot_file, dpi=300)
    print(f"Confusion matrix plot saved to {cm_plot_file}")
    
    # Create a directory for tree visualizations
    trees_dir = RESULTS_DIR / f"decision_trees_h{H3_RESOLUTION}"
    trees_dir.mkdir(exist_ok=True)
    
    # Visualize a sample of trees from the Random Forest
    # Get the Random Forest classifier from the pipeline
    rf_classifier = model.named_steps['classifier']
    feature_names = X_test.columns
    
    # Select a sample of trees to visualize (e.g., first 3 trees)
    num_trees_to_visualize = min(3, len(rf_classifier.estimators_))
    print(f"\nVisualizing {num_trees_to_visualize} sample trees from the Random Forest:")
    
    def export_tree_to_png(estimator, output_path, feature_names, max_depth=3):
        """Export a decision tree to PNG format"""
        # Create a temporary dot file
        dot_data = export_graphviz(
            estimator,
            out_file=None,
            feature_names=feature_names,
            class_names=['No Station', 'Has Station'],
            filled=True,
            rounded=True,
            precision=2,
            max_depth=max_depth
        )
        
        # Convert to PNG using graphviz
        graph = graphviz.Source(dot_data)
        graph.format = 'png'
        graph.render(filename=output_path.stem, directory=str(output_path.parent), cleanup=True)
        print(f"Tree visualization saved to {output_path}")
    
    # Export individual trees
    for i in range(num_trees_to_visualize):
        tree_png_file = trees_dir / f"tree_{i}.png"
        try:
            export_tree_to_png(
                rf_classifier.estimators_[i],
                tree_png_file,
                feature_names,
                max_depth=3
            )
        except Exception as e:
            print(f"Error exporting tree {i} to PNG: {e}")
            # Fallback to DOT file if PNG conversion fails
            tree_dot_file = trees_dir / f"tree_{i}.dot"
            export_graphviz(
                rf_classifier.estimators_[i],
                out_file=str(tree_dot_file),
                feature_names=feature_names,
                class_names=['No Station', 'Has Station'],
                filled=True,
                rounded=True,
                precision=2,
                max_depth=3
            )
            print(f"Fallback: Tree {i} visualization saved as DOT file to {tree_dot_file}")
    
    # Also export a tree visualization that includes feature importance information
    # Get the most important tree (one with highest feature importance for top feature)
    if len(rf_classifier.estimators_) > 0:
        top_feature = feature_importances.iloc[0]['Feature']
        top_feature_idx = list(feature_names).index(top_feature) if top_feature in feature_names else 0
        
        # Find tree with highest importance for this feature
        tree_importances = [tree.feature_importances_[top_feature_idx] 
                           for tree in rf_classifier.estimators_]
        best_tree_idx = np.argmax(tree_importances)
        
        # Export the most important tree
        important_tree_png_file = trees_dir / "most_important_tree.png"
        
        try:
            export_tree_to_png(
                rf_classifier.estimators_[best_tree_idx],
                important_tree_png_file,
                feature_names,
                max_depth=4  # Slightly deeper for the important tree
            )
        except Exception as e:
            print(f"Error exporting most important tree to PNG: {e}")
            # Fallback to DOT file
            important_tree_dot_file = trees_dir / "most_important_tree.dot"
            export_graphviz(
                rf_classifier.estimators_[best_tree_idx],
                out_file=str(important_tree_dot_file),
                feature_names=feature_names,
                class_names=['No Station', 'Has Station'],
                filled=True,
                rounded=True,
                precision=2,
                max_depth=4
            )
            print(f"Fallback: Most important tree visualization saved as DOT file to {important_tree_dot_file}")


def run_model_training_pipeline():
    """
    Run the complete model training pipeline with city-level evaluation.
    """
    print("=== Starting Model Training Pipeline ===")
    
    # Load data
    features_gdf = load_feature_data()
    
    # Prepare data with city-level split
    X_train, X_test, y_train, y_test, feature_names, train_cities, test_cities = prepare_data_for_training(features_gdf)
    
    # Train model
    model, grid_search, feature_importances = train_model(X_train, y_train, feature_names)
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Additional city-specific evaluation
    print("\n=== CITY-SPECIFIC EVALUATION ===")
    test_data = features_gdf[features_gdf['city'].isin(test_cities)].copy()
    test_data = test_data.dropna(subset=['population', 'has_station', 'city'])
    
    city_metrics = {}
    for city in test_cities:
        city_data = test_data[test_data['city'] == city]
        if len(city_data) > 0:
            X_city = city_data[feature_names]
            y_city = city_data['has_station']
            
            y_city_pred = model.predict(X_city)
            y_city_proba = model.predict_proba(X_city)[:, 1]
            
            city_metrics[city] = {
                'accuracy': accuracy_score(y_city, y_city_pred),
                'precision': precision_score(y_city, y_city_pred, zero_division=0),
                'recall': recall_score(y_city, y_city_pred, zero_division=0),
                'f1': f1_score(y_city, y_city_pred, zero_division=0),
                'samples': len(city_data),
                'stations': y_city.sum(),
                'avg_probability': y_city_proba.mean()
            }
            
            print(f"\n{city.upper()} Performance:")
            print(f"  Samples: {city_metrics[city]['samples']:,}")
            print(f"  Stations: {city_metrics[city]['stations']:,}")
            print(f"  Accuracy: {city_metrics[city]['accuracy']:.3f}")
            print(f"  Precision: {city_metrics[city]['precision']:.3f}")
            print(f"  Recall: {city_metrics[city]['recall']:.3f}")
            print(f"  F1-Score: {city_metrics[city]['f1']:.3f}")
            print(f"  Avg Probability: {city_metrics[city]['avg_probability']:.3f}")
    
    # Save results with city information
    save_results(model, feature_importances, metrics, X_test, y_test, y_pred, train_cities, test_cities, city_metrics)
    
    print("\n=== Model Training Pipeline Complete ===")
    print(f"Training cities: {', '.join(train_cities)}")
    print(f"Test cities: {', '.join(test_cities)}")
    
    return model


if __name__ == '__main__':
    run_model_training_pipeline()
