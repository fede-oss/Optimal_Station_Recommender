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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
import graphviz

# Define directories (consistent with other scripts)
BASE_DIR = Path(__file__).resolve().parent.parent  # Should point to Optimal_Station_Recommender directory
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
    Prepare the feature data for model training.
    
    Args:
        features_gdf (gpd.GeoDataFrame): The combined feature data
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
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
    required_columns = ['population', 'has_station']
    # Check for at least one amenity column
    has_amenity_columns = any(col.startswith('count_amenity_') for col in features_gdf.columns)
    if not has_amenity_columns:
        required_columns.append('at least one amenity count column')
    
    missing_columns = [col for col in required_columns if col not in features_gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Drop any rows with NaN values
    features_gdf = features_gdf.dropna(subset=required_columns)
    print(f"Using {len(features_gdf)} valid cells after dropping NaN values")
    
    # Select feature columns (excluding geometry, h3_index, and target variable)
    feature_columns = [col for col in features_gdf.columns 
                      if col not in ['geometry', 'h3_index', 'has_station', 'city']]
    
    # Print data overview
    print(f"Feature columns: {feature_columns}")
    print("Target distribution:")
    print(features_gdf['has_station'].value_counts())
    
    # Split into features (X) and target (y)
    X = features_gdf[feature_columns]
    y = features_gdf['has_station']
    
    # Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_columns


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
    
    # Create a pipeline with scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
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
    
    # Get feature importances
    rf_model = best_model.named_steps['classifier']
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importances)
    
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
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Print metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred


def save_results(model, feature_importances, metrics, X_test, y_test, y_pred):
    """
    Save the model, feature importances, and evaluation results.
    
    Args:
        model: Trained model
        feature_importances (pd.DataFrame): Feature importances
        metrics (dict): Evaluation metrics
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        y_pred (np.array): Predicted values
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
    Run the complete model training pipeline.
    """
    print("=== Starting Model Training Pipeline ===")
    
    # Load data
    features_gdf = load_feature_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_training(features_gdf)
    
    # Train model
    model, grid_search, feature_importances = train_model(X_train, y_train, feature_names)
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Save results
    save_results(model, feature_importances, metrics, X_test, y_test, y_pred)
    
    print("\n=== Model Training Pipeline Complete ===")
    
    return model


if __name__ == '__main__':
    run_model_training_pipeline()
