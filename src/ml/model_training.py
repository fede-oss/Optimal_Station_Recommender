# Enhanced model training with BalancedRandomForestClassifier + RandomizedSearchCV
# This implementation focuses on better class imbalance handling and faster hyperparameter search

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import os
import time
from datetime import datetime

# Enhanced imports for improved training
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, 
    average_precision_score, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz


# Define directories (consistent with other scripts)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
    # Define city splits (same as baseline for fair comparison)
    TRAIN_CITIES = [
        'paris', 'berlin', 'madrid', 'stockholm', 'rome', 
        'seoul', 'singapore', 'hong_kong', 'new_york', 
        'mexico_city', 'sao_paulo', 'buenos_aires'
    ]
    
    TEST_CITIES = ['warsaw', 'toronto', 'sydney']
    
    print("\n--- ENHANCED CITY-LEVEL TRAIN/TEST SPLIT ---")
    print(f"Training cities ({len(TRAIN_CITIES)}): {', '.join(TRAIN_CITIES)}")
    print(f"Testing cities ({len(TEST_CITIES)}): {', '.join(TEST_CITIES)}")
    
    # Convert population column to numeric if needed
    print("\n--- POPULATION DATA PROCESSING ---")
    print(f"Population column data type: {features_gdf['population'].dtype}")
    if features_gdf['population'].dtype == 'object':
        print("Converting population column to numeric...")
        features_gdf['population'] = pd.to_numeric(features_gdf['population'], errors='coerce')
        features_gdf['population'] = features_gdf['population'].fillna(0)
        print(f"Conversion complete. New data type: {features_gdf['population'].dtype}")
    
    # Data quality summary
    print(f"Total cells: {len(features_gdf):,}")
    print(f"Cells with population data: {features_gdf['population'].notna().sum():,}")
    print(f"Population statistics:\n{features_gdf['population'].describe()}")
    
    # Class distribution analysis
    print(f"\n--- CLASS DISTRIBUTION ANALYSIS ---")
    total_stations = features_gdf['has_station'].sum()
    total_cells = len(features_gdf)
    station_rate = total_stations / total_cells
    print(f"Total stations: {total_stations:,}")
    print(f"Total cells: {total_cells:,}")
    print(f"Station density: {station_rate:.4f} ({station_rate*100:.2f}%)")
    print(f"Class imbalance ratio: {(1-station_rate)/station_rate:.1f}:1 (negative:positive)")
    
    # Verify cities exist in data
    available_cities = set(features_gdf['city'].unique())
    missing_train_cities = set(TRAIN_CITIES) - available_cities
    missing_test_cities = set(TEST_CITIES) - available_cities
    
    if missing_train_cities:
        print(f"Warning: Missing training cities: {missing_train_cities}")
        TRAIN_CITIES = [city for city in TRAIN_CITIES if city in available_cities]
    
    if missing_test_cities:
        print(f"Warning: Missing test cities: {missing_test_cities}")
        TEST_CITIES = [city for city in TEST_CITIES if city in available_cities]
    
    # Split data by cities
    train_data = features_gdf[features_gdf['city'].isin(TRAIN_CITIES)].copy()
    test_data = features_gdf[features_gdf['city'].isin(TEST_CITIES)].copy()
    
    # Clean data
    required_columns = ['population', 'has_station', 'city']
    train_data = train_data.dropna(subset=required_columns)
    test_data = test_data.dropna(subset=required_columns)
    
    print(f"\n--- FINAL DATASET DISTRIBUTION ---")
    print(f"Training data: {len(train_data):,} cells from {len(TRAIN_CITIES)} cities")
    print(f"Test data: {len(test_data):,} cells from {len(TEST_CITIES)} cities")
    
    # Training set class distribution
    train_stations = train_data['has_station'].sum()
    train_rate = train_stations / len(train_data)
    print(f"Training set: {train_stations:,} stations ({train_rate:.3f} rate)")
    
    # Test set class distribution  
    test_stations = test_data['has_station'].sum()
    test_rate = test_stations / len(test_data)
    print(f"Test set: {test_stations:,} stations ({test_rate:.3f} rate)")
    
    # Select feature columns
    feature_columns = [col for col in features_gdf.columns 
                      if col not in ['geometry', 'h3_index', 'has_station', 'city']]
    
    print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")
    
    # Create feature matrices and target vectors
    X_train = train_data[feature_columns]
    y_train = train_data['has_station']
    X_test = test_data[feature_columns]
    y_test = test_data['has_station']
    
    return X_train, X_test, y_train, y_test, feature_columns, TRAIN_CITIES, TEST_CITIES


def train_enhanced_model(X_train, y_train, feature_names):
    """
    Train a BalancedRandomForestClassifier with RandomizedSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        feature_names (list): Names of the feature columns
        
    Returns:
        tuple: Best model, search results, and feature importances
    """
    print("\n" + "="*60)
    print("ENHANCED MODEL TRAINING")
    print("BalancedRandomForestClassifier + RandomizedSearchCV")
    print("="*60)
    
    start_time = time.time()
    
    # Create pipeline with BalancedRandomForestClassifier
    pipeline = Pipeline([
        ('classifier', BalancedRandomForestClassifier(
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            oob_score=True,  # Out-of-bag scoring
            sampling_strategy='auto',  # Automatic class balancing
            replacement=False  # Sample without replacement for diversity
        ))
    ])
    
    # Enhanced parameter distributions for RandomizedSearch
    print("\nParameter search space:")
    param_distributions = {
        'classifier__n_estimators': randint(100, 500),  # 100-499 trees
        'classifier__max_depth': randint(10, 50),  # 10-49 depth
        'classifier__min_samples_split': randint(2, 20),  # 2-19 samples
        'classifier__min_samples_leaf': randint(1, 10),  # 1-9 samples  
        'classifier__max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],  # Feature randomness
        'classifier__criterion': ['gini', 'entropy'],  # Split quality
        'classifier__bootstrap': [True, False],  # Sampling strategy
        'classifier__max_samples': [None, 0.7, 0.8, 0.9]  # Subsample size per tree
    }
    
    for param, distribution in param_distributions.items():
        print(f"  {param}: {distribution}")
    
    # RandomizedSearchCV - much faster than GridSearchCV
    print(f"\nStarting RandomizedSearchCV with 250 parameter combinations...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=250,  # Test 250 combinations (vs 1296 in GridSearch)
        cv=5,  # 5-fold cross-validation
        scoring='f1',  # Optimize for F1-score (good for imbalanced data)
        n_jobs=-1,  # Parallel processing
        verbose=1,  # Progress updates
        random_state=42,  # Reproducible results
        return_train_score=True  # Get training scores too
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Get best model
    best_model = random_search.best_estimator_
    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation F1 score: {random_search.best_score_:.4f}")
    
    # BalancedRandomForest specific information
    brf_model = best_model.named_steps['classifier']
    
    # Out-of-bag score
    if hasattr(brf_model, 'oob_score_'):
        print(f"Out-of-bag score: {brf_model.oob_score_:.4f}")
    
    # Sampling information
    print(f"\nBalanced sampling information:")
    print(f"  Number of estimators: {brf_model.n_estimators}")
    print(f"  Sampling strategy: {brf_model.sampling_strategy}")
    print(f"  Replacement: {brf_model.replacement}")
    
    # Feature importances
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': brf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances:")
    for i, (_, row) in enumerate(feature_importances.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['Feature']:35s} {row['Importance']:.4f}")
    
    return best_model, random_search, feature_importances


def evaluate_enhanced_model(model, X_test, y_test):
    """
    Comprehensive evaluation of the enhanced model.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("ENHANCED MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    # Print results with enhanced formatting
    print(f"\n🎯 PERFORMANCE METRICS")
    print(f"{'Metric':<15} {'Score':<8} {'Interpretation'}")
    print(f"{'-'*15} {'-'*8} {'-'*30}")
    print(f"{'Accuracy':<15} {metrics['accuracy']:<8.4f} Overall correctness")
    print(f"{'Precision':<15} {metrics['precision']:<8.4f} Positive prediction accuracy")
    print(f"{'Recall':<15} {metrics['recall']:<8.4f} Actual positive detection")
    print(f"{'F1-Score':<15} {metrics['f1']:<8.4f} Balanced precision/recall")
    print(f"{'ROC-AUC':<15} {metrics['roc_auc']:<8.4f} Overall discrimination")
    print(f"{'PR-AUC':<15} {metrics['pr_auc']:<8.4f} Imbalanced data performance")
    
    # Class distribution context
    positive_rate = y_test.mean()
    negative_rate = 1 - positive_rate
    print(f"\n📊 TEST SET DISTRIBUTION")
    print(f"Positive class (stations): {positive_rate:.3f} ({positive_rate*100:.1f}%)")
    print(f"Negative class (no station): {negative_rate:.3f} ({negative_rate*100:.1f}%)")
    print(f"Class imbalance ratio: {negative_rate/positive_rate:.1f}:1")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🔍 CONFUSION MATRIX ANALYSIS")
    print(f"True Negatives (TN):  {tn:,} - Correctly predicted no station")
    print(f"False Positives (FP): {fp:,} - Incorrectly predicted station")
    print(f"False Negatives (FN): {fn:,} - Missed actual stations")
    print(f"True Positives (TP):  {tp:,} - Correctly predicted station")
    
    # Prediction distribution analysis
    print(f"\n📈 PREDICTION DISTRIBUTION")
    print(f"Predicted stations: {(y_pred == 1).sum():,}")
    print(f"Actual stations: {(y_test == 1).sum():,}")
    print(f"Average prediction probability: {y_pred_proba.mean():.4f}")
    print(f"Std prediction probability: {y_pred_proba.std():.4f}")
    
    return metrics, y_pred, y_pred_proba


def save_enhanced_results(model, feature_importances, metrics, X_test, y_test, y_pred, y_pred_proba, 
                         search_results, train_cities, test_cities, city_metrics=None):
    """
    Save comprehensive results from enhanced training.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 SAVING ENHANCED RESULTS")
    
    # Save model with enhanced naming
    model_file = MODELS_DIR / f"balanced_rf_h{H3_RESOLUTION}_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"✅ Model saved: {model_file}")
    
    # Save feature importances
    importance_file = RESULTS_DIR / f"feature_importances_balanced_h{H3_RESOLUTION}_{timestamp}.csv"
    feature_importances.to_csv(importance_file, index=False)
    print(f"✅ Feature importances: {importance_file}")
    
    # Save evaluation metrics
    metrics_file = RESULTS_DIR / f"evaluation_metrics_balanced_h{H3_RESOLUTION}_{timestamp}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"✅ Evaluation metrics: {metrics_file}")
    
    # Save hyperparameter search results
    search_results_df = pd.DataFrame(search_results.cv_results_)
    search_file = RESULTS_DIR / f"hyperparameter_search_balanced_h{H3_RESOLUTION}_{timestamp}.csv"
    search_results_df.to_csv(search_file, index=False)
    print(f"✅ Search results: {search_file}")
    
    # Save city-specific metrics if provided
    if city_metrics:
        city_file = RESULTS_DIR / f"city_metrics_balanced_h{H3_RESOLUTION}_{timestamp}.csv"
        pd.DataFrame.from_dict(city_metrics, orient='index').to_csv(city_file)
        print(f"✅ City metrics: {city_file}")
    
    # Enhanced training configuration
    config_file = RESULTS_DIR / f"training_config_balanced_h{H3_RESOLUTION}_{timestamp}.txt"
    with open(config_file, 'w') as f:
        f.write(f"Enhanced Training Configuration - H3 Resolution {H3_RESOLUTION}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: BalancedRandomForestClassifier\n")
        f.write(f"Search: RandomizedSearchCV (250 iterations)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Split Method: City-level split\n\n")
        
        f.write(f"Training Cities ({len(train_cities)}):\n")
        f.write(f"  {', '.join(train_cities)}\n\n")
        
        f.write(f"Test Cities ({len(test_cities)}):\n")
        f.write(f"  {', '.join(test_cities)}\n\n")
        
        f.write("Best Parameters:\n")
        for param, value in search_results.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
    
    print(f"✅ Training config: {config_file}")
    
    # Create enhanced visualizations
    create_enhanced_visualizations(feature_importances, metrics, y_test, y_pred, y_pred_proba, timestamp)
    
    return {
        'model_file': model_file,
        'timestamp': timestamp,
        'config_file': config_file
    }


def create_enhanced_visualizations(feature_importances, metrics, y_test, y_pred, y_pred_proba, timestamp):
    """
    Create enhanced visualizations for the balanced model results.
    """
    print(f"\n🎨 CREATING ENHANCED VISUALIZATIONS")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Enhanced Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importances.head(10)
    colors = sns.color_palette("viridis", len(top_features))
    
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances - BalancedRandomForest', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    importance_plot = RESULTS_DIR / f"feature_importance_balanced_h{H3_RESOLUTION}_{timestamp}.png"
    plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot: {importance_plot}")
    
    # 2. Enhanced Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations combining counts and percentages
    annotations = np.array([[f'{cm[i,j]:,}\n({cm_percent[i,j]:.1f}%)' 
                           for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['No Station', 'Has Station'],
                yticklabels=['No Station', 'Has Station'],
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Actual Class', fontsize=12)
    plt.title('Enhanced Confusion Matrix - BalancedRandomForest', fontsize=14, fontweight='bold')
    
    confusion_plot = RESULTS_DIR / f"confusion_matrix_balanced_h{H3_RESOLUTION}_{timestamp}.png"
    plt.savefig(confusion_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix plot: {confusion_plot}")
    
    # 3. Prediction Probability Distribution
    plt.figure(figsize=(12, 6))
    
    # Plot distributions for both classes
    plt.subplot(1, 2, 1)
    no_station_probs = y_pred_proba[y_test == 0]
    station_probs = y_pred_proba[y_test == 1]
    
    plt.hist(no_station_probs, bins=50, alpha=0.7, label='No Station', color='lightcoral', density=True)
    plt.hist(station_probs, bins=50, alpha=0.7, label='Has Station', color='skyblue', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot precision-recall curve
    plt.subplot(1, 2, 2)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color='darkorange', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.3, color='darkorange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n(AUC = {metrics["pr_auc"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    prob_plot = RESULTS_DIR / f"probability_analysis_balanced_h{H3_RESOLUTION}_{timestamp}.png"
    plt.savefig(prob_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Probability analysis plot: {prob_plot}")


def run_enhanced_training_pipeline():
    """
    Run the complete enhanced training pipeline.
    """
    print("🚀 ENHANCED MODEL TRAINING PIPELINE")
    print("BalancedRandomForestClassifier + RandomizedSearchCV")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load data
    print("\n📊 Loading feature data...")
    features_gdf = load_feature_data()
    
    # Prepare data
    print("\n🔄 Preparing data for training...")
    X_train, X_test, y_train, y_test, feature_names, train_cities, test_cities = prepare_data_for_training(features_gdf)
    
    # Train enhanced model
    print("\n🤖 Training enhanced model...")
    model, search_results, feature_importances = train_enhanced_model(X_train, y_train, feature_names)
    
    # Evaluate model
    print("\n📊 Evaluating enhanced model...")
    metrics, y_pred, y_pred_proba = evaluate_enhanced_model(model, X_test, y_test)
    
    # City-specific evaluation
    print("\n🏙️ City-specific evaluation...")
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
                'roc_auc': roc_auc_score(y_city, y_city_proba) if len(y_city.unique()) > 1 else 0,
                'pr_auc': average_precision_score(y_city, y_city_proba) if len(y_city.unique()) > 1 else 0,
                'samples': len(city_data),
                'stations': y_city.sum(),
                'avg_probability': y_city_proba.mean()
            }
            
            print(f"\n{city.upper()} Performance:")
            print(f"  📊 Samples: {city_metrics[city]['samples']:,}")
            print(f"  🚉 Stations: {city_metrics[city]['stations']:,}")
            print(f"  🎯 F1-Score: {city_metrics[city]['f1']:.3f}")
            print(f"  🔍 Precision: {city_metrics[city]['precision']:.3f}")
            print(f"  📈 Recall: {city_metrics[city]['recall']:.3f}")
            print(f"  📊 PR-AUC: {city_metrics[city]['pr_auc']:.3f}")
    
    # Save all results
    print("\n💾 Saving results...")
    save_info = save_enhanced_results(
        model, feature_importances, metrics, X_test, y_test, y_pred, y_pred_proba,
        search_results, train_cities, test_cities, city_metrics
    )
    
    total_time = time.time() - start_time
    print(f"\n✅ ENHANCED TRAINING PIPELINE COMPLETE")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🎯 Best F1-Score: {metrics['f1']:.4f}")
    print(f"📈 Best PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"💾 Results timestamp: {save_info['timestamp']}")
    
    return model, save_info


if __name__ == '__main__':
    run_enhanced_training_pipeline()
