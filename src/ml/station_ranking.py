#!/usr/bin/env python3
"""
station_ranking.py - Station Ranking System

This script builds on the trained model from model_training.py but instead of binary
classification, it generates a ranking of potential station locations based on
probability scores. The output is a ranked list of H3 cells and visualizations
including heatmaps of station suitability.

Differences from model_training.py:
- Uses predict_proba() instead of predict() to get probability scores
- Ranks cells by probability (creates a suitability score)
- Generates heatmaps and visualizations of potential station locations
- Exports ranked locations for further analysis and planning
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import folium
from folium.plugins import HeatMap
from branca.colormap import linear
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import contextily as ctx
from shapely.geometry import Point
import h3
from pathlib import Path
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RANKING_DIR = RESULTS_DIR / "rankings"
HEATMAPS_DIR = RANKING_DIR / "heatmaps"

# Create directories if they don't exist
RANKING_DIR.mkdir(exist_ok=True, parents=True)
HEATMAPS_DIR.mkdir(exist_ok=True, parents=True)

def load_data_and_model(model_path=None, h3_resolution=9):
    """
    Load the features data and the trained model.
    
    Parameters:
    -----------
    model_path : str or Path, optional
        Specific model path to load. If None, uses the most recent balanced_rf model
    h3_resolution : int
        H3 resolution used for the grid
        
    Returns:
    --------
    tuple
        (features_gdf, model, feature_columns, model_name) - GeoDataFrame with features, trained model, feature columns, and model name
    """
    # Load feature data
    features_path = PROCESSED_DATA_DIR / f"all_cities_features_h{h3_resolution}.gpkg"
    print(f"Loading feature data from {features_path}")
    features_gdf = gpd.read_file(features_path)
    
    # Load trained model
    if model_path is None:
        # Use the most recent balanced_rf model by default
        model_files = list(MODELS_DIR.glob(f"balanced_rf_h{h3_resolution}_*.joblib"))
        if model_files:
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        else:
            model_path = MODELS_DIR / f"station_recommender_h{h3_resolution}.joblib"
    else:
        model_path = Path(model_path)
    
    print(f"Loading trained model from {model_path}")
    model = load(model_path)
    
    # Extract model name from filename (without extension)
    model_name = model_path.stem
    
    # Define feature columns (excluding non-feature columns)
    feature_columns = [col for col in features_gdf.columns 
                      if col not in ['geometry', 'hex_id', 'city', 'lat', 'lng']]
    
    return features_gdf, model, feature_columns, model_name

def get_all_available_models(h3_resolution):
    """
    Get all available models for the given H3 resolution.
    
    Parameters:
    -----------
    h3_resolution : int
        H3 resolution used for the grid
        
    Returns:
    --------
    list
        List of model file paths
    """
    model_files = []
    
    # Look for all model patterns
    patterns = [
        f"balanced_rf_h{h3_resolution}_*.joblib",
        f"station_recommender_h{h3_resolution}_*.joblib",
        f"station_recommender_h{h3_resolution}.joblib"
    ]
    
    for pattern in patterns:
        model_files.extend(MODELS_DIR.glob(pattern))
    
    # Remove duplicates and sort by modification time (newest first)
    model_files = list(set(model_files))
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return model_files

def prepare_data_for_ranking(features_gdf):
    """
    Prepare data for ranking by handling data types and preparing feature columns.
    Similar to prepare_data_for_training but without train/test split.
    
    Parameters:
    -----------
    features_gdf : GeoDataFrame
        GeoDataFrame with features
        
    Returns:
    --------
    tuple
        (X, feature_columns) - Feature matrix and list of feature column names
    """
    print("Preparing data for ranking...")
    
    # Convert population to numeric if it's stored as string
    if features_gdf['population'].dtype == 'object':
        print("Converting population column from object to numeric")
        features_gdf['population'] = pd.to_numeric(features_gdf['population'], errors='coerce')
        features_gdf['population'] = features_gdf['population'].fillna(0)
    
    # Drop rows with NaN values
    features_gdf = features_gdf.dropna()
    print(f"Using {len(features_gdf)} valid cells after dropping NaN values")
    
    # Define feature columns dynamically (same as in model_training.py)
    feature_columns = [col for col in features_gdf.columns 
                      if col not in ['geometry', 'h3_index', 'has_station', 'city']]
    
    # Create feature matrix
    X = features_gdf[feature_columns].copy()
    
    return X, feature_columns

def generate_station_rankings(features_gdf, model, feature_columns, h3_resolution, model_name, model_dir):
    """
    Generate rankings for potential station locations based on model probabilities.
    
    Parameters:
    -----------
    features_gdf : GeoDataFrame
        GeoDataFrame with features
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    feature_columns : list
        List of feature column names
    h3_resolution : int
        H3 resolution used for the grid
    model_name : str
        Name of the model for file naming
    model_dir : Path
        Directory to save model-specific results
        
    Returns:
    --------
    GeoDataFrame
        GeoDataFrame with station probability scores and rankings
    """
    print(f"Generating station rankings for {model_name}...")
    
    # Create a copy of the features GeoDataFrame
    ranked_gdf = features_gdf.copy()
    
    # Get feature matrix
    X = ranked_gdf[feature_columns]
    
    # Generate probability scores (probability of class 1 - station)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add probability scores to GeoDataFrame
    ranked_gdf['station_probability'] = probabilities
    
    # Add ranking (1 is highest probability)
    ranked_gdf['station_rank'] = ranked_gdf['station_probability'].rank(ascending=False, method='min')
    
    # Sort by probability (highest first)
    ranked_gdf = ranked_gdf.sort_values('station_probability', ascending=False)
    
    # Calculate percentile ranks (0-100 scale, 100 is best)
    ranked_gdf['percentile_rank'] = 100 - (ranked_gdf['station_rank'] / len(ranked_gdf) * 100)
    
    # Create station_rankings subfolder
    rankings_dir = model_dir / "station_rankings"
    rankings_dir.mkdir(exist_ok=True)
    
    # Save ranked GeoDataFrame in station_rankings subfolder
    output_path = rankings_dir / f"station_rankings_h{h3_resolution}_{model_name}.gpkg"
    ranked_gdf.to_file(output_path, driver="GPKG")
    print(f"Station rankings saved to {output_path}")
    
    # Also save a CSV version for easier analysis
    csv_path = rankings_dir / f"station_rankings_h{h3_resolution}_{model_name}.csv"
    ranked_gdf.drop('geometry', axis=1).to_csv(csv_path, index=False)
    print(f"Station rankings CSV saved to {csv_path}")
    
    return ranked_gdf

def create_static_heatmap(ranked_gdf, h3_resolution, model_name, model_dir, city=None):
    """
    Create a static heatmap of station probability scores.
    
    Parameters:
    -----------
    ranked_gdf : GeoDataFrame
        GeoDataFrame with station probability scores
    h3_resolution : int
        H3 resolution used for the grid
    model_name : str
        Name of the model for file naming
    model_dir : Path
        Directory to save model-specific results
    city : str, optional
        City name to filter data (if None, use all cities)
    """
    print(f"Creating static heatmap for {model_name}{'' if city is None else f' - {city}'}")
    
    # Filter by city if specified
    if city is not None and 'city' in ranked_gdf.columns:
        city_gdf = ranked_gdf[ranked_gdf['city'] == city.lower()]
        if len(city_gdf) == 0:
            print(f"No data found for city: {city}")
            return
    else:
        city_gdf = ranked_gdf
    
    # For global heatmap (all cities), create a map showing top locations only
    # to avoid the issue of spanning continents
    if city is None:
        print(f"Creating global overview with top 1000 highest-scoring locations...")
        # Get top 1000 locations globally for better visualization
        city_gdf = city_gdf.sort_values('station_probability', ascending=False).head(1000)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot hexagons colored by probability
    city_gdf.plot(
        column='station_probability',
        ax=ax,
        cmap='YlOrRd',
        legend=True,
        alpha=0.8,
        legend_kwds={'label': 'Station Suitability Score', 'orientation': 'horizontal'},
        markersize=50 if city is None else None  # Larger markers for global view
    )
    
    # Add basemap (skip for global view to avoid projection issues)
    if city is not None:
        try:
            ctx.add_basemap(ax, crs=city_gdf.crs.to_string())
        except Exception as e:
            print(f"Could not add basemap: {e}")
    
    # Add title and labels
    title = f"Station Suitability Heatmap - {model_name}"
    if city is not None:
        title += f" - {city.capitalize()}"
    else:
        title += " - Global Top 1000 Locations"
    ax.set_title(title, fontsize=16, pad=20)
    
    # Set axis labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Create heatmaps subfolder if it doesn't exist
    heatmaps_dir = model_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)
    
    # Save figure - ALWAYS put in heatmaps subfolder for consistency
    filename = f"station_heatmap_h{h3_resolution}_{model_name}"
    if city is not None:
        filename += f"_{city.lower()}"
    else:
        filename += "_global"
    
    output_path = heatmaps_dir / f"{filename}.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    plt.close()

def create_interactive_map(ranked_gdf, h3_resolution, model_name, model_dir, city=None):
    """
    Create an interactive folium map with station probability heatmap.
    
    Parameters:
    -----------
    ranked_gdf : GeoDataFrame
        GeoDataFrame with station probability scores
    h3_resolution : int
        H3 resolution used for the grid
    model_name : str
        Name of the model for file naming
    model_dir : Path
        Directory to save model-specific results
    city : str, optional
        City name to filter data (if None, use all cities)
    """
    print(f"Creating interactive map for {model_name}{'s' if city is None else f' - {city}'}")
    
    # Filter by city if specified
    if city is not None and 'city' in ranked_gdf.columns:
        city_gdf = ranked_gdf[ranked_gdf['city'] == city.lower()]
        if len(city_gdf) == 0:
            print(f"No data found for city: {city}")
            return
    else:
        city_gdf = ranked_gdf
    
    # Convert to EPSG:4326 for folium
    if city_gdf.crs != 'EPSG:4326':
        city_gdf = city_gdf.to_crs('EPSG:4326')
    
    # Use the bounds to determine the center of the map (avoids centroid calculation warnings)
    bounds = city_gdf.total_bounds  # (min_x, min_y, max_x, max_y)
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    center = [center_lat, center_lon]
    
    # Create a base map without standard controls
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles=None)
    
    # Create layer groups for controls
    fg_info = folium.FeatureGroup(name="Cell Information", show=True)
    fg_actual_stations = folium.FeatureGroup(name="Actual Stations", show=False) # Default hidden
    
    # Create a choropleth layer for the heatmap (must be added directly to the map)
    choropleth = folium.Choropleth(
        geo_data=city_gdf.__geo_interface__,
        data=city_gdf,
        columns=['h3_index', 'station_probability'],
        key_on='feature.properties.h3_index',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Station Suitability Score',
        name='Station Suitability Heatmap',
        highlight=True
    ).add_to(m)
    
    # Add a GeoJSON layer with popup info for cells
    folium.GeoJson(
        city_gdf,
        tooltip=folium.GeoJsonTooltip(
            fields=['h3_index', 'station_probability', 'station_rank', 'population'],
            aliases=['H3 Index', 'Suitability Score', 'Rank', 'Population'],
            localize=True,
            sticky=False
        ),
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'transparent',
            'fillOpacity': 0.0,
            'weight': 0
        }
    ).add_to(fg_info)
    
    # Add actual station locations
    # If city is specified, only add that city's stations
    # Otherwise, add stations from all cities
    cities_to_load = [city.lower()] if city is not None else ['london', 'paris', 'berlin']
    
    total_stations = 0
    
    for current_city in cities_to_load:
        # Path to station data
        station_file = DATA_RAW_DIR / f"{current_city}_stations.geojson"
        
        # Check if station data exists
        if station_file.exists():
            try:
                # Load station data
                stations_gdf = gpd.read_file(station_file)
                
                # Ensure it's in the right CRS
                if stations_gdf.crs != 'EPSG:4326':
                    stations_gdf = stations_gdf.to_crs('EPSG:4326')
                
                # Add station points to the map
                for _, station in stations_gdf.iterrows():
                    city_label = f"[{current_city.capitalize()}] " if city is None else ""
                    popup_text = f"{city_label}Station: {station.get('name', 'Unknown')}"
                    
                    # Handle different geometry types
                    if station.geometry.geom_type == 'Point':
                        # For point geometries
                        folium.CircleMarker(
                            location=[station.geometry.y, station.geometry.x],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.7,
                            popup=popup_text
                        ).add_to(fg_actual_stations)
                    else:
                        # For polygon or other geometries, use the centroid
                        centroid = station.geometry.centroid
                        folium.CircleMarker(
                            location=[centroid.y, centroid.x],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.7,
                            popup=popup_text
                        ).add_to(fg_actual_stations)
                
                total_stations += len(stations_gdf)
                print(f"Added {len(stations_gdf)} stations from {current_city}")
            except Exception as e:
                print(f"Error loading {current_city} station data: {e}")
        else:
            print(f"No station data found at {station_file}")
    
    if total_stations > 0:
        print(f"Total of {total_stations} actual stations added to the map")
    
    # Add all feature groups to the map
    fg_info.add_to(m)
    fg_actual_stations.add_to(m)
    
    # Add CartoDB positron tile layer as a base layer but with custom name and no option to toggle
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Base Map',  # This name should be hidden in controls
        overlay=False,
        control=False,
    ).add_to(m)
    
    # Add custom layer control that only shows overlay layers
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    
    # Add custom JavaScript to remove all base layer controls
    custom_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Remove base layer section from layer control
            setTimeout(function() {
                var baseLayersDiv = document.querySelector('.leaflet-control-layers-base');
                if (baseLayersDiv) {
                    baseLayersDiv.style.display = 'none';
                }
                
                // Also hide separator if it exists
                var separator = document.querySelector('.leaflet-control-layers-separator');
                if (separator) {
                    separator.style.display = 'none';
                }
            }, 100);
        });
        </script>
    """
    m.get_root().html.add_child(folium.Element(custom_js))
    
    # Add a title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 450px; height: 65px; 
                background-color: white; border-radius: 5px; padding: 10px; 
                z-index: 9999; font-size: 16px; font-weight: bold;">
        Station Suitability Map - {} {}
        <div style="font-size: 12px; font-weight: normal;">
            Model: {}<br>
            Toggle layers using the control in the top-right corner
        </div>
    </div>
    '''.format(f'{city.capitalize()}' if city else 'All Cities', '', model_name)
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create interactive_maps subfolder if it doesn't exist
    interactive_dir = model_dir / "interactive_maps"
    interactive_dir.mkdir(exist_ok=True)
    
    # Save map
    filename = f"interactive_station_map_h{h3_resolution}_{model_name}"
    if city is not None:
        filename += f"_{city.lower()}"
    output_path = interactive_dir / f"{filename}.html"
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")
    
    return output_path

def export_top_stations(ranked_gdf, top_n, h3_resolution, model_name, model_dir, city=None):
    """
    Export the top N potential station locations in GeoJSON format.
    
    Parameters:
    -----------
    ranked_gdf : GeoDataFrame
        GeoDataFrame with station probability scores
    top_n : int
        Number of top-ranked stations to export
    h3_resolution : int
        H3 resolution used for the grid
    model_name : str
        Name of the model for file naming
    model_dir : Path
        Directory to save model-specific results
    city : str, optional
        City name to filter data (if None, use all cities)
    """
    print(f"Exporting top {top_n} potential station locations for {model_name}")
    
    # Filter by city if specified
    if city is not None and 'city' in ranked_gdf.columns:
        city_gdf = ranked_gdf[ranked_gdf['city'] == city.lower()]
        if len(city_gdf) == 0:
            print(f"No data found for city: {city}")
            return
    else:
        city_gdf = ranked_gdf
    
    # Get top N cells
    top_stations = city_gdf.sort_values('station_probability', ascending=False).head(top_n)
    
    # Convert to EPSG:4326 for GeoJSON
    if top_stations.crs != 'EPSG:4326':
        top_stations = top_stations.to_crs('EPSG:4326')
    
    # Add h3 cell center point (for easier visualization)
    top_stations['center_point'] = top_stations['h3_index'].apply(
        lambda idx: Point(h3.cell_to_latlng(idx)[::-1])
    )
    
    # Create a GeoDataFrame from center points
    center_points_gdf = gpd.GeoDataFrame(
        top_stations.drop('geometry', axis=1),
        geometry='center_point',
        crs='EPSG:4326'
    )
    
    # Create station_rankings subfolder if it doesn't exist
    rankings_dir = model_dir / "station_rankings"
    rankings_dir.mkdir(exist_ok=True)
    
    # Save as GeoJSON in station_rankings folder
    filename = f"top_{top_n}_stations_h{h3_resolution}_{model_name}"
    if city is not None:
        filename += f"_{city.lower()}"
    
    # Save hexagons
    hex_output_path = rankings_dir / f"{filename}_hexagons.geojson"
    top_stations.drop('center_point', axis=1).to_file(hex_output_path, driver='GeoJSON')
    print(f"Top station hexagons saved to {hex_output_path}")
    
    # Save center points
    points_output_path = rankings_dir / f"{filename}_points.geojson"
    center_points_gdf.to_file(points_output_path, driver='GeoJSON')
    print(f"Top station points saved to {points_output_path}")

def run_model_comparison_pipeline(h3_resolution=9):
    """
    Run the complete station ranking pipeline for all available models.
    
    Parameters:
    -----------
    h3_resolution : int
        H3 resolution used for the grid
    """
    print("🚀 MULTI-MODEL STATION RANKING COMPARISON PIPELINE")
    print("=" * 60)
    
    # Get all available models
    model_files = get_all_available_models(h3_resolution)
    
    if not model_files:
        print(f"❌ No models found for H3 resolution {h3_resolution}")
        return
    
    print(f"📊 Found {len(model_files)} models to compare:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file.name}")
    
    # Load feature data once (same for all models)
    print(f"\n📊 Loading feature data...")
    features_path = PROCESSED_DATA_DIR / f"all_cities_features_h{h3_resolution}.gpkg"
    features_gdf = gpd.read_file(features_path)
    X, feature_columns = prepare_data_for_ranking(features_gdf)
    
    # Get unique cities for individual analysis
    cities = features_gdf['city'].unique() if 'city' in features_gdf.columns else []
    print(f"📍 Found {len(cities)} cities: {', '.join(cities)}")
    
    model_results = {}
    
    # Process each model
    for model_file in model_files:
        model_name = model_file.stem
        print(f"\n{'='*60}")
        print(f"🤖 PROCESSING MODEL: {model_name}")
        print(f"{'='*60}")
        
        # Create model-specific directory structure
        model_dir = RANKING_DIR / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Create required subfolders for organization
        heatmaps_dir = model_dir / "heatmaps"
        interactive_dir = model_dir / "interactive_maps" 
        rankings_dir = model_dir / "station_rankings"
        
        heatmaps_dir.mkdir(exist_ok=True)
        interactive_dir.mkdir(exist_ok=True)
        rankings_dir.mkdir(exist_ok=True)
        
        print(f"📁 Created folder structure: {model_dir.name}/{{heatmaps,interactive_maps,station_rankings}}")
        
        try:
            # Load model
            print(f"Loading model from {model_file}")
            model = load(model_file)
            
            # Get model's expected features
            try:
                # Try to predict with current features to detect feature mismatch
                test_prediction = model.predict_proba(X.head(1))
                model_feature_columns = feature_columns
                print(f"✅ Model compatible with {len(model_feature_columns)} features")
            except Exception as feature_error:
                print(f"⚠️ Feature mismatch detected: {feature_error}")
                
                # Try to determine model's expected features by inspecting the model
                # For now, create a reduced feature set by excluding problematic features
                if "feature names should match" in str(feature_error).lower():
                    print("🔧 Attempting to fix feature mismatch...")
                    
                    # Common problematic features that might not exist in older models
                    problematic_features = [
                        'count_amenity_accommodation',
                        'count_amenity_entertainment', 
                        'count_amenity_religious',
                        'count_amenity_tourism',
                        'count_amenity_transportation'
                    ]
                    
                    # Create reduced feature set
                    reduced_feature_columns = [col for col in feature_columns 
                                             if col not in problematic_features]
                    
                    print(f"🔄 Trying with {len(reduced_feature_columns)} features (removed {len(feature_columns) - len(reduced_feature_columns)} problematic features)")
                    
                    # Test with reduced features
                    X_reduced = features_gdf[reduced_feature_columns]
                    test_prediction = model.predict_proba(X_reduced.head(1))
                    model_feature_columns = reduced_feature_columns
                    print(f"✅ Model now compatible with reduced feature set")
                else:
                    raise feature_error
            
            # Generate rankings with appropriate feature set
            X_model = features_gdf[model_feature_columns]
            
            # Create a copy for ranking
            ranking_features_gdf = features_gdf.copy()
            
            # Generate probability scores
            probabilities = model.predict_proba(X_model)[:, 1]
            
            # Add probability scores to GeoDataFrame
            ranking_features_gdf['station_probability'] = probabilities
            ranking_features_gdf['station_rank'] = ranking_features_gdf['station_probability'].rank(ascending=False, method='min')
            ranking_features_gdf = ranking_features_gdf.sort_values('station_probability', ascending=False)
            ranking_features_gdf['percentile_rank'] = 100 - (ranking_features_gdf['station_rank'] / len(ranking_features_gdf) * 100)
            
            # Save rankings in station_rankings folder
            rankings_output_path = rankings_dir / f"station_rankings_h{h3_resolution}_{model_name}.gpkg"
            ranking_features_gdf.to_file(rankings_output_path, driver="GPKG")
            print(f"Station rankings saved to {rankings_output_path}")
            
            rankings_csv_path = rankings_dir / f"station_rankings_h{h3_resolution}_{model_name}.csv"
            ranking_features_gdf.drop('geometry', axis=1).to_csv(rankings_csv_path, index=False)
            print(f"Station rankings CSV saved to {rankings_csv_path}")
            
            ranked_gdf = ranking_features_gdf
            
            # Store results for comparison
            model_results[model_name] = {
                'model': model,
                'ranked_gdf': ranked_gdf,
                'model_dir': model_dir,
                'top_score': ranked_gdf['station_probability'].max(),
                'mean_score': ranked_gdf['station_probability'].mean(),
                'std_score': ranked_gdf['station_probability'].std(),
                'features_used': len(model_feature_columns)
            }
            
            print(f"\n📊 Model Statistics:")
            print(f"  🎯 Max Probability: {model_results[model_name]['top_score']:.4f}")
            print(f"  📈 Mean Probability: {model_results[model_name]['mean_score']:.4f}")
            print(f"  📊 Std Probability: {model_results[model_name]['std_score']:.4f}")
            print(f"  🔧 Features Used: {model_results[model_name]['features_used']}")
            
            # Create visualizations
            print(f"\n🎨 Creating visualizations for {model_name}...")
            
            # Create static heatmaps for all data (global)
            create_static_heatmap(ranked_gdf, h3_resolution, model_name, model_dir)
            
            # Create static heatmaps for each city
            print(f"🏙️  Creating individual city heatmaps...")
            for city in cities:
                create_static_heatmap(ranked_gdf, h3_resolution, model_name, model_dir, city)
            
            # Create interactive maps for all data (global)
            create_interactive_map(ranked_gdf, h3_resolution, model_name, model_dir)
            
            # Create interactive maps for each city
            print(f"🗺️  Creating individual city interactive maps...")
            for city in cities:
                create_interactive_map(ranked_gdf, h3_resolution, model_name, model_dir, city)
            
            # Export top stations
            print(f"💾 Exporting top station locations...")
            export_top_stations(ranked_gdf, 100, h3_resolution, model_name, model_dir)
            
            # Export top stations per city
            for city in cities:
                export_top_stations(ranked_gdf, 50, h3_resolution, model_name, model_dir, city)
            
            print(f"✅ Model {model_name} processing complete!")
            print(f"📁 Results saved to: {model_dir}")
            
        except Exception as e:
            print(f"❌ Error processing model {model_name}: {e}")
            print(f"📁 Folder structure still created at: {model_dir}")
            continue
    
    # Create comparison summary
    print(f"\n{'='*60}")
    print("📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    comparison_data = []
    for model_name, results in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Max_Probability': results['top_score'],
            'Mean_Probability': results['mean_score'],
            'Std_Probability': results['std_score'],
            'Features_Used': results['features_used'],
            'Results_Directory': str(results['model_dir'])
        })
    
    # Sort by max probability
    comparison_data.sort(key=lambda x: x['Max_Probability'], reverse=True)
    
    print(f"\n🏆 RANKING BY MAXIMUM PROBABILITY:")
    for i, data in enumerate(comparison_data, 1):
        print(f"  {i}. {data['Model']:<35} | Max: {data['Max_Probability']:.4f} | Mean: {data['Mean_Probability']:.4f} | Features: {data['Features_Used']}")
    
    # Save comparison summary
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = RANKING_DIR / f"model_comparison_summary_h{h3_resolution}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Comparison summary saved to: {comparison_file}")
    
    print(f"\n✅ MULTI-MODEL COMPARISON PIPELINE COMPLETE!")
    print(f"📁 All results organized in: {RANKING_DIR}")
    print(f"🎯 Processed {len(model_results)} models successfully")
    
    return model_results

def run_ranking_pipeline(h3_resolution=9):
    """
    Legacy function for backward compatibility - now runs the multi-model pipeline.
    """
    return run_model_comparison_pipeline(h3_resolution)

if __name__ == "__main__":
    run_model_comparison_pipeline()
