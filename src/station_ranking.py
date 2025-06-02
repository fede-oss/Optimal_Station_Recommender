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
PROJECT_ROOT = Path(__file__).parent.parent
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

def load_data_and_model(h3_resolution):
    """
    Load the features data and the trained model.
    
    Parameters:
    -----------
    h3_resolution : int
        H3 resolution used for the grid
        
    Returns:
    --------
    tuple
        (features_gdf, model) - GeoDataFrame with features and trained model
    """
    # Load feature data
    features_path = PROCESSED_DATA_DIR / f"all_cities_features_h{h3_resolution}.gpkg"
    print(f"Loading feature data from {features_path}")
    features_gdf = gpd.read_file(features_path)
    
    # Load trained model
    model_path = MODELS_DIR / f"station_recommender_h{h3_resolution}.joblib"
    print(f"Loading trained model from {model_path}")
    model = load(model_path)
    
    return features_gdf, model

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
    
    # Define feature columns (same as in model_training.py)
    feature_columns = [
        'population',
        'count_amenity_education',
        'count_amenity_healthcare',
        'count_amenity_shopping_food',
        'count_amenity_leisure_recreation',
        'count_amenity_workplaces',
        'count_amenity_public_services'
    ]
    
    # Create feature matrix
    X = features_gdf[feature_columns].copy()
    
    return X, feature_columns

def generate_station_rankings(features_gdf, model, feature_columns, h3_resolution):
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
        
    Returns:
    --------
    GeoDataFrame
        GeoDataFrame with station probability scores and rankings
    """
    print("Generating station rankings...")
    
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
    
    # Save ranked GeoDataFrame
    output_path = RANKING_DIR / f"station_rankings_h{h3_resolution}.gpkg"
    ranked_gdf.to_file(output_path, driver="GPKG")
    print(f"Station rankings saved to {output_path}")
    
    # Also save a CSV version for easier analysis
    csv_path = RANKING_DIR / f"station_rankings_h{h3_resolution}.csv"
    ranked_gdf.drop('geometry', axis=1).to_csv(csv_path, index=False)
    print(f"Station rankings CSV saved to {csv_path}")
    
    return ranked_gdf

def create_static_heatmap(ranked_gdf, h3_resolution, city=None):
    """
    Create a static heatmap of station probability scores.
    
    Parameters:
    -----------
    ranked_gdf : GeoDataFrame
        GeoDataFrame with station probability scores
    h3_resolution : int
        H3 resolution used for the grid
    city : str, optional
        City name to filter data (if None, use all cities)
    """
    print(f"Creating static heatmap{'s' if city is None else f' for {city}'}")
    
    # Filter by city if specified
    if city is not None and 'city' in ranked_gdf.columns:
        city_gdf = ranked_gdf[ranked_gdf['city'] == city.lower()]
        if len(city_gdf) == 0:
            print(f"No data found for city: {city}")
            return
    else:
        city_gdf = ranked_gdf
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot hexagons colored by probability
    city_gdf.plot(
        column='station_probability',
        ax=ax,
        cmap='YlOrRd',
        legend=True,
        alpha=0.7,
        legend_kwds={'label': 'Station Suitability Score', 'orientation': 'horizontal'}
    )
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=city_gdf.crs.to_string())
    except Exception as e:
        print(f"Could not add basemap: {e}")
    
    # Add title and labels
    title = f"Station Suitability Heatmap"
    if city is not None:
        title += f" - {city.capitalize()}"
    ax.set_title(title, fontsize=16)
    
    # Save figure
    filename = f"station_heatmap_h{h3_resolution}"
    if city is not None:
        filename += f"_{city.lower()}"
        # Save city heatmaps in dedicated folder
        output_path = HEATMAPS_DIR / f"{filename}.png"
    else:
        # Save combined heatmap in main rankings folder
        output_path = RANKING_DIR / f"{filename}.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")
    plt.close()

def create_interactive_map(ranked_gdf, h3_resolution, city=None):
    """
    Create an interactive folium map with station probability heatmap.
    
    Parameters:
    -----------
    ranked_gdf : GeoDataFrame
        GeoDataFrame with station probability scores
    h3_resolution : int
        H3 resolution used for the grid
    city : str, optional
        City name to filter data (if None, use all cities)
    """
    print(f"Creating interactive map{'s' if city is None else f' for {city}'}")
    
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
                top: 10px; left: 50px; width: 400px; height: 45px; 
                background-color: white; border-radius: 5px; padding: 10px; 
                z-index: 9999; font-size: 16px; font-weight: bold;">
        Station Suitability Map {}
        <div style="font-size: 12px; font-weight: normal;">
            Toggle layers using the control in the top-right corner
        </div>
    </div>
    '''.format(f'for {city.capitalize()}' if city else '')
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    filename = f"interactive_station_map_h{h3_resolution}"
    if city is not None:
        filename += f"_{city.lower()}"
    output_path = RANKING_DIR / f"{filename}.html"
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")
    
    return output_path

def export_top_stations(ranked_gdf, top_n=100, h3_resolution=9, city=None):
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
    city : str, optional
        City name to filter data (if None, use all cities)
    """
    print(f"Exporting top {top_n} potential station locations")
    
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
    
    # Save as GeoJSON
    filename = f"top_{top_n}_stations_h{h3_resolution}"
    if city is not None:
        filename += f"_{city.lower()}"
    
    # Save hexagons
    hex_output_path = RANKING_DIR / f"{filename}_hexagons.geojson"
    top_stations.drop('center_point', axis=1).to_file(hex_output_path, driver='GeoJSON')
    print(f"Top station hexagons saved to {hex_output_path}")
    
    # Save center points
    points_output_path = RANKING_DIR / f"{filename}_points.geojson"
    center_points_gdf.to_file(points_output_path, driver='GeoJSON')
    print(f"Top station points saved to {points_output_path}")

def run_ranking_pipeline(h3_resolution=9):
    """
    Run the complete station ranking pipeline.
    
    Parameters:
    -----------
    h3_resolution : int
        H3 resolution used for the grid
    """
    print("=== Starting Station Ranking Pipeline ===")
    
    # Load data and model
    features_gdf, model = load_data_and_model(h3_resolution)
    
    # Prepare data for ranking
    X, feature_columns = prepare_data_for_ranking(features_gdf)
    
    # Generate station rankings
    ranked_gdf = generate_station_rankings(features_gdf, model, feature_columns, h3_resolution)
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    
    # Create static heatmaps for all data
    create_static_heatmap(ranked_gdf, h3_resolution)
    
    # Create static heatmaps for each city if city info is available
    if 'city' in ranked_gdf.columns:
        print("\n=== Creating Individual City Heatmaps ===")
        for city in ranked_gdf['city'].unique():
            create_static_heatmap(ranked_gdf, h3_resolution, city)
    
    # Create interactive maps for all data
    create_interactive_map(ranked_gdf, h3_resolution)
    
    # Create interactive maps for each city if city info is available
    if 'city' in ranked_gdf.columns:
        for city in ranked_gdf['city'].unique():
            create_interactive_map(ranked_gdf, h3_resolution, city)
    
    # Export top stations
    print("\n=== Exporting Top Station Locations ===")
    export_top_stations(ranked_gdf, top_n=100, h3_resolution=h3_resolution)
    
    # Export top stations per city if city info is available
    if 'city' in ranked_gdf.columns:
        for city in ranked_gdf['city'].unique():
            export_top_stations(ranked_gdf, top_n=50, h3_resolution=h3_resolution, city=city)
    
    print("\n=== Station Ranking Pipeline Complete ===")
    print(f"All ranking results saved to {RANKING_DIR}")
    print(f"Individual city heatmaps saved to {HEATMAPS_DIR}")

if __name__ == "__main__":
    run_ranking_pipeline()
