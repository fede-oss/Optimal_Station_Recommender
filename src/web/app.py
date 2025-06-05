#!/usr/bin/env python3
"""
app.py - Flask Backend for Station Recommender Interactive Map

This script provides the backend API for the interactive station suitability map.
It handles requests to predict station suitability for cities, checks if data exists,
downloads missing data, and returns GeoJSON for the predictions and actual stations.
"""

import os
import sys
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from shapely.geometry import Polygon, mapping
from joblib import load

# Add the src directory to the path so we can import our modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

# Import our modules
from config.config import CITIES
from data.enhanced_osm_fetcher import OSMFetcher
from data.worldpop_downloader import download_population_data_for_cities, COUNTRY_DATA
from ml.feature_engineering import create_h3_grid_for_city, create_features_for_city
from ml.station_ranking import generate_station_rankings
from config.config import STATION_TAGS, AMENITY_TAGS_CONFIG, DATA_RAW_DIR

# Helper functions for OSM data fetching
def fetch_city_boundary(city_key, city_name_query):
    """Fetch and save boundary for a city using enhanced fetcher."""
    boundary_output_path = DATA_RAW_DIR / f"{city_key}_boundary.geojson"
    
    if boundary_output_path.exists():
        print(f"✅ Boundary data for {city_key} already exists. Skipping.")
        return

    print(f"🔍 Fetching boundary for {city_name_query}...")
    
    fetcher = OSMFetcher(use_progress_bars=False)
    try:
        boundary_gdf = fetcher.get_city_boundary(city_name_query)
        if boundary_gdf is not None and not boundary_gdf.empty:
            if boundary_gdf.crs is None:
                boundary_gdf = boundary_gdf.set_crs("EPSG:4326", allow_override=True)
            elif boundary_gdf.crs.to_string() != "EPSG:4326":
                boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
            
            for col in boundary_gdf.columns:
                if boundary_gdf[col].apply(lambda x: isinstance(x, list)).any():
                    boundary_gdf[col] = boundary_gdf[col].astype(str)

            boundary_output_path.parent.mkdir(parents=True, exist_ok=True)
            boundary_gdf.to_file(boundary_output_path, driver="GeoJSON")
            print(f"✅ Saved boundary for {city_key} to {boundary_output_path}")
        else:
            print(f"❌ No boundary found for {city_key}.")
    except Exception as e:
        print(f"❌ Error fetching/saving boundary for {city_key}: {e}")

def fetch_city_stations(city_key, city_name_query):
    """Fetch and save stations for a city using enhanced fetcher."""
    output_path = DATA_RAW_DIR / f"{city_key}_stations.geojson"
    
    if output_path.exists():
        print(f"✅ Stations data for {city_key} already exists. Skipping.")
        return

    print(f"🔍 Fetching stations for {city_name_query}...")
    
    fetcher = OSMFetcher(use_progress_bars=False)
    try:
        gdf = fetcher.fetch_amenities_smart(city_name_query, STATION_TAGS, prefer_chunking=False)
        
        if not gdf.empty:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            for col in gdf.columns:
                if gdf[col].apply(lambda x: isinstance(x, list)).any():
                    gdf[col] = gdf[col].astype(str)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"✅ Saved {len(gdf)} stations for {city_key} to {output_path}")
        else:
            print(f"⚠️ No stations found for {city_key}")
            empty_gdf = gpd.GeoDataFrame()
            empty_gdf.to_file(output_path, driver="GeoJSON")
    except Exception as e:
        print(f"❌ Error fetching stations for {city_key}: {e}")

# Initialize Flask app
app = Flask(__name__, static_folder=None, template_folder='.')

# Register multiple static folders
app.add_url_rule('/static/<path:filename>', endpoint='static', view_func=app.send_static_file)
app.static_folder = 'static'  # For files in src/static

# Add another static folder for the rankings directory
from flask import Blueprint
rankings_static = Blueprint('rankings_static', __name__, static_url_path='/rankings', static_folder='../results/rankings')
app.register_blueprint(rankings_static)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RANKING_DIR = RESULTS_DIR / "rankings"

# H3 resolution for predictions
H3_RESOLUTION = 9

# Dictionary to keep track of processing status for each city
# This prevents multiple simultaneous processes for the same city
processing_status = {}

@app.route('/')
def index():
    """Serve the city search HTML file"""
    return send_from_directory(Path(__file__).parent, 'city_search.html')

@app.route('/map')
def map_view():
    """Serve the interactive map HTML file with optional city parameter"""
    city = request.args.get('city')
    
    # Read the HTML file and add our script tag
    with open(RANKING_DIR / 'interactive_station_map_h9.html', 'r') as f:
        html_content = f.read()
    
    # Add our connector script right before the closing </head> tag
    script_tag = '<script src="/static/map_connector.js"></script>'
    html_content = html_content.replace('</head>', f'{script_tag}\n</head>')
    
    # Create a response with the modified HTML
    response = app.response_class(
        response=html_content,
        status=200,
        mimetype='text/html'
    )
    
    # Add a header that the JS can read to know which city to load
    if city:
        response.headers['X-City-To-Load'] = city
    
    return response

@app.route('/predict_city', methods=['POST'])
def predict_city():
    """
    Process a request to predict station suitability for a city.
    
    Steps:
    1. Check if the city is supported
    2. Check if we already have data for the city
    3. If not, download the necessary data
    4. Generate features for the city
    5. Run the model to get predictions
    6. Return GeoJSON for the predictions and actual stations
    """
    try:
        # Get the city name from the request
        data = request.json
        city = data.get('city', '').lower().strip()
        
        if not city:
            return jsonify({"error": True, "detail": "City name is required"}), 400
        
        # Check if we're already processing this city
        if city in processing_status and processing_status[city]['status'] == 'processing':
            # If it's been processing for more than 5 minutes, reset the status
            if time.time() - processing_status[city]['start_time'] > 300:
                processing_status[city] = {
                    'status': 'processing',
                    'start_time': time.time()
                }
            else:
                return jsonify({
                    "error": True, 
                    "detail": f"Already processing data for {city}. Please wait and try again."
                }), 409  # 409 Conflict
        
        # Set status to processing
        processing_status[city] = {
            'status': 'processing',
            'start_time': time.time()
        }
        
        # Check if the city is in our supported list
        city_key = city.lower()
        
        # If city isn't in the predefined CITIES dict, check if it's in COUNTRY_DATA
        if city_key not in CITIES and city_key in COUNTRY_DATA:
            # Add it to CITIES for this session
            CITIES[city_key] = city.title() + ", " + COUNTRY_DATA[city_key].get('country', '')
        
        if city_key not in CITIES and city_key not in COUNTRY_DATA:
            processing_status[city] = {'status': 'error', 'detail': f"City '{city}' is not supported"}
            return jsonify({
                "error": True, 
                "detail": f"City '{city}' is not supported. Supported cities: {', '.join(sorted(COUNTRY_DATA.keys()))}"
            }), 400
        
        # Load the trained model
        model_path = MODELS_DIR / f"station_recommender_h{H3_RESOLUTION}.joblib"
        if not model_path.exists():
            processing_status[city] = {'status': 'error', 'detail': "Model file not found"}
            return jsonify({"error": True, "detail": "Model file not found. Please train the model first."}), 500
        
        model = load(model_path)
        
        # Check if we have the required data for this city
        city_boundary_path = DATA_RAW_DIR / f"{city_key}_boundary.geojson"
        city_stations_path = DATA_RAW_DIR / f"{city_key}_stations.geojson"
        
        # Check if we have WorldPop data for this city
        if city_key in COUNTRY_DATA:
            country_iso = COUNTRY_DATA[city_key]['iso']
            year = COUNTRY_DATA[city_key]['year']
            worldpop_filename = f"{country_iso.lower()}_ppp_{year}_100m_unconstrained.tif"
            worldpop_path = DATA_RAW_DIR / worldpop_filename
            
            if not worldpop_path.exists():
                print(f"WorldPop data not found for {city}. Downloading...")
                success = download_population_data_for_cities(city_filter=city_key)
                if not success:
                    processing_status[city] = {'status': 'error', 'detail': f"Failed to download WorldPop data for {city}"}
                    return jsonify({"error": True, "detail": f"Failed to download WorldPop data for {city}"}), 500
        else:
            processing_status[city] = {'status': 'error', 'detail': f"No WorldPop data configuration for {city}"}
            return jsonify({"error": True, "detail": f"No WorldPop data configuration for {city}"}), 400
        
        # Fetch boundary if needed
        if not city_boundary_path.exists():
            print(f"City boundary not found for {city}. Fetching from OpenStreetMap...")
            city_name_query = CITIES.get(city_key, city.title())
            fetch_city_boundary(city_key, city_name_query)
            
            if not city_boundary_path.exists():
                processing_status[city] = {'status': 'error', 'detail': f"Failed to fetch boundary for {city}"}
                return jsonify({"error": True, "detail": f"Failed to fetch boundary for {city}"}), 500
        
        # Fetch stations if needed
        if not city_stations_path.exists():
            print(f"Stations not found for {city}. Fetching from OpenStreetMap...")
            city_name_query = CITIES.get(city_key, city.title())
            fetch_city_stations(city_key, city_name_query)
            
            if not city_stations_path.exists():
                processing_status[city] = {'status': 'error', 'detail': f"Failed to fetch stations for {city}"}
                return jsonify({"error": True, "detail": f"Failed to fetch stations for {city}"}), 500
        
        # Process population data for this city
        city_pop_processed_path = DATA_PROCESSED_DIR / f"{city_key}_population_2020_100m.tif"
        if not city_pop_processed_path.exists():
            print(f"Processed population data not found for {city}. Processing...")
            # Load the boundary
            city_boundary_gdf = gpd.read_file(city_boundary_path)
            
            # Get the WorldPop filename from COUNTRY_DATA
            if city_key in COUNTRY_DATA:
                country_iso = COUNTRY_DATA[city_key]['iso']
                year = COUNTRY_DATA[city_key]['year']
                worldpop_filename = f"{country_iso.lower()}_ppp_{year}_100m_unconstrained.tif"
                worldpop_path = DATA_RAW_DIR / worldpop_filename
                
                # Import the necessary function to process population data
                from data.population_utils import clip_raster_with_boundary
                
                # Clip the raster to the city boundary
                clip_raster_with_boundary(worldpop_path, city_boundary_gdf, city_pop_processed_path)
                
                if not city_pop_processed_path.exists():
                    processing_status[city] = {'status': 'error', 'detail': f"Failed to process population data for {city}"}
                    return jsonify({"error": True, "detail": f"Failed to process population data for {city}"}), 500
            else:
                processing_status[city] = {'status': 'error', 'detail': f"No WorldPop data configuration for {city}"}
                return jsonify({"error": True, "detail": f"No WorldPop data configuration for {city}"}), 400
        
        # Generate features for the city
        print(f"Generating features for {city}...")
        city_features_gdf = create_features_for_city(city_key, H3_RESOLUTION)
        
        if city_features_gdf is None or city_features_gdf.empty:
            processing_status[city] = {'status': 'error', 'detail': f"Failed to generate features for {city}"}
            return jsonify({"error": True, "detail": f"Failed to generate features for {city}"}), 500
        
        # Run the model to get predictions
        print(f"Generating predictions for {city}...")
        
        # Dynamically determine feature columns from the data
        # This matches the logic in model_training.py for consistency
        feature_columns = [col for col in city_features_gdf.columns 
                          if col not in ['geometry', 'h3_index', 'has_station', 'city']]
        
        print(f"Using feature columns: {feature_columns}")
        
        # Check if all feature columns exist
        missing_columns = [col for col in feature_columns if col not in city_features_gdf.columns]
        if missing_columns:
            # Handle missing feature columns
            for col in missing_columns:
                city_features_gdf[col] = 0
            print(f"Warning: Added missing feature columns: {missing_columns}")
        
        # Generate the rankings
        ranked_gdf = generate_station_rankings(city_features_gdf, model, feature_columns, H3_RESOLUTION)
        
        if ranked_gdf is None or ranked_gdf.empty:
            processing_status[city] = {'status': 'error', 'detail': f"Failed to generate rankings for {city}"}
            return jsonify({"error": True, "detail": f"Failed to generate rankings for {city}"}), 500
        
        # Ensure all geometries are valid and convert to EPSG:4326 for GeoJSON
        ranked_gdf = ranked_gdf.to_crs('EPSG:4326')
        
        # Create a helper function to convert row values to JSON-serializable types
        def make_json_serializable(value):
            if hasattr(value, 'isoformat'):  # Handle datetime objects
                return value.isoformat()
            elif pd.isna(value):  # Handle NaN/None
                return None
            elif isinstance(value, (int, float, str, bool)) or value is None:
                return value
            else:
                return str(value)  # Convert any other type to string
        
        # Generate predictions for the city
        predictions_geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for idx, row in ranked_gdf.iterrows():
            # Skip cells with missing geometries
            if row.geometry is None:
                continue
                
            properties = {}
            # Convert all properties to JSON-serializable types
            for key, value in row.items():
                if key != 'geometry' and key != 'h3_index':
                    properties[key] = make_json_serializable(value)
            
            # Always include these critical properties
            properties['h3_index'] = row.h3_index
            properties['probability'] = float(row.station_probability)
            properties['has_station'] = int(row.has_station) if 'has_station' in row else 0
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": properties
            }
            predictions_geojson["features"].append(feature)
        
        # Load actual stations
        stations_gdf = gpd.read_file(city_stations_path)
        
        # Convert to EPSG:4326 for GeoJSON
        if stations_gdf.crs != 'EPSG:4326':
            stations_gdf = stations_gdf.to_crs('EPSG:4326')
        
        # Convert all timestamp columns to strings
        for col in stations_gdf.select_dtypes(include=['datetime64']).columns:
            stations_gdf[col] = stations_gdf[col].astype(str)
        
        # Create a custom JSON encoder that can handle special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif hasattr(obj, '__str__'):
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        
        # Generate GeoJSON for actual stations
        stations_json = stations_gdf.to_json()
        stations_geojson = json.loads(stations_json)
        
        # Load city boundary
        boundary_gdf = gpd.read_file(city_boundary_path)
        
        # Convert to EPSG:4326 for GeoJSON
        if boundary_gdf.crs != 'EPSG:4326':
            boundary_gdf = boundary_gdf.to_crs('EPSG:4326')
        
        # Convert all timestamp columns to strings
        for col in boundary_gdf.select_dtypes(include=['datetime64']).columns:
            boundary_gdf[col] = boundary_gdf[col].astype(str)
        
        # Generate GeoJSON for city boundary
        boundary_json = boundary_gdf.to_json()
        boundary_geojson = json.loads(boundary_json)
        
        # Update status
        processing_status[city] = {'status': 'complete'}
        
        # Return the GeoJSON data
        return jsonify({
            "city_boundary_geojson": boundary_geojson,
            "prediction_h3_geojson": predictions_geojson,
            "actual_stations_geojson": stations_geojson
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_status[city] = {'status': 'error', 'detail': str(e)}
        return jsonify({"error": True, "detail": f"Error processing request: {str(e)}"}), 500

@app.route('/processing_status/<city>', methods=['GET'])
def get_processing_status(city):
    """Get the processing status for a city"""
    city = city.lower().strip()
    if city in processing_status:
        return jsonify(processing_status[city])
    else:
        return jsonify({"status": "unknown"})

@app.route('/supported_cities', methods=['GET'])
def get_supported_cities():
    """Get a list of supported cities"""
    return jsonify({
        "cities": sorted(COUNTRY_DATA.keys())
    })

if __name__ == '__main__':
    # Ensure the processed data directory exists
    DATA_PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

    # Run the Flask app. 8082 is used to avoid conflicts with other services.
    app.run(debug=True, host='0.0.0.0', port=8082)
