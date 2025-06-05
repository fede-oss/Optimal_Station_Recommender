"""
Simple OSM data fetching utilities.

This module contains straightforward functions for fetching basic OSM data
like city boundaries, stations, and amenities without the complex chunking logic.
"""

import osmnx as ox
import geopandas as gpd
import concurrent.futures
from pathlib import Path

from config.config import STATION_TAGS, AMENITY_TAGS_CONFIG, DATA_RAW_DIR
from osm_utils import get_osm_geometries


def fetch_and_save_osm_data(city_short_name, city_full_name, data_type, tags, output_dir):
    """Enhanced OSM data fetching with intelligent parameter optimization."""
    output_filename = f"{city_short_name}_{data_type}.geojson"
    output_path = output_dir / output_filename
    
    # Skip if file already exists
    if output_path.exists():
        print(f"✅ {data_type.title()} data for {city_short_name} already exists. Skipping.")
        return

    print(f"🔍 Fetching {data_type} for {city_full_name}...")
    
    # Determine optimal parameters based on data type
    if data_type == "stations":
        # Stations are usually fewer, so chunking might not be needed
        use_chunking = False
        max_workers = 1
        chunk_size = 10  # Not used for non-chunked queries
    elif data_type.startswith("amenities_"):
        # Amenities can be numerous, especially for large cities
        use_chunking = True
        max_workers = 3
        chunk_size = 7  # Smaller chunks for detailed amenity data
    else:
        # Default for boundaries or other data types
        use_chunking = False
        max_workers = 1
        chunk_size = 10
    
    try:
        gdf = get_osm_geometries(
            city_full_name, 
            tags, 
            timeout=180,
            use_chunking=use_chunking,
            chunk_size_km=chunk_size,
            max_workers=max_workers
        )
        
        if not gdf.empty:
            # Standardize CRS to EPSG:4326 (WGS84) for consistency
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Handle any list columns that might cause issues with GeoJSON export
            for col in gdf.columns:
                if gdf[col].apply(lambda x: isinstance(x, list)).any():
                    gdf[col] = gdf[col].astype(str)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"✅ Saved {len(gdf)} {data_type} for {city_short_name} to {output_path}")
        else:
            print(f"⚠️ No {data_type} found for {city_short_name}")
            # Save empty GeoDataFrame to indicate we tried
            empty_gdf = gdf  # Already empty
            empty_gdf.to_file(output_path, driver="GeoJSON")
            
    except Exception as e:
        print(f"❌ Error fetching {data_type} for {city_short_name}: {e}")


def fetch_city_boundary(city_key, city_name_query):
    """Fetch and save boundary for a city."""
    try:
        print(f"Fetching boundary for {city_name_query}...")
        boundary_gdf = ox.geocode_to_gdf(city_name_query)
        if not boundary_gdf.empty:
            # Standardize CRS to EPSG:4326 (WGS84) for consistency
            if boundary_gdf.crs is None:
                boundary_gdf = boundary_gdf.set_crs("EPSG:4326", allow_override=True)
            elif boundary_gdf.crs.to_string() != "EPSG:4326":
                boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
            
            for col in boundary_gdf.columns:
                if boundary_gdf[col].apply(lambda x: isinstance(x, list)).any():
                    boundary_gdf[col] = boundary_gdf[col].astype(str)

            boundary_output_path = DATA_RAW_DIR / f"{city_key}_boundary.geojson"
            boundary_output_path.parent.mkdir(parents=True, exist_ok=True)
            boundary_gdf.to_file(boundary_output_path, driver="GeoJSON")
            print(f"✅ Saved boundary for {city_name_query} to {boundary_output_path}")
        else:
            print(f"❌ No boundary found for {city_name_query}.")
    except Exception as e:
        print(f"❌ Error fetching/saving boundary for {city_name_query}: {e}")


def fetch_city_amenity(city_key, city_name_query, amenity_category, tags):
    """Fetch and save a specific amenity type for a city."""
    data_type_name = f"amenities_{amenity_category}"
    fetch_and_save_osm_data(city_key, city_name_query, data_type_name, tags, DATA_RAW_DIR)


def fetch_city_stations(city_key, city_name_query):
    """Fetch and save stations for a city."""
    fetch_and_save_osm_data(city_key, city_name_query, "stations", STATION_TAGS, DATA_RAW_DIR)


def fetch_city_data(city_key, city_name_query, max_workers=4):
    """Fetch all data for one city using parallel processing for amenities."""
    print(f"\n🌆 Processing city: {city_name_query}")
    
    # First fetch the boundary (not parallelized)
    fetch_city_boundary(city_key, city_name_query)
    
    # Then fetch stations
    fetch_city_stations(city_key, city_name_query)
    
    # Then fetch amenities in parallel
    print(f"Fetching {len(AMENITY_TAGS_CONFIG)} amenity types for {city_name_query} in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for amenity_category, tags in AMENITY_TAGS_CONFIG.items():
            future = executor.submit(fetch_city_amenity, city_key, city_name_query, amenity_category, tags)
            futures.append(future)
            
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will re-raise any exceptions that occurred
            except Exception as e:
                print(f"❌ Error in amenity processing: {e}")
    
    print(f"✅ Completed data fetching for {city_name_query}")


def run_osm_data_pipeline(parallel_cities=False, max_workers_per_city=3, max_cities=3):
    """
    Runs the OSM data fetching pipeline for all configured cities.
    
    Args:
        parallel_cities (bool): Whether to process multiple cities in parallel
        max_workers_per_city (int): Number of parallel workers for amenities per city
        max_cities (int): Maximum number of cities to process in parallel
    """
    from config.config import CITIES
    
    print("\n--- Starting OSM Data Pipeline ---")
    print(f"Cities to process: {', '.join(CITIES.keys())}")
    
    if parallel_cities and len(CITIES) > 1:
        # Process multiple cities in parallel
        print(f"Processing cities in parallel with max {max_cities} cities at once")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_cities) as executor:
            futures = []
            for city_key, city_name_query in CITIES.items():
                future = executor.submit(fetch_city_data, city_key, city_name_query, max_workers_per_city)
                futures.append(future)
                
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will re-raise any exceptions that occurred
                except Exception as e:
                    print(f"❌ Error in parallel city processing: {e}")
    else:
        # Process cities sequentially, but amenities for each city in parallel
        for city_key, city_name_query in CITIES.items():
            fetch_city_data(city_key, city_name_query, max_workers_per_city)
    
    print("\n--- OSM Data fetching pipeline complete. ---")
