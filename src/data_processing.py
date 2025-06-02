import osmnx as ox
import geopandas as gpd
from pathlib import Path
import rasterio # Added for raster processing
from rasterio.mask import mask # Specifically for clipping

def get_osm_geometries(place_name, tags, timeout=300, use_subdiv=True):
    """
    Fetches geometries from OpenStreetMap for a given place and tags.

    Args:
        place_name (str or dict): The name of the place to query (e.g., "Manhattan, New York City, USA")
                                  or a dictionary for structured queries.
        tags (dict): A dictionary of OSM tags to filter by (e.g., {"amenity": "restaurant"}).
        timeout (int): Timeout in seconds for the Overpass API request. Default is 300 seconds (5 minutes).
        use_subdiv (bool): Whether to use polygon subdivision for large areas. Default is True.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the geometries.
                                Returns an empty GeoDataFrame if no features are found or an error occurs.
    """
    # Use thread-local settings to avoid conflicts in parallel processing
    if not hasattr(thread_local, 'ox_configured'):
        # Configure OSMnx settings for this thread
        ox.settings.timeout = timeout
        ox.settings.use_cache = True
        
        # For very numerous features like bus stops, platforms etc., we can use subdivision
        if use_subdiv:
            # This will automatically split large areas into smaller chunks for the query
            ox.settings.max_query_area_size = 25*1000*1000  # 25 sq km chunks (smaller than default)
        
        # Mark this thread as configured
        thread_local.ox_configured = True
    
    try:
        # Try to fetch the data
        gdf = ox.features_from_place(place_name, tags=tags)
        
        if len(gdf) == 0:
            print(f"No features found for {place_name} with tags {tags}")
        elif len(gdf) > 10000:
            print(f"⚠️ Found a very large number of features: {len(gdf)} for {place_name} with tags {tags}")
            print(f"   Consider filtering this dataset more specifically if processing is slow")
        
        return gdf
    except Exception as e:
        print(f"Error fetching OSM data for {place_name} with tags {tags}: {e}")
        print(f"   This might be because the area is too large or there are too many features.")
        print(f"   Try running with a more specific area or fewer tags.")
        return gpd.GeoDataFrame() # Return an empty GeoDataFrame on error

import time
from datetime import datetime, timedelta
import concurrent.futures
import threading

# Thread-local storage for OSMnx settings to avoid conflicts during parallel processing
thread_local = threading.local()

# --- Configuration for OSM Data --- #
CITIES = {
    # Original cities
    "london": "London, UK",
    "paris": "Paris, France",
    "berlin": "Berlin, Germany",
    # Additional cities with excellent public transport
    "tokyo": "Tokyo, Japan",
    "singapore": "Singapore",
    "hong_kong": "Hong Kong",
    "seoul": "Seoul, South Korea",
    "madrid": "Madrid, Spain"
}

STATION_TAGS = {'railway': 'station', 'station': 'subway'}

# Define amenity tags for feature extraction
# Extended list to provide more granular features for the model
AMENITY_TAGS_CONFIG = {
    # Education facilities
    "education": {'amenity': ['school', 'university', 'college', 'library', 'kindergarten']},
    
    # Healthcare facilities
    "healthcare": {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy', 'dentist']},
    
    # Shopping and food establishments
    "shopping_food": {'shop': ['supermarket', 'mall', 'convenience', 'department_store'], 
                    'amenity': ['restaurant', 'cafe', 'fast_food', 'food_court', 'marketplace']},
    
    # Leisure and recreation places
    "leisure_recreation": {'leisure': ['park', 'sports_centre', 'fitness_centre', 'stadium', 'theater', 'cinema'],
                         'tourism': ['museum', 'gallery', 'zoo', 'theme_park']},
    
    # Workplaces
    "workplaces": {'office': 'yes', 'building': ['office', 'commercial']},
    
    # Public services
    "public_services": {'amenity': ['bank', 'post_office', 'police', 'fire_station', 'townhall', 'courthouse', 'community_centre']},
    
    # Transportation hubs (excluding train/subway stations already covered)
    "transportation": {'amenity': ['bus_station'], 'public_transport': ['stop_position', 'platform'], 'highway': 'bus_stop'},
    
    # Accommodation
    "accommodation": {'tourism': ['hotel', 'hostel', 'guest_house', 'apartment']},
    
    # Entertainment venues
    "entertainment": {'amenity': ['bar', 'pub', 'nightclub', 'casino'], 'leisure': ['dance']},
    
    # Religious places
    "religious": {'amenity': 'place_of_worship'},
    
    # Tourist attractions
    "tourism": {'tourism': ['attraction', 'viewpoint', 'artwork']}
}

DATA_RAW_DIR = Path("data/raw")
# Ensure the directory exists
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_PROCESSED_DIR = Path("data/processed") # Added for processed data
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

# --- Configuration for WorldPop Data --- #
# Filenames are generated using the ISO3 country code pattern by worldpop_downloader.py
WORLDPOP_CONFIG = {
    # Original cities
    "london": "gbr_ppp_2020_100m_unconstrained.tif",  # UK file for London
    "paris": "fra_ppp_2020_100m_unconstrained.tif",   # France file for Paris
    "berlin": "deu_ppp_2020_100m_unconstrained.tif",  # Germany file for Berlin
    # Additional cities
    "tokyo": "jpn_ppp_2020_100m_unconstrained.tif",   # Japan file for Tokyo
    "singapore": "sgp_ppp_2020_100m_unconstrained.tif", # Singapore file
    "hong_kong": "hkg_ppp_2020_100m_unconstrained.tif", # Hong Kong file
    "seoul": "kor_ppp_2020_100m_unconstrained.tif",   # South Korea file for Seoul
    "madrid": "esp_ppp_2020_100m_unconstrained.tif"   # Spain file for Madrid
}

def load_geojson_file(filepath):
    """Loads a GeoJSON file into a GeoDataFrame."""
    if not Path(filepath).exists():
        print(f"Error: File not found {filepath}")
        return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_file(filepath)
        # Ensure CRS is consistent, e.g., EPSG:4326, if applicable
        if gdf.crs is None:
            print(f"Warning: CRS for {filepath} is None. Assuming EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        print(f"Error loading GeoJSON {filepath}: {e}")
        return gpd.GeoDataFrame()

def clip_raster_with_boundary(raster_path, boundary_gdf, output_path):
    """
    Clips a raster to the given boundary GeoDataFrame and saves the output.
    Ensures the boundary is in the same CRS as the raster before clipping.
    """
    if not Path(raster_path).exists():
        print(f"Error: Raster file not found {raster_path}")
        return
    if boundary_gdf.empty:
        print(f"Error: Boundary GeoDataFrame is empty. Cannot clip raster {raster_path}")
        return

    try:
        with rasterio.open(raster_path) as src:
            # Ensure boundary is in the same CRS as the raster
            if boundary_gdf.crs != src.crs:
                print(f"Reprojecting boundary from {boundary_gdf.crs} to {src.crs} for clipping.")
                boundary_gdf = boundary_gdf.to_crs(src.crs)
            
            # Get shapes for masking
            shapes = [geom for geom in boundary_gdf.geometry]

            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            print(f"Successfully clipped {raster_path} to {output_path}")

    except Exception as e:
        print(f"Error clipping raster {raster_path}: {e}")

def process_population_data():
    """Processes raw population rasters: clips them to city boundaries and saves them."""
    print("--- Starting Population Data Processing --- ")
    for city_key, city_name_query in CITIES.items():
        print(f"Processing population for {city_name_query}...")
        
        # 1. Load city boundary
        boundary_filename = f"{city_key}_boundary.geojson"
        boundary_path = DATA_RAW_DIR / boundary_filename
        boundary_gdf = load_geojson_file(boundary_path)

        if boundary_gdf.empty:
            print(f"Skipping population processing for {city_key} due to missing boundary.")
            continue

        # 2. Get corresponding WorldPop raster path
        worldpop_raster_name = WORLDPOP_CONFIG.get(city_key)
        if not worldpop_raster_name:
            print(f"WorldPop raster not configured for {city_key}. Skipping.")
            continue
        
        country_raster_path = DATA_RAW_DIR / worldpop_raster_name
        if not country_raster_path.exists():
            print(f"WorldPop raster {country_raster_path} not found. Skipping {city_key}.")
            print(f"Please ensure the file is named correctly and placed in {DATA_RAW_DIR}.")
            continue

        # 3. Define output path for clipped raster
        clipped_raster_filename = f"{city_key}_population_2020_100m.tif"
        clipped_raster_output_path = DATA_PROCESSED_DIR / clipped_raster_filename

        # 4. Clip and save
        print(f"Clipping {country_raster_path} to {city_key} boundary...")
        clip_raster_with_boundary(country_raster_path, boundary_gdf, clipped_raster_output_path)
    
    print("--- Population Data Processing complete. ---")

def fetch_and_save_osm_data(city_short_name, city_full_name, data_type, tags, output_dir):
    """
    Fetches specified OSM data for a city and saves it as a GeoJSON file.
    Skips download if the file already exists.

    Args:
        city_short_name (str): Short name for the city (used for filename).
        city_full_name (str): Full name for OSM query.
        data_type (str): Type of data being fetched (e.g., "stations", "amenities_education").
        tags (dict): OSM tags to query.
        output_dir (Path): Directory to save the GeoJSON file.
    """
    import time
    from datetime import datetime, timedelta
    
    output_filename = f"{city_short_name}_{data_type}.geojson"
    output_path = output_dir / output_filename

    # Check if file already exists and has content
    if output_path.exists() and output_path.stat().st_size > 100:  # More than 100 bytes should have valid content
        print(f"✅ {data_type} data for {city_full_name} already exists at {output_path}. Skipping download.")
        print(f"   To force re-download, delete the file and run this script again.")
        return

    print(f"🔄 Fetching {data_type} for {city_full_name}...")
    
    # Estimate download time based on city size and data type
    # City-specific factor (arbitrary units based on relative size/complexity)
    city_size_factor = {
        "london": 10.0,
        "paris": 7.0,
        "berlin": 8.0,
        "tokyo": 12.0,
        "singapore": 5.0,
        "hong_kong": 5.0,
        "seoul": 9.0,
        "madrid": 7.0
    }.get(city_short_name.lower(), 5.0)  # Default factor for unknown cities
    
    # Set the data type factor based on the type of data
    if data_type == "stations":
        data_type_factor = 1.0
    elif data_type == "boundaries":
        data_type_factor = 0.5
    # For amenities, estimate based on category
    elif data_type.startswith("amenities_"):
        category = data_type.split("_", 1)[1]
        category_factors = {
            "education": 2.0,
            "healthcare": 1.5,
            "shopping_food": 3.5,  # Restaurants and shops are numerous
            "leisure_recreation": 2.0,
            "workplaces": 2.5,
            "public_services": 1.0,
            "transportation": 3.0,
            "accommodation": 1.5,
            "entertainment": 2.0,
            "religious": 1.0,
            "tourism": 1.0
        }
        data_type_factor = category_factors.get(category, 2.0)
    else:
        data_type_factor = 1.0  # Default factor for other data types
    
    # Calculate estimated time in seconds
    estimated_time = city_size_factor * data_type_factor
    
    # Print ETA
    eta = datetime.now() + timedelta(seconds=estimated_time)
    print(f"   Estimated time: {estimated_time:.1f} seconds (ETA: {eta.strftime('%H:%M:%S')})")
    
    start_time = time.time()
    
    try:
        # For transportation and other potentially large datasets, use longer timeout
        extended_timeout = 600 if data_type in ['amenities_transportation', 'stations'] else 300
        
        # Get data from OSM with appropriate settings
        gdf = get_osm_geometries(city_full_name, tags, 
                               timeout=extended_timeout, 
                               use_subdiv=True)
        
        actual_time = time.time() - start_time
        
        # Check if we got any data
        if gdf.empty:
            print(f"❌ No {data_type} features found for {city_full_name} after {actual_time:.1f} seconds")
            return
        
        # Handle potential CRS issues
        if gdf.crs is None:
            print(f"⚠️ Warning: CRS is None, assuming EPSG:4326 for {data_type} in {city_full_name}")
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Convert list columns to string to avoid GeoJSON serialization issues
        for col in gdf.columns:
            if gdf[col].apply(lambda x: isinstance(x, list)).any():
                gdf[col] = gdf[col].astype(str)
        
        # Save to GeoJSON
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ Saved {len(gdf)} {data_type} features to {output_path} in {actual_time:.1f} seconds")
        
        # Calculate time difference for better future estimates
        time_diff = actual_time - estimated_time
        if abs(time_diff) > 5:  # Only log significant differences
            print(f"   Time estimate was off by {time_diff:.1f} seconds")
    
    except Exception as e:
        actual_time = time.time() - start_time
        print(f"❌ Error fetching/saving {data_type} for {city_full_name}: {e}")
        print(f"   Failed after {actual_time:.1f} seconds")
        
        # Create empty file to prevent repeated failures
        try:
            with open(output_path, 'w') as f:
                f.write('{"type": "FeatureCollection", "features": []}')
            print(f"   Created empty GeoJSON file at {output_path} to prevent repeated failures")
        except Exception as inner_e:
            print(f"   Could not create placeholder file: {inner_e}")

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
    print(f"\n🌆 Processing city: {city_name_query} --- ")
    
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
                print(f"❌ Error in parallel amenity fetch: {e}")
    
    print(f"✅ Completed all data fetching for {city_name_query}")

def run_osm_data_pipeline(parallel_cities=False, max_workers_per_city=3, max_cities=3):
    """Runs the full OSM data fetching pipeline for all configured cities and tags.
    
    Args:
        parallel_cities (bool): Whether to process multiple cities in parallel
        max_workers_per_city (int): Maximum number of parallel amenity downloads per city
        max_cities (int): Maximum number of cities to process in parallel if parallel_cities is True
    """
    print(f"\n--- Starting OSM Data Pipeline ---")
    print(f"Processing {len(CITIES)} cities with {len(AMENITY_TAGS_CONFIG)} amenity types each")
    print(f"Parallel processing: {'Enabled' if parallel_cities else 'Disabled'} for cities, "
          f"{max_workers_per_city} workers per city for amenities")
    
    if parallel_cities:
        # Process cities in parallel (be careful with API rate limits)
        print(f"Processing cities in parallel (max {max_cities} at once)...")
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

def download_worldpop_data():
    """Downloads WorldPop data using the worldpop_downloader script"""
    print("\n--- Starting WorldPop Data Download ---")
    try:
        # Import the downloader module
        from worldpop_downloader import download_population_data_for_cities
        download_population_data_for_cities()
    except ImportError as e:
        print(f"Error importing worldpop_downloader: {e}")
        print("Please run 'python src/worldpop_downloader.py' to download population data.")
    except Exception as e:
        print(f"Error downloading WorldPop data: {e}")
    print("--- WorldPop Data Download Complete ---")

def main():
    """Run the full data acquisition pipeline"""
    global CITIES  # Ensure we're modifying the global dictionary
    import argparse
    parser = argparse.ArgumentParser(description="Run the data acquisition pipeline")
    
    # Pipeline step options
    parser.add_argument('--skip-osm', action='store_true', help='Skip OSM data download')
    parser.add_argument('--skip-population', action='store_true', help='Skip population data processing')
    parser.add_argument('--skip-worldpop-download', action='store_true', help='Skip WorldPop data download')
    parser.add_argument('--city', help='Process only a specific city (e.g., tokyo)')
    
    # Parallel processing options
    parser.add_argument('--parallel-cities', action='store_true', 
                      help='Process multiple cities in parallel (use with caution)')
    parser.add_argument('--parallel-amenities', type=int, default=3, 
                      help='Number of amenity types to download in parallel per city (default: 3)')
    parser.add_argument('--max-parallel-cities', type=int, default=2, 
                      help='Maximum number of cities to process in parallel (default: 2)')
    
    args = parser.parse_args()
    
    # If a specific city is requested, filter the CITIES dictionary
    if args.city:
        city = args.city.lower()
        if city not in CITIES:
            print(f"Error: City '{city}' not found. Available cities:")
            for c in CITIES.keys():
                print(f"- {c}")
            return
        
        # Filter to only include the specified city
        city_value = CITIES[city]
        cities_copy = CITIES.copy()  # Create a copy
        CITIES.clear()
        CITIES[city] = city_value
    
    print("\n=== Starting Data Acquisition Pipeline ===")
    print(f"Processing {len(CITIES)} cities: {', '.join(CITIES.keys())}")
    print(f"Parallel processing: {'Enabled' if args.parallel_cities else 'Disabled'} for cities, "
          f"{args.parallel_amenities} workers per city for amenities")
    
    # Step 1: Download WorldPop data (before OSM since we need population data for processing)
    if not args.skip_worldpop_download:
        download_worldpop_data()
    else:
        print("Skipping WorldPop data download")
    
    # Step 2: Download OSM data
    if not args.skip_osm:
        run_osm_data_pipeline(
            parallel_cities=args.parallel_cities,
            max_workers_per_city=args.parallel_amenities,
            max_cities=args.max_parallel_cities
        )
    else:
        print("Skipping OSM data download")
    
    # Step 3: Process population data
    if not args.skip_population:
        process_population_data()
    else:
        print("Skipping population data processing")
    
    # Restore original CITIES dictionary if we filtered it
    if args.city:
        CITIES.clear()
        CITIES.update(cities_copy)
        
    print("\n=== Data Acquisition Pipeline Complete ===")
    print(f"Data saved to: {DATA_RAW_DIR} (raw) and {DATA_PROCESSED_DIR} (processed)")

if __name__ == '__main__':
    # This allows running the script directly to fetch data
    main()