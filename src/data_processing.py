import osmnx as ox
import geopandas as gpd
from pathlib import Path
import rasterio # Added for raster processing
from rasterio.mask import mask # Specifically for clipping

def get_osm_geometries(place_name, tags):
    """
    Fetches geometries from OpenStreetMap for a given place and tags.

    Args:
        place_name (str or dict): The name of the place to query (e.g., "Manhattan, New York City, USA")
                                  or a dictionary for structured queries.
        tags (dict): A dictionary of OSM tags to filter by (e.g., {"amenity": "restaurant"}).

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the geometries.
                                Returns an empty GeoDataFrame if no features are found or an error occurs.
    """
    try:
        # ox.geocode_to_gdf is good for boundaries, features_from_place for other POIs
        # Ensuring the place is valid first might be good, but osmnx usually handles it.
        gdf = ox.features_from_place(place_name, tags=tags)
        return gdf
    except Exception as e:
        print(f"Error fetching OSM data for {place_name} with tags {tags}: {e}")
        return gpd.GeoDataFrame() # Return an empty GeoDataFrame on error

# Placeholder for data processing functions

# --- Configuration for OSM Data --- #
CITIES = {
    "london": "London, UK",
    "paris": "Paris, France",
    "berlin": "Berlin, Germany"
}

STATION_TAGS = {'railway': 'station', 'station': 'subway'}

AMENITY_TAGS_CONFIG = {
    "education": {'amenity': ['school', 'university']},
    "healthcare": {'amenity': ['hospital', 'clinic']},
    "shopping_food": {'shop': ['supermarket', 'mall'], 'amenity': ['restaurant', 'cafe']},
    "leisure_recreation": {'leisure': ['park', 'sports_centre']},
    "workplaces": {'office': 'yes'}, # Using 'yes' to capture various office types
    "public_services": {'amenity': ['bank', 'post_office']}
}

DATA_RAW_DIR = Path("data/raw")
# Ensure the directory exists
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_PROCESSED_DIR = Path("data/processed") # Added for processed data
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

# --- Configuration for WorldPop Data --- #
# !!! PLEASE VERIFY THESE FILENAMES MATCH YOUR DOWNLOADED GeoTIFFs IN data/raw/ !!!
WORLDPOP_CONFIG = {
    "london": "gbr_ppp_2020_100m_unconstrained.tif",  # Assuming UK file for London
    "paris": "fra_ppp_2020_100m_unconstrained.tif",   # Assuming France file for Paris
    "berlin": "deu_ppp_2020_100m_unconstrained.tif"  # Assuming Germany file for Berlin
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

    Args:
        city_short_name (str): Short name for the city (used for filename).
        city_full_name (str): Full name for OSM query.
        data_type (str): Type of data being fetched (e.g., "stations", "amenities_education").
        tags (dict): OSM tags to query.
        output_dir (Path): Directory to save the GeoJSON file.
    """
    print(f"Fetching {data_type} for {city_full_name}...")
    gdf = get_osm_geometries(city_full_name, tags)

    if not gdf.empty:
        # Standardize CRS to EPSG:4326 (WGS84) for consistency if not already
        if gdf.crs is None:
            print(f"Warning: CRS for {data_type} in {city_full_name} is None. Assuming EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Ensure geometry column exists and is valid
        if 'geometry' not in gdf.columns or gdf.is_empty.all():
            print(f"No valid geometries found for {data_type} in {city_full_name}. Skipping save.")
            return

        # Filter out non-point geometries for amenities/stations if necessary, or ensure they are handled
        # For now, assuming all fetched geometries are relevant or will be filtered later
        
        # Some features might have array-like values in columns which GeoJSON doesn't like
        # Convert list/array columns to string representations
        for col in gdf.columns:
            if gdf[col].apply(lambda x: isinstance(x, list)).any():
                gdf[col] = gdf[col].astype(str)

        output_path = output_dir / f"{city_short_name}_{data_type}.geojson"
        try:
            gdf.to_file(output_path, driver="GeoJSON")
            print(f"Saved {data_type} for {city_full_name} to {output_path}")
        except Exception as e:
            print(f"Error saving {output_path}: {e}. GDF info:\n{gdf.info()}")
            # Try saving problematic columns individually or further cleaning
            # For example, if 'nodes' or 'ways' columns cause issues:
            # cols_to_drop = [col for col in ['nodes', 'ways', 'relations'] if col in gdf.columns]
            # if cols_to_drop:
            #     gdf_simplified = gdf.drop(columns=cols_to_drop)
            #     try:
            #         gdf_simplified.to_file(output_path, driver="GeoJSON")
            #         print(f"Saved simplified {data_type} for {city_full_name} to {output_path}")
            #     except Exception as e2:
            #         print(f"Error saving simplified {output_path}: {e2}")

    else:
        print(f"No {data_type} found for {city_full_name}.")

def run_osm_data_pipeline():
    """Runs the full OSM data fetching pipeline for all configured cities and tags."""
    for city_key, city_name_query in CITIES.items():
        print(f"--- Processing city: {city_name_query} --- ")
        # Fetch and save boundaries (useful for context or clipping later)
        # Using geocode_to_gdf for administrative boundaries
        try:
            print(f"Fetching boundary for {city_name_query}...")
            boundary_gdf = ox.geocode_to_gdf(city_name_query)
            if not boundary_gdf.empty:
                if boundary_gdf.crs is None:
                    boundary_gdf = boundary_gdf.set_crs("EPSG:4326", allow_override=True)
                elif boundary_gdf.crs.to_string() != "EPSG:4326":
                    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
                
                for col in boundary_gdf.columns:
                     if boundary_gdf[col].apply(lambda x: isinstance(x, list)).any():
                        boundary_gdf[col] = boundary_gdf[col].astype(str)

                boundary_output_path = DATA_RAW_DIR / f"{city_key}_boundary.geojson"
                boundary_gdf.to_file(boundary_output_path, driver="GeoJSON")
                print(f"Saved boundary for {city_name_query} to {boundary_output_path}")
            else:
                print(f"No boundary found for {city_name_query}.")
        except Exception as e:
            print(f"Error fetching/saving boundary for {city_name_query}: {e}")

        # Fetch and save stations
        fetch_and_save_osm_data(city_key, city_name_query, "stations", STATION_TAGS, DATA_RAW_DIR)

        # Fetch and save all configured amenities
        for amenity_category, tags in AMENITY_TAGS_CONFIG.items():
            data_type_name = f"amenities_{amenity_category}"
            fetch_and_save_osm_data(city_key, city_name_query, data_type_name, tags, DATA_RAW_DIR)
    print("--- OSM Data fetching pipeline complete. ---")

if __name__ == '__main__':
    # This allows running the script directly to fetch data
    run_osm_data_pipeline() # Fetches OSM data
    process_population_data() # Processes downloaded population data
    # Example: How to load one of the saved files
    # london_stations_path = DATA_RAW_DIR / "london_stations.geojson"
    # if london_stations_path.exists():
    #     stations_gdf = gpd.read_file(london_stations_path)
    #     print("\nLoaded London stations:")
    #     print(stations_gdf.head()) 