import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from h3 import LatLngPoly
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, Point, MultiPolygon

# Define directories (consistent with data_processing.py)
BASE_DIR = Path(__file__).resolve().parent.parent # Should point to Optimal_Station_Recommender directory
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# H3 resolution to use for gridding the city
# Res 9: ~175m avg edge length, ~74k m^2 area
# Res 10: ~66m avg edge length, ~11k m^2 area
# Let's start with resolution 9 as a balance
H3_RESOLUTION = 9 

def create_h3_grid_for_city(city_boundary_gdf, h3_resolution):
    """
    Creates an H3 grid covering the city boundary.

    Args:
        city_boundary_gdf (gpd.GeoDataFrame): GeoDataFrame of the city boundary (single polygon expected).
        h3_resolution (int): The H3 resolution for the grid.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with H3 cell IDs and their geometries.
    """
    if city_boundary_gdf.empty or city_boundary_gdf.iloc[0].geometry is None:
        print("Error: City boundary is empty or has no geometry.")
        return gpd.GeoDataFrame()

    # Ensure boundary is in WGS84 (EPSG:4326) as H3 uses it
    if city_boundary_gdf.crs.to_string() != "EPSG:4326":
        city_boundary_gdf = city_boundary_gdf.to_crs("EPSG:4326")

    # --- H3 Library Test with H3 4.2.2 Polygon Format ---
    print("\n--- Testing h3.polygon_to_cells with proper input format for version 4.2.2 ---")
    # Define a simple rectangle in London
    coords = [(-0.1, 51.5), (-0.1, 51.6), (0.0, 51.6), (0.0, 51.5), (-0.1, 51.5)]
    simple_shapely_polygon = Polygon(coords)
    
    try:
        test_resolution = h3_resolution
        print(f"Attempting h3.polygon_to_cells with formatted input for Polygon: {simple_shapely_polygon.wkt[:100]}..., resolution: {test_resolution}")
        
        # For h3 version 4.2.2, we need to convert the Shapely polygon to the expected format:
        # - Outer ring coordinates as a list of [lat, lng] pairs (note h3 expects [lat, lng] order)
        # - Optional list of hole coordinates, each as a list of [lat, lng] pairs
        
        # Extract exterior coordinates from the Shapely polygon
        shapely_coords = list(simple_shapely_polygon.exterior.coords)
        
        # Convert to the format h3 expects: list of [lat, lng] pairs (note the swap)
        # We also need to skip the last coordinate as shapely duplicates first/last point
        h3_polygon = [[y, x] for x, y in shapely_coords[:-1]]
        
        # Create a list of holes (empty for this simple test polygon)
        h3_holes = []
        
        print(f"Formatted h3 polygon with {len(h3_polygon)} exterior points and {len(h3_holes)} holes")
        
        # Create a LatLngPoly object and call h3.polygon_to_cells
        h3_poly = LatLngPoly(h3_polygon)
        test_hexagons = h3.polygon_to_cells(h3_poly, test_resolution)
        
        print(f"H3 test SUCCESSFUL. Found {len(test_hexagons)} H3 cells for the polygon.")
    except Exception as e:
        print(f"H3 test FAILED with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print("--- End H3 Library Test ---\n")
    # --- End of H3 Library Test ---

    # Get the first geometry (assuming it's the main city boundary)
    boundary_polygon = city_boundary_gdf.geometry.iloc[0]
    
    # Polyfill the boundary with H3 cells
    all_hexagons = set() # Use a set to store unique hexagon IDs

    def polygon_to_h3_cells(poly, h3_resolution):
        # For h3 4.2.2, polygon_to_cells expects a specific format:
        # - A list of [lat, lng] coordinates for the outer ring
        # - An optional list of holes, each as a list of [lat, lng] coordinates
        # - No closing point (i.e., don't repeat the first point at the end)
        
        # Ensure the polygon is valid
        if not poly.is_valid:
            print(f"Warning: Input polygon is not valid. Attempting to buffer by 0 to fix.")
            poly = poly.buffer(0)
            if not poly.is_valid:
                print(f"Error: Polygon remains invalid after buffer(0). Cannot proceed with H3 conversion.")
                return set() # Return an empty set for invalid polygons
        
        print(f"Converting {poly.geom_type} to h3 format (valid: {poly.is_valid})")
        
        # Container for all H3 cells
        all_hexagons = set()
        
        # Handle both Polygon and MultiPolygon types
        polygons_to_process = []
        if poly.geom_type == 'Polygon':
            polygons_to_process = [poly]
        elif poly.geom_type == 'MultiPolygon':
            print(f"Processing MultiPolygon with {len(poly.geoms)} parts")
            polygons_to_process = [p for p in poly.geoms if p.is_valid and p.geom_type == 'Polygon']
        else:
            print(f"Warning: Unexpected geometry type: {poly.geom_type}. Skipping.")
            return set()
        
        # Process each polygon
        for p in polygons_to_process:
            try:
                # Get exterior ring coordinates (skipping the closing point)
                ext_coords = list(p.exterior.coords)[:-1]
                
                # Convert to h3's expected format: list of [lat, lng] pairs
                # Shapely stores as (x,y) = (lng,lat) but h3 needs [lat,lng]
                h3_polygon_coords = [[y, x] for x, y in ext_coords]
                
                # Process holes (interior rings) if any
                h3_holes = []
                for interior in p.interiors:
                    int_coords = list(interior.coords)[:-1]  # Skip closing point
                    # Convert to h3's [lat, lng] format
                    h3_hole_coords = [[y, x] for x, y in int_coords]
                    h3_holes.append(h3_hole_coords)
                
                # Call h3.polygon_to_cells with proper format
                try:
                    # Create a LatLngPoly object for the exterior and holes
                    # Only include holes if they have at least 3 points
                    valid_holes = [hole for hole in h3_holes if len(hole) >= 3]
                    h3_poly = LatLngPoly(h3_polygon_coords, valid_holes if valid_holes else None)
                    polygon_cells = h3.polygon_to_cells(h3_poly, h3_resolution)
                    
                    # Add cells to our result set
                    all_hexagons.update(polygon_cells)
                    print(f"  Added {len(polygon_cells)} H3 cells from polygon")
                    
                except Exception as e:
                    print(f"  Error calling h3.polygon_to_cells: {type(e).__name__}: {e}")
                    print(f"  Formatted polygon: {len(h3_polygon_coords)} exterior points, {len(h3_holes)} holes")
                    print(f"  First few points: {h3_polygon_coords[:3]}")
                    
                    # If regular conversion fails, fall back to point sampling approach
                    print("  Falling back to point sampling approach...")
                    
                    # Point sampling as fallback
                    minx, miny, maxx, maxy = p.bounds
                    step_size = 0.001  # Small step size for decent coverage
                    sampled_cells = set()
                    sampled_count = 0
                    
                    for x in np.arange(minx, maxx, step_size):
                        for y in np.arange(miny, maxy, step_size):
                            point = Point(x, y)
                            if p.contains(point):
                                h3_cell = h3.latlng_to_cell(y, x, h3_resolution)  # lat, lng order
                                sampled_cells.add(h3_cell)
                                sampled_count += 1
                    
                    print(f"  Point sampling found {len(sampled_cells)} cells from {sampled_count} points")
                    all_hexagons.update(sampled_cells)
                
            except Exception as e:
                print(f"Error processing polygon: {type(e).__name__}: {e}")
        
        print(f"Total unique H3 cells found across all polygons: {len(all_hexagons)}")
        return all_hexagons

    if boundary_polygon.geom_type == 'Polygon':
        if boundary_polygon.is_valid and not boundary_polygon.is_empty:
            hexagons = polygon_to_h3_cells(boundary_polygon, h3_resolution)
            all_hexagons.update(hexagons)
        else:
            print("Skipping invalid or empty main boundary polygon.")
    elif boundary_polygon.geom_type == 'MultiPolygon':
        print(f"Boundary is a MultiPolygon. Processing {len(list(boundary_polygon.geoms))} sub-polygons.")
        for poly in boundary_polygon.geoms:
            if poly.is_valid and not poly.is_empty:
                hexagons = polygon_to_h3_cells(poly, h3_resolution)
                all_hexagons.update(hexagons)
            else:
                print("Skipping invalid or empty sub-polygon in MultiPolygon.")
    else:
        print(f"Unsupported geometry type: {boundary_polygon.geom_type}. Cannot create H3 grid.")
        return gpd.GeoDataFrame()
    
    if not all_hexagons:
        print(f"No H3 cells found for the boundary at resolution {h3_resolution}.")
        return gpd.GeoDataFrame()

    # Create a GeoDataFrame from the H3 cells
    h3_gdf = gpd.GeoDataFrame({
        'h3_index': list(all_hexagons) # Convert set to list for GeoDataFrame
    })
    # Convert H3 indices to polygons for spatial operations
    # CRITICAL FIX: H3 returns coordinates as (lat, lng), but Shapely expects (lng, lat)
    # We need to swap coordinates to create valid polygons
    h3_gdf['geometry'] = h3_gdf['h3_index'].apply(lambda x: Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(x)]))
    h3_gdf.set_crs("EPSG:4326", inplace=True)
    
    return h3_gdf

# Placeholder for feature engineering functions

# --- Helper to load data for a city ---
def load_city_data_for_fe(city_key, raw_dir_path, processed_dir_path):
    """Loads all necessary data for a city for feature engineering."""
    data = {}
    data['boundary'] = gpd.read_file(raw_dir_path / f"{city_key}_boundary.geojson")
    data['stations'] = gpd.read_file(raw_dir_path / f"{city_key}_stations.geojson")
    
    data['amenities'] = {}
    # AMENITY_TAGS_CONFIG is defined in data_processing.py, we might need to redefine or import it
    # For now, let's assume we know the amenity categories used during data fetching
    amenity_categories = ["education", "healthcare", "shopping_food", "leisure_recreation", "workplaces", "public_services"]
    for cat in amenity_categories:
        amenity_file = raw_dir_path / f"{city_key}_amenities_{cat}.geojson"
        if amenity_file.exists():
            data['amenities'][cat] = gpd.read_file(amenity_file)
        else:
            print(f"Warning: Amenity file not found: {amenity_file}")
            data['amenities'][cat] = gpd.GeoDataFrame() # Empty GDF

    # Population raster
    pop_raster_file = processed_dir_path / f"{city_key}_population_2020_100m.tif"
    if pop_raster_file.exists():
        data['population_raster_path'] = pop_raster_file
    else:
        print(f"Warning: Population raster not found: {pop_raster_file}")
        data['population_raster_path'] = None
    return data

def aggregate_raster_data_to_grid(grid_gdf, raster_path, band=1, stat_func=np.sum, nodata_val=None):
    """
    Aggregates raster data to a vector grid using zonal statistics concept.
    For each polygon in grid_gdf, it extracts raster pixels falling within it
    and applies stat_func to them.

    Args:
        grid_gdf (gpd.GeoDataFrame): The vector grid (e.g., H3 cells).
        raster_path (str or Path): Path to the raster file.
        band (int): Raster band to process.
        stat_func (function): Function to apply to pixel values (e.g., np.sum, np.mean).
        nodata_val (float, optional): NoData value from raster. If None, tries to get from raster meta.

    Returns:
        list: A list of aggregated values, corresponding to each geometry in grid_gdf.
    """
    aggregated_values = []

    if not Path(raster_path).exists():
        print(f"Error: Raster file not found: {raster_path}")
        return [0] * len(grid_gdf) # Return default values

    with rasterio.open(raster_path) as src:
        if nodata_val is None:
            nodata_val = src.nodata # Get nodata value from raster if not specified
        
        # Ensure grid_gdf is in the same CRS as the raster
        if grid_gdf.crs != src.crs:
            print(f"Reprojecting grid from {grid_gdf.crs} to {src.crs} for aggregation.")
            grid_gdf_reprojected = grid_gdf.to_crs(src.crs)
        else:
            grid_gdf_reprojected = grid_gdf.copy()

        for i, geom in enumerate(grid_gdf_reprojected.geometry):
            try:
                # Mask (clip) the raster to the geometry
                out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, all_touched=True)
                
                # Extract valid pixel values
                # The out_image might have multiple bands if the source did, select the specified one
                # Also, after mask, nodata values might be 0 or a fill value. Check src.meta for fill.
                # For WorldPop, population counts are usually floats, nodata often negative.
                pixel_values = out_image[band-1].flatten() # Get the specified band and flatten
                
                # Filter out NoData values
                if nodata_val is not None:
                    valid_pixels = pixel_values[pixel_values != nodata_val]
                else:
                    # If no specific nodata_val, assume all are valid unless they are e.g. NaN
                    # This might need adjustment based on raster properties
                    valid_pixels = pixel_values[~np.isnan(pixel_values)] 
                
                if valid_pixels.size > 0:
                    aggregated_values.append(stat_func(valid_pixels))
                else:
                    aggregated_values.append(0) # Default if no valid pixels or empty mask
            except Exception as e:
                # print(f"Error processing geometry {i} for H3 index {grid_gdf.iloc[i]['h3_index'] if 'h3_index' in grid_gdf.columns else 'N/A'}: {e}")
                aggregated_values.append(0) # Default on error
    return aggregated_values

def count_points_in_polygons(polygons_gdf, points_gdf, count_column_name):
    """
    Count the number of points from points_gdf that are within each polygon from polygons_gdf.
    
    Args:
        polygons_gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygons
        points_gdf (geopandas.GeoDataFrame): GeoDataFrame containing points
        count_column_name (str): Name of the column to store the counts
        
    Returns:
        geopandas.GeoDataFrame: polygons_gdf with additional column containing counts
    """
    # Initialize count column with zeros
    polygons_gdf[count_column_name] = 0
    
    # Return early if no points to count
    if points_gdf.empty:
        return polygons_gdf
    
    # Ensure CRS match
    if polygons_gdf.crs != points_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    try:
        # Use spatial join with 'within' predicate to find points within polygons
        # We join polygons (left) to points (right) so we can count points per polygon
        joined = gpd.sjoin(polygons_gdf, points_gdf, how='left', predicate='contains')
        
        # Group by polygon index and count points (non-null index_right values)
        if 'index_right' in joined.columns:
            # Count points per polygon (those with non-null index_right)
            counts = joined.groupby(level=0)['index_right'].count()
            
            # Update count column where counts exist
            for idx, count in counts.items():
                if idx in polygons_gdf.index:
                    polygons_gdf.loc[idx, count_column_name] = count
    
    except Exception as e:
        print(f"Error in count_points_in_polygons: {e}")
    
    return polygons_gdf

def create_features_for_city(city_key, h3_resolution):
    """
    Orchestrates feature engineering for a single city.
    1. Loads data.
    2. Creates H3 grid.
    3. Aggregates population to grid.
    4. Counts amenities in grid cells.
    5. Identifies grid cells with stations (target variable).
    6. Saves the final feature GeoDataFrame.
    """
    print(f"--- Starting Feature Engineering for {city_key} (H3 Res: {h3_resolution}) ---")

    # 1. Load data
    city_data = load_city_data_for_fe(city_key, DATA_RAW_DIR, DATA_PROCESSED_DIR)
    if city_data['boundary'].empty:
        print(f"Skipping {city_key}: Boundary data not found or empty.")
        return None
    if not city_data['population_raster_path']:
        # For now, we'll allow processing even if population is missing, feature will be 0
        print(f"Warning: Population raster not found for {city_key}. Population features will be 0.")

    # 2. Create H3 grid
    print(f"Creating H3 grid for {city_key}...")
    h3_grid_gdf = create_h3_grid_for_city(city_data['boundary'], h3_resolution)
    if h3_grid_gdf.empty:
        print(f"Skipping {city_key}: H3 grid could not be created.")
        return None
    print(f"Created {len(h3_grid_gdf)} H3 cells for {city_key}.")

    # 3. Aggregate population to grid
    if city_data['population_raster_path']:
        print(f"Aggregating population for {city_key}...")
        population_counts = aggregate_raster_data_to_grid(h3_grid_gdf, city_data['population_raster_path'], stat_func=np.sum)
        h3_grid_gdf['population'] = population_counts
    else:
        h3_grid_gdf['population'] = 0

    # 4. Count amenities in grid cells
    print(f"Counting amenities for {city_key}...")
    for amenity_cat, amenity_gdf in city_data['amenities'].items():
        col_name = f"count_amenity_{amenity_cat}"
        if not amenity_gdf.empty:
            h3_grid_gdf = count_points_in_polygons(h3_grid_gdf, amenity_gdf, col_name)
        else:
            h3_grid_gdf[col_name] = 0 # Add column with zeros if no amenities of this type
    
    # 5. Identify grid cells with stations (target variable)
    print(f"Identifying cells with stations for {city_key}...")
    stations_gdf = city_data['stations']
    if not stations_gdf.empty:
        # Ensure stations_gdf has geometry and is not empty
        if 'geometry' not in stations_gdf.columns or stations_gdf.geometry.isnull().all():
            print(f"Warning: Stations GeoDataFrame for {city_key} has no valid geometries. Target will be all 0.")
            h3_grid_gdf['has_station'] = 0
        else:
            # Debug print statements to help diagnose station identification issues
            print(f"  H3 grid CRS: {h3_grid_gdf.crs}")
            print(f"  Stations CRS: {stations_gdf.crs}")
            print(f"  Number of stations: {len(stations_gdf)}")
            print(f"  First station geometry type: {stations_gdf.geometry.iloc[0].geom_type if len(stations_gdf) > 0 else 'N/A'}")
            
            # Ensure both dataframes are in the same CRS before joining
            if h3_grid_gdf.crs != stations_gdf.crs:
                print(f"  Reprojecting stations from {stations_gdf.crs} to {h3_grid_gdf.crs}")
                stations_gdf = stations_gdf.to_crs(h3_grid_gdf.crs)
            
            # Try a different approach for station identification
            print("  DEBUG: Looking at sample station and hexagon...")
            if len(stations_gdf) > 0 and len(h3_grid_gdf) > 0:
                # Print sample station point and hexagon to investigate spatial relationship
                sample_station = stations_gdf.iloc[0]
                sample_hexagon = h3_grid_gdf.iloc[0]
                print(f"  Sample station: {sample_station.geometry}")
                print(f"  Sample hexagon: {sample_hexagon.geometry}")
                
                # Try different join predicates
                print("  Attempting spatial join with 'intersects' predicate...")
                try:
                    # First try: hexagons contain stations (traditional approach)
                    joined1 = gpd.sjoin(h3_grid_gdf, stations_gdf, how='inner', predicate='intersects')
                    print(f"  Method 1: Found {len(joined1)} intersections between hexagons and stations")
                    print(f"  Method 1: Found {len(joined1['index_right'].unique())} unique stations")
                    print(f"  Method 1: Found {len(joined1.index.unique())} unique hexagons with stations")
                    
                    # Set the target variable based on this join
                    if len(joined1) > 0:
                        h3_grid_gdf['has_station'] = 0
                        h3_grid_gdf.loc[joined1.index.unique(), 'has_station'] = 1
                        print(f"  Successfully identified {h3_grid_gdf['has_station'].sum()} hexagons with stations")
                    else:
                        # If no matches, try a buffer approach
                        print("  No intersections found, trying with buffered stations...")
                        # Create a small buffer around stations to ensure intersection
                        buffered_stations = stations_gdf.copy()
                        buffer_size = 0.0001  # ~10m buffer in degrees
                        buffered_stations['geometry'] = buffered_stations.geometry.buffer(buffer_size)
                        joined_buffer = gpd.sjoin(h3_grid_gdf, buffered_stations, how='inner', predicate='intersects')
                        print(f"  Buffer method: Found {len(joined_buffer)} intersections")
                        
                        if len(joined_buffer) > 0:
                            h3_grid_gdf['has_station'] = 0
                            h3_grid_gdf.loc[joined_buffer.index.unique(), 'has_station'] = 1
                            print(f"  Buffer method identified {h3_grid_gdf['has_station'].sum()} hexagons with stations")
                        else:
                            print("  Still no matches with buffer approach")
                            h3_grid_gdf['has_station'] = 0
                except Exception as e:
                    print(f"  Error in spatial join: {e}")
                    h3_grid_gdf['has_station'] = 0
            else:
                print("  Either stations or hexagons dataframe is empty")
                h3_grid_gdf['has_station'] = 0

            # Note: The spatial join logic is now handled in the code block above
            # We're keeping this block commented for reference
            # If needed, we can uncomment and adapt it for additional processing
            
            # # If a h3 cell contains multiple stations, it will be duplicated.
            # # We only care if *any* station is present.
            # # Group by the original h3_grid_gdf index and check if 'index_right' (from stations_gdf) is not NaN.
            # if 'index_right' in joined_stations.columns:
            #     # Call notna() before groupby, not after
            #     station_presence = joined_stations['index_right'].notna().groupby(joined_stations.index).any()
            #     h3_grid_gdf['has_station'] = station_presence.astype(int)
            # else:
            #      h3_grid_gdf['has_station'] = 0
    else:
        h3_grid_gdf['has_station'] = 0
    
    # Fill any NaN population/amenity counts that might have occurred due to errors with 0
    # (though functions are designed to return 0 already in many error cases)
    cols_to_fill_na = ['population'] + [col for col in h3_grid_gdf.columns if 'count_amenity_' in col]
    for col in cols_to_fill_na:
        if col in h3_grid_gdf.columns:
            h3_grid_gdf[col] = h3_grid_gdf[col].fillna(0)
        else:
            h3_grid_gdf[col] = 0 # If a category had no file, it might not have a column yet

    # 6. Save the final feature GeoDataFrame
    output_filename = f"{city_key}_features_h{h3_resolution}.gpkg"
    output_path = DATA_PROCESSED_DIR / output_filename
    try:
        h3_grid_gdf.to_file(output_path, driver="GPKG")
        print(f"Successfully saved features for {city_key} to {output_path}")
    except Exception as e:
        print(f"Error saving features for {city_key} to {output_path}: {e}")
        # Attempt to save as GeoJSON as a fallback, converting list-like columns
        for col in h3_grid_gdf.columns:
            if h3_grid_gdf[col].apply(lambda x: isinstance(x, list)).any():
                h3_grid_gdf[col] = h3_grid_gdf[col].astype(str)
        output_filename_geojson = f"{city_key}_features_h{h3_resolution}.geojson"
        output_path_geojson = DATA_PROCESSED_DIR / output_filename_geojson
        try:
            h3_grid_gdf.to_file(output_path_geojson, driver="GeoJSON")
            print(f"Successfully saved features for {city_key} to {output_path_geojson} (GeoJSON fallback)")
        except Exception as e2:
            print(f"Error saving features for {city_key} to GeoJSON as well: {e2}")

    return h3_grid_gdf

def run_feature_engineering_pipeline():
    """Runs the full feature engineering pipeline for all configured cities."""
    # CITIES config should be accessible here (e.g. import from data_processing or redefine)
    # For now, hardcoding the keys we expect were processed based on user output
    # In a more robust setup, this would come from a shared config or by listing processed data
    processed_city_keys = []
    if (DATA_PROCESSED_DIR / "london_population_2020_100m.tif").exists():
        processed_city_keys.append("london")
    if (DATA_PROCESSED_DIR / "berlin_population_2020_100m.tif").exists():
        processed_city_keys.append("berlin")
    # Add Paris if its population data exists
    if (DATA_PROCESSED_DIR / "paris_population_2020_100m.tif").exists():
         processed_city_keys.append("paris")
    elif Path(DATA_RAW_DIR / WORLDPOP_CONFIG.get("paris", "")).exists(): # Check if raw Paris pop data is there now
        print("Paris raw population data found. Consider re-running data_processing.py to generate its processed raster.")

    if not processed_city_keys:
        print("No cities with processed population data found. Aborting feature engineering.")
        print(f"Please check {DATA_PROCESSED_DIR} for city-specific population rasters.")
        return

    all_features_gdfs = []
    for city_key in processed_city_keys:
        city_features_gdf = create_features_for_city(city_key, H3_RESOLUTION)
        if city_features_gdf is not None and not city_features_gdf.empty:
            all_features_gdfs.append(city_features_gdf)
    
    if all_features_gdfs:
        # Combine features from all cities into one GeoDataFrame
        # This is useful for training a single model
        combined_features_gdf = pd.concat(all_features_gdfs, ignore_index=True)
        print(f"\nTotal combined features from {len(all_features_gdfs)} cities: {len(combined_features_gdf)} H3 cells.")
        
        # Save combined features
        combined_output_filename = f"all_cities_features_h{H3_RESOLUTION}.gpkg"
        combined_output_path = DATA_PROCESSED_DIR / combined_output_filename
        try:
            combined_features_gdf.to_file(combined_output_path, driver="GPKG")
            print(f"Successfully saved combined features to {combined_output_path}")
        except Exception as e:
            print(f"Error saving combined features: {e}")
            # Fallback to GeoJSON for combined features
            for col in combined_features_gdf.columns:
                 if combined_features_gdf[col].apply(lambda x: isinstance(x, list)).any():
                    combined_features_gdf[col] = combined_features_gdf[col].astype(str)
            combined_output_filename_geojson = f"all_cities_features_h{H3_RESOLUTION}.geojson"
            combined_output_path_geojson = DATA_PROCESSED_DIR / combined_output_filename_geojson
            try:
                combined_features_gdf.to_file(combined_output_path_geojson, driver="GeoJSON")
                print(f"Successfully saved combined features to {combined_output_path_geojson} (GeoJSON fallback)")
            except Exception as e2:
                print(f"Error saving combined features to GeoJSON: {e2}")

    else:
        print("No features were generated for any city.")
    print("--- Feature Engineering pipeline complete. ---")

# Need WORLDPOP_CONFIG from data_processing.py for the check in run_feature_engineering_pipeline
# This is a bit of a circular dependency if not managed well. For now, let's redefine it or assume it's available.
# A better way would be to have a shared config module.
WORLDPOP_CONFIG = {
    "london": "gbr_ppp_2020_100m_unconstrained.tif",
    "paris": "fra_ppp_2020_100m_unconstrained.tif",
    "berlin": "deu_ppp_2020_100m_unconstrained.tif"
}

if __name__ == '__main__':
    run_feature_engineering_pipeline() 