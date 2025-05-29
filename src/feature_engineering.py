#!/opt/homebrew/Caskroom/miniconda/base/envs/optimal_station/bin/python
import geopandas as gpd
import pandas as pd
import numpy as np
import h3
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon

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

    # Get the first geometry (assuming it's the main city boundary)
    boundary_polygon = city_boundary_gdf.geometry.iloc[0]
    
    # Polyfill the boundary with H3 cells
    all_hexagons = set() # Use a set to store unique hexagon IDs

    def polygon_to_h3_cells(poly, h3_resolution):
        coords = [list(poly.exterior.coords)]
        if poly.interiors:
            coords += [list(ring.coords) for ring in poly.interiors]
        geojson_dict = {
            "type": "Polygon",
            "coordinates": coords
            }
        return h3.polygon_to_cells(coords, h3_resolution)  # Pass the list of coordinate rings directly

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
    # Convert H3 indices to Shapely polygons
    h3_gdf['geometry'] = h3_gdf['h3_index'].apply(lambda x: Polygon(h3.h3_to_geo_boundary(x, geo_json=True)))
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

def count_points_in_polygons(polygons_gdf, points_gdf, count_col_name):
    """
    Counts the number of points from points_gdf that fall within each polygon of polygons_gdf.

    Args:
        polygons_gdf (gpd.GeoDataFrame): GeoDataFrame of polygons.
        points_gdf (gpd.GeoDataFrame): GeoDataFrame of points.
        count_col_name (str): Name for the new column in polygons_gdf with the counts.

    Returns:
        gpd.GeoDataFrame: polygons_gdf with an added column for point counts.
    """
    if polygons_gdf.empty:
        return polygons_gdf
    if points_gdf.empty:
        polygons_gdf[count_col_name] = 0
        return polygons_gdf

    # Ensure CRSs match
    if polygons_gdf.crs != points_gdf.crs:
        # print(f"Reprojecting points from {points_gdf.crs} to {polygons_gdf.crs} for spatial join.")
        try:
            points_gdf = points_gdf.to_crs(polygons_gdf.crs)
        except Exception as e:
            print(f"Error reprojecting points for {count_col_name}: {e}. Returning zero counts.")
            polygons_gdf[count_col_name] = 0
            return polygons_gdf

    # Perform spatial join
    # 'sjoin' with 'op=contains' for polygons containing points, or 'op=within' for points within polygons
    # We want to count points per polygon, so we do a left join from polygons to points
    joined_gdf = gpd.sjoin(polygons_gdf, points_gdf, how="left", predicate="contains")

    # Group by polygon index and count points
    # The join will create duplicate polygon rows if multiple points are in one polygon
    # The index of points_gdf (e.g., 'index_right') can be used for counting
    counts = joined_gdf.groupby(joined_gdf.index).size()
    # If a polygon had no points, it might result in a count of 1 due to the left join (with NaN point cols)
    # or not appear in groupby if there were truly no points. Let's refine:
    
    # A more robust way for point counts after left sjoin:
    # Count non-NA values in a column from the right GeoDataFrame (points_gdf)
    # Pick a column that should always be present in points_gdf, e.g., its original index if it was reset
    # Or simply count the occurrences of each polygon's original index in the joined table.
    # If points_gdf has an 'index_right' after join, this indicates a match.
    if 'index_right' in joined_gdf.columns: # 'index_right' is the default name for the index of the right GDF
        counts = joined_gdf.groupby(polygons_gdf.index)['index_right'].count() # Count actual joined points
    else: # If 'index_right' is not present, implies no points were joined or sjoin had issues
        counts = pd.Series(0, index=polygons_gdf.index)

    polygons_gdf[count_col_name] = counts.fillna(0).astype(int)
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
            # Perform a spatial join. H3 cells that contain a station.
            # A cell 'has_station' if its geometry contains the geometry of any station.
            # We can use a temporary column to mark joined cells.
            joined_stations = gpd.sjoin(h3_grid_gdf, stations_gdf, how='left', predicate='contains')
            # If a h3 cell contains multiple stations, it will be duplicated.
            # We only care if *any* station is present.
            # Group by the original h3_grid_gdf index and check if 'index_right' (from stations_gdf) is not NaN.
            if 'index_right' in joined_stations.columns:
                station_presence = joined_stations.groupby(h3_grid_gdf.index)['index_right'].notna().any(level=0)
                h3_grid_gdf['has_station'] = station_presence.astype(int)
            else:
                 h3_grid_gdf['has_station'] = 0
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