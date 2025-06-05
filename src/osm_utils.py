"""
OSM (OpenStreetMap) utilities for fetching and processing geographic data.

This module contains all the complex OSM data fetching logic including:
- Enhanced progress tracking with persistence
- Intelligent chunking for large areas
- Robust retry mechanisms
- Caching functionality
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import time
import threading
import json
import requests
import concurrent.futures
from pathlib import Path
from shapely.geometry import box
from shapely.ops import unary_union
import pickle
import hashlib

# Thread-local storage for OSMnx settings to avoid conflicts during parallel processing
thread_local = threading.local()


class OSMProgressTracker:
    """Enhanced progress tracker for OSM data fetching operations with persistence."""
    
    def __init__(self, operation_name: str, total_chunks: int = 1, cache_dir: Path = None):
        self.operation_name = operation_name
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self.failed_chunks = 0
        self.skipped_chunks = 0
        self.start_time = time.time()
        self.last_update = 0
        self.lock = threading.Lock()
        self.update_interval = 1.0  # Update every 1 second
        self.cache_dir = cache_dir
        self.chunk_details = {}  # Track individual chunk progress
        self.progress_file = cache_dir / f"{operation_name.replace(' ', '_')}_progress.json" if cache_dir else None
        self.completed_chunk_ids = set()
        
        # Load previous progress if exists
        self._load_progress()
        
    def _load_progress(self):
        """Load progress from previous session."""
        if self.progress_file and self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_chunk_ids = set(data.get('completed_chunks', []))
                    self.completed_chunks = len(self.completed_chunk_ids)
                    print(f"📂 Resuming from previous session: {self.completed_chunks} chunks already completed")
            except Exception as e:
                print(f"⚠️ Could not load progress file: {e}")
    
    def _save_progress(self):
        """Save current progress to file."""
        if self.progress_file:
            try:
                self.progress_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.progress_file, 'w') as f:
                    json.dump({
                        'completed_chunks': list(self.completed_chunk_ids),
                        'total_chunks': self.total_chunks,
                        'operation': self.operation_name,
                        'last_updated': time.time()
                    }, f)
            except Exception as e:
                print(f"⚠️ Could not save progress: {e}")
        
    def update(self, chunk_completed: bool = True, chunk_id: str = None, chunk_failed: bool = False, chunk_skipped: bool = False):
        """Update progress with thread safety and persistence."""
        with self.lock:
            if chunk_completed and chunk_id:
                if chunk_id not in self.completed_chunk_ids:
                    self.completed_chunk_ids.add(chunk_id)
                    self.completed_chunks = len(self.completed_chunk_ids)
                    self._save_progress()
            elif chunk_failed:
                self.failed_chunks += 1
            elif chunk_skipped:
                self.skipped_chunks += 1
                
            current_time = time.time()
            
            if current_time - self.last_update >= self.update_interval or self.completed_chunks == self.total_chunks:
                self._print_progress()
                self.last_update = current_time
    
    def is_chunk_completed(self, chunk_id: str) -> bool:
        """Check if a chunk was already completed."""
        return chunk_id in self.completed_chunk_ids
    
    def _print_progress(self):
        """Print formatted progress bar with enhanced information."""
        if self.total_chunks == 0:
            return
            
        progress = self.completed_chunks / self.total_chunks
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.completed_chunks > 0 and elapsed > 0:
            rate = self.completed_chunks / elapsed  # chunks per second
            remaining_chunks = self.total_chunks - self.completed_chunks
            eta_seconds = remaining_chunks / rate if rate > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Additional stats
        stats = f"F:{self.failed_chunks} S:{self.skipped_chunks}" if (self.failed_chunks > 0 or self.skipped_chunks > 0) else ""
        
        # Print with carriage return to overwrite
        print(f"\r🗺️ {self.operation_name}: [{bar}] {progress:.1%} ({self.completed_chunks}/{self.total_chunks}) ETA: {eta_str} {stats}", 
              end='', flush=True)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def finish(self):
        """Print completion message."""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {self.operation_name} completed in {self._format_time(elapsed)}")


def create_cache_key(place_name, tags, chunk_info=None):
    """Create a unique cache key for OSM queries."""
    # Convert everything to string and create hash
    key_data = {
        'place': str(place_name),
        'tags': str(sorted(tags.items())),
        'chunk': str(chunk_info) if chunk_info else 'full'
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def save_cache(cache_key, data, cache_dir):
    """Save data to cache."""
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not save cache: {e}")
        return False


def load_cache(cache_key, cache_dir):
    """Load data from cache."""
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load cache: {e}")
            # Delete corrupt cache file
            cache_file.unlink(missing_ok=True)
    return None


def get_place_boundary(place_name, max_retries=3):
    """Get the boundary of a place with retry logic."""
    for attempt in range(max_retries):
        try:
            # Configure OSMnx for this query
            ox.settings.timeout = 60
            ox.settings.use_cache = True
            boundary_gdf = ox.geocode_to_gdf(place_name)
            if not boundary_gdf.empty:
                # Use shapely.ops.unary_union for geometry union
                return unary_union(boundary_gdf.geometry.tolist())
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed to get boundary for {place_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"❌ Failed to get boundary for {place_name} after {max_retries} attempts")
    return None


def create_grid_chunks(boundary, chunk_size_km=5):
    """Create a grid of chunks to query OSM data in smaller pieces."""
    if boundary is None:
        return []
    
    # Get bounds
    minx, miny, maxx, maxy = boundary.bounds
    
    # Convert km to degrees (very rough approximation)
    # 1 degree ≈ 111 km at equator
    chunk_size_deg = chunk_size_km / 111.0
    
    chunks = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            # Create chunk boundary
            chunk_bounds = box(x, y, min(x + chunk_size_deg, maxx), min(y + chunk_size_deg, maxy))
            
            # Only include chunks that intersect with the actual boundary
            if chunk_bounds.intersects(boundary):
                intersection = chunk_bounds.intersection(boundary)
                if not intersection.is_empty:
                    chunks.append({
                        'bounds': chunk_bounds,
                        'intersection': intersection,
                        'id': f"chunk_{len(chunks)}"
                    })
            
            y += chunk_size_deg
        x += chunk_size_deg
    
    return chunks


def create_smart_chunks(boundary, base_chunk_size_km=5, max_chunks=2000, min_chunk_size_km=1):
    """
    Create smart adaptive chunks that balance performance and thoroughness.
    Uses larger chunks first, then subdivides dense areas if needed.
    """
    # For very large cities like Mexico City, start with even larger chunks
    if max_chunks > 5000:  # Very large dataset expected
        initial_chunk_size = min(base_chunk_size_km * 3, 15)
    else:
        initial_chunk_size = min(base_chunk_size_km * 2, 10)
    
    # Create initial grid
    chunks = create_grid_chunks(boundary, initial_chunk_size)
    
    print(f"🧠 Smart chunking: Starting with {len(chunks)} large chunks ({initial_chunk_size}km each)")
    
    # If we have too many chunks, use even larger chunks
    if len(chunks) > max_chunks:
        larger_chunk_size = initial_chunk_size * 1.5
        chunks = create_grid_chunks(boundary, larger_chunk_size)
        print(f"🧠 Reduced to {len(chunks)} chunks ({larger_chunk_size:.1f}km each) to stay under {max_chunks} limit")
    
    # Skip water areas if possible
    try:
        water_chunks_skipped = 0
        filtered_chunks = []
        for chunk in chunks:
            # Simple heuristic: if chunk is mostly water, skip it
            if not _is_likely_water_area(chunk, boundary):
                filtered_chunks.append(chunk)
            else:
                water_chunks_skipped += 1
        
        if water_chunks_skipped > 0:
            print(f"🌊 Skipped {water_chunks_skipped} chunks likely to be water areas")
            chunks = filtered_chunks
    except Exception as e:
        print(f"⚠️ Could not filter water areas: {e}")
    
    return chunks


def _is_likely_water_area(chunk, boundary):
    """
    Simple heuristic to detect if a chunk is likely to be mostly water.
    This is a basic implementation that can be enhanced with actual water body data.
    """
    try:
        # If the chunk intersection with boundary is very small compared to chunk size,
        # it might be mostly water/outside the city
        chunk_area = chunk['bounds'].area
        intersection_area = chunk['intersection'].area
        
        # If less than 20% of chunk is actual city area, likely water/outside
        if intersection_area / chunk_area < 0.2:
            return True
            
        # Additional heuristic: check chunk position relative to boundary centroid
        boundary_centroid = boundary.centroid
        chunk_centroid = chunk['bounds'].centroid
        
        # If chunk is very far from city center, might be water
        distance = chunk_centroid.distance(boundary_centroid)
        boundary_radius = (boundary.area ** 0.5) / 2  # Rough radius estimate
        
        if distance > boundary_radius * 1.5:
            return True
            
    except Exception:
        pass
    
    return False


def query_osm_chunk(chunk, tags, timeout=120, max_retries=5):
    """Query OSM data for a single chunk with enhanced retry logic and error handling."""
    chunk_id = chunk.get('id', 'unknown')
    
    for attempt in range(max_retries):
        try:
            # Configure OSMnx for this query with adaptive timeout
            adaptive_timeout = min(timeout * (attempt + 1), timeout * 3)  # Increase timeout on retries
            ox.settings.timeout = adaptive_timeout
            ox.settings.use_cache = True
            ox.settings.max_query_area_size = 50*1000*1000  # 50 sq km
            
            # Query using the chunk bounds
            bounds = chunk['bounds']
            north, south, east, west = bounds.bounds[3], bounds.bounds[1], bounds.bounds[2], bounds.bounds[0]
            
            # Create bbox tuple: (left, bottom, right, top) = (west, south, east, north)
            bbox = (west, south, east, north)
            
            # Add small delay to avoid overwhelming the server
            if attempt > 0:
                time.sleep(min(2 ** attempt, 30))  # Cap at 30 seconds
            
            gdf = ox.features_from_bbox(bbox, tags)
            
            if not gdf.empty:
                # Filter to only features that actually intersect with our area of interest
                gdf = gdf[gdf.geometry.intersects(chunk['intersection'])]
            
            return gdf
            
        except requests.exceptions.Timeout as e:
            print(f"⏱️ Chunk {chunk_id} attempt {attempt + 1}/{max_retries} timed out after {adaptive_timeout}s")
            if attempt < max_retries - 1:
                time.sleep(min(5 * (attempt + 1), 60))  # Longer delay for timeouts
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Chunk {chunk_id} attempt {attempt + 1}/{max_retries} connection error")
            if attempt < max_retries - 1:
                time.sleep(min(10 * (attempt + 1), 120))  # Even longer delay for connection issues
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"⚠️ Chunk {chunk_id} attempt {attempt + 1}/{max_retries} failed: {error_msg}...")
            
            # Special handling for specific error types
            if "too many requests" in error_msg.lower() or "rate limit" in error_msg.lower():
                print(f"🚫 Rate limit detected for chunk {chunk_id}, waiting longer...")
                if attempt < max_retries - 1:
                    time.sleep(min(30 * (attempt + 1), 300))  # Up to 5 minutes for rate limits
            elif "server error" in error_msg.lower() or "502" in error_msg or "503" in error_msg:
                print(f"🔧 Server error detected for chunk {chunk_id}, waiting for server recovery...")
                if attempt < max_retries - 1:
                    time.sleep(min(15 * (attempt + 1), 180))  # Up to 3 minutes for server errors
            else:
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))  # Standard exponential backoff
    
    print(f"❌ Failed to fetch chunk {chunk_id} after {max_retries} attempts")
    return gpd.GeoDataFrame()


def get_osm_geometries(place_name, tags, timeout=300, use_chunking=True, chunk_size_km=5, max_workers=4):
    """
    Fetches geometries from OpenStreetMap for a given place and tags with enhanced robustness.

    Args:
        place_name (str or dict): The name of the place to query (e.g., "London, UK")
        tags (dict): A dictionary of OSM tags to filter by (e.g., {"amenity": "restaurant"})
        timeout (int): Timeout in seconds per chunk. Default is 300 seconds
        use_chunking (bool): Whether to break large areas into smaller chunks. Default is True
        chunk_size_km (float): Size of each chunk in kilometers. Default is 5km
        max_workers (int): Number of parallel workers for chunked queries. Default is 4

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the geometries.
                                Returns an empty GeoDataFrame if no features are found or an error occurs.
    """
    # Set up caching
    cache_dir = Path("cache/osm_data")
    cache_key = create_cache_key(place_name, tags, {'chunking': use_chunking, 'chunk_size': chunk_size_km})
    
    # Try to load from cache first
    cached_data = load_cache(cache_key, cache_dir)
    if cached_data is not None:
        print(f"✅ Loaded OSM data from cache for {place_name}")
        return cached_data
    
    print(f"🔍 Fetching OSM data for {place_name} with tags {tags}")
    
    # Use thread-local settings to avoid conflicts in parallel processing
    if not hasattr(thread_local, 'ox_configured'):
        ox.settings.use_cache = True
        ox.settings.log_console = False  # Reduce noise
        thread_local.ox_configured = True
    
    try:
        if use_chunking:
            # Get place boundary first
            print(f"📍 Getting boundary for {place_name}...")
            boundary = get_place_boundary(place_name)
            
            if boundary is None:
                print(f"❌ Could not get boundary for {place_name}, falling back to simple query")
                return get_osm_geometries(place_name, tags, timeout, use_chunking=False)
            
            # Create smart chunks with optimization
            print(f"🗂️ Creating optimized chunks ({chunk_size_km}km base size)...")
            chunks = create_smart_chunks(boundary, chunk_size_km, max_chunks=3000)
            
            if len(chunks) == 0:
                print(f"❌ No valid chunks created for {place_name}")
                return gpd.GeoDataFrame()
            
            print(f"📊 Created {len(chunks)} optimized chunks for {place_name}")
            
            # Set up enhanced progress tracking with persistence
            operation_name = f"Fetching {place_name} {list(tags.keys())[0] if tags else 'data'}"
            progress = OSMProgressTracker(operation_name, len(chunks), cache_dir)
            
            # Query chunks with resume capability
            all_gdfs = []
            failed_chunks = []
            
            def query_chunk_with_enhanced_progress(chunk):
                chunk_id = chunk.get('id', f"chunk_{len(all_gdfs)}")
                
                # Check if chunk already completed
                if progress.is_chunk_completed(chunk_id):
                    progress.update(chunk_skipped=True)
                    return None
                
                # Query the chunk with enhanced retry logic
                gdf = query_osm_chunk(chunk, tags, timeout, max_retries=5)
                
                if not gdf.empty:
                    progress.update(chunk_completed=True, chunk_id=chunk_id)
                    return gdf
                else:
                    # Even empty results count as completed to avoid re-querying
                    progress.update(chunk_completed=True, chunk_id=chunk_id)
                    return None
            
            if len(chunks) == 1 or max_workers == 1:
                # Single-threaded for small areas or if requested
                for i, chunk in enumerate(chunks):
                    chunk['id'] = f"chunk_{i}"  # Ensure chunk has ID
                    gdf = query_chunk_with_enhanced_progress(chunk)
                    if gdf is not None and not gdf.empty:
                        all_gdfs.append(gdf)
            else:
                # Multi-threaded for larger areas with controlled concurrency
                print(f"🔄 Processing chunks with {max_workers} parallel workers...")
                
                # Add IDs to chunks
                for i, chunk in enumerate(chunks):
                    chunk['id'] = f"chunk_{i}"
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all chunks
                    future_to_chunk = {
                        executor.submit(query_chunk_with_enhanced_progress, chunk): chunk 
                        for chunk in chunks
                    }
                    
                    # Process completed futures
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        try:
                            gdf = future.result()
                            if gdf is not None and not gdf.empty:
                                all_gdfs.append(gdf)
                        except Exception as e:
                            chunk = future_to_chunk[future]
                            print(f"❌ Error processing chunk {chunk.get('id', 'unknown')}: {e}")
                            failed_chunks.append(chunk)
            
            progress.finish()
            
            # Report on failed chunks
            if failed_chunks:
                print(f"⚠️ {len(failed_chunks)} chunks failed completely but download continued")
                print(f"   You can retry these chunks later if needed")
            
            # Combine all results
            if all_gdfs:
                print(f"🔄 Combining {len(all_gdfs)} chunk results...")
                combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
                
                # Remove duplicates (features might appear in multiple chunks)
                if 'osmid' in combined_gdf.columns:
                    original_count = len(combined_gdf)
                    combined_gdf = combined_gdf.drop_duplicates(subset=['osmid'])
                    deduplicated_count = len(combined_gdf)
                    if original_count != deduplicated_count:
                        print(f"🔄 Removed {original_count - deduplicated_count} duplicate features")
                
                print(f"✅ Successfully fetched {len(combined_gdf)} features for {place_name}")
                
                # Cache the result
                save_cache(cache_key, combined_gdf, cache_dir)
                
                return combined_gdf
            else:
                print(f"❌ No features found for {place_name}")
                return gpd.GeoDataFrame()
        
        else:
            # Simple single query (fallback mode)
            print(f"🔄 Using simple query for {place_name}...")
            ox.settings.timeout = timeout
            ox.settings.max_query_area_size = 50*1000*1000  # 50 sq km
            
            gdf = ox.features_from_place(place_name, tags)
            
            if len(gdf) == 0:
                print(f"❌ No features found for {place_name}")
            elif len(gdf) > 10000:
                print(f"⚠️ Large dataset ({len(gdf)} features) - consider using chunking for better performance")
            else:
                print(f"✅ Successfully fetched {len(gdf)} features for {place_name}")
            
            # Cache the result
            save_cache(cache_key, gdf, cache_dir)
            
            return gdf
    
    except Exception as e:
        print(f"❌ Error fetching OSM data for {place_name} with tags {tags}: {e}")
        print(f"   This might be because the area is too large or there are connection issues.")
        print(f"   The system will attempt to resume from where it left off on the next run.")
        return gpd.GeoDataFrame()
