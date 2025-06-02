#!/usr/bin/env python3
"""
worldpop_downloader.py - Automatic WorldPop Data Downloader

This script automatically downloads population data from WorldPop for specified countries.
It retrieves the unconstrained population per pixel (ppp) datasets at 100m resolution
directly from WorldPop's public S3 bucket.

Note: Previously tried using the REST API but it returned 500 errors.
"""

import requests
import os
import sys
import argparse
import json
from pathlib import Path
import time

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# WorldPop data URLs
WORLDPOP_HUB = "https://hub.worldpop.org/geodata/summary?id=24777"
WORLDPOP_DATA = "https://data.worldpop.org"

# Global dataset fallback URL (100m resolution)
WORLDPOP_GLOBAL_DATASET = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated_UNadj.tif"

# Country data (ISO code and year of latest available data)
COUNTRY_DATA = {
    # Original cities
    "london": {"iso": "GBR", "year": 2020},    # United Kingdom
    "paris": {"iso": "FRA", "year": 2020},     # France
    "berlin": {"iso": "DEU", "year": 2020},    # Germany
    
    # New cities
    "tokyo": {"iso": "JPN", "year": 2020},     # Japan
    "singapore": {"iso": "SGP", "year": 2020}, # Singapore
    "hong_kong": {"iso": "HKG", "year": 2020}, # Hong Kong
    "seoul": {"iso": "KOR", "year": 2020},     # South Korea
    "madrid": {"iso": "ESP", "year": 2020},    # Spain
}

def download_from_s3(country_iso, year, output_path):
    """
    Download WorldPop dataset directly from the WorldPop server.
    
    Args:
        country_iso (str): ISO 3166-1 alpha-3 country code (e.g., "GBR" for UK)
        year (int): Year of the dataset
        output_path (Path): Path to save the downloaded dataset
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    country_lower = country_iso.lower()
    
    # Construct various URL patterns based on WorldPop's different data structures
    url_patterns = [
        # This is the pattern that works for Japan and likely others (2018 structure)
        f"{WORLDPOP_DATA}/GIS/Population/Global_2000_2020/2020/{country_iso}/{country_lower}_ppp_{year}.tif",
        
        # 2019 Global Mosaic (100m)
        f"{WORLDPOP_DATA}/GIS/Population/Global_2000_2020_Constrained/2020/{country_lower}_ppp_{year}_constrained_100m.tif",
        
        # UN-adjusted datasets (100m) - newer format
        f"{WORLDPOP_DATA}/GIS/Population/Global_2000_2020/2020/UNadj/{country_iso}/{country_lower}_ppp_{year}_UNadj.tif",
        
        # Individual country unconstrained datasets (100m)
        f"{WORLDPOP_DATA}/GIS/Population/Global_2000_2020/2020/{country_iso}/{country_lower}_ppp_{year}_unconstrained.tif",
        
        # Individual country constrained datasets (100m)
        f"{WORLDPOP_DATA}/GIS/Population/Global_2000_2020/2020/{country_iso}/{country_lower}_ppp_{year}_constrained.tif",
        
        # Individual country constrained datasets (100m) - alternative format
        f"{WORLDPOP_DATA}/GIS/Population/Global/2000_2020/{year}/{country_iso}/{country_lower}_ppp_{year}_constrained.tif",
        
        # Individual country constrained datasets (100m) - alternative format 2
        f"{WORLDPOP_DATA}/GIS/Population/Global/2000_2020/{year}/BSGM/{country_iso}/{country_lower}_ppp_{year}_1km_Aggregated_UNadj.tif",
        
        # Individual country constrained datasets (1km) - general format
        f"{WORLDPOP_DATA}/GIS/Population/Global_2000_2020/2020/0_Mosaicked/{country_lower}_ppp_{year}_1km_Aggregated.tif",
        
        # Alternative format for some Asian countries (tested with Japan)
        f"{WORLDPOP_DATA}/GIS/Population/JPN_POP/{country_lower}_ppp_{year}_1km_Aggregated.tif",
        
        # GPWv4 alternative source - some countries
        f"https://geodata.ucdavis.edu/gpw/gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_{year}.tif"
    ]
    
    # Try each URL until one works
    for url in url_patterns:
        try:
            print(f"Trying to download from: {url}")
            response = requests.head(url)
            
            # If the file exists, download it
            if response.status_code == 200:
                print(f"Found valid file at: {url}")
                return download_file(url, output_path)
            else:
                print(f"File not found at: {url} (Status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Error checking URL {url}: {e}")
    
    # Try downloading global dataset as a fallback
    print("\nTrying to download global population dataset as fallback...")
    try:
        response = requests.head(WORLDPOP_GLOBAL_DATASET)
        if response.status_code == 200:
            print(f"Found global dataset at: {WORLDPOP_GLOBAL_DATASET}")
            print(f"WARNING: This is a global dataset at 1km resolution rather than a country-specific one.")
            print(f"It will work for feature extraction but may be less precise.")
            return download_file(WORLDPOP_GLOBAL_DATASET, output_path)
    except requests.exceptions.RequestException as e:
        print(f"Error checking global dataset: {e}")
    
    # As a last resort, suggest manual download
    print(f"\nCould not find any valid WorldPop dataset for {country_iso}")
    print("\nSuggested manual download options:")
    print(f"1. Visit WorldPop Hub: https://hub.worldpop.org/")
    print(f"2. Search for '{country_iso}' or the full country name")
    print(f"3. Filter for 'Population' datasets from {year}")
    print(f"4. Download a suitable dataset and place it in: {DATA_RAW_DIR}")
    print(f"5. Rename the file to: {country_lower}_ppp_{year}_100m_unconstrained.tif")
    
    return False

def download_file(url, output_path):
    """
    Download a file with progress reporting.
    
    Args:
        url (str): URL to download from
        output_path (Path): Path to save the downloaded file
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        print(f"Downloading from {url} to {output_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Download the file with progress reporting
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                total_size_mb = total_size / (1024 * 1024)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_size)
                        progress = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {progress:.1f}/{total_size_mb:.1f} MB")
                        sys.stdout.flush()
                sys.stdout.write('\n')
        
        print(f"Successfully downloaded to {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def download_population_data_for_cities():
    """
    Download population data for all cities in the COUNTRY_DATA dictionary.
    """
    print("=== Starting WorldPop Data Download ===")
    
    for city, country_info in COUNTRY_DATA.items():
        country_iso = country_info['iso']
        year = country_info['year']
        
        print(f"\nProcessing data for {city.capitalize()} ({country_iso})...")
        
        # Check if we already have a file for this country
        expected_filename = f"{country_iso.lower()}_ppp_{year}_100m_unconstrained.tif"
        output_path = DATA_RAW_DIR / expected_filename
        
        if output_path.exists():
            print(f"Found existing population file: {expected_filename}")
            print(f"Skipping download for {city}. Delete the file if you want to re-download.")
            continue
        
        # Download the dataset
        success = download_from_s3(country_iso, year, output_path)
        
        if success:
            print(f"Successfully downloaded population data for {city}")
        else:
            print(f"Failed to download population data for {city}")
            # Try to create an empty file as a placeholder to indicate we tried
            try:
                with open(output_path, 'w') as f:
                    f.write("Download failed - please download manually")
            except:
                pass
        
        # Be nice to the server
        time.sleep(2)
    
    print("\n=== WorldPop Data Download Complete ===")

def main():
    parser = argparse.ArgumentParser(description="Download WorldPop population data for specified cities")
    parser.add_argument('--city', help='Specific city to download data for (e.g., tokyo)')
    parser.add_argument('--list', action='store_true', help='List available cities and their country codes')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output including all attempted URLs')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available cities and their country codes:")
        for city, info in COUNTRY_DATA.items():
            print(f"- {city.capitalize()}: {info['iso']} (year: {info['year']})")
        return
        
    if args.city:
        city = args.city.lower()
        if city not in COUNTRY_DATA:
            print(f"Error: City '{args.city}' not found in the configuration.")
            print("Available cities:")
            for city in COUNTRY_DATA.keys():
                print(f"- {city}")
            return
            
        # Filter to only process the specified city
        city_info = COUNTRY_DATA[city]
        temp_country_data = {city: city_info}
        COUNTRY_DATA.clear()
        COUNTRY_DATA.update(temp_country_data)
        
    # Download population data for the specified or all cities
    download_population_data_for_cities()

if __name__ == "__main__":
    main()
