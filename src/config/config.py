"""
Configuration module for the Optimal Station Recommender project.

This module contains all the configuration constants and data structures
used across the project, providing a central place for managing settings.
"""
from pathlib import Path

# --- Directory Configuration --- #
DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_PROCESSED_DIR = Path("data/processed")
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Cities Configuration --- #
CITIES = {
    # Original cities
    "london": "London, UK",
    "paris": "Paris, France",
    "berlin": "Berlin, Germany",
    # Additional cities with excellent public transport
    "singapore": "Singapore",
    "hong_kong": "Hong Kong",
    "seoul": "Seoul, South Korea",
    "madrid": "Madrid, Spain",
    # North American cities
    "new_york": "New York City, New York, USA",
    "toronto": "Toronto, Ontario, Canada",
    "mexico_city": "Mexico City, Mexico",
    # South American cities
    "sao_paulo": "São Paulo, Brazil",
    "buenos_aires": "Buenos Aires, Argentina",
    "lima": "Lima, Peru",
    # European cities
    "stockholm": "Stockholm, Sweden",
    "rome": "Rome, Italy",
    "warsaw": "Warsaw, Poland",
    # Oceania cities
    "sydney": "Sydney, Australia"
}

# --- OSM Tag Configurations --- #
STATION_TAGS = {'railway': 'station', 'station': 'subway'}

# Define amenity tags for feature extraction
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

# --- WorldPop Configuration --- #
# Filenames are generated using the ISO3 country code pattern by worldpop_downloader.py
WORLDPOP_CONFIG = {
    # Original cities
    "london": "gbr_ppp_2020_100m_unconstrained.tif",
    "paris": "fra_ppp_2020_100m_unconstrained.tif",
    "berlin": "deu_ppp_2020_100m_unconstrained.tif",
    "singapore": "sgp_ppp_2020_100m_unconstrained.tif",
    "hong_kong": "hkg_ppp_2020_100m_unconstrained.tif",
    "seoul": "kor_ppp_2020_100m_unconstrained.tif",
    "madrid": "esp_ppp_2020_100m_unconstrained.tif",
    "new_york": "usa_ppp_2020_100m_unconstrained.tif",
    "toronto": "can_ppp_2020_100m_unconstrained.tif",
    "mexico_city": "mex_ppp_2020_100m_unconstrained.tif",
    "sao_paulo": "bra_ppp_2020_100m_unconstrained.tif",
    "buenos_aires": "arg_ppp_2020_100m_unconstrained.tif",
    "lima": "per_ppp_2020_100m_unconstrained.tif",
    "stockholm": "swe_ppp_2020_100m_unconstrained.tif",
    "rome": "ita_ppp_2020_100m_unconstrained.tif",
    "warsaw": "pol_ppp_2020_100m_unconstrained.tif",
    "sydney": "aus_ppp_2020_100m_unconstrained.tif"
}
