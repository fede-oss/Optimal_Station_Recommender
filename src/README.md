# Source Code Organization

This directory contains the refactored and organized source code for the Optimal Station Recommender project.

## Directory Structure

```
src/
├── config/          # Configuration files and constants
│   ├── __init__.py
│   ├── config.py    # Main configuration (cities, amenities, paths)
│   └── ...
├── data/            # Data processing modules
│   ├── __init__.py
│   ├── worldpop_downloader.py    # WorldPop population data downloading
│   ├── simple_osm_fetcher.py     # Basic OSM data fetching
│   ├── osm_utils.py              # Advanced OSM utilities (chunking, caching)
│   ├── population_utils.py       # Population data processing
│   ├── data_processing.py        # Main data processing pipeline
│   └── ...
├── ml/              # Machine learning modules
│   ├── __init__.py
│   ├── feature_engineering.py    # H3 grid generation and feature creation
│   ├── model_training.py         # Model training and evaluation
│   ├── station_ranking.py        # Station suitability ranking
│   └── ...
├── web/             # Web application
│   ├── __init__.py
│   ├── app.py                    # Flask backend API
│   ├── city_search.html          # City search interface
│   └── static/                   # Static web assets
└── utils/           # Utility modules
    ├── __init__.py
    └── ...
```

## Key Improvements from Refactoring

### 1. **Modular Organization**
- Separated concerns into logical modules
- Each directory has a specific responsibility
- Easier to navigate and maintain

### 2. **Reduced Complexity**
- Split the original 1100+ line `data_processing.py` into smaller, focused modules
- Removed redundant WorldPop downloader code (already existed in separate file)
- Cleaner, more maintainable codebase

### 3. **Better Import Structure**
- Each module properly imports dependencies
- Path resolution works from any directory
- Consistent import patterns across all modules

### 4. **Configuration Management**
- Centralized configuration in `config/config.py`
- Consistent path definitions across all modules
- Easy to modify settings in one place

## Usage

### Running the Web Application
```bash
cd src/web
python app.py
```

### Running Feature Engineering
```bash
cd src
python -m data.worldpop_downloader --city london
python -m data.data_processing
```

### Training Models
```bash
# Train models and generate rankings
python src/ml/model_training.py
python src/ml/station_ranking.py
```

## Import Guidelines

When importing from these modules in your scripts:

1. **From within src directory:**
   ```python
   from config.config import CITIES
   from ml.feature_engineering import create_features_for_city
   ```

2. **From subdirectories (like web/):**
   ```python
   import sys, os
   src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   sys.path.insert(0, src_dir)
   
   from config.config import CITIES
   # ... other imports
   ```

## Backward Compatibility

The refactored modules maintain the same public API as the original files, so existing scripts that import these functions should continue to work with minimal changes to their import statements.
