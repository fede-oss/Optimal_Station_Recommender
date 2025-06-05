# Optimal Station Recommender

A data science pipeline for recommending optimal station locations in cities using population density data and advanced algorithms.

## Quick Start

### Adding a New City

To add a new city to the analysis pipeline, use the flag-based approach:

```bash
# Add a new city (interactive)
python src/data_processing.py --add-city paris "Paris, France" FRA

# List all configured cities
python src/data_processing.py --list-cities

# Process data for a specific city
python src/data_processing.py --city paris
```

### Downloading Population Data

The pipeline uses WorldPop population data. To download data for cities:

```bash
# Download data for all configured cities
python src/worldpop_downloader.py

# Download for a specific city
python src/data/worldpop_downloader.py --city madrid

# Add new city to downloader and download immediately
python src/data/worldpop_downloader.py --add-city vancouver CAN

# List cities and their download status
python src/worldpop_downloader.py --list
```

## Features

- **Easy City Addition**: Add new cities using command-line flags without manual dictionary editing
- **Robust Downloads**: Multi-threaded chunked downloads with retry logic and progress tracking
- **Real-time Progress**: Detailed progress bars with download speed and ETA
- **Flexible Configuration**: Support for different data years and country codes
- **Resume Capability**: Resume interrupted downloads automatically

## Pipeline Architecture

1. **City Configuration**: Define cities with their search queries and country codes
2. **Population Data**: Download WorldPop raster data for demographic analysis
3. **Data Processing**: Process and analyze population density patterns
4. **Station Optimization**: Apply algorithms to recommend optimal station locations

## Configuration

### Supported Cities

The pipeline currently supports:
- London, UK  
- Paris, France
- Berlin, Germany
- Madrid, Spain
- New York, USA
- Singapore
- Hong Kong
- Seoul, South Korea
- Toronto, Canada
- Mexico City, Mexico
- São Paulo, Brazil
- Buenos Aires, Argentina
- Lima, Peru
- Stockholm, Sweden
- Rome, Italy
- Warsaw, Poland
- Sydney, Australia

### Adding New Cities

1. **Via Command Line** (Recommended):
   ```bash
   python src/data_processing.py --add-city CITY_KEY "City Name, Country" ISO_CODE
   ```

2. **Manual Configuration**: Edit the `CITIES` and `WORLDPOP_CONFIG` dictionaries in `src/data_processing.py`

### WorldPop Data

- **Source**: WorldPop 100m resolution population datasets
- **Years**: 2020 (default), configurable per city
- **Format**: GeoTIFF raster files
- **Storage**: Downloaded to `data/raw/` directory

## Advanced Usage

### Download Options

```bash
# Use single-threaded downloads (if multi-threaded fails)
python src/worldpop_downloader.py --no-chunked

# Adjust number of download threads
python src/data/worldpop_downloader.py --workers 8

# Set maximum retry attempts
python src/data/worldpop_downloader.py --max-retries 5

# Verbose output with all attempted URLs
python src/data/worldpop_downloader.py --verbose
```

### Data Processing Options

```bash
# Interactive mode with prompts
python src/data_processing.py

# Process specific city with custom parameters
python src/data/data_processing.py --city madrid

# Batch processing for multiple cities
python src/data_processing.py --batch
```

## Requirements

- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`):
  - requests
  - rasterio
  - geopandas
  - osmnx
  - numpy
  - matplotlib

## Directory Structure

```
Optimal_Station_Recommender/
├── src/
│   ├── data/
│   │   ├── data_processing.py      # Main pipeline with city management
│   │   ├── worldpop_downloader.py  # Population data downloader
│   │   └── ...
│   ├── config/
│   │   └── config.py              # Configuration settings
│   └── ...
├── data/
│   ├── raw/                    # Downloaded population datasets
│   ├── processed/              # Processed city data
│   └── results/                # Analysis results
└── README.md
```

## Troubleshooting

### Download Issues

1. **Connection Timeouts**: Use `--no-chunked` for single-threaded downloads
2. **File Not Found**: Check if the country/year combination is available on WorldPop
3. **Network Restrictions**: Some networks block multi-threaded downloads

### City Addition Issues

1. **Invalid ISO Code**: Use 3-letter ISO country codes (e.g., USA, GBR, JPN)
2. **City Not Found**: Ensure the city query is specific and includes country
3. **Data Availability**: Check WorldPop Hub for data availability by year

## Contributing

1. Add new cities using the flag-based approach
2. Test the complete pipeline before submitting changes
3. Update documentation for new features
4. Follow the existing code style and structure

## License

This project is part of a data science team project at KU.

data science team project
