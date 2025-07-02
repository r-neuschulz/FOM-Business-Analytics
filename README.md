# FOM-Business-Analytics

This repository contains materials and resources for the Business Analytics presentation at FOM Hochschule for the academic year 2025.

## Project Overview

We're correlating pollution data from OpenWeatherMap with traffic data from BASt to potentially guide policymakers, inform traffic planners, and enable data-driven urban development strategies. This analysis aims to provide insights into the relationship between environmental factors and transportation patterns.

## Course Information
- **Institution**: FOM Hochschule
- **Course**: Business Analytics
- **Academic Year**: 2025
- **Project Type**: Semester Project
- **Authors**: Richard Neuschulz and Björn Kerneker

## Contents
- Scripts for reference (01 to 10) 
- Instructions on how to run the BA Scripts (`README.md`, this file)
- Pre-generated Plots (everything in Graphs)
- Documentation and findings (`Presentation in Material/PP_Business_Analytics.pdf`)

## Getting Started

This repository is organized to support the application of the semester project.

### Prerequisites

To verify the downloaded OpenWeatherMap data, an API key is required. To obtain one:

1. Visit [OpenWeatherMap API Keys](https://home.openweathermap.org/api_keys)
2. Create an account or sign in
3. Generate a new API key

### Setup Instructions

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Complete Analytics Pipeline

The main application (`app.py`) provides a comprehensive 11-step pipeline for retrieving, visualizing, analyzing, and correlating BASt traffic data with OpenWeatherMap pollution data. By default, it filters BASt data to focus on three major German cities: Cologne, Berlin, and Düsseldorf.

**BASt Data Volume Information (as of 2025-06-24):**
- **Default (3 cities)**: 1,076 years worth of hourly data for 51 distinct stations (approximately 1.5 GB)
- **Germany-wide**: Over 35,000 years worth of data across all stations (approximately 70 GB)
- **Data availability**: Traffic data available up to and including 2023
- **Source**: Real-Life Measurements by BASt

**OpenWeatherMap API Integration:**
- **Educational API limit**: 50,000 requests per day, but yearly queries possible.
- **Pollution data available**: Starting from 2020-11-25
- **Pollutants tracked**: SO₂, NO₂, PM10, PM2.5, O₃, and CO. Also Air Quality Index, AQI.
- **Data granularity**: Hourly data
- **Source**: SILAM atmospheric composition model

#### Complete Pipeline Steps

The pipeline executes the following 11 steps in sequence:
- **Step 0**: Create BASt-OpenWeatherMap overlap visualization
- **Step 1**: Retrieve BASt traffic counting station coordinates from 2003-2023
- **Step 2**: Generate BASt locations visualizations (heatmap, year comparison)
- **Step 3**: Download and extract hourly traffic data from BASt servers
- **Step 4**: Create city-by-year stacked bar visualization
- **Step 5**: Create city heatmap visualization
- **Step 6**: Download OpenWeather hourly pollution data
- **Step 7**: Create download quality visualization
- **Step 8**: Create traffic vs pollution analysis
- **Step 9**: Perform association and correlation analysis
- **Step 10**: Perform deseasoned correlation analysis

#### Basic Usage

To run the complete analytics pipeline:

```bash
python app.py --api-key YOUR_API_KEY_HERE
```

This will execute all 11 steps sequentially, providing comprehensive analysis of traffic-pollution correlations.

#### Command Line Options

The pipeline supports command line options for different use cases:

```bash
# Basic usage with API key (required)
python app.py --api-key YOUR_API_KEY_HERE

# Force fresh data retrieval for coordinates
python app.py --api-key YOUR_API_KEY_HERE --fresh

# Run in test mode (limited URLs for testing)
python app.py --api-key YOUR_API_KEY_HERE --test

# Skip hourly data download (only creates city mapping and visualizations)
python app.py --api-key YOUR_API_KEY_HERE --skip-hourly

# Filter by specific cities (can specify multiple)
python app.py --api-key YOUR_API_KEY_HERE --city cologne berlin
python app.py --api-key YOUR_API_KEY_HERE --city berlin
python app.py --api-key YOUR_API_KEY_HERE --city cologne berlin duesseldorf

# Customize parallel processing
python app.py --api-key YOUR_API_KEY_HERE --workers 20

# Skip specific steps (0-10)
python app.py --api-key YOUR_API_KEY_HERE --skip-steps 6 7 8 9 10  # Skip OpenWeather and analysis steps
python app.py --api-key YOUR_API_KEY_HERE --skip-steps 0 4 5       # Skip visualization steps

# Combine options
python app.py --api-key YOUR_API_KEY_HERE --fresh --test --workers 15 --city cologne berlin --skip-steps 6 7 8 9 10
```

#### Available Cities

The `--city` option accepts the following cities (can specify multiple):
- `cologne` - Cologne area stations
- `berlin` - Berlin area stations  
- `duesseldorf` - Düsseldorf area stations

Default behavior includes all three cities: `cologne`, `berlin`, `duesseldorf`

#### Test Mode

For testing or development purposes, use the `--test` flag:

```bash
python app.py --api-key YOUR_API_KEY_HERE --test
```

This will:
- Skip actual downloads in Step 3, only create city mapping
- For testing the pipeline without overwhelming the servers
- Still generates all visualizations and city mappings

#### Skip Hourly Data Mode

Use the `--skip-hourly` flag to run the pipeline without downloading hourly data:

```bash
python app.py --api-key YOUR_API_KEY_HERE --skip-hourly
```

This will:
- Execute Steps 0, 1, 2, 4, and 5 (coordinates, visualizations, city-specific visualizations)
- Skip Step 3 (hourly data download) and Steps 6-10 (OpenWeather and analysis)
- Create city mapping for visualization purposes only
- Useful for quick testing or when you only need coordinate data and visualizations

#### Skip Specific Steps

Use the `--skip-steps` option to skip specific pipeline steps:

```bash
# Skip OpenWeather data download and analysis steps
python app.py --api-key YOUR_API_KEY_HERE --skip-steps 6 7 8 9 10

# Skip only visualization steps
python app.py --api-key YOUR_API_KEY_HERE --skip-steps 0 2 4 5

# Skip correlation analysis steps
python app.py --api-key YOUR_API_KEY_HERE --skip-steps 9 10
```

#### Parallel Processing

```bash
# Conservative approach (fewer workers, usually more respectful towards providing servers)
python app.py --api-key YOUR_API_KEY_HERE --workers 5

# Aggressive approach (more workers, faster downloads)
python app.py --api-key YOUR_API_KEY_HERE --workers 20
```

#### Graceful Termination

The pipeline supports graceful termination with Ctrl+C:

- Press `Ctrl+C` at any time to safely stop the pipeline
- The script will complete the current operation and exit cleanly
- No partial downloads or corrupted files will be left behind
- Useful for long-running downloads that need to be interrupted

#### Generated Outputs

The pipeline creates the following files:
- `BASt Station Files/bast_locations.csv` - Coordinate data for all traffic counting stations
- `Graphs/bast_openweather_overlap.png` - BASt-OpenWeatherMap overlap visualization
- `Graphs/bast_locations_heatmap.png` - Heatmap visualization with Germany borders
- `Graphs/bast_stations_by_year.png` - Bar chart showing station counts by year
- `Graphs/bast_locations_by_year_by_city.png` - City-by-year stacked bar visualization
- `Graphs/bast_locations_by_city_heatmap.png` - City heatmap visualization
- `Graphs/download_quality_analysis.png` - Download quality visualization
- `Graphs/traffic_vs_pollution_analysis.png` - Traffic vs pollution analysis
- `Graphs/correlation_analysis_results.png` - Association and correlation analysis results
- `Graphs/deseasoned_correlation_analysis.png` - Deseasoned correlation analysis results
- `BASt Hourly Data/` - Directory containing extracted hourly traffic data files
- `BASt Hourly Data/zip_file_existence_check.csv` - Download status report
- `BASt Hourly Data/bast_stations_by_city.csv` - Station-city mapping
- `owm Hourly Data/` - Directory containing OpenWeather API downloaded hourly pollution data

#### Individual Scripts

You can also run the individual components separately:

```bash
# Step 0: Create overlap visualization
python Helpers/00_DrawBastOpenWeatherMapOverlap.py

# Step 1: Retrieve coordinates only
python Helpers/01_GetBastStationGeneralData.py [--fresh]

# Step 2: Generate visualizations only
python Helpers/02_DrawBastLocations.py

# Step 3: Download hourly data only (with parallel processing)
python Helpers/03_GetBastStationHourlyData.py [--test] [--workers N] [--city cologne berlin duesseldorf]

# Step 4: Create city-specific visualizations
python Helpers/04_DrawBastLocationsByCity.py
python Helpers/05_DrawBastLocationsByCityHeatmap.py

# Step 6: Download OpenWeather hourly data (requires API key)
python Helpers/06_GetOpenWeatherHourlyData.py --api-key YOUR_API_KEY_HERE

# Step 7: Create download quality visualization
python Helpers/07_DrawDownloadQuality.py

# Step 8: Create traffic vs pollution analysis
python Helpers/08_DrawTrafficVsPollution.py

# Step 9: Perform association and correlation analysis
python Helpers/09_PerformAssociationCorrelationAnalysis.py

# Step 10: Perform deseasoned correlation analysis
python Helpers/10_PerformDeseasonedCorrelationAnalysis.py
```

#### Performance Considerations

For downloading the full dataset (35,000+ files):
- **Estimated time**: ~1-2 hours with 10 worker processes
- **Monitoring**: Watch for 429 errors (rate limiting)
- **Resume capability**: Script skips already downloaded files

#### Error Handling

The pipeline includes robust error handling:
- **Critical steps** (0-3): Pipeline stops if these fail
- **Non-critical steps** (4-10): Pipeline continues with warnings if these fail
- **Graceful termination**: Ctrl+C safely stops the pipeline
- **Real-time output**: All script output is forwarded in real-time

## Licenses:

- [BASt](https://www.bast.de/): The data provided form bast can be used freely and is subject to the "Creative Commons Attribution - 4.0 International" (CC BY 4.0).
- [OpenWeatherMap API](https://openweathermap.org/api): This project uses OpenWeatherMap API data under a commercial license agreement with OpenWeather Ltd. The license grants non-exclusive, non-transferable rights for internal usage and distribution of non-retrievable value-added services. Users must provide attribution to OpenWeatherMap in their products or services. The complete license terms are available in the "Licenses/OpenWeatherAPI License Agreement.pdf" file in this repository.