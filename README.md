# FOM-Business-Analytics

This repository contains materials and resources for the Business Analytics presentation at FOM Hochschule for the academic year 2025.

## Project Overview

We're correlating pollution data from OpenWeatherMap with traffic data from BASt to potentially guide policymakers,inform traffic planners, and enable data-driven urban development strategies. This analysis aims to provide insights into the relationship between environmental factors and transportation patterns.

## Course Information
- **Institution**: FOM Hochschule
- **Course**: Business Analytics
- **Academic Year**: 2025
- **Project Type**: Semester Project

## Contents
- Script for reference
- Instructions on how to run the BA Scripts
- Pre-generated Plots
- Documentation and findings

## Getting Started

This repository is organized to support the application of the semester project.

### Prerequisites

To verify the downloaded OpenWeatherMap data, an API key is required. To obtain one:

1. Visit [OpenWeatherMap API Keys](https://home.openweathermap.org/api_keys)
2. Create an account or sign in
3. Generate a new API key
4. Add the API key to the `.env` file in the project root

### Setup Instructions

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your `.env` file with your OpenWeatherMap API key

### Running the BASt Data Pipeline

The main application (`app.py`) provides a complete pipeline for retrieving, visualizing, and downloading all data. By default, it filters BASt data to focus on three major German cities: Cologne, Berlin, and Düsseldorf.

**BASt Data Volume Information (as of 2025-06-24):**
- **Default (3 cities)**: 1,076 years worth of hourly data for 51 distinct stations (approximately 1.5 GB)
- **Germany-wide**: Over 35,000 years worth of data across all stations (approximately 70 GB)
- **Data availability**: Traffic data available up to and including 2023
- **Source**: Real-Life Measurements by BASt

**OpenWeatherMap API Integration:**
- **Educational API limit**: 50,000 requests per day
- **Pollution data available**: Starting from 2020-11-26
- **Pollutants tracked**: SO₂, NO₂, PM10, PM2.5, O₃, and CO
- **Data granularity**: Hourly measurements
- **Source**: SILAM atmospheric composition model

#### Basic Usage

To run the complete BASt data processing pipeline:

```bash
python app.py
```

This will execute the following steps:
1. **Step 1**: Retrieve BASt traffic counting station coordinates from 2003-2023 (skipping existing files)
2. **Step 2**: Generate visualizations including:
   - Heatmap of station locations across Germany
   - Year-by-year comparison of station counts
3. **Step 3**: Download and extract hourly traffic data from BASt servers
4. **Step 3b**: Create city-by-year stacked bar visualization
5. **Step 3c**: Create city heatmap visualization
6. **Step 4**: Format data for OpenWeather API (creates files with Unix timestamps and location data)

#### Command Line Options

The pipeline supports various command line options for different use cases:

```bash
# Force fresh data retrieval for coordinates
python app.py --fresh

# Run in test mode (limited URLs for testing)
python app.py --test

# Customize parallel processing
python app.py --workers 20

# Skip hourly data download (only creates city mapping and visualizations)
python app.py --skip-hourly

# Filter by specific cities (can specify multiple)
python app.py --city cologne berlin
python app.py --city berlin
python app.py --city cologne berlin duesseldorf

# Combine options
python app.py --fresh --test --workers 15 --city cologne berlin
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
python app.py --test
```

This will:
- Skip actual downloads, only create city mapping
- For testing the pipeline without overwhelming the servers
- Still generates all visualizations and city mappings

#### Skip Hourly Data Mode

Use the `--skip-hourly` flag to run the pipeline without downloading hourly data:

```bash
python app.py --skip-hourly
```

This will:
- Execute Steps 1, 2, 3b, and 3c (coordinates, visualizations, city-specific visualizations)
- Skip Step 3 (hourly data download) and Step 4 (OpenWeather formatting)
- Create city mapping for visualization purposes only
- Useful for quick testing or when you only need coordinate data and visualizations

#### Parallel Processing

```bash
# Conservative approach (fewer workers, usually more respectful towards providing servers)
python app.py --workers 5

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
- `Graphs/bast_locations_heatmap.png` - Heatmap visualization with Germany borders
- `Graphs/bast_stations_by_year.png` - Bar chart showing station counts by year
- `Graphs/bast_locations_by_year_by_city.png` - City-by-year stacked bar visualization
- `Graphs/bast_locations_by_city_heatmap.png` - City heatmap visualization
- `BASt Hourly Data/` - Directory containing extracted hourly traffic data files
- `BASt Hourly Data/zip_file_existence_check.csv` - Download status report
- `BASt Hourly Data/bast_stations_by_city.csv` - Station-city mapping
- `BASt Hourly Data/*.csv` - Files with Unix timestamps and location data for OpenWeather API (Step 4 output)

#### Individual Scripts

You can also run the individual components separately:

```bash
# Retrieve coordinates only
python Helpers/01_GetBastStationGeneralData.py [--fresh]

# Generate visualizations only
python Helpers/02_DrawBastLocations.py

# Download hourly data only (with parallel processing)
python Helpers/03_GetBastStationHourlyData.py [--test] [--workers N] [--city cologne berlin duesseldorf]

# Create city-specific visualizations
python Helpers/04_DrawBastLocationsByCity.py
python Helpers/05_DrawBastLocationsByCityHeatmap.py

# Format data for OpenWeather API
python Helpers/06_FormatForOpenWeather.py
```

#### Performance Considerations

For downloading the full dataset (35,000+ files):
- **Estimated time**: ~2-4 hours with 10 workers (vs ~20 hours sequential)
- **Monitoring**: Watch for 429 errors (rate limiting)
- **Resume capability**: Script skips already downloaded files

## Licenses:

- [BASt](https://www.bast.de/): The data provided form bast can be used freely and is subject to the "Creative Commons Attribution - 4.0 International" (CC BY 4.0).
- [OpenWeatherMap API](https://openweathermap.org/api): This project uses OpenWeatherMap API data under a commercial license agreement with OpenWeather Ltd. The license grants non-exclusive, non-transferable rights for internal usage and distribution of non-retrievable value-added services. Users must provide attribution to OpenWeatherMap in their products or services. The complete license terms are available in the "OpenWeatherAPI License Agreement.pdf" file in this repository.