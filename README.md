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

The main application (`app.py`) provides a complete pipeline for retrieving, visualizing, and downloading all data. By default, it is reducing the BASt data to Cologne, Berlin, and Düsseldorf. Ad of 2025-06-24 this will download 1076 years worth of hourly data for 51 distinct stations (1.5 GB of Data). The Germany-wide download consumes roughly 70 GB across more than 35.000 years worth of data.

As of 2025-06-24, that data is available up to, and including 2023.

If run with an educational OpenWeatherMap API, key 50.000 requests per day can be processed. It includes pollution data up starting on the 2020-11-26, at hourly granularity for S02, NO2,PM10, PM2.5, O3, and CO.

#### Basic Usage

To run the complete BASt data processing pipeline:

```bash
python app.py
```

This will:
1. Retrieve BASt traffic counting station coordinates from 2003-2023 (skipping existing files)
2. Generate visualizations including:
   - Heatmap of station locations across Germany
   - Year-by-year comparison of station counts
3. Download and extract hourly traffic data from BASt servers

#### Command Line Options

The pipeline supports various command line options for different use cases:

```bash
# Force fresh data retrieval for coordinates
python app.py --fresh

# Run in test mode (limited URLs for testing)
python app.py --test

# Customize batch processing settings
python app.py --batch-size 50 --batch-delay 30

# Skip hourly data download
python app.py --skip-hourly

# Combine options
python app.py --fresh --test --batch-size 100
```

#### Test Mode

For testing or development purposes, use the `--test` flag:

```bash
python app.py --test
```

This will:
- Limit URL checking to 20 BASt Station URLs
- Limit downloads to 5 files
- Perfect for testing the pipeline without overwhelming the servers

#### Batch Processing

For large-scale downloads (35,000+ files), you can customize batch processing:

```bash
# Conservative approach (smaller batches, longer delays)
python app.py --batch-size 50 --batch-delay 60

# Aggressive approach (larger batches, shorter delays)
python app.py --batch-size 200 --batch-delay 15
```

#### Download Strategy

The pipeline includes robust download measures:
- **Immediate first attempts** for maximum speed
- **Exponential backoff** with jitter for failed requests
- **User agent rotation** across 5 different browsers
- **Random delays** between requests (1-3 seconds)
- **Batch processing** with configurable breaks
- **Session management** with automatic retries

#### Generated Outputs

The pipeline creates the following files:
- `BASt Station Files/bast_locations.csv` - Coordinate data for all traffic counting stations
- `Graphs/bast_locations_heatmap.png` - Heatmap visualization with Germany borders
- `Graphs/bast_stations_by_year.png` - Bar chart showing station counts by year
- `BASt Hourly Data/` - Directory containing extracted hourly traffic data files
- `BASt Hourly Data/zip_file_existence_check.csv` - Download status report

#### Individual Scripts

You can also run the individual components separately:

```bash
# Retrieve coordinates only
python Helpers/01_GetBastStationGeneralData.py [--fresh]

# Generate visualizations only
python Helpers/02_DrawBastLocations.py

# Download hourly data only
python Helpers/03_GetBastStationHourlyData.py [--test] [--batch-size N] [--batch-delay N]
```

#### Performance Considerations

For downloading the full dataset (35,000+ files):
- **Estimated time**: ~20 hours with default settings
- **Recommended**: Run during off-peak hours (night time in Germany)
- **Monitoring**: Watch for 429 errors (rate limiting)
- **Resume capability**: Script skips already downloaded files

## Licenses:

- [BASt](https://www.bast.de/): The data provided form bast can be used freely and is subject to the "Creative Commons Attribution - 4.0 International" (CC BY 4.0).
- [OpenWeatherMap API](https://openweathermap.org/api): This project uses OpenWeatherMap API data under a commercial license agreement with OpenWeather Ltd. The license grants non-exclusive, non-transferable rights for internal usage and distribution of non-retrievable value-added services. Users must provide attribution to OpenWeatherMap in their products or services. The complete license terms are available in the "OpenWeatherAPI License Agreement.pdf" file in this repository.