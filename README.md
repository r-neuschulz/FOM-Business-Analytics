# FOM-Business-Analytics

This repository contains materials and resources for the Business Analytics presentation at FOM Hochschule for the academic year 2025.

## Project Overview

We're correlating pollution data from OpenWeatherMap with traffic data to potentially guide policymakers, traffic planners, and urban development strategies. This analysis aims to provide insights into the relationship between environmental factors and transportation patterns.

## Course Information
- **Institution**: FOM Hochschule
- **Course**: Business Analytics
- **Academic Year**: 2025
- **Project Type**: Semester Project

## Contents
- Script for reference
- Instructions on how to run the BA Scripts
- Data analysis tools and methodologies
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

The main application (`app.py`) provides a complete pipeline for retrieving and visualizing BASt (German Federal Highway Research Institute) traffic counting station data.

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

#### Fresh Data Retrieval

To force re-download of all BASt HTML files and regenerate the coordinate data:

```bash
python app.py --fresh
```

#### Generated Outputs

The pipeline creates the following files:
- `BASt Station Files/bast_locations.csv` - Coordinate data for all traffic counting stations
- `Graphs/bast_locations_heatmap.png` - Heatmap visualization with Germany borders
- `Graphs/bast_stations_by_year.png` - Bar chart showing station counts by year

#### Individual Scripts

You can also run the individual components separately:

```bash
# Retrieve coordinates only
python 01_GetBastCoords.py [--fresh]

# Generate visualizations only
python 02_DrawBastLocations.py
```

## Licenses:

[BASt](https://www.bast.de/): The data provided form bast can be used freely and is subject to the "Creative Commons Attribution - 4.0 International" (CC BY 4.0).
[OpenWeatherMap API](https://openweathermap.org/api): This project uses OpenWeatherMap API data under a commercial license agreement with OpenWeather Ltd. The license grants non-exclusive, non-transferable rights for internal usage and distribution of non-retrievable value-added services. Users must provide attribution to OpenWeatherMap in their products or services. The complete license terms are available in the "OpenWeatherAPI License Agreement.pdf" file in this repository.