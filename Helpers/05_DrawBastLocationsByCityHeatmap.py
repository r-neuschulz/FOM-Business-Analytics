import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.ticker as mticker

# Define city colors
CITY_COLORS = {
    'berlin': '#000000',
    'duesseldorf': '#3c74b9',
    'cologne': '#e1141c'
}

# File paths
LOCATIONS_FILE = 'BASt Hourly Data/bast_stations_by_city.csv'
OUTPUT_FILE = 'Graphs/bast_locations_by_city_heatmap.png'

# Load data
def load_locations():
    if not os.path.exists(LOCATIONS_FILE):
        print(f"Error: {LOCATIONS_FILE} not found.")
        return None
    df = pd.read_csv(LOCATIONS_FILE)
    # Expect columns: city, lat, lon (or similar)
    col_map = {c.lower(): c for c in df.columns}
    lat_col = col_map.get('lat', col_map.get('latitude'))
    lon_col = col_map.get('lon', col_map.get('longitude'))
    city_col = col_map.get('city')
    if not (lat_col and lon_col and city_col):
        print(f"Error: Could not find required columns in {LOCATIONS_FILE}.")
        return None
    return df.rename(columns={lat_col: 'lat', lon_col: 'lon', city_col: 'city'})

def deg_min_formatter(x, pos):
    deg = int(x)
    min_ = int(abs((x - deg) * 60))
    return f"{deg}°{min_}'"

def get_consistent_ticks(min_val, max_val, interval=0.02):
    """
    Generate consistent tick positions at regular intervals (default 2 minutes = 0.02 degrees)
    """
    # Round down to the nearest interval boundary
    start = np.floor(min_val / interval) * interval
    # Round up to the nearest interval boundary
    end = np.ceil(max_val / interval) * interval
    # Generate ticks at regular intervals
    ticks = np.arange(start, end + interval, interval)
    return ticks

def plot_city_heatmaps(df):
    cities = ['duesseldorf', 'cologne', 'berlin']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    # Compute width/height for each city (in degrees)
    city_extents = {}
    for city in cities:
        city_df = df[df['city'].str.lower() == city]
        if city_df.empty:
            continue
        min_lon, max_lon = city_df['lon'].min(), city_df['lon'].max()
        min_lat, max_lat = city_df['lat'].min(), city_df['lat'].max()
        width = max_lon - min_lon
        height = max_lat - min_lat
        center_lon = (max_lon + min_lon) / 2
        center_lat = (max_lat + min_lat) / 2
        city_extents[city] = dict(center_lon=center_lon, center_lat=center_lat, width=width, height=height)
    # Find the max width/height across all cities (in degrees)
    max_width = max(ext['width'] for ext in city_extents.values())
    max_height = max(ext['height'] for ext in city_extents.values())
    # Add a small margin
    margin_factor = 0.05
    max_width *= (1 + margin_factor)
    max_height *= (1 + margin_factor)
    for i, city in enumerate(cities):
        ax = axes[i]
        city_df = df[df['city'].str.lower() == city]
        if city_df.empty:
            ax.set_title(f"{city.title()}\n(No Data)")
            ax.axis('off')
            continue
        center_lon = city_extents[city]['center_lon']
        center_lat = city_extents[city]['center_lat']
        # Set limits to same real-world area for all
        xlim = (center_lon - max_width/2, center_lon + max_width/2)
        ylim = (center_lat - max_height/2, center_lat + max_height/2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Convert to GeoDataFrame and project to Web Mercator
        gdf = gpd.GeoDataFrame(city_df.copy(), geometry=[Point(xy) for xy in zip(city_df['lon'], city_df['lat'])], crs='EPSG:4326')
        gdf = gdf.to_crs(epsg=3857)
        # Plot basemap
        ax.set_aspect('equal', adjustable='box')
        bounds = gpd.GeoSeries([Point(xlim[0], ylim[0]), Point(xlim[1], ylim[1])], crs='EPSG:4326').to_crs(epsg=3857).total_bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ctx.add_basemap(ax, crs='EPSG:3857', source='OpenStreetMap.Mapnik', attribution_size=6, alpha=0.7)
        # Plot points (after basemap, so they're on top)
        ax.scatter(gdf.geometry.x, gdf.geometry.y, s=20, color=CITY_COLORS[city], alpha=0.8, zorder=10)
        ax.set_title(city.title(), fontsize=16, color=CITY_COLORS[city])
        ax.set_xlabel('Longitude (°)')
        if i == 0:
            ax.set_ylabel('Latitude (°)')
        else:
            ax.set_ylabel('')
        # Format axes as decimal degrees with two decimals
        def merc_to_lon(x):
            return gpd.GeoSeries([Point(x, 0)], crs='EPSG:3857').to_crs(epsg=4326).x.iloc[0]
        def merc_to_lat(y):
            return gpd.GeoSeries([Point(0, y)], crs='EPSG:3857').to_crs(epsg=4326).y.iloc[0]
        # Helper to get lowest multiple of interval
        def get_tick_start(val, interval):
            return np.floor(val / interval) * interval
        # Use tight data limits for tick calculation
        interval = 0.04  # 2 hundredths of a degree
        lon_tick_start = get_tick_start(xlim[0], interval)
        lon_tick_end = get_tick_start(xlim[1], interval) + interval
        lat_tick_start = get_tick_start(ylim[0], interval)
        lat_tick_end = get_tick_start(ylim[1], interval) + interval
        lon_tick_vals = np.arange(lon_tick_start, lon_tick_end, interval)
        lat_tick_vals = np.arange(lat_tick_start, lat_tick_end, interval)
        merc_lon_ticks = [gpd.GeoSeries([Point(lon, 0)], crs='EPSG:4326').to_crs(epsg=3857).x.iloc[0] for lon in lon_tick_vals]
        merc_lat_ticks = [gpd.GeoSeries([Point(0, lat)], crs='EPSG:4326').to_crs(epsg=3857).y.iloc[0] for lat in lat_tick_vals]
        ax.set_xticks(merc_lon_ticks)
        ax.set_yticks(merc_lat_ticks)
        # Format ticks as 0.00 decimal degrees
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{merc_to_lon(x):.2f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{merc_to_lat(y):.2f}"))
        # Rotate longitude labels by 90°
        plt.setp(ax.get_xticklabels(), rotation=90)
        # Add gridlines
        ax.grid(which='major', color='grey', linewidth=0.7, alpha=0.5)
    plt.suptitle('BASt Station Locations by City', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.subplots_adjust(bottom=0.18)
    Path('Graphs').mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.close()
    print(f"Point map saved to: {OUTPUT_FILE}")

def main():
    df = load_locations()
    if df is not None:
        plot_city_heatmaps(df)

if __name__ == "__main__":
    main() 