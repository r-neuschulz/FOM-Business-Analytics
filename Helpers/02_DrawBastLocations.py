import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
from pyproj import Transformer
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore')

def create_bast_heatmap():
    """
    Create a heatmap of BASt traffic counting locations overlaid on a German map
    """
    print("Loading BASt location data...")
    
    # Load the BASt locations data
    try:
        df = pd.read_csv('BASt Files/bast_locations.csv')
        print(f"Loaded {len(df)} locations")
    except FileNotFoundError:
        print("Error: bast_locations.csv not found in BASt Files directory")
        return
    
    # Convert coordinates to numeric, handling any non-numeric values
    df['x_coordinate'] = pd.to_numeric(df['x_coordinate'], errors='coerce')
    df['y_coordinate'] = pd.to_numeric(df['y_coordinate'], errors='coerce')
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['x_coordinate', 'y_coordinate'])
    print(f"Valid coordinates: {len(df)} locations")
    
    # Print coordinate range to understand the data
    print(f"X coordinate range: {df['x_coordinate'].min():.0f} to {df['x_coordinate'].max():.0f}")
    print(f"Y coordinate range: {df['y_coordinate'].min():.0f} to {df['y_coordinate'].max():.0f}")
    
    # Create a GeoDataFrame from the coordinates
    geometry = [Point(xy) for xy in zip(df['x_coordinate'], df['y_coordinate'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:25832')  # ETRS89 / UTM zone 32N
    
    # Convert to Web Mercator for contextily
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Determine bounds from the actual data in Web Mercator
    x_min, x_max = gdf_web.geometry.x.min(), gdf_web.geometry.x.max()
    y_min, y_max = gdf_web.geometry.y.min(), gdf_web.geometry.y.max()
    
    # Add some padding to the bounds
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    print(f"Web Mercator bounds: X({x_min:.0f}, {x_max:.0f}), Y({y_min:.0f}, {y_max:.0f})")
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Add the base map FIRST
    print("Adding base map...")
    try:
        if gdf_web.crs is not None:
            ctx.add_basemap(ax, crs=gdf_web.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        else:
            ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik)
    except Exception as e:
        print(f"Warning: Could not load base map: {e}")
        pass

    # Overlay German borders
    germany = gpd.read_file('Graphs/ne_10m_admin_0_countries.shp')
    germany = germany[germany['NAME'] == 'Germany']
    germany = germany.to_crs(epsg=3857)
    germany.boundary.plot(ax=ax, color='white', linewidth=2, alpha=0.5, zorder=10)

    # Formatter for axis ticks (Web Mercator to lon/lat)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    def format_lon(x, pos):
        lon, _ = transformer.transform(x, y_min)
        deg = int(lon)
        min_ = int(abs((lon - deg) * 60))
        return f"{deg}°{min_}'"
    def format_lat(y, pos):
        _, lat = transformer.transform(x_min, y)
        deg = int(lat)
        min_ = int(abs((lat - deg) * 60))
        return f"{deg}°{min_}'"
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Set axis limits to match the heatmap extent
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot the points with a heatmap effect
    print("Creating heatmap...")
    
    # Create a 2D histogram for the heatmap
    x_coords = gdf_web.geometry.x
    y_coords = gdf_web.geometry.y
    
    print(f"Number of coordinates to plot: {len(x_coords)}")
    print(f"Sample coordinates: X={x_coords.iloc[:5].tolist()}, Y={y_coords.iloc[:5].tolist()}")
    
    # Create histogram bins
    bins = 150
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=bins, 
                                            range=[[x_min, x_max], [y_min, y_max]])
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap value range: {heatmap.min()} to {heatmap.max()}")
    print(f"Non-zero values in heatmap: {np.count_nonzero(heatmap)}")
    
    # Smooth the heatmap
    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap, sigma=1)
    
    # Use magma colormap instead of custom colormap
    cmap = plt.cm.magma
    
    # Plot the heatmap AFTER the base map
    extent = (x_min, x_max, y_min, y_max)
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, alpha=0.8, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Traffic Counting Stations', fontsize=12)
    
    # Set title and labels
    ax.set_title('BASt Traffic Counting Stations Heatmap (2003-2023)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add some statistics as text
    total_stations = len(df)
    unique_years = df['year'].nunique()
    avg_per_year = total_stations / unique_years
    
    stats_text = f'Total Stations: {total_stations:,}\nYears: {unique_years}\nAvg per Year: {avg_per_year:.0f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create Graphs directory if it doesn't exist
    graphs_dir = "Graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        print(f"Created directory: {graphs_dir}")
    
    # Save the plot
    output_file = os.path.join(graphs_dir, 'bast_locations_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    
    # Close the plot to free memory (no popup)
    plt.close()
    
    return gdf

def create_year_comparison_plot():
    """
    Create a comparison plot showing station density by year
    """
    print("Creating year comparison plot...")
    
    # Load the data
    df = pd.read_csv('BASt Files/bast_locations.csv')
    
    # Count stations by year
    year_counts = df['year'].value_counts().sort_index()
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Convert to numpy arrays to fix type issues
    years = np.array(year_counts.index)
    counts = np.array(year_counts.values)
    
    # Create magma colormap for the bars
    cmap = plt.cm.magma
    colors = cmap(np.linspace(0.2, 0.8, len(years)))
    
    # Plot the data with magma colors
    bars = ax.bar(years, counts, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Customize the plot
    ax.set_title('BASt Traffic Counting Stations by Year (2003-2023)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Stations', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add trend line
    z = np.polyfit(years, counts, 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), "r--", alpha=0.8, linewidth=2)
    
    # Create Graphs directory if it doesn't exist
    graphs_dir = "Graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        print(f"Created directory: {graphs_dir}")
    
    # Save the plot
    output_file = os.path.join(graphs_dir, 'bast_stations_by_year.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Year comparison plot saved to: {output_file}")
    
    # Close the plot to free memory (no popup)
    plt.close()

if __name__ == "__main__":
    print("BASt Locations Visualization")
    print("=" * 40)
    
    # Create the main heatmap
    gdf = create_bast_heatmap()
    
    # Create the year comparison plot
    create_year_comparison_plot()
    
    print("\nVisualization complete!")

