import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import warnings
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
    
    # Create a GeoDataFrame from the coordinates
    geometry = [Point(xy) for xy in zip(df['x_coordinate'], df['y_coordinate'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:25832')  # ETRS89 / UTM zone 32N
    
    # Convert to Web Mercator for contextily
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot the points with a heatmap effect
    print("Creating heatmap...")
    
    # Create a 2D histogram for the heatmap
    x_coords = gdf_web.geometry.x
    y_coords = gdf_web.geometry.y
    
    # Define the extent of Germany (approximate bounds in Web Mercator)
    # These are rough bounds for Germany
    x_min, x_max = 500000, 1500000
    y_min, y_max = 5500000, 6500000
    
    # Create histogram bins
    bins = 100
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=bins, 
                                            range=[[x_min, x_max], [y_min, y_max]])
    
    # Smooth the heatmap
    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap, sigma=1)
    
    # Create custom colormap (blue to red)
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot the heatmap
    extent = [x_min, x_max, y_min, y_max]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Traffic Counting Stations', fontsize=12)
    
    # Add the base map
    print("Adding base map...")
    try:
        ctx.add_basemap(ax, crs=gdf_web.crs.to_string(), 
                       source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Warning: Could not load base map: {e}")
        # Fallback: just show the heatmap without base map
        pass
    
    # Set title and labels
    ax.set_title('BASt Traffic Counting Stations Heatmap (2003-2023)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude (Web Mercator)', fontsize=12)
    ax.set_ylabel('Latitude (Web Mercator)', fontsize=12)
    
    # Add some statistics as text
    total_stations = len(df)
    unique_years = df['year'].nunique()
    avg_per_year = total_stations / unique_years
    
    stats_text = f'Total Stations: {total_stations:,}\nYears: {unique_years}\nAvg per Year: {avg_per_year:.0f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    output_file = 'BASt Files/bast_locations_heatmap.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
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
    
    # Plot the data
    bars = ax.bar(year_counts.index, year_counts.values, 
                  color='steelblue', alpha=0.7, edgecolor='navy')
    
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
    z = np.polyfit(year_counts.index, year_counts.values, 1)
    p = np.poly1d(z)
    ax.plot(year_counts.index, p(year_counts.index), "r--", alpha=0.8, linewidth=2)
    
    # Save the plot
    output_file = 'BASt Files/bast_stations_by_year.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Year comparison plot saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    print("BASt Locations Visualization")
    print("=" * 40)
    
    # Create the main heatmap
    gdf = create_bast_heatmap()
    
    # Create the year comparison plot
    create_year_comparison_plot()
    
    print("\nVisualization complete!")
