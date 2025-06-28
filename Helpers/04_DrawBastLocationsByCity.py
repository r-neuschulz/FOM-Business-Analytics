import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from pathlib import Path
import matplotlib.image as mpimg

from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AuxTransformBox, VPacker, HPacker, TextArea
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredOffsetbox
warnings.filterwarnings('ignore')

def create_city_year_stacked_bar():
    """
    Create a stacked bar chart showing the number of BASt stations per city for each year
    """
    print("Loading BASt city location data...")
    
    # Load the BASt city locations data
    try:
        df = pd.read_csv('BASt Hourly Data/bast_stations_by_city.csv')
        print(f"Loaded {len(df)} station-year combinations")
    except FileNotFoundError:
        print("Error: bast_stations_by_city.csv not found in BASt Hourly Data directory")
        return
    
    # Check if required columns exist
    required_columns = ['year', 'city']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Convert year to numeric, handling any non-numeric values
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Remove rows with invalid years
    df = df.dropna(subset=['year'])
    print(f"Valid years: {len(df)} station-year combinations")
    
    # Get unique cities and years
    cities = sorted(df['city'].unique())
    years = sorted(df['year'].unique())
    
    print(f"Found {len(cities)} cities: {', '.join(cities)}")
    print(f"Found {len(years)} years: {min(years)} to {max(years)}")
    
    # Create pivot table: cities as columns, years as index, count as values
    pivot_data = df.groupby(['year', 'city']).size().unstack(fill_value=0)
    
    # Ensure all cities are present (in case some years don't have all cities)
    for city in cities:
        if city not in pivot_data.columns:
            pivot_data[city] = 0
    
    # Reorder columns to match the original city order
    pivot_data = pivot_data[cities]
    
    print(f"Pivot table shape: {pivot_data.shape}")
    print(f"Sample data:")
    print(pivot_data[:5])
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define specific colors for the three cities
    city_colors = {
        'berlin': '#000000',
        'duesseldorf': '#3c74b9', 
        'cologne': '#e1141c'
    }
    
    # Create stacked bar chart
    bottom = np.zeros(len(pivot_data))
    
    for i, city in enumerate(cities):
        # Use specific color if available, otherwise use default
        color = city_colors.get(city, '#1f77b4')
        values = pivot_data[city]
        
        # Create bars for this city
        bars = ax.bar(range(len(pivot_data)), values, bottom=bottom, 
                     label=city.title(), color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars if they're not zero
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                # Calculate position for label
                bar_height = bar.get_height()
                bar_bottom = bar.get_y()
                
                # Only show label if bar is tall enough
                if bar_height > 2:
                    ax.text(bar.get_x() + bar.get_width()/2, bar_bottom + bar_height/2,
                           str(int(value)), ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white' if bar_height > 5 else 'black')
        
        bottom += values
    
    # Customize the plot
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of BASt Stations', fontsize=14, fontweight='bold')
    ax.set_title('BASt Traffic Counting Stations by City and Year (2003-2023)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(range(len(pivot_data)))
    ax.set_xticklabels(years, rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create Graphs directory if it doesn't exist
    graphs_dir = Path("Graphs")
    graphs_dir.mkdir(exist_ok=True)
    print(f"Ensured directory exists: {graphs_dir}")
    
    # Create custom legend with colors and logos
    from matplotlib.patches import Patch
    
    # Check for existing PNG logo files
    logo_png_files = {}
    for city in cities:
        png_path = f"Graphs/Logo_{city}.png"
        
        if os.path.exists(png_path):
            logo_png_files[city] = png_path
            print(f"Found logo: {png_path}")
        else:
            print(f"PNG logo not found: {png_path}")
    
    # Create a separate legend with logos
    boxes = []
    for city in cities:
        # Bar
        bar_box = AuxTransformBox(ax.transData)
        bar_patch = Rectangle((0,0), 0.6, 3.3, color=city_colors[city], alpha=0.8)
        bar_box.add_artist(bar_patch)
        # Logo
        if city in logo_png_files:
            img = mpimg.imread(logo_png_files[city])
            logo_box = OffsetImage(img, zoom=0.45, alpha=0.8)
        else:
            logo_box = TextArea(city.title())
        # Combine bar and logo
        pair = HPacker(children=[bar_box, logo_box], align='center', pad=0, sep=10)
        boxes.append(pair)
    
    legend_box = HPacker(children=boxes, align='center', pad=0, sep=50)
    anchored_box = AnchoredOffsetbox(loc='lower center', child=legend_box, pad=0.5, frameon=False, bbox_to_anchor=(0.5, -0.18), bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_box)
    
    # Add some statistics as text
    total_stations = len(df)
    total_years = len(years)
    avg_per_year = total_stations / total_years
    
    # Calculate city totals
    city_totals = df['city'].value_counts()
    city_stats = []
    for city in cities:
        count = city_totals.get(city, 0)
        city_stats.append(f"{city.title()}: {count}")
    
    stats_text = f'Total Station-Years: {total_stations:,}\nYears: {total_years}\nAvg per Year: {avg_per_year:.0f}\n\nCity Totals:\n' + '\n'.join(city_stats[:5])  # Show first 5 cities
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save the plot
    output_file = graphs_dir / 'bast_locations_by_year_by_city.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Stacked bar chart saved to: {output_file}")
    
    # Close the plot to free memory (no popup)
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total station-year combinations: {total_stations:,}")
    print(f"Years covered: {min(years)} to {max(years)} ({len(years)} years)")
    print(f"Average stations per year: {avg_per_year:.1f}")
    print(f"\nStations per city (total across all years):")
    for city in cities:
        count = city_totals.get(city, 0)
        print(f"  {city.title()}: {count:,}")
    
    return pivot_data

def main():
    """
    Main function to create the city-year stacked bar chart
    """
    print("Creating BASt city locations stacked bar chart...")
    
    try:
        pivot_data = create_city_year_stacked_bar()
        print("Successfully created city-year stacked bar chart!")
        return pivot_data
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

if __name__ == "__main__":
    main() 