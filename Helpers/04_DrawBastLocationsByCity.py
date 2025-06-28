import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from pathlib import Path

from matplotlib.patches import Rectangle
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
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    # Check for existing PNG logo files
    logo_png_files = {}
    for city in cities:
        png_path = f"Graphs/Logo_{city}.png"
        
        if os.path.exists(png_path):
            logo_png_files[city] = png_path
            print(f"Found logo: {png_path}")
        else:
            print(f"PNG logo not found: {png_path}")
    
    # Load PNG logos for legend
    def load_png_logo(png_path):
        """Load PNG logo as numpy array"""
        try:
            import matplotlib.image as mpimg
            return mpimg.imread(png_path)
        except Exception as e:
            print(f"Error loading PNG {png_path}: {e}")
            return None
    
    # Create simple legend for main plot with logos
    legend_handles = []
    legend_labels = []
    
    for city in cities:
        color = city_colors.get(city, '#1f77b4')
        legend_patch = Patch(color=color, label=city.title())
        legend_handles.append(legend_patch)
        legend_labels.append(city.title())
    
    # Add legend to main plot
    ax.legend(handles=legend_handles, labels=legend_labels, title='Cities', title_fontsize=12, fontsize=11, 
             loc='upper left', bbox_to_anchor=(1, 1))
    
    # Create a separate legend with logos
    logo_fig, logo_ax = plt.subplots(figsize=(10, 3))
    logo_ax.axis('off')
    
    # Position logos horizontally
    logo_width = 0.25
    spacing = 0.33
    
    for i, city in enumerate(cities):
        color = city_colors.get(city, '#1f77b4')
        x_pos = 0.05 + i * spacing
        
        # Create color patch
        logo_ax.add_patch(Rectangle((x_pos, 0.3), logo_width, 0.4, color=color, edgecolor='black', linewidth=2))
        
        # Try to load and add logo
        png_path = logo_png_files.get(city)
        if png_path:
            img_array = load_png_logo(png_path)
            if img_array is not None:
                # Add logo above color patch
                logo_box = OffsetImage(img_array, zoom=0.3)
                ab = AnnotationBbox(logo_box, (x_pos + logo_width/2, 0.8), 
                                  frameon=False, box_alignment=(0.5, 0.5))
                logo_ax.add_artist(ab)
        
        # Add city name below color patch
        logo_ax.text(x_pos + logo_width/2, 0.15, city.title(), 
                    fontsize=14, va='center', ha='center', fontweight='bold')
    
    # Save logo legend
    logo_legend_file = graphs_dir / 'city_logo_legend.png'
    logo_fig.savefig(logo_legend_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(logo_fig)
    print(f"Logo legend saved to: {logo_legend_file}")
    
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