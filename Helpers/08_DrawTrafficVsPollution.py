import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import random
import re
from datetime import datetime
from matplotlib.ticker import NullFormatter

def find_matching_files():
    """
    Find matching OWM and BASt files for the same station and year.
    Returns a list of tuples (station_number, year, owm_file, bast_file).
    """
    matching_files = []
    
    # Get OWM files
    owm_dir = "owm Hourly Data"
    bast_dir = "BASt Hourly Data"
    
    if not os.path.exists(owm_dir) or not os.path.exists(bast_dir):
        print("Data directories not found!")
        return []
    
    # Get all OWM files
    owm_files = [f for f in os.listdir(owm_dir) if f.startswith("owm") and f.endswith(".csv")]
    
    for owm_file in owm_files:
        # Extract station and year from OWM filename (owm{station}_{year}.csv)
        match = re.match(r'owm(\d+)_(\d{4})\.csv', owm_file)
        if match:
            station = match.group(1)
            year = match.group(2)
            
            # Look for corresponding BASt file
            bast_file = f"zst{station}_{year}.csv"
            bast_path = os.path.join(bast_dir, bast_file)
            
            if os.path.exists(bast_path):
                matching_files.append((station, year, owm_file, bast_file))
    
    return matching_files

def load_and_process_data(owm_file, bast_file):
    """
    Load and process OWM and BASt data for plotting.
    """
    # Load OWM data
    owm_path = os.path.join("owm Hourly Data", owm_file)
    owm_data = pd.read_csv(owm_path)
    
    # Convert timestamp to datetime
    owm_data['datetime'] = pd.to_datetime(owm_data['dt'], unit='s')
    owm_data['date'] = owm_data['datetime'].dt.date
    
    # Load BASt data
    bast_path = os.path.join("BASt Hourly Data", bast_file)
    bast_data = pd.read_csv(bast_path, sep=';')
    
    # Convert date column to datetime
    bast_data['date'] = pd.to_datetime(bast_data['Datum'], format='%y%m%d').dt.date
    
    return owm_data, bast_data

def create_traffic_vs_pollution_plot(owm_data, bast_data, station, year):
    """
    Create a plot comparing pollution (CO) with traffic data.
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = ax1.twinx()
    
    # Process OWM data - aggregate all relevant pollutant columns by date
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    owm_daily = owm_data.groupby('date')[pollutant_cols].mean().reset_index()
    owm_daily['datetime'] = pd.to_datetime(owm_daily['date'])
    
    # Process BASt data - aggregate traffic by date
    bast_daily = bast_data.groupby('date').agg({
        'KFZ_R1': 'sum',
        'KFZ_R2': 'sum', 
        'Lkw_R1': 'sum',
        'Lkw_R2': 'sum'
    }).reset_index()
    bast_daily['datetime'] = pd.to_datetime(bast_daily['date'])
    
    # Find common date range
    start_date = max(owm_daily['datetime'].min(), bast_daily['datetime'].min())
    end_date = min(owm_daily['datetime'].max(), bast_daily['datetime'].max())
    
    # Filter data to common range
    owm_filtered = owm_daily[(owm_daily['datetime'] >= start_date) & (owm_daily['datetime'] <= end_date)]
    bast_filtered = bast_daily[(bast_daily['datetime'] >= start_date) & (bast_daily['datetime'] <= end_date)]
    
    # Plot traffic data as stacked bars on left y-axis (ax1), trucks at bottom
    x_dates = bast_filtered['datetime']
    width = 1  # days
    bars1 = ax1.bar(x_dates, bast_filtered['Lkw_R1'], width, 
                    bottom=np.zeros(len(bast_filtered)), color='#a05c60', alpha=0.8, label='Lkw_R1', zorder=2)
    bars2 = ax1.bar(x_dates, bast_filtered['Lkw_R2'], width,
                    bottom=bast_filtered['Lkw_R1'], color='#4b2a3b', alpha=0.8, label='Lkw_R2', zorder=2)
    bars3 = ax1.bar(x_dates, bast_filtered['KFZ_R1'], width,
                    bottom=bast_filtered['Lkw_R1'] + bast_filtered['Lkw_R2'], color='#bdb6b6', alpha=0.8, label='KFZ_R1', zorder=2)
    bars4 = ax1.bar(x_dates, bast_filtered['KFZ_R2'], width,
                    bottom=bast_filtered['Lkw_R1'] + bast_filtered['Lkw_R2'] + bast_filtered['KFZ_R1'], color='#7c6e7f', alpha=0.8, label='KFZ_R2', zorder=2)
    ax1.set_ylabel('Total BASt Traffic Count (vehicles/day)', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot OWM pollutant lines on right y-axis (ax2, so they're on top)
    pollutant_colors = {
        'co':    '#39ff14',  # toxic green
        'no':    '#3b528b',  # viridis dark blue
        'no2':   '#21918c',  # viridis teal
        'o3':    '#5ec962',  # viridis green
        'so2':   '#fde725',  # viridis yellow
        'pm2_5': '#440154',  # viridis purple
        'pm10':  '#31688e',  # viridis blue
        'nh3':   '#b5de2b',  # viridis lime
    }
    pollutant_labels = {
        'co': 'CO (μg/m³)',
        'no': 'NO (μg/m³)',
        'no2': 'NO₂ (μg/m³)',
        'o3': 'O₃ (μg/m³)',
        'so2': 'SO₂ (μg/m³)',
        'pm2_5': 'PM2.5 (μg/m³)',
        'pm10': 'PM10 (μg/m³)',
        'nh3': 'NH₃ (μg/m³)',
    }
    for i, pol in enumerate(['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']):
        if pol in owm_filtered:
            ax2.plot(owm_filtered['datetime'], owm_filtered[pol],
                     color=pollutant_colors[pol], linewidth=2 if pol=='co' else 1, label=pollutant_labels[pol],
                     marker='o' if pol=='co' else None, markersize=4 if pol=='co' else None, zorder=10-i)
    ax2.set_ylabel('Mean OpenWeatherMap API Pollutant Density (μg/m³)', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(bottom=0)
    
    # Set x-axis
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_xlim(start_date, end_date)
    
    # Major ticks at the 1st of each month (with tick marks, no label)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax1.xaxis.set_major_formatter(NullFormatter())

    # Minor ticks at the 15th of each month (no tick mark, but with label)
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%B'))

    # Set minor tick length to 0 (no visible tick)
    ax1.tick_params(axis='x', which='minor', length=0)
    # Set major tick length as desired (default or e.g. 5)
    ax1.tick_params(axis='x', which='major', length=5)

    # Center the minor tick labels
    plt.setp(ax1.xaxis.get_minorticklabels(), ha='center')
    
    # Set title
    plt.title(f'Traffic vs Pollution Analysis - Station {station}, Year {year}\n'
              f'CO & Other Pollutants vs Daily Traffic Volume', fontsize=14, fontweight='bold')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def main():
    """
    Main function to run the traffic vs pollution analysis.
    """
    print("Starting Traffic vs Pollution Analysis...")
    
    # Find matching files
    matching_files = find_matching_files()
    
    if not matching_files:
        print("No matching OWM and BASt files found!")
        return
    
    print(f"Found {len(matching_files)} matching file pairs.")
    
    # Randomly select one pair
    random.seed(42)  # Set seed for reproducible results
    station, year, owm_file, bast_file = random.choice(matching_files)
    print(f"Selected: Station {station}, Year {year}")
    print(f"OWM file: {owm_file}")
    print(f"BASt file: {bast_file}")
    
    # Load and process data
    try:
        owm_data, bast_data = load_and_process_data(owm_file, bast_file)
        print(f"Loaded OWM data: {len(owm_data)} records")
        print(f"Loaded BASt data: {len(bast_data)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create the plot
    try:
        fig = create_traffic_vs_pollution_plot(owm_data, bast_data, station, year)
        
        # Save the plot
        output_path = f"Graphs/traffic_vs_pollution_station_{station}_{year}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 