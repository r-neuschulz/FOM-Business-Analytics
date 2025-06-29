import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
import argparse
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
    Create separate plots for traffic-only and traffic+pollutants - both hourly and daily averages.
    """
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    pollutant_colors = {
        'co': 'black',
        'pm2_5': 'grey',
        'pm10': '#404040',  # Dark grey
        'no': '#D2B48C',    # Light brown
        'no2': '#8B4513',   # Dark brown
        'nh3': 'purple',
        'o3': '#00008B',    # Dark blue
        'so2': '#B8860B'    # Dark yellow
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
    
    # ===== HOURLY PLOTS =====
    
    # Process data for hourly plots
    owm_hourly = owm_data[['datetime'] + pollutant_cols].copy()
    bast_hourly = bast_data[['Datum', 'Stunde', 'KFZ_R1', 'KFZ_R2', 'Lkw_R1', 'Lkw_R2']].copy()
    bast_hourly['datetime'] = pd.to_datetime(bast_hourly['Datum'], format='%y%m%d') + pd.to_timedelta(bast_hourly['Stunde'], unit='h')
    
    start_date_hourly = max(owm_hourly['datetime'].min(), bast_hourly['datetime'].min())
    end_date_hourly = min(owm_hourly['datetime'].max(), bast_hourly['datetime'].max())
    
    owm_filtered_hourly = owm_hourly[(owm_hourly['datetime'] >= start_date_hourly) & (owm_hourly['datetime'] <= end_date_hourly)]
    bast_filtered_hourly = bast_hourly[(bast_hourly['datetime'] >= start_date_hourly) & (bast_hourly['datetime'] <= end_date_hourly)]
    
    # 1. Hourly Traffic Only Plot
    fig_hourly_traffic, ax_hourly_traffic = plt.subplots(figsize=(15, 10))  # Same size as combined plot
    ax2_hourly_traffic = ax_hourly_traffic.twinx()  # Add second y-axis for invisible pollutants
    x_dates_hourly = bast_filtered_hourly['datetime']
    width_hourly = pd.Timedelta(hours=1)
    
    bars1_h = ax_hourly_traffic.bar(x_dates_hourly, bast_filtered_hourly['Lkw_R1'], width_hourly, 
                    bottom=np.zeros(len(bast_filtered_hourly)), color='#a05c60', alpha=0.8, label='Lkw_R1', zorder=2)
    bars2_h = ax_hourly_traffic.bar(x_dates_hourly, bast_filtered_hourly['Lkw_R2'], width_hourly,
                    bottom=bast_filtered_hourly['Lkw_R1'], color='#4b2a3b', alpha=0.8, label='Lkw_R2', zorder=2)
    bars3_h = ax_hourly_traffic.bar(x_dates_hourly, bast_filtered_hourly['KFZ_R1'], width_hourly,
                    bottom=bast_filtered_hourly['Lkw_R1'] + bast_filtered_hourly['Lkw_R2'], color='#bdb6b6', alpha=0.8, label='KFZ_R1', zorder=2)
    bars4_h = ax_hourly_traffic.bar(x_dates_hourly, bast_filtered_hourly['KFZ_R2'], width_hourly,
                    bottom=bast_filtered_hourly['Lkw_R1'] + bast_filtered_hourly['Lkw_R2'] + bast_filtered_hourly['KFZ_R1'], color='#7c6e7f', alpha=0.8, label='KFZ_R2', zorder=2)
    
    # Add invisible pollutant lines for seamless overlay
    for i, pol in enumerate(['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']):
        if pol in owm_filtered_hourly:
            if pol == 'co':
                continue
            ax2_hourly_traffic.plot(owm_filtered_hourly['datetime'], owm_filtered_hourly[pol],
                     color=pollutant_colors[pol], linewidth=1, alpha=0.0, zorder=10-i)  # Completely transparent
    
    ax_hourly_traffic.set_ylabel('Total BASt Traffic Count (vehicles/hour)', color='black', fontsize=12)
    ax_hourly_traffic.set_xlabel('Date', fontsize=12)
    ax_hourly_traffic.set_xlim(start_date_hourly, end_date_hourly)
    ax_hourly_traffic.set_title(f'Traffic Analysis - Station {station}, Year {year}\nHourly Traffic Volume', fontsize=14, fontweight='bold')
    ax_hourly_traffic.legend(loc='upper left')
    ax_hourly_traffic.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits for pollutants to match the combined plot
    ax2_hourly_traffic.set_ylim(bottom=0)
    ax2_hourly_traffic.set_ylabel('OpenWeatherMap API Pollutant Density (μg/m³)', color='black', fontsize=12)
    ax2_hourly_traffic.tick_params(axis='y', labelcolor='black')
    
    # Set x-axis to show days rotated 45°
    ax_hourly_traffic.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_hourly_traffic.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax_hourly_traffic.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    # 2. Hourly Traffic + Pollutants Plot
    fig_hourly_combined, ax1_hourly = plt.subplots(figsize=(15, 10))
    ax2_hourly = ax1_hourly.twinx()
    
    # Plot traffic data
    bars1_hc = ax1_hourly.bar(x_dates_hourly, bast_filtered_hourly['Lkw_R1'], width_hourly, 
                    bottom=np.zeros(len(bast_filtered_hourly)), color='#a05c60', alpha=0.8, label='Lkw_R1', zorder=2)
    bars2_hc = ax1_hourly.bar(x_dates_hourly, bast_filtered_hourly['Lkw_R2'], width_hourly,
                    bottom=bast_filtered_hourly['Lkw_R1'], color='#4b2a3b', alpha=0.8, label='Lkw_R2', zorder=2)
    bars3_hc = ax1_hourly.bar(x_dates_hourly, bast_filtered_hourly['KFZ_R1'], width_hourly,
                    bottom=bast_filtered_hourly['Lkw_R1'] + bast_filtered_hourly['Lkw_R2'], color='#bdb6b6', alpha=0.8, label='KFZ_R1', zorder=2)
    bars4_hc = ax1_hourly.bar(x_dates_hourly, bast_filtered_hourly['KFZ_R2'], width_hourly,
                    bottom=bast_filtered_hourly['Lkw_R1'] + bast_filtered_hourly['Lkw_R2'] + bast_filtered_hourly['KFZ_R1'], color='#7c6e7f', alpha=0.8, label='KFZ_R2', zorder=2)
    
    ax1_hourly.set_ylabel('Total BASt Traffic Count (vehicles/hour)', color='black', fontsize=12)
    ax1_hourly.tick_params(axis='y', labelcolor='black')
    
    # Plot pollutant lines
    for i, pol in enumerate(['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']):
        if pol in owm_filtered_hourly:
            if pol == 'co':
                continue
            ax2_hourly.plot(owm_filtered_hourly['datetime'], owm_filtered_hourly[pol],
                     color=pollutant_colors[pol], linewidth=1, label=pollutant_labels[pol], zorder=10-i)
    
    ax2_hourly.set_ylabel('OpenWeatherMap API Pollutant Density (μg/m³)', color='black', fontsize=12)
    ax2_hourly.tick_params(axis='y', labelcolor='black')
    ax2_hourly.set_ylim(bottom=0)
    
    ax1_hourly.set_xlabel('Date', fontsize=12)
    ax1_hourly.set_xlim(start_date_hourly, end_date_hourly)
    ax1_hourly.set_title(f'Traffic vs Pollution Analysis - Station {station}, Year {year}\nPollutants vs Hourly Traffic Volume', fontsize=14, fontweight='bold')
    ax1_hourly.legend(loc='upper left')
    ax2_hourly.legend(loc='upper right')
    ax1_hourly.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis to show days rotated 45°
    ax1_hourly.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1_hourly.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1_hourly.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    # ===== DAILY PLOTS =====
    
    # Process data for daily plots
    owm_daily = owm_data.groupby('date')[pollutant_cols].mean().reset_index()
    owm_daily['datetime'] = pd.to_datetime(owm_daily['date'])
    
    bast_daily = bast_data.groupby('date').agg({
        'KFZ_R1': 'sum',
        'KFZ_R2': 'sum', 
        'Lkw_R1': 'sum',
        'Lkw_R2': 'sum'
    }).reset_index()
    bast_daily['datetime'] = pd.to_datetime(bast_daily['date'])
    
    start_date_daily = max(owm_daily['datetime'].min(), bast_daily['datetime'].min())
    end_date_daily = min(owm_daily['datetime'].max(), bast_daily['datetime'].max())
    
    owm_filtered_daily = owm_daily[(owm_daily['datetime'] >= start_date_daily) & (owm_daily['datetime'] <= end_date_daily)]
    bast_filtered_daily = bast_daily[(bast_daily['datetime'] >= start_date_daily) & (bast_daily['datetime'] <= end_date_daily)]
    
    # 3. Daily Traffic Only Plot
    fig_daily_traffic, ax_daily_traffic = plt.subplots(figsize=(15, 10))  # Same size as combined plot
    ax2_daily_traffic = ax_daily_traffic.twinx()  # Add second y-axis for invisible pollutants
    x_dates_daily = bast_filtered_daily['datetime']
    width_daily = 1  # days
    
    bars1_d = ax_daily_traffic.bar(x_dates_daily, bast_filtered_daily['Lkw_R1'], width_daily, 
                    bottom=np.zeros(len(bast_filtered_daily)), color='#a05c60', alpha=0.8, label='Lkw_R1', zorder=2)
    bars2_d = ax_daily_traffic.bar(x_dates_daily, bast_filtered_daily['Lkw_R2'], width_daily,
                    bottom=bast_filtered_daily['Lkw_R1'], color='#4b2a3b', alpha=0.8, label='Lkw_R2', zorder=2)
    bars3_d = ax_daily_traffic.bar(x_dates_daily, bast_filtered_daily['KFZ_R1'], width_daily,
                    bottom=bast_filtered_daily['Lkw_R1'] + bast_filtered_daily['Lkw_R2'], color='#bdb6b6', alpha=0.8, label='KFZ_R1', zorder=2)
    bars4_d = ax_daily_traffic.bar(x_dates_daily, bast_filtered_daily['KFZ_R2'], width_daily,
                    bottom=bast_filtered_daily['Lkw_R1'] + bast_filtered_daily['Lkw_R2'] + bast_filtered_daily['KFZ_R1'], color='#7c6e7f', alpha=0.8, label='KFZ_R2', zorder=2)
    
    # Add invisible pollutant lines for seamless overlay
    for i, pol in enumerate(['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']):
        if pol in owm_filtered_daily:
            if pol == 'co':
                continue
            ax2_daily_traffic.plot(owm_filtered_daily['datetime'], owm_filtered_daily[pol],
                     color=pollutant_colors[pol], linewidth=2 if pol=='co' else 1, alpha=0.0, zorder=10-i)  # Completely transparent
    
    ax_daily_traffic.set_ylabel('Total BASt Traffic Count (vehicles/day)', color='black', fontsize=12)
    ax_daily_traffic.set_xlabel('Date', fontsize=12)
    ax_daily_traffic.set_xlim(start_date_daily, end_date_daily)
    ax_daily_traffic.set_title(f'Traffic Analysis - Station {station}, Year {year}\nDaily Traffic Volume', fontsize=14, fontweight='bold')
    ax_daily_traffic.legend(loc='upper left')
    ax_daily_traffic.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits for pollutants to match the combined plot
    ax2_daily_traffic.set_ylim(bottom=0)
    ax2_daily_traffic.set_ylabel('Mean OpenWeatherMap API Pollutant Density (μg/m³)', color='black', fontsize=12)
    ax2_daily_traffic.tick_params(axis='y', labelcolor='black')
    
    # Set x-axis to show days rotated 45°
    ax_daily_traffic.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax_daily_traffic.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax_daily_traffic.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    # 4. Daily Traffic + Pollutants Plot
    fig_daily_combined, ax1_daily = plt.subplots(figsize=(15, 10))
    ax2_daily = ax1_daily.twinx()
    
    # Plot traffic data
    bars1_dc = ax1_daily.bar(x_dates_daily, bast_filtered_daily['Lkw_R1'], width_daily, 
                    bottom=np.zeros(len(bast_filtered_daily)), color='#a05c60', alpha=0.8, label='Lkw_R1', zorder=2)
    bars2_dc = ax1_daily.bar(x_dates_daily, bast_filtered_daily['Lkw_R2'], width_daily,
                    bottom=bast_filtered_daily['Lkw_R1'], color='#4b2a3b', alpha=0.8, label='Lkw_R2', zorder=2)
    bars3_dc = ax1_daily.bar(x_dates_daily, bast_filtered_daily['KFZ_R1'], width_daily,
                    bottom=bast_filtered_daily['Lkw_R1'] + bast_filtered_daily['Lkw_R2'], color='#bdb6b6', alpha=0.8, label='KFZ_R1', zorder=2)
    bars4_dc = ax1_daily.bar(x_dates_daily, bast_filtered_daily['KFZ_R2'], width_daily,
                    bottom=bast_filtered_daily['Lkw_R1'] + bast_filtered_daily['Lkw_R2'] + bast_filtered_daily['KFZ_R1'], color='#7c6e7f', alpha=0.8, label='KFZ_R2', zorder=2)
    
    ax1_daily.set_ylabel('Total BASt Traffic Count (vehicles/day)', color='black', fontsize=12)
    ax1_daily.tick_params(axis='y', labelcolor='black')
    
    # Plot pollutant lines
    for i, pol in enumerate(['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']):
        if pol in owm_filtered_daily:
            if pol == 'co':
                continue
            ax2_daily.plot(owm_filtered_daily['datetime'], owm_filtered_daily[pol],
                     color=pollutant_colors[pol], linewidth=2 if pol=='co' else 1, label=pollutant_labels[pol],
                     marker='o' if pol=='co' else None, markersize=4 if pol=='co' else None, zorder=10-i)
    
    ax2_daily.set_ylabel('Mean OpenWeatherMap API Pollutant Density (μg/m³)', color='black', fontsize=12)
    ax2_daily.tick_params(axis='y', labelcolor='black')
    ax2_daily.set_ylim(bottom=0)
    
    ax1_daily.set_xlabel('Date', fontsize=12)
    ax1_daily.set_xlim(start_date_daily, end_date_daily)
    ax1_daily.set_title(f'Traffic vs Pollution Analysis - Station {station}, Year {year}\nPollutants vs Daily Traffic Volume', fontsize=14, fontweight='bold')
    ax1_daily.legend(loc='upper left')
    ax2_daily.legend(loc='upper right')
    ax1_daily.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis to show days rotated 45°
    ax1_daily.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1_daily.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1_daily.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    return fig_hourly_traffic, fig_hourly_combined, fig_daily_traffic, fig_daily_combined

def main():
    """
    Main function to run the traffic vs pollution analysis.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Traffic vs Pollution Analysis')
    parser.add_argument('--station', type=str, default='2002', 
                       help='Station number to analyze (default: 2002)')
    args = parser.parse_args()
    
    print("Starting Traffic vs Pollution Analysis...")
    print(f"Target station: {args.station}")
    
    # Find matching files
    matching_files = find_matching_files()
    
    if not matching_files:
        print("No matching OWM and BASt files found!")
        return
    
    print(f"Found {len(matching_files)} matching file pairs.")
    
    # Filter files for the specified station
    station_files = [file_tuple for file_tuple in matching_files if file_tuple[0] == args.station]
    
    if not station_files:
        print(f"No matching files found for station {args.station}")
        print("Available stations:", sorted(list(set([f[0] for f in matching_files]))))
        return
    
    # Use the first pair from the specified station
    station, year, owm_file, bast_file = station_files[0]
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
        fig_hourly_traffic, fig_hourly_combined, fig_daily_traffic, fig_daily_combined = create_traffic_vs_pollution_plot(owm_data, bast_data, station, year)
        
        # Save the hourly plot
        output_path_hourly_traffic = f"Graphs/traffic_vs_pollution_station_{station}_{year}_hourly_traffic.png"
        fig_hourly_traffic.savefig(output_path_hourly_traffic, dpi=300, bbox_inches='tight')
        print(f"Hourly traffic plot saved to: {output_path_hourly_traffic}")

        # Save the hourly combined plot
        output_path_hourly_combined = f"Graphs/traffic_vs_pollution_station_{station}_{year}_hourly_combined.png"
        fig_hourly_combined.savefig(output_path_hourly_combined, dpi=300, bbox_inches='tight')
        print(f"Hourly combined plot saved to: {output_path_hourly_combined}")

        # Save the daily traffic plot
        output_path_daily_traffic = f"Graphs/traffic_vs_pollution_station_{station}_{year}_daily_traffic.png"
        fig_daily_traffic.savefig(output_path_daily_traffic, dpi=300, bbox_inches='tight')
        print(f"Daily traffic plot saved to: {output_path_daily_traffic}")

        # Save the daily combined plot
        output_path_daily_combined = f"Graphs/traffic_vs_pollution_station_{station}_{year}_daily_combined.png"
        fig_daily_combined.savefig(output_path_daily_combined, dpi=300, bbox_inches='tight')
        print(f"Daily combined plot saved to: {output_path_daily_combined}")

    except Exception as e:
        print(f"Error creating plot: {e}")
        return
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 