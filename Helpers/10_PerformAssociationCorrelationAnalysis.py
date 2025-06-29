import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import warnings
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy import stats
warnings.filterwarnings('ignore')

def load_and_prepare_data(station_number=2002):
    """
    Load and prepare traffic and air quality data for a specific station across all available years
    """
    print(f"Loading data for station {station_number} across all available years...")
    
    # Load all years of traffic data for the station
    traffic_data_list = []
    air_data_list = []
    
    # Find all traffic files for this station
    traffic_dir = "BASt Hourly Data"
    traffic_files = [f for f in os.listdir(traffic_dir) if f.startswith(f'zst{station_number}_') and f.endswith('.csv')]
    
    # Find all air quality files for this station
    air_dir = "owm Hourly Data"
    air_files = [f for f in os.listdir(air_dir) if f.startswith(f'owm{station_number}_') and f.endswith('.csv')]
    
    print(f"Found {len(traffic_files)} traffic files: {traffic_files}")
    print(f"Found {len(air_files)} air quality files: {air_files}")
    
    # Load all traffic data
    for filename in sorted(traffic_files):
        file_path = os.path.join(traffic_dir, filename)
        try:
            df = pd.read_csv(file_path, sep=';')
            if not df.empty:
                traffic_data_list.append(df)
                print(f"Loaded traffic data: {filename} ({len(df)} records)")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Load all air quality data
    for filename in sorted(air_files):
        file_path = os.path.join(air_dir, filename)
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                air_data_list.append(df)
                print(f"Loaded air quality data: {filename} ({len(df)} records)")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not traffic_data_list:
        raise ValueError(f"No traffic data found for station {station_number}")
    if not air_data_list:
        raise ValueError(f"No air quality data found for station {station_number}")
    
    # Combine all years of traffic data
    traffic_df = pd.concat(traffic_data_list, ignore_index=True)
    print(f"Combined traffic data: {len(traffic_df)} total records")
    
    # Combine all years of air quality data
    air_df = pd.concat(air_data_list, ignore_index=True)
    print(f"Combined air quality data: {len(air_df)} total records")
    
    # Remove duplicates if any
    traffic_df = traffic_df.drop_duplicates()
    air_df = air_df.drop_duplicates()
    
    print(f"After deduplication - Traffic: {len(traffic_df)} records, Air: {len(air_df)} records")
    
    return traffic_df, air_df

def parse_german_date_time(datum, stunde):
    """
    Parse German date and hour to datetime object.
    
    Args:
        datum (str): Date in YYMMDD format (e.g., "30101")
        stunde (str): Hour in HH format (e.g., "01")
    
    Returns:
        datetime: Parsed datetime object in German timezone
    
    Raises:
        ValueError: If date or hour cannot be parsed
    """
    datum = str(datum).zfill(6) # fix a weird bug where leading 0es are omited in the source data, argh
    year = 2000 + int(datum[:2])
    month = int(datum[2:4])
    day = int(datum[4:6])
    hour = int(stunde)
    if hour == 24:
        hour = 0
        temp_dt = datetime(year, month, day)
        next_day = temp_dt + timedelta(days=1)
        year, month, day = next_day.year, next_day.month, next_day.day
    dt = datetime(year, month, day, hour, 0, 0, tzinfo=pytz.timezone('Europe/Berlin'))
    return dt

def convert_to_unix_start(dt):
    """
    Convert datetime to Unix timestamp (start of hour)
    """
    utc_dt = dt.astimezone(pytz.UTC)
    return int((utc_dt - timedelta(hours=1)).timestamp())

def convert_to_unix_end(dt):
    """
    Convert datetime to Unix timestamp (end of hour)
    """
    utc_dt = dt.astimezone(pytz.UTC)
    return int((utc_dt - timedelta(seconds=1)).timestamp())

def process_traffic_data(traffic_df):
    """
    Process traffic data to create total traffic counts (KFZ + LKW)
    Using the same datetime conversion logic as 06_FormatForOpenWeather.py
    """
    print("Processing traffic data...")
    
    # Create datetime column using the same logic as 06_FormatForOpenWeather.py
    traffic_df['datetime'] = traffic_df.apply(
        lambda row: parse_german_date_time(row['Datum'], row['Stunde']), 
        axis=1
    )
    
    # Convert to UTC for consistent matching
    traffic_df['datetime_utc'] = traffic_df['datetime'].dt.tz_convert('UTC')
    
    # Calculate total traffic (KFZ + LKW) for both directions
    # KFZ_R1 + KFZ_R2 + Lkw_R1 + Lkw_R2
    traffic_df['total_traffic'] = (
        traffic_df['KFZ_R1'].fillna(0) + 
        traffic_df['KFZ_R2'].fillna(0) + 
        traffic_df['Lkw_R1'].fillna(0) + 
        traffic_df['Lkw_R2'].fillna(0)
    )
    
    # Group by hour to get hourly totals
    traffic_hourly = traffic_df.groupby('datetime_utc').agg({
        'total_traffic': 'sum'
    }).reset_index()
    
    print(f"Hourly traffic data shape: {traffic_hourly.shape}")
    return traffic_hourly

def process_air_quality_data(air_df):
    """
    Process air quality data to match with traffic data
    Using proper Unix timestamp conversion
    """
    print("Processing air quality data...")
    
    # Convert dt (Unix timestamp) to datetime in UTC
    air_df['datetime_utc'] = pd.to_datetime(air_df['dt'], unit='s', utc=True)
    
    # Select relevant air quality columns
    air_columns = ['datetime_utc', 'aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    air_processed = air_df[air_columns].copy()
    
    print(f"Processed air quality data shape: {air_processed.shape}")
    return air_processed

def merge_datasets(traffic_hourly, air_processed):
    """
    Simple merge of traffic and air quality data on datetime
    """
    print("Merging datasets...")
    
    # Round both datasets to hour for exact matching
    traffic_hourly['datetime_hour'] = traffic_hourly['datetime_utc'].dt.floor('H')
    air_processed['datetime_hour'] = air_processed['datetime_utc'].dt.floor('H')
    
    # Merge on hour
    merged_df = pd.merge(traffic_hourly, air_processed, on='datetime_hour', how='inner')
    
    # Clean up
    merged_df = merged_df.drop('datetime_hour', axis=1)
    merged_df = merged_df.rename(columns={'datetime_utc_x': 'datetime_utc'})
    merged_df = merged_df.drop('datetime_utc_y', axis=1)
    
    # Remove rows with negative values in any numeric column
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'total_traffic':  # Keep traffic data as is
            merged_df = merged_df[merged_df[col] >= 0]
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Date range: {merged_df['datetime_utc'].min()} to {merged_df['datetime_utc'].max()}")
    
    return merged_df

def perform_correlation_analysis(merged_df):
    """
    Perform correlation analysis between traffic and air pollutants using established statistical methods
    """
    print("Performing correlation analysis...")
    
    # Focus on traffic vs pollutants correlation
    pollutant_columns = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    # Calculate correlation between traffic and each pollutant
    correlation_data = []
    
    for pollutant in pollutant_columns:
        if pollutant in merged_df.columns:
            # Remove NaN values for this specific pollutant
            mask = ~(merged_df['total_traffic'].isna() | merged_df[pollutant].isna())
            traffic_clean = merged_df['total_traffic'][mask]
            pollutant_clean = merged_df[pollutant][mask]
            
            if len(traffic_clean) > 2:  # Need at least 3 points for correlation
                try:
                    # Calculate correlation using pandas (more reliable for type checking)
                    corr = traffic_clean.corr(pollutant_clean)
                    
                    # Calculate p-value and confidence intervals using scipy
                    if not np.isnan(corr) and abs(corr) < 1.0:
                        # Calculate t-statistic for p-value
                        n = len(traffic_clean)
                        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        
                        # Calculate 95% confidence interval using Fisher's z-transformation
                        # This is the standard method for correlation confidence intervals
                        z_corr = np.arctanh(corr)  # Fisher's z-transformation
                        z_se = 1 / np.sqrt(n - 3)  # Standard error of z
                        z_ci_lower = z_corr - 1.96 * z_se  # 95% CI lower bound
                        z_ci_upper = z_corr + 1.96 * z_se  # 95% CI upper bound
                        
                        # Transform back to correlation scale
                        ci_lower = np.tanh(z_ci_lower)
                        ci_upper = np.tanh(z_ci_upper)
                        
                        # Calculate standard error of correlation
                        corr_se = np.sqrt((1 - corr**2) / (n - 2))
                        
                        # Determine significance using standard thresholds
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"
                        
                        correlation_data.append({
                            'pollutant': pollutant,
                            'correlation': corr,
                            'p_value': p_value,
                            'std_error': corr_se,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'sample_size': n,
                            'significance': significance
                        })
                except (ValueError, RuntimeWarning):
                    # Skip if correlation calculation fails
                    continue
    
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df = correlation_df.sort_values('correlation', ascending=False)
    
    # Print results using standard format
    print("\n" + "="*90)
    print("PEARSON CORRELATION ANALYSIS: TRAFFIC VS POLLUTANTS")
    print("="*90)
    print(f"{'Pollutant':<8} {'r':<8} {'SE':<8} {'95% CI':<20} {'p-value':<10} {'N':<6} {'Sig':<4}")
    print("-" * 90)
    
    for _, row in correlation_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(f"{row['pollutant']:<8} {row['correlation']:<8.3f} {row['std_error']:<8.3f} "
              f"{ci_str:<20} {row['p_value']:<10.4f} {row['sample_size']:<6} {row['significance']:<4}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("SE = Standard Error, CI = Confidence Interval")
    print("="*90)
    
    # Summary of significant correlations
    significant_correlations = correlation_df[correlation_df['p_value'] < 0.05]
    print(f"\nSignificant correlations (p < 0.05): {len(significant_correlations)} out of {len(correlation_df)}")
    
    if len(significant_correlations) > 0:
        print("Significant correlations with confidence intervals:")
        for _, row in significant_correlations.iterrows():
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            print(f"  {row['pollutant']}: r = {row['correlation']:.3f} ± {row['std_error']:.3f}, "
                  f"95% CI = {ci_str}, p = {row['p_value']:.4f} {row['significance']}")
    
    # Null hypothesis rejection summary
    print(f"\nNull hypothesis rejection summary:")
    print(f"  Total correlations tested: {len(correlation_df)}")
    print(f"  Null hypothesis rejected (p < 0.05): {len(significant_correlations)}")
    print(f"  Null hypothesis not rejected (p ≥ 0.05): {len(correlation_df) - len(significant_correlations)}")
    
    if len(significant_correlations) > 0:
        print(f"\nReasons to reject null hypothesis:")
        for _, row in significant_correlations.iterrows():
            direction = "positive" if row['correlation'] > 0 else "negative"
            strength = "strong" if abs(row['correlation']) > 0.7 else "moderate" if abs(row['correlation']) > 0.3 else "weak"
            print(f"  {row['pollutant']}: {strength} {direction} correlation (r = {row['correlation']:.3f}, p = {row['p_value']:.4f})")
            print(f"    - 95% CI does not include zero: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            print(f"    - Standard error: ±{row['std_error']:.3f}")
            print(f"    - Sample size: n = {row['sample_size']}")
    
    print("\n" + "="*90)
    
    # Also calculate full correlation matrix for reference
    full_correlation_columns = ['total_traffic'] + pollutant_columns
    correlation_df_full = merged_df[full_correlation_columns].corr()
    
    return correlation_df_full, correlation_df, merged_df

def create_correlation_visualization(correlation_matrix):
    """
    Create correlation matrix visualization
    """
    print("Creating correlation visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap using matplotlib
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Set axis ticks and labels
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    
    # Annotate correlation values
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            if not mask[i, j]:
                val = correlation_matrix.iloc[i, j]
                plt.text(j, i, f'{val:.3f}', ha='center', va='center', 
                        color='black', fontsize=9, fontweight='bold')
    
    plt.title('Correlation Matrix: Traffic vs Air Pollutants', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig('Graphs/traffic_air_quality_correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    
    return correlation_matrix

def print_correlation_summary(correlation_df):
    """
    Print summary of correlations with traffic including statistical significance
    """
    print("\n" + "="*60)
    print("CORRELATION SUMMARY WITH TRAFFIC")
    print("="*60)
    
    # Sort by absolute correlation strength
    correlation_df['abs_corr'] = correlation_df['correlation'].abs()
    sorted_df = correlation_df.sort_values('abs_corr', ascending=False)
    
    print("\nCorrelations with Total Traffic (sorted by strength):")
    print("-" * 60)
    print(f"{'Pollutant':<8} {'Correlation':<12} {'P-value':<10} {'Significance':<12}")
    print("-" * 60)
    
    for _, row in sorted_df.iterrows():
        if not np.isnan(row['correlation']):
            strength = "Strong" if abs(row['correlation']) > 0.7 else "Moderate" if abs(row['correlation']) > 0.3 else "Weak"
            direction = "Positive" if row['correlation'] > 0 else "Negative"
            significance = row['significance']
            
            print(f"{row['pollutant']:<8} {row['correlation']:<12.3f} {row['p_value']:<10.4f} {significance:<12}")
            print(f"{'':<8} ({strength} {direction})")
        else:
            print(f"{row['pollutant']:<8} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
    
    # Summary of significant correlations
    significant_correlations = correlation_df[correlation_df['p_value'] < 0.05]
    print(f"\nSignificant correlations (p < 0.05): {len(significant_correlations)} out of {len(correlation_df)}")
    
    if len(significant_correlations) > 0:
        print("Significant correlations:")
        for _, row in significant_correlations.iterrows():
            print(f"  {row['pollutant']}: r = {row['correlation']:.3f}, p = {row['p_value']:.4f} {row['significance']}")
    
    print("\n" + "="*60)

def create_scatter_plots(merged_df):
    """
    Create scatter plots of traffic vs each pollutant with trend lines
    """
    print("Creating scatter plots...")
    
    # Create custom AQI colormap
    aqi_cmap = create_aqi_colormap()
    
    # Define pollutants in the requested 3x3 grid order
    pollutants = ['aqi', 'no', 'co', 'pm2_5', 'no2', 'o3', 'pm10', 'nh3', 'so2']
    titles = ['Air Quality Index (AQI) /1', 'Nitric Oxide (NO) / μm m⁻³', 'Carbon Monoxide (CO) / μm m⁻³', 
              'Fine Particulate Matter (PM2.5) / μm m⁻³', 'Nitrogen Dioxide (NO2) / μm m⁻³', 'Ozone (O3) / μm m⁻³', 
              'Coarse Particulate Matter (PM10) / μm m⁻³', 'Ammonia (NH3) / μm m⁻³', 'Sulfur Dioxide (SO2) / μm m⁻³']
    # Define colors for each pollutant
    colors = {
        'aqi': 'custom',  # Will be handled specially
        'pm2_5': 'grey',
        'pm10': '#404040',  # Dark grey
        'no': '#D2B48C',    # Light brown
        'no2': '#8B4513',   # Dark brown
        'nh3': 'purple',
        'co': 'black',
        'o3': '#00008B',
        'so2': '#B8860B'    # Dark yellow
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, (pollutant, title) in enumerate(zip(pollutants, titles)):
        if pollutant in merged_df.columns:
            ax = axes[i]
            
            x = merged_df['total_traffic']
            y = merged_df[pollutant]
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                # Calculate correlation
                corr = np.corrcoef(x_clean, y_clean)[0, 1]
                r_squared = corr ** 2
                
                # Create scatter plot with specific colors
                if pollutant == 'aqi':
                    # Color gradient for AQI: green (1) -> yellow (3) -> red (5)
                    # Map AQI values directly to colors: 1=green, 3=yellow, 5=red
                    scatter = ax.scatter(x_clean, y_clean, c=y_clean, cmap=aqi_cmap, 
                                       alpha=0.6, s=20, vmin=1, vmax=5)
                    # Set y-axis to only show integer ticks
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                else:
                    ax.scatter(x_clean, y_clean, color=colors[pollutant], alpha=0.6, s=20)
                
                # Add trend line (skip for AQI)
                if pollutant != 'aqi':
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)
                
                # Add correlation coefficient and R²
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\nR²: {r_squared:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Total Traffic / vehicles h⁻¹')
                ax.set_ylabel(title)
                # Complete mapping for all pollutant titles (full names)
                chemical_titles = {
                    'aqi': 'Air Quality Index (AQI) vs Traffic',
                    'no': 'Nitric Oxide (NO) vs Traffic',
                    'co': 'Carbon Monoxide (CO) vs Traffic',
                    'pm2_5': 'Fine Particulate Matter (PM₂.₅) vs Traffic',
                    'no2': 'Nitrogen Dioxide (NO₂) vs Traffic',
                    'o3': 'Ozone (O₃) vs Traffic',
                    'pm10': 'Coarse Particulate Matter (PM₁₀) vs Traffic',
                    'nh3': 'Ammonia (NH₃) vs Traffic',
                    'so2': 'Sulfur Dioxide (SO₂) vs Traffic'
                }
                # Mapping for axis labels (chemical symbols + units)
                axis_labels = {
                    'aqi': 'AQI / 1',
                    'no': 'NO / μg m⁻³',
                    'co': 'CO / μg m⁻³',
                    'pm2_5': 'PM2.5 / μg m⁻³',
                    'no2': 'NO₂ / μg m⁻³',
                    'o3': 'O₃ / μg m⁻³',
                    'pm10': 'PM10 / μg m⁻³',
                    'nh3': 'NH₃ / μg m⁻³',
                    'so2': 'SO₂ / μg m⁻³'
                }
                plot_title = chemical_titles[pollutant]
                axis_label = axis_labels[pollutant]
                ax.set_title(plot_title, fontsize=12, fontweight='bold')
                ax.set_ylabel(axis_label)
                ax.grid(True, alpha=0.3)
        
    
    plt.tight_layout()
    plt.savefig('Graphs/traffic_vs_pollutants_scatter.png', dpi=300, bbox_inches='tight')

def analyze_temporal_coverage(merged_df):
    """
    Analyze the temporal coverage of the merged dataset
    """
    print("\n" + "="*60)
    print("TEMPORAL COVERAGE ANALYSIS")
    print("="*60)
    
    # Extract year from datetime
    merged_df['year'] = merged_df['datetime_utc'].dt.year
    merged_df['month'] = merged_df['datetime_utc'].dt.month
    merged_df['hour'] = merged_df['datetime_utc'].dt.hour
    
    # Yearly coverage
    yearly_counts = merged_df['year'].value_counts().sort_index()
    print(f"\nData coverage by year:")
    for year, count in yearly_counts.items():
        print(f"  {year}: {count:,} records")
    
    # Overall statistics
    print(f"\nOverall temporal statistics:")
    print(f"  Date range: {merged_df['datetime_utc'].min()} to {merged_df['datetime_utc'].max()}")
    print(f"  Total time span: {(merged_df['datetime_utc'].max() - merged_df['datetime_utc'].min()).days} days")
    print(f"  Total records: {len(merged_df):,}")

def create_aqi_colormap():
    """
    Create a custom colormap for AQI: 1=green (good), 3=yellow (medium), 5=red (bad)
    """
    colors = ['green', 'yellow', 'red']
    n_bins = 5  # AQI scale from 1 to 5
    cmap = LinearSegmentedColormap.from_list('AQI', colors, N=n_bins)
    return cmap

def create_timeseries_overview_plot(merged_df, station_number):
    """
    Create a timeseries plot with:
    - Total Traffic (daily sum, bar, black)
    - CO (daily average, line, black or scatter color)
    - AQI (daily max, scatter, AQI colormap)
    - All other pollutants (daily average, line, scatter colors)
    Add a legend to the Total Traffic subplot with station info.
    """
    print("Creating timeseries overview plot...")
    aqi_cmap = create_aqi_colormap()
    colors = {
        'co': 'black',
        'pm2_5': 'grey',
        'pm10': '#404040',
        'no': '#D2B48C',
        'no2': '#8B4513',
        'nh3': 'purple',
        'o3': '#00008B',
        'so2': '#B8860B'
    }
    # Resample to daily
    df = merged_df.copy()
    df['date'] = df['datetime_utc'].dt.floor('D')
    # Total Traffic: daily sum
    traffic_daily = df.groupby('date')['total_traffic'].sum()
    # CO: daily mean
    co_daily = df.groupby('date')['co'].mean()
    # AQI: daily max (worst)
    aqi_daily = df.groupby('date')['aqi'].max()
    # Other pollutants: daily mean
    other_pollutants = ['no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    pollutant_daily = {p: df.groupby('date')[p].mean() for p in other_pollutants if p in df.columns}

    # Get station info from bast_stations_by_city.csv
    try:
        bast_info = pd.read_csv('BASt Hourly Data/bast_stations_by_city.csv')
        # Use the first matching year for the station
        info_row = bast_info[bast_info['station_number'] == int(station_number)].iloc[0]
        lat = info_row['latitude']
        lon = info_row['longitude']
        city = info_row['city']
    except Exception as e:
        lat, lon, city = 'N/A', 'N/A', 'N/A'
        print(f"Warning: Could not get geolocation for station {station_number}: {e}")
    n_timepoints = len(traffic_daily)
    # Prepare aligned infobox text
    label_pad = 11  # Adjust as needed for alignment
    legend_text = (
        f"{'Station:'.ljust(label_pad)} {station_number}\n"
        f"{'City:'.ljust(label_pad)} {city}\n"
        f"{'Timepoints:'.ljust(label_pad)} {n_timepoints}\n"
        f"{'Lat:'.ljust(label_pad)} {lat if lat == 'N/A' else f'{lat:.5f}'}\n"
        f"{'Lon:'.ljust(label_pad)} {lon if lon == 'N/A' else f'{lon:.5f}'}"
    )

    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
    # 1. Total Traffic
    axes[0].bar(traffic_daily.index, traffic_daily.values, color='black', width=1)
    axes[0].set_ylabel('Total Traffic\n(vehicles/day)')
    axes[0].set_title('Total Traffic (Daily Sum)')
    axes[0].grid(True, alpha=0.3)
    # 2. CO
    axes[1].plot(co_daily.index, co_daily.values, color=colors['co'], linewidth=1)
    axes[1].set_ylabel('CO / μg m⁻³')
    axes[1].set_title('CO (Daily Average)')
    axes[1].grid(True, alpha=0.3)
    # 3. AQI
    scatter = axes[2].scatter(aqi_daily.index, aqi_daily.values, c=aqi_daily.values, cmap=aqi_cmap, vmin=1, vmax=5, s=20)
    axes[2].set_ylabel('AQI (Daily Max)')
    axes[2].set_title('AQI (Daily Worst)')
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].grid(True, alpha=0.3)
    # 4. Other pollutants
    for p in pollutant_daily:
        axes[3].plot(pollutant_daily[p].index, pollutant_daily[p].values, label=p.upper() if p not in ['no2','so2','o3','nh3'] else p.replace('2','₂').replace('3','₃').upper(), color=colors[p], linewidth=1)
    axes[3].set_ylabel('Pollutant / μg m⁻³')
    axes[3].set_title('Other Pollutants (Daily Average)')
    axes[3].legend(loc='upper left', ncol=4, fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    # Add overall title to the figure
    fig.suptitle(f'Timeseries Overview Station {station_number} {city.title()}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add infobox with station info AFTER tight_layout to ensure proper positioning
    # I wasted far too much time on this, it doesn't work but I'm not going to change it now
    axes[0].text(
        0.01, 0.98, legend_text,
        transform=fig.transFigure,
        fontsize=10,
        va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    plt.savefig('Graphs/traffic_pollutants_timeseries_overview.png', dpi=300)
    print('Saved: Graphs/traffic_pollutants_timeseries_overview.png')

def main(station_number=5670):
    """
    Main function to execute the correlation analysis for a specific station
    """
    print("Starting Traffic-Air Quality Correlation Analysis")
    print(f"Station: {station_number}")
    print("="*60)
    
    try:
        # Load data for the specified station across all years
        traffic_df, air_df = load_and_prepare_data(station_number)
        
        # Process data
        traffic_hourly = process_traffic_data(traffic_df)
        air_processed = process_air_quality_data(air_df)
        
        # Merge datasets
        merged_df = merge_datasets(traffic_hourly, air_processed)
        
        # Analyze temporal coverage
        analyze_temporal_coverage(merged_df)
        
        # Perform correlation analysis
        correlation_df_full, correlation_df, merged_df = perform_correlation_analysis(merged_df)
        
        # Create visualization
        correlation_matrix = create_correlation_visualization(correlation_df_full)
        
        # Print summary
        print_correlation_summary(correlation_df)
        
        # Save correlation matrix to CSV
        correlation_df_full.to_csv(f'Graphs/station_{station_number}_correlation_matrix.csv')
        print(f"\nCorrelation matrix saved to: Graphs/station_{station_number}_correlation_matrix.csv")
        
        # Save merged dataset for further analysis
        merged_df.to_csv(f'Graphs/station_{station_number}_merged_data.csv', index=False)
        print(f"Merged dataset saved to: Graphs/station_{station_number}_merged_data.csv")
        
        # Create scatter plots
        create_scatter_plots(merged_df)
        
        # Create timeseries overview plot
        create_timeseries_overview_plot(merged_df, station_number)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # You can change the station number here
    station_number = 2002  # Default station
    main(station_number) 