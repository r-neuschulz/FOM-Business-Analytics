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
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.gridspec import GridSpec
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
    Using the same datetime conversion logic as 06_GetOpenWeatherHourlyData.py
    """
    print("Processing traffic data...")
    
    # Create datetime column using the same logic as 06_GetOpenWeatherHourlyData.py
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
                    
                    # Calculate descriptive statistics
                    traffic_mean = traffic_clean.mean()
                    traffic_std = traffic_clean.std()
                    pollutant_mean = pollutant_clean.mean()
                    pollutant_std = pollutant_clean.std()
                    
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
                            'significance': significance,
                            'traffic_mean': traffic_mean,
                            'traffic_std': traffic_std,
                            'pollutant_mean': pollutant_mean,
                            'pollutant_std': pollutant_std
                        })
                except (ValueError, RuntimeWarning):
                    # Skip if correlation calculation fails
                    continue
    
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df = correlation_df.sort_values('correlation', ascending=False)
    
    # Print results using standard format with descriptive statistics
    print("\n" + "="*120)
    print("PEARSON CORRELATION ANALYSIS: TRAFFIC VS POLLUTANTS")
    print("="*120)
    print(f"{'Pollutant':<8} {'r':<8} {'SE':<8} {'95% CI':<20} {'p-value':<10} {'N':<6} {'Sig':<4} {'Traffic Mean':<15} {'Pollutant Mean':<15}")
    print("-" * 120)
    
    for _, row in correlation_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(f"{row['pollutant']:<8} {row['correlation']:<8.3f} {row['std_error']:<8.3f} "
              f"{ci_str:<20} {row['p_value']:<10.4f} {row['sample_size']:<6} {row['significance']:<4} "
              f"{row['traffic_mean']:<15.1f} {row['pollutant_mean']:<15.3f}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("SE = Standard Error, CI = Confidence Interval")
    print("Traffic Mean = Mean vehicles per hour, Pollutant Mean = Mean concentration")
    print("="*120)
    
    # Summary of significant correlations
    significant_correlations = correlation_df[correlation_df['p_value'] < 0.05]
    print(f"\nSignificant correlations (p < 0.05): {len(significant_correlations)} out of {len(correlation_df)}")
    
    if len(significant_correlations) > 0:
        print("Significant correlations with confidence intervals:")
        for _, row in significant_correlations.iterrows():
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            print(f"  {row['pollutant']}: r = {row['correlation']:.3f} ± {row['std_error']:.3f}, "
                  f"95% CI = {ci_str}, p = {row['p_value']:.4f} {row['significance']}")
            print(f"    Traffic: {row['traffic_mean']:.1f} ± {row['traffic_std']:.1f} vehicles/h")
            print(f"    {row['pollutant']}: {row['pollutant_mean']:.3f} ± {row['pollutant_std']:.3f}")
    
    # Null hypothesis rejection summary
    print(f"\nNull hypothesis rejection summary:")
    print(f"  Total correlations tested: {len(correlation_df)}")
    print(f"  Null hypothesis rejected (p < 0.05): {len(significant_correlations)}")
    print(f"  Null hypothesis not rejected (p >= 0.05): {len(correlation_df) - len(significant_correlations)}")
    
    if len(significant_correlations) > 0:
        print(f"\nReasons to reject null hypothesis:")
        for _, row in significant_correlations.iterrows():
            direction = "positive" if row['correlation'] > 0 else "negative"
            strength = "strong" if abs(row['correlation']) > 0.7 else "moderate" if abs(row['correlation']) > 0.3 else "weak"
            print(f"  {row['pollutant']}: {strength} {direction} correlation (r = {row['correlation']:.3f}, p = {row['p_value']:.4f})")
            print(f"    - 95% CI does not include zero: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            print(f"    - Standard error: ±{row['std_error']:.3f}")
            print(f"    - Sample size: n = {row['sample_size']}")
            print(f"    - Traffic levels: {row['traffic_mean']:.1f} ± {row['traffic_std']:.1f} vehicles/h")
            print(f"    - {row['pollutant']} levels: {row['pollutant_mean']:.3f} ± {row['pollutant_std']:.3f}")
    
    print("\n" + "="*120)
    
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
    
    plt.savefig(f'Graphs/station_{station_number}_traffic_air_quality_correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    
    return correlation_matrix

def print_correlation_summary(correlation_df):
    """
    Print summary of correlations with traffic including statistical significance
    """
    print("\n" + "="*80)
    print("CORRELATION SUMMARY WITH TRAFFIC")
    print("="*80)
    
    # Sort by absolute correlation strength
    correlation_df['abs_corr'] = correlation_df['correlation'].abs()
    sorted_df = correlation_df.sort_values('abs_corr', ascending=False)
    
    print("\nCorrelations with Total Traffic (sorted by strength):")
    print("-" * 80)
    print(f"{'Pollutant':<8} {'Correlation':<12} {'P-value':<10} {'Significance':<12} {'Traffic Mean':<15} {'Pollutant Mean':<15}")
    print("-" * 80)
    
    for _, row in sorted_df.iterrows():
        if not np.isnan(row['correlation']):
            strength = "Strong" if abs(row['correlation']) > 0.7 else "Moderate" if abs(row['correlation']) > 0.3 else "Weak"
            direction = "Positive" if row['correlation'] > 0 else "Negative"
            significance = row['significance']
            
            print(f"{row['pollutant']:<8} {row['correlation']:<12.3f} {row['p_value']:<10.4f} {significance:<12} "
                  f"{row['traffic_mean']:<15.1f} {row['pollutant_mean']:<15.3f}")
            print(f"{'':<8} ({strength} {direction})")
        else:
            print(f"{row['pollutant']:<8} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<15}")
    
    # Summary of significant correlations
    significant_correlations = correlation_df[correlation_df['p_value'] < 0.05]
    print(f"\nSignificant correlations (p < 0.05): {len(significant_correlations)} out of {len(correlation_df)}")
    
    if len(significant_correlations) > 0:
        print("Significant correlations with descriptive statistics:")
        for _, row in significant_correlations.iterrows():
            print(f"  {row['pollutant']}: r = {row['correlation']:.3f}, p = {row['p_value']:.4f} {row['significance']}")
            print(f"    Traffic: {row['traffic_mean']:.1f} ± {row['traffic_std']:.1f} vehicles/h")
            print(f"    {row['pollutant']}: {row['pollutant_mean']:.3f} ± {row['pollutant_std']:.3f}")
    
    # Practical significance assessment
    print(f"\nPractical Significance Assessment:")
    print(f"  Traffic levels: Mean = {correlation_df['traffic_mean'].iloc[0]:.1f} vehicles/h")
    print(f"  This represents typical hourly traffic volume at this station")
    print(f"  Correlations show how traffic changes relate to pollutant changes")
    
    # Effect size interpretation
    print(f"\nEffect Size Interpretation:")
    print(f"  Strong correlations (|r| > 0.7): {len(correlation_df[correlation_df['abs_corr'] > 0.7])}")
    print(f"  Moderate correlations (0.3 < |r| <= 0.7): {len(correlation_df[(correlation_df['abs_corr'] > 0.3) & (correlation_df['abs_corr'] <= 0.7)])}")
    print(f"  Weak correlations (|r| <= 0.3): {len(correlation_df[correlation_df['abs_corr'] <= 0.3])}")
    
    print("\n" + "="*80)

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
                    'pm2_5': 'Fine Particulate Matter (PM2.5) vs Traffic',
                    'no2': 'Nitrogen Dioxide (NO₂) vs Traffic',
                    'o3': 'Ozone (O₃) vs Traffic',
                    'pm10': 'Coarse Particulate Matter (PM10) vs Traffic',
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
    plt.savefig(f'Graphs/station_{station_number}_traffic_vs_pollutants_scatter.png', dpi=300, bbox_inches='tight')

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
    
    plt.savefig(f'Graphs/station_{station_number}_traffic_pollutants_timeseries_overview.png', dpi=300)
    print(f'Saved: Graphs/station_{station_number}_traffic_pollutants_timeseries_overview.png')

def perform_linear_regression_analysis(merged_df):
    """
    Perform linear regression analysis between traffic and air pollutants to get effect sizes
    """
    print("\n" + "="*120)
    print("LINEAR REGRESSION ANALYSIS: TRAFFIC VS POLLUTANTS")
    print("="*120)
    
    # Focus on traffic vs pollutants regression
    pollutant_columns = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    # Calculate regression for each pollutant
    regression_data = []
    
    for pollutant in pollutant_columns:
        if pollutant in merged_df.columns:
            # Remove NaN values for this specific pollutant
            mask = ~(merged_df['total_traffic'].isna() | merged_df[pollutant].isna())
            traffic_clean = merged_df['total_traffic'][mask]
            pollutant_clean = merged_df[pollutant][mask]
            
            if len(traffic_clean) > 2:  # Need at least 3 points for regression
                try:
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(traffic_clean, pollutant_clean)
                    r_squared = float(r_value) ** 2 # type: ignore
                    
                    # Calculate confidence intervals for slope
                    # Degrees of freedom
                    df = len(traffic_clean) - 2
                    
                    # t-critical value for 95% confidence interval
                    t_critical = stats.t.ppf(0.975, df)
                    
                    # Confidence interval for slope
                    slope_ci_lower = float(slope) - t_critical * float(std_err) #type: ignore
                    slope_ci_upper = float(slope) + t_critical * float(std_err) #type: ignore
                    
                    # Calculate predicted values for R-squared interpretation
                    predicted = slope * traffic_clean + intercept
                    
                    # Calculate mean values for context
                    traffic_mean = traffic_clean.mean()
                    pollutant_mean = pollutant_clean.mean()
                    
                    # Determine significance
                    if float(p_value) < 0.001: #type: ignore
                        significance = "***"
                    elif float(p_value) < 0.01: #type: ignore
                        significance = "**"
                    elif float(p_value) < 0.05: #type: ignore
                        significance = "*"
                    else:
                        significance = "ns"
                    
                    regression_data.append({
                        'pollutant': pollutant,
                        'slope': float(slope), #type: ignore
                        'slope_std_err': float(std_err), #type: ignore
                        'slope_ci_lower': slope_ci_lower,
                        'slope_ci_upper': slope_ci_upper,
                        'intercept': float(intercept), #type: ignore
                        'r_squared': r_squared,
                        'p_value': float(p_value), #type: ignore
                        'significance': significance,
                        'sample_size': len(traffic_clean),
                        'traffic_mean': traffic_mean,
                        'pollutant_mean': pollutant_mean
                    })
                    
                except (ValueError, RuntimeWarning):
                    # Skip if regression calculation fails
                    continue
    
    regression_df = pd.DataFrame(regression_data)
    regression_df = regression_df.sort_values('r_squared', ascending=False)
    
    # Print results
    print(f"{'Pollutant':<8} {'Slope':<12} {'SE':<10} {'95% CI':<25} {'R²':<8} {'p-value':<10} {'Sig':<4} {'Pollutant Mean':<15} {'Traffic Mean':<15}")
    print("-" * 120)
    
    for _, row in regression_df.iterrows():
        ci_str = f"[{row['slope_ci_lower']:.6f}, {row['slope_ci_upper']:.6f}]"
        
        print(f"{row['pollutant']:<8} {row['slope']:<12.6f} {row['slope_std_err']:<10.6f} "
              f"{ci_str:<25} {row['r_squared']:<8.4f} {row['p_value']:<10.4f} {row['significance']:<4} "
              f"{row['pollutant_mean']:<15.3f} {row['traffic_mean']:<15.1f}")
    
    print("\nSlope = Change in pollutant per additional vehicle per hour")
    print("SE = Standard Error of the slope")
    print("95% CI = 95% Confidence Interval for the slope")
    print("R² = Proportion of variance explained by traffic")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("Pollutant Mean = Average concentration in μg m⁻³")
    print("Traffic Mean = Average vehicles per hour")
    print("="*120)
    
    # Summary of significant regressions
    significant_regressions = regression_df[regression_df['p_value'] < 0.05]
    print(f"\nSignificant regressions (p < 0.05): {len(significant_regressions)} out of {len(regression_df)}")
    
    if len(significant_regressions) > 0:
        print("Significant regressions with effect sizes:")
        for _, row in significant_regressions.iterrows():
            ci_str = f"[{row['slope_ci_lower']:.6f}, {row['slope_ci_upper']:.6f}]"
            direction = "increase" if row['slope'] > 0 else "decrease"
            print(f"  {row['pollutant']}: {row['slope']:.6f} ± {row['slope_std_err']:.6f} per vehicle/h")
            print(f"    - 95% CI: {ci_str}")
            print(f"    - For each additional vehicle/h, {row['pollutant']} {direction}s by {abs(row['slope']):.6f} μg m⁻³")
            print(f"    - R² = {row['r_squared']:.4f} ({row['r_squared']*100:.1f}% of variance explained)")
            print(f"    - p = {row['p_value']:.4f} {row['significance']}")
    
    # Practical interpretation
    print(f"\nPractical Interpretation:")
    print(f"  Traffic levels: Mean = {regression_df['traffic_mean'].iloc[0]:.1f} vehicles/h")
    print(f"  Slopes show the change in pollutant concentration per additional vehicle per hour")
    print(f"  Positive slopes: pollutant increases with traffic")
    print(f"  Negative slopes: pollutant decreases with traffic (may indicate dilution or other factors)")
    
    return regression_df

def perform_autocorrelation_analysis(merged_df, station_number):
    """
    Perform autocorrelation analysis on total traffic and all pollutants
    to identify temporal patterns and seasonality
    """
    print("\n" + "="*80)
    print("AUTOCORRELATION ANALYSIS: TEMPORAL PATTERNS")
    print("="*80)
    
    # Sort by datetime to ensure proper time series order
    df_sorted = merged_df.sort_values('datetime_utc').reset_index(drop=True)
    
    # Variables to analyze
    variables = ['total_traffic'] + ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    # Colors for each variable
    colors = {
        'total_traffic': 'black',
        'aqi': 'green',
        'co': 'black',
        'pm2_5': 'grey',
        'pm10': '#404040',
        'no': '#D2B48C',
        'no2': '#8B4513',
        'nh3': 'purple',
        'o3': '#00008B',
        'so2': '#B8860B'
    }
    
    # Titles for each variable
    titles = {
        'total_traffic': 'Total Traffic (vehicles/h)',
        'aqi': 'Air Quality Index (AQI)',
        'co': 'Carbon Monoxide (CO) / μg m⁻³',
        'no': 'Nitric Oxide (NO) / μg m⁻³',
        'no2': 'Nitrogen Dioxide (NO₂) / μg m⁻³',
        'o3': 'Ozone (O₃) / μg m⁻³',
        'so2': 'Sulfur Dioxide (SO₂) / μg m⁻³',
        'pm2_5': 'Fine Particulate Matter (PM2.5) / μg m⁻³',
        'pm10': 'Coarse Particulate Matter (PM10) / μg m⁻³',
        'nh3': 'Ammonia (NH₃) / μg m⁻³'
    }
    
    # Calculate maximum lag (up to 240 hours = 10 days)
    max_lag = min(240, len(df_sorted) // 4)  # Use 1/4 of data length or 10 days, whichever is smaller
    
    print(f"Performing autocorrelation analysis with max lag: {max_lag} hours ({max_lag/24:.1f} days)")
    
    # Create subplots for autocorrelation analysis - use GridSpec for custom layout
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.5, 1, 1, 1])
    
    # Total Traffic gets the top 3 slots (spanning all columns)
    ax_traffic = fig.add_subplot(gs[0, :])
    
    # Other variables get individual slots starting from row 1
    axes = []
    for i in range(1, 4):  # Rows 1, 2, 3
        for j in range(3):  # Columns 0, 1, 2
            axes.append(fig.add_subplot(gs[i, j]))
    
    autocorr_results = {}
    
    # Process Total Traffic first (special handling)
    variable = 'total_traffic'
    if variable in df_sorted.columns:
        ax = ax_traffic
        
        # Get clean data (remove NaN values)
        data = df_sorted[variable].dropna()
        
        if len(data) > max_lag + 10:  # Need sufficient data for autocorrelation
            try:
                # Calculate autocorrelation function - handle return values properly
                acf_result = acf(data, nlags=max_lag, alpha=0.05, fft=True)
                
                # Handle different return formats from acf function
                if isinstance(acf_result, tuple):
                    if len(acf_result) == 2:
                        acf_values, confint = acf_result
                    elif len(acf_result) == 3:
                        acf_values, confint, _ = acf_result
                    elif len(acf_result) == 4:
                        acf_values, confint, _, _ = acf_result
                    else:
                        acf_values = acf_result[0]
                        confint = None
                else:
                    acf_values = acf_result
                    confint = None
                
                # Create lag array
                lags = np.arange(0, max_lag + 1)
                
                # Plot autocorrelation
                ax.plot(lags, acf_values, color=colors[variable], linewidth=3, label=variable)
                
                # Add confidence intervals (95%) if available
                if confint is not None:
                    ax.fill_between(lags, confint[:, 0] - acf_values, confint[:, 1] - acf_values, 
                                   alpha=0.3, color=colors[variable])
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Add significance threshold lines
                ax.axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.7, label='95% CI')
                ax.axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.7)
                
                # Set labels and title
                ax.set_xlabel('Lag (hours)')
                ax.set_ylabel('Autocorrelation')
                ax.set_title(f'{titles[variable]} - Autocorrelation Analysis', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set x-axis ticks every 24 hours
                ax.set_xticks(range(0, max_lag + 1, 24))
                ax.set_xticklabels([f'{i//24}d' for i in range(0, max_lag + 1, 24)])
                
                # Store results
                autocorr_results[variable] = {
                    'acf_values': acf_values,
                    'confint': confint,
                    'lags': lags,
                    'sample_size': len(data)
                }
                
                # Print significant lags
                significant_lags = []
                for lag in range(1, min(25, len(acf_values))):  # Check first 24 hours
                    if abs(acf_values[lag]) > 1.96/np.sqrt(len(data)):
                        significant_lags.append(lag)
                
                if significant_lags:
                    print(f"{variable}: Significant autocorrelation at lags {significant_lags} hours")
                else:
                    print(f"{variable}: No significant autocorrelation in first 24 hours")
                    
            except Exception as e:
                print(f"Error calculating autocorrelation for {variable}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'Insufficient data\n({len(data)} points)', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10)
            print(f"{variable}: Insufficient data for autocorrelation ({len(data)} points)")
    else:
        ax_traffic.text(0.5, 0.5, f'Variable not found\n{variable}', transform=ax_traffic.transAxes, 
                       ha='center', va='center', fontsize=10)
    
    # Process other variables
    other_variables = ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    for i, variable in enumerate(other_variables):
        if variable in df_sorted.columns:
            ax = axes[i]
            
            # Get clean data (remove NaN values)
            data = df_sorted[variable].dropna()
            
            if len(data) > max_lag + 10:  # Need sufficient data for autocorrelation
                try:
                    # Calculate autocorrelation function - handle return values properly
                    acf_result = acf(data, nlags=max_lag, alpha=0.05, fft=True)
                    
                    # Handle different return formats from acf function
                    if isinstance(acf_result, tuple):
                        if len(acf_result) == 2:
                            acf_values, confint = acf_result
                        elif len(acf_result) == 3:
                            acf_values, confint, _ = acf_result
                        elif len(acf_result) == 4:
                            acf_values, confint, _, _ = acf_result
                        else:
                            acf_values = acf_result[0]
                            confint = None
                    else:
                        acf_values = acf_result
                        confint = None
                    
                    # Create lag array
                    lags = np.arange(0, max_lag + 1)
                    
                    # Plot autocorrelation
                    ax.plot(lags, acf_values, color=colors[variable], linewidth=2, label=variable)
                    
                    # Add confidence intervals (95%) if available
                    if confint is not None:
                        ax.fill_between(lags, confint[:, 0] - acf_values, confint[:, 1] - acf_values, 
                                       alpha=0.3, color=colors[variable])
                    
                    # Add horizontal line at zero
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    
                    # Add significance threshold lines
                    ax.axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.7, label='95% CI')
                    ax.axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.7)
                    
                    # Set labels and title
                    ax.set_xlabel('Lag (hours)')
                    ax.set_ylabel('Autocorrelation')
                    ax.set_title(titles[variable], fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Set x-axis ticks every 24 hours
                    ax.set_xticks(range(0, max_lag + 1, 24))
                    ax.set_xticklabels([f'{i//24}d' for i in range(0, max_lag + 1, 24)])
                    
                    # Store results
                    autocorr_results[variable] = {
                        'acf_values': acf_values,
                        'confint': confint,
                        'lags': lags,
                        'sample_size': len(data)
                    }
                    
                    # Print significant lags
                    significant_lags = []
                    for lag in range(1, min(25, len(acf_values))):  # Check first 24 hours
                        if abs(acf_values[lag]) > 1.96/np.sqrt(len(data)):
                            significant_lags.append(lag)
                    
                    if significant_lags:
                        print(f"{variable}: Significant autocorrelation at lags {significant_lags} hours")
                    else:
                        print(f"{variable}: No significant autocorrelation in first 24 hours")
                        
                except Exception as e:
                    print(f"Error calculating autocorrelation for {variable}: {e}")
                    ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Insufficient data\n({len(data)} points)', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10)
                print(f"{variable}: Insufficient data for autocorrelation ({len(data)} points)")
        else:
            ax.text(0.5, 0.5, f'Variable not found\n{variable}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'Graphs/station_{station_number}_autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Autocorrelation plot saved to: Graphs/station_{station_number}_autocorrelation_analysis.png")
    
    # Print summary of autocorrelation patterns
    print(f"\nAutocorrelation Analysis Summary:")
    print(f"  Data points analyzed: {len(df_sorted)}")
    print(f"  Time span: {df_sorted['datetime_utc'].min()} to {df_sorted['datetime_utc'].max()}")
    print(f"  Max lag tested: {max_lag} hours ({max_lag/24:.1f} days)")
    
    # Identify common patterns
    print(f"\nCommon temporal patterns identified:")
    
    # Check for daily patterns (lag ~24 hours)
    daily_patterns = []
    for variable, results in autocorr_results.items():
        if 'acf_values' in results:
            # Check lag 24 (daily pattern)
            if 24 < len(results['acf_values']):
                daily_corr = results['acf_values'][24]
                if abs(daily_corr) > 1.96/np.sqrt(results['sample_size']):
                    daily_patterns.append((variable, daily_corr))
    
    if daily_patterns:
        print(f"  Daily patterns (24-hour lag):")
        for variable, corr in sorted(daily_patterns, key=lambda x: abs(x[1]), reverse=True):
            print(f"    {variable}: r = {corr:.3f}")
    
    # Check for short-term patterns (lags 1-6 hours)
    short_term_patterns = []
    for variable, results in autocorr_results.items():
        if 'acf_values' in results:
            # Average of lags 1-6 hours
            short_lags = results['acf_values'][1:7]
            avg_short_corr = np.mean(short_lags)
            if abs(avg_short_corr) > 1.96/np.sqrt(results['sample_size']):
                short_term_patterns.append((variable, avg_short_corr))
    
    if short_term_patterns:
        print(f"  Short-term patterns (1-6 hour average):")
        for variable, corr in sorted(short_term_patterns, key=lambda x: abs(x[1]), reverse=True):
            print(f"    {variable}: r = {corr:.3f}")
    
    print("="*80)
    
    return autocorr_results

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
        
        # Perform linear regression analysis
        regression_df = perform_linear_regression_analysis(merged_df)
        
        # Save regression results to CSV
        regression_df.to_csv(f'Graphs/station_{station_number}_regression_results.csv', index=False)
        print(f"Regression results saved to: Graphs/station_{station_number}_regression_results.csv")
        
        # Perform autocorrelation analysis
        autocorr_results = perform_autocorrelation_analysis(merged_df, station_number)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # You can change the station number here
    station_number = 2002  # Default station
    main(station_number) 