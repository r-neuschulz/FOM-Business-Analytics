import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import re
import argparse
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import pytz
import warnings
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')

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

def load_and_merge_data(station_number):
    # Load all years of traffic data for the station
    traffic_data_list = []
    air_data_list = []
    traffic_dir = "BASt Hourly Data"
    air_dir = "owm Hourly Data"
    traffic_files = [f for f in os.listdir(traffic_dir) if f.startswith(f'zst{station_number}_') and f.endswith('.csv')]
    air_files = [f for f in os.listdir(air_dir) if f.startswith(f'owm{station_number}_') and f.endswith('.csv')]
    for filename in sorted(traffic_files):
        file_path = os.path.join(traffic_dir, filename)
        try:
            df = pd.read_csv(file_path, sep=';')
            if not df.empty:
                traffic_data_list.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    for filename in sorted(air_files):
        file_path = os.path.join(air_dir, filename)
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                air_data_list.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    if not traffic_data_list or not air_data_list:
        raise ValueError(f"No data found for station {station_number}")
    traffic_df = pd.concat(traffic_data_list, ignore_index=True).drop_duplicates()
    air_df = pd.concat(air_data_list, ignore_index=True).drop_duplicates()
    # Process traffic data
    traffic_df['datetime'] = traffic_df.apply(
        lambda row: parse_german_date_time(row['Datum'], row['Stunde']), axis=1)
    traffic_df['datetime_utc'] = traffic_df['datetime'].dt.tz_convert('UTC')
    traffic_df['total_traffic'] = (
        traffic_df['KFZ_R1'].fillna(0) + 
        traffic_df['KFZ_R2'].fillna(0) + 
        traffic_df['Lkw_R1'].fillna(0) + 
        traffic_df['Lkw_R2'].fillna(0)
    )
    traffic_hourly = traffic_df.groupby('datetime_utc').agg({'total_traffic': 'sum'}).reset_index()
    # Process air quality data
    air_df['datetime_utc'] = pd.to_datetime(air_df['dt'], unit='s', utc=True)
    air_columns = ['datetime_utc', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    air_processed = air_df[air_columns].copy()
    # Merge datasets
    # Ensure datetime columns are pandas datetime objects
    traffic_hourly['datetime_utc'] = pd.to_datetime(traffic_hourly['datetime_utc'], utc=True)
    air_processed['datetime_utc'] = pd.to_datetime(air_processed['datetime_utc'], utc=True)
    
    # Create hourly floor timestamps for merging
    traffic_hourly['datetime_hour'] = traffic_hourly['datetime_utc'].dt.floor('H')
    air_processed['datetime_hour'] = air_processed['datetime_utc'].dt.floor('H')  # type: ignore
    
    merged_df = pd.merge(traffic_hourly, air_processed, on='datetime_hour', how='inner')
    merged_df = merged_df.drop('datetime_hour', axis=1)
    merged_df = merged_df.rename(columns={'datetime_utc_x': 'datetime_utc'})
    merged_df = merged_df.drop('datetime_utc_y', axis=1)
    # Remove rows with negative values in any numeric column except total_traffic
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'total_traffic':
            merged_df = merged_df[merged_df[col] >= 0]
    return merged_df

def deseason_data_comprehensive(data, datetime_col, value_col):
    """
    Remove both daily (hourly) and weekly seasonality from hourly data.
    
    With hourly granularity, this removes:
    - Daily seasonality: Hour-of-day patterns (rush hours, etc.)
    - Weekly seasonality: Day-of-week patterns (weekday vs weekend)
    
    Note: Cannot remove sub-hourly patterns since data is only hourly.
    
    Args:
        data: DataFrame containing the data
        datetime_col: Name of the datetime column
        value_col: Name of the value column to deseason
    
    Returns:
        DataFrame with deseasoned values added as '{value_col}_deseasoned'
    """
    # Ensure datetime_col is a pandas datetime
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        data = data.copy()
        data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Create hour and day of week columns
    data = data.copy()
    data['hour'] = data[datetime_col].dt.hour
    data['day_of_week'] = data[datetime_col].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Calculate hourly means (daily seasonality - e.g., rush hour patterns)
    hourly_means = data.groupby('hour')[value_col].mean()
    
    # Calculate day-of-week means (weekly seasonality - e.g., weekday vs weekend)
    dow_means = data.groupby('day_of_week')[value_col].mean()
    
    # Calculate overall mean
    overall_mean = data[value_col].mean()
    
    # Remove both daily and weekly seasonality
    # Formula: Deseasoned = Original - Hourly_Mean - DayOfWeek_Mean + Overall_Mean
    deseasoned = data.copy()
    deseasoned[f'{value_col}_deseasoned'] = (
        data[value_col] - 
        data['hour'].map(hourly_means) - 
        data['day_of_week'].map(dow_means) + 
        overall_mean
    )
    
    return deseasoned

def deseason_data(data, group_col, value_col):
    """
    Remove seasonal (daily) patterns from the data by subtracting the mean for each hour.
    """
    # Calculate hourly means
    hourly_means = data.groupby(group_col)[value_col].mean()
    
    # Subtract hourly means from each observation
    deseasoned = data.copy()
    deseasoned[f'{value_col}_deseasoned'] = data[value_col] - data[group_col].map(hourly_means)
    
    return deseasoned

def create_heatmap(ax, corr_matrix, title, cmap='RdBu_r'):
    """
    Create a heatmap using matplotlib only.
    """
    im = ax.imshow(corr_matrix.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.index)
    
    # Add text annotations
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    return im

def print_deseasoned_traffic_correlations(merged_df_clean, pollutant_cols):
    """
    Print Pearson correlation, standard error, 95% CI, p-value, and significance for each pollutant vs total_traffic_deseasoned.
    """
    print("\n" + "="*90)
    print("PEARSON CORRELATION ANALYSIS: Deseasoned Traffic vs Pollutants")
    print("="*90)
    print(f"{'Pollutant':<10} {'r':<8} {'SE':<8} {'95% CI':<20} {'p-value':<10} {'N':<6} {'Sig':<4}")
    print("-" * 90)
    for pollutant in pollutant_cols:
        x = merged_df_clean['total_traffic_deseasoned']
        y = merged_df_clean[f'{pollutant}_deseasoned']
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)
        if n > 2:
            corr = x_clean.corr(y_clean)
            if not np.isnan(corr) and abs(corr) < 1.0:
                t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                z_corr = np.arctanh(corr)
                z_se = 1 / np.sqrt(n - 3)
                z_ci_lower = z_corr - 1.96 * z_se
                z_ci_upper = z_corr + 1.96 * z_se
                ci_lower = np.tanh(z_ci_lower)
                ci_upper = np.tanh(z_ci_upper)
                corr_se = np.sqrt((1 - corr**2) / (n - 2))
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                print(f"{pollutant:<10} {corr:<8.3f} {corr_se:<8.3f} {ci_str:<20} {p_value:<10.4f} {n:<6} {significance:<4}")
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("SE = Standard Error, CI = Confidence Interval")
    print("="*90)

def create_deseasoned_correlation_analysis(merged_df, station, year):
    """
    Create deseasoned correlation analysis between traffic and pollution data.
    """
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    pollutant_labels = {
        'co': 'CO',
        'no': 'NO',
        'no2': 'NO₂',
        'o3': 'O₃',
        'so2': 'SO₂',
        'pm2_5': 'PM2.5',
        'pm10': 'PM10',
        'nh3': 'NH₃',
    }
    if len(merged_df) == 0:
        print("No overlapping data found between traffic and air quality data")
        return None, None, None
    print(f"Merged dataset contains {len(merged_df)} records")
    # Calculate total traffic
    merged_df['total_traffic'] = merged_df['total_traffic']
    # Deseason the data
    print("Deseasoning hourly data (removing daily and weekly patterns)...")
    print("  - Daily: Removing hour-of-day patterns (rush hours, etc.)")
    print("  - Weekly: Removing day-of-week patterns (weekday vs weekend)")
    print("  - Note: Cannot remove sub-hourly patterns (data is hourly only)")
    # Deseason traffic data using comprehensive deseasoning
    merged_df = deseason_data_comprehensive(merged_df, 'datetime_utc', 'total_traffic')
    # Deseason pollutant data using comprehensive deseasoning
    for pollutant in pollutant_cols:
        if pollutant in merged_df.columns:
            merged_df = deseason_data_comprehensive(merged_df, 'datetime_utc', pollutant)
    # Remove rows with NaN values
    deseasoned_cols = ['total_traffic_deseasoned'] + [f'{pol}_deseasoned' for pol in pollutant_cols if pol in merged_df.columns]
    merged_df_clean = merged_df.dropna(subset=deseasoned_cols)
    print(f"Clean dataset contains {len(merged_df_clean)} records after removing NaN values")
    # Print detailed correlation summary (only vs total_traffic_deseasoned)
    print_deseasoned_traffic_correlations(merged_df_clean, [pol for pol in pollutant_cols if pol in merged_df.columns])
    return None, None, None

def perform_deseasoned_correlation_analysis(merged_df_clean):
    """
    Perform comprehensive correlation analysis between deseasoned traffic and air pollutants
    with adjusted thresholds for large sample sizes (N=26,000)
    """
    print("Performing deseasoned correlation analysis...")
    print(f"Sample size: N = {len(merged_df_clean):,}")
    print("Note: With large sample sizes, focus on effect size (correlation magnitude) over p-values")
    print("Adjusted thresholds: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    print("Effect size: Strong |r|≥0.7, Moderate 0.3≤|r|<0.7, Weak 0.1≤|r|<0.3, Negligible |r|<0.1")
    
    # Focus on traffic vs pollutants correlation (deseasoned data)
    pollutant_columns = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    # Calculate correlation between deseasoned traffic and each pollutant
    correlation_data = []
    
    for pollutant in pollutant_columns:
        if pollutant in merged_df_clean.columns:
            # Remove NaN values for this specific pollutant
            mask = ~(merged_df_clean['total_traffic_deseasoned'].isna() | 
                    merged_df_clean[f'{pollutant}_deseasoned'].isna())
            traffic_clean = merged_df_clean['total_traffic_deseasoned'][mask]
            pollutant_clean = merged_df_clean[f'{pollutant}_deseasoned'][mask]
            
            if len(traffic_clean) > 2:  # Need at least 3 points for correlation
                try:
                    # Calculate correlation using pandas
                    corr = traffic_clean.corr(pollutant_clean)
                    
                    # Calculate p-value and confidence intervals using scipy
                    if not np.isnan(corr) and abs(corr) < 1.0:
                        # Calculate t-statistic for p-value
                        n = len(traffic_clean)
                        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        
                        # Calculate 95% confidence interval using Fisher's z-transformation
                        z_corr = np.arctanh(corr)  # Fisher's z-transformation
                        z_se = 1 / np.sqrt(n - 3)  # Standard error of z
                        z_ci_lower = z_corr - 1.96 * z_se  # 95% CI lower bound
                        z_ci_upper = z_corr + 1.96 * z_se  # 95% CI upper bound
                        
                        # Transform back to correlation scale
                        ci_lower = np.tanh(z_ci_lower)
                        ci_upper = np.tanh(z_ci_upper)
                        
                        # Calculate standard error of correlation
                        corr_se = np.sqrt((1 - corr**2) / (n - 2))
                        
                        # Determine significance using adjusted thresholds for large N
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"
                        
                        # Determine effect size classification
                        if abs(corr) >= 0.7:
                            effect_size = "Strong"
                        elif abs(corr) >= 0.3:
                            effect_size = "Moderate"
                        elif abs(corr) >= 0.1:
                            effect_size = "Weak"
                        else:
                            effect_size = "Negligible"
                        
                        # Determine practical significance (minimum meaningful correlation)
                        practical_significance = abs(corr) >= 0.05
                        
                        correlation_data.append({
                            'pollutant': pollutant,
                            'correlation': corr,
                            'p_value': p_value,
                            'std_error': corr_se,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'sample_size': n,
                            'significance': significance,
                            'effect_size': effect_size,
                            'practical_significance': practical_significance
                        })
                except (ValueError, RuntimeWarning):
                    # Skip if correlation calculation fails
                    continue
    
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df = correlation_df.sort_values('correlation', ascending=False)
    
    # Print results using standard format with effect size
    print("\n" + "="*100)
    print("DESEASONED PEARSON CORRELATION ANALYSIS: TRAFFIC VS POLLUTANTS (N=26,000)")
    print("="*100)
    print(f"{'Pollutant':<8} {'r':<8} {'SE':<8} {'95% CI':<20} {'p-value':<10} {'Effect':<10} {'Practical':<10}")
    print("-" * 100)
    
    for _, row in correlation_df.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        practical = "Yes" if bool(row['practical_significance']) else "No"
        print(f"{row['pollutant']:<8} {row['correlation']:<8.3f} {row['std_error']:<8.3f} "
              f"{ci_str:<20} {row['p_value']:<10.4f} {row['effect_size']:<10} {practical:<10}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    print("Effect Size: Strong |r|≥0.7, Moderate 0.3≤|r|<0.7, Weak 0.1≤|r|<0.3, Negligible |r|<0.1")
    print("Practical: Minimum meaningful correlation |r|≥0.05 for N=26,000")
    print("Seasonality removed: Daily (hourly) + Weekly (day-of-week) patterns")
    print("="*100)
    
    # Summary of significant correlations
    significant_correlations = correlation_df[correlation_df['p_value'] < 0.05]
    practical_correlations = correlation_df[correlation_df['practical_significance'].astype(bool)]
    
    print(f"\nStatistical significance (p < 0.05): {len(significant_correlations)} out of {len(correlation_df)}")
    print(f"Practical significance (|r| ≥ 0.05): {len(practical_correlations)} out of {len(correlation_df)}")
    
    if not practical_correlations.empty:
        print("\nPractically significant correlations (|r| ≥ 0.05):")
        for _, row in practical_correlations.iterrows():
            ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            direction = "positive" if row['correlation'] > 0 else "negative"
            print(f"  {row['pollutant']}: {row['effect_size']} {direction} correlation (r = {row['correlation']:.3f}, p = {row['p_value']:.4f})")
            print(f"    - 95% CI: {ci_str}")
            print(f"    - Standard error: ±{row['std_error']:.3f}")
    
    # Effect size distribution
    effect_counts = correlation_df['effect_size'].value_counts()
    print(f"\nEffect size distribution:")
    for effect, count in effect_counts.items():
        print(f"  {effect}: {count} correlations")
    
    # Null hypothesis rejection summary
    print(f"\nNull hypothesis rejection summary:")
    print(f"  Total correlations tested: {len(correlation_df)}")
    print(f"  Null hypothesis rejected (p < 0.05): {len(significant_correlations)}")
    print(f"  Null hypothesis not rejected (p ≥ 0.05): {len(correlation_df) - len(significant_correlations)}")
    print(f"  Practically meaningful (|r| ≥ 0.05): {len(practical_correlations)}")
    
    if not practical_correlations.empty:
        print(f"\nKey findings for N=26,000:")
        for _, row in practical_correlations.iterrows():
            direction = "positive" if row['correlation'] > 0 else "negative"
            strength = str(row['effect_size']).lower()
            print(f"  {row['pollutant']}: {strength} {direction} correlation (r = {row['correlation']:.3f})")
            print(f"    - 95% CI does not include zero: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
            print(f"    - Standard error: ±{row['std_error']:.3f}")
            print(f"    - Sample size: n = {row['sample_size']:,}")
    
    print("\n" + "="*100)
    
    # Also calculate full correlation matrix for reference
    full_correlation_columns = ['total_traffic_deseasoned'] + [f'{p}_deseasoned' for p in pollutant_columns if p in merged_df_clean.columns]
    correlation_df_full = merged_df_clean[full_correlation_columns].corr()
    
    return correlation_df_full, correlation_df, merged_df_clean

def create_deseasoned_correlation_visualization(correlation_matrix, station_number):
    """
    Create correlation matrix visualization for deseasoned data
    """
    print("Creating deseasoned correlation visualization...")
    
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
    
    plt.title(f'Deseasoned Correlation Matrix: Traffic vs Air Pollutants (Station {station_number})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(f'Graphs/station_{station_number}_deseasoned_traffic_air_quality_correlation_matrix.png', 
                dpi=300, bbox_inches='tight')
    
    return correlation_matrix

def create_deseasoned_scatter_plots(merged_df_clean, station_number):
    """
    Create scatter plots of deseasoned traffic vs each pollutant with trend lines
    """
    print("Creating deseasoned scatter plots...")
    
    # Define pollutants in the requested 3x3 grid order
    pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    titles = ['Carbon Monoxide (CO)', 'Nitric Oxide (NO)', 'Nitrogen Dioxide (NO₂)', 
              'Ozone (O₃)', 'Sulfur Dioxide (SO₂)', 'Fine Particulate Matter (PM2.5)', 
              'Coarse Particulate Matter (PM10)', 'Ammonia (NH₃)']
    
    # Define colors for each pollutant
    colors = {
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
        if pollutant in merged_df_clean.columns:
            ax = axes[i]
            
            x = merged_df_clean['total_traffic_deseasoned']
            y = merged_df_clean[f'{pollutant}_deseasoned']
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                # Calculate correlation
                corr = np.corrcoef(x_clean, y_clean)[0, 1]
                r_squared = corr ** 2
                
                # Create scatter plot with specific colors
                ax.scatter(x_clean, y_clean, color=colors[pollutant], alpha=0.6, s=20)
                
                # Add trend line
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)
                
                # Add correlation coefficient and R²
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\nR²: {r_squared:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Deseasoned Traffic / vehicles h⁻¹')
                ax.set_ylabel(f'Deseasoned {title} / μg m⁻³')
                ax.set_title(f'Deseasoned {title} vs Traffic', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Graphs/station_{station_number}_deseasoned_traffic_vs_pollutants_scatter.png', dpi=300, bbox_inches='tight')
    print(f'Saved: Graphs/station_{station_number}_deseasoned_traffic_vs_pollutants_scatter.png')

def analyze_deseasoned_temporal_coverage(merged_df_clean, station_number):
    """
    Analyze the temporal coverage of the deseasoned dataset
    """
    print("\n" + "="*60)
    print("DESEASONED TEMPORAL COVERAGE ANALYSIS")
    print("="*60)
    
    # Extract year from datetime
    merged_df_clean['year'] = merged_df_clean['datetime_utc'].dt.year
    merged_df_clean['month'] = merged_df_clean['datetime_utc'].dt.month
    merged_df_clean['hour'] = merged_df_clean['datetime_utc'].dt.hour
    
    # Yearly coverage
    yearly_counts = merged_df_clean['year'].value_counts().sort_index()
    print(f"\nData coverage by year:")
    for year, count in yearly_counts.items():
        print(f"  {year}: {count:,} records")
    
    # Overall statistics
    print(f"\nOverall temporal statistics:")
    print(f"  Date range: {merged_df_clean['datetime_utc'].min()} to {merged_df_clean['datetime_utc'].max()}")
    print(f"  Total time span: {(merged_df_clean['datetime_utc'].max() - merged_df_clean['datetime_utc'].min()).days} days")
    print(f"  Total records: {len(merged_df_clean):,}")
    print(f"  Seasonality removed: Daily (hourly) + Weekly (day-of-week) patterns")

def create_deseasoned_timeseries_overview_plot(merged_df_clean, station_number):
    """
    Create a timeseries plot with deseasoned data:
    - Deseasoned Total Traffic (daily sum, bar, black)
    - Deseasoned CO (daily average, line, black)
    - All other deseasoned pollutants (daily average, line, scatter colors)
    """
    print("Creating deseasoned timeseries overview plot...")
    
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
    df = merged_df_clean.copy()
    df['date'] = df['datetime_utc'].dt.floor('D')
    
    # Deseasoned Total Traffic: daily sum
    traffic_daily = df.groupby('date')['total_traffic_deseasoned'].sum()
    
    # Deseasoned CO: daily mean
    co_daily = df.groupby('date')['co_deseasoned'].mean()
    
    # Other deseasoned pollutants: daily mean
    other_pollutants = ['no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    pollutant_daily = {p: df.groupby('date')[f'{p}_deseasoned'].mean() 
                      for p in other_pollutants if f'{p}_deseasoned' in df.columns}

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
        f"{'Lon:'.ljust(label_pad)} {lon if lon == 'N/A' else f'{lon:.5f}'}\n"
        f"{'Seasonality:'.ljust(label_pad)} Removed"
    )

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    
    # 1. Deseasoned Total Traffic
    axes[0].bar(traffic_daily.index, traffic_daily.values, color='black', width=1)
    axes[0].set_ylabel('Deseasoned Traffic\n(vehicles/day)')
    axes[0].set_title('Deseasoned Total Traffic (Daily Sum)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Deseasoned CO
    axes[1].plot(co_daily.index, co_daily.values, color=colors['co'], linewidth=1)
    axes[1].set_ylabel('Deseasoned CO / μg m⁻³')
    axes[1].set_title('Deseasoned CO (Daily Average)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Other deseasoned pollutants
    for p in pollutant_daily:
        axes[2].plot(pollutant_daily[p].index, pollutant_daily[p].values, 
                    label=p.upper() if p not in ['no2','so2','o3','nh3'] else p.replace('2','₂').replace('3','₃').upper(), 
                    color=colors[p], linewidth=1)
    axes[2].set_ylabel('Deseasoned Pollutant / μg m⁻³')
    axes[2].set_title('Other Deseasoned Pollutants (Daily Average)')
    axes[2].legend(loc='upper left', ncol=4, fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    # Add overall title to the figure
    fig.suptitle(f'Deseasoned Timeseries Overview Station {station_number} {city.title()}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add infobox with station info
    axes[0].text(
        0.01, 0.98, legend_text,
        transform=fig.transFigure,
        fontsize=10,
        va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    plt.savefig(f'Graphs/station_{station_number}_deseasoned_traffic_pollutants_timeseries_overview.png', dpi=300)
    print(f'Saved: Graphs/station_{station_number}_deseasoned_traffic_pollutants_timeseries_overview.png')

def print_deseasoned_correlation_summary(correlation_df):
    """
    Print summary of deseasoned correlations with traffic including statistical significance
    """
    print("\n" + "="*60)
    print("DESEASONED CORRELATION SUMMARY WITH TRAFFIC")
    print("="*60)
    
    # Sort by absolute correlation strength
    correlation_df['abs_corr'] = correlation_df['correlation'].abs()
    sorted_df = correlation_df.sort_values('abs_corr', ascending=False)
    
    print("\nDeseasoned correlations with Total Traffic (sorted by strength):")
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

def perform_deseasoned_linear_regression_analysis(merged_df_clean):
    """
    Perform linear regression analysis between deseasoned traffic and air pollutants to get effect sizes
    """
    print("\n" + "="*120)
    print("LINEAR REGRESSION ANALYSIS: DESEASONED TRAFFIC VS POLLUTANTS")
    print("="*120)
    
    # Focus on deseasoned traffic vs deseasoned pollutants regression
    pollutant_columns = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    # Calculate regression for each pollutant
    regression_data = []
    
    for pollutant in pollutant_columns:
        deseasoned_pollutant = f'{pollutant}_deseasoned'
        if deseasoned_pollutant in merged_df_clean.columns:
            # Remove NaN values for this specific pollutant
            mask = ~(merged_df_clean['total_traffic_deseasoned'].isna() | merged_df_clean[deseasoned_pollutant].isna())
            traffic_clean = merged_df_clean['total_traffic_deseasoned'][mask]
            pollutant_clean = merged_df_clean[deseasoned_pollutant][mask]
            
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
    
    print("\nSlope = Change in deseasoned pollutant per additional deseasoned vehicle per hour")
    print("SE = Standard Error of the slope")
    print("95% CI = 95% Confidence Interval for the slope")
    print("R² = Proportion of variance explained by deseasoned traffic")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("Pollutant Mean = Average deseasoned concentration in μg m⁻³")
    print("Traffic Mean = Average deseasoned vehicles per hour")
    print("="*120)
    
    # Summary of significant regressions
    significant_regressions = regression_df[regression_df['p_value'] < 0.05]
    print(f"\nSignificant regressions (p < 0.05): {len(significant_regressions)} out of {len(regression_df)}")
    
    if len(significant_regressions) > 0:
        print("Significant regressions with effect sizes:")
        for _, row in significant_regressions.iterrows():
            ci_str = f"[{row['slope_ci_lower']:.6f}, {row['slope_ci_upper']:.6f}]"
            direction = "increase" if row['slope'] > 0 else "decrease"
            print(f"  {row['pollutant']}: {row['slope']:.6f} ± {row['slope_std_err']:.6f} per deseasoned vehicle/h")
            print(f"    - 95% CI: {ci_str}")
            print(f"    - For each additional deseasoned vehicle/h, {row['pollutant']} {direction}s by {abs(row['slope']):.6f} μg m⁻³")
            print(f"    - R² = {row['r_squared']:.4f} ({row['r_squared']*100:.1f}% of variance explained)")
            print(f"    - p = {row['p_value']:.4f} {row['significance']}")
    
    # Practical interpretation
    print(f"\nPractical Interpretation:")
    print(f"  Deseasoned traffic levels: Mean = {regression_df['traffic_mean'].iloc[0]:.1f} vehicles/h")
    print(f"  Slopes show the change in deseasoned pollutant concentration per additional deseasoned vehicle per hour")
    print(f"  Positive slopes: deseasoned pollutant increases with deseasoned traffic")
    print(f"  Negative slopes: deseasoned pollutant decreases with deseasoned traffic (may indicate dilution or other factors)")
    print(f"  Note: Seasonal patterns have been removed from both traffic and pollutant data")
    
    return regression_df

def main():
    """
    Main function to run the enhanced deseasoned correlation analysis.
    """
    parser = argparse.ArgumentParser(description='Enhanced Deseasoned Correlation Analysis between Traffic and Pollution')
    parser.add_argument('--station', type=str, default='2002', 
                       help='Station number to analyze (default: 2002)')
    args = parser.parse_args()
    
    print("Starting Enhanced Deseasoned Correlation Analysis...")
    print(f"Target station: {args.station}")
    print("="*60)
    
    try:
        # Load and merge data
        merged_df = load_and_merge_data(args.station)
        print(f"Merged dataset contains {len(merged_df)} records")
    except Exception as e:
        print(f"Error loading or merging data: {e}")
        return
    
    # Ensure merged_df is a DataFrame
    if not isinstance(merged_df, pd.DataFrame) or merged_df.empty:
        print("No merged data available for analysis.")
        return
    
    print("Applying comprehensive deseasoning (removing daily and weekly patterns)...")
    print("  - Daily: Removing hour-of-day patterns (rush hours, etc.)")
    print("  - Weekly: Removing day-of-week patterns (weekday vs weekend)")
    print("  - Note: Cannot remove sub-hourly patterns (data is hourly only)")
    
    # Deseason traffic and pollutants using comprehensive deseasoning
    deseasoned_cols = ['total_traffic'] + [col for col in merged_df.columns if col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']]
    for col in deseasoned_cols:
        merged_df = deseason_data_comprehensive(merged_df, 'datetime_utc', col)
    
    # Prepare deseasoned columns for correlation
    deseasoned_corr_cols = [f'{col}_deseasoned' for col in deseasoned_cols]
    merged_df_clean = merged_df.dropna(subset=deseasoned_corr_cols)
    
    print(f"Clean deseasoned dataset contains {len(merged_df_clean)} records after removing NaN values")
    
    # Analyze temporal coverage
    analyze_deseasoned_temporal_coverage(merged_df_clean, args.station)
    
    # Perform comprehensive correlation analysis
    correlation_df_full, correlation_df, merged_df_clean = perform_deseasoned_correlation_analysis(merged_df_clean)
    
    # Create visualizations
    correlation_matrix = create_deseasoned_correlation_visualization(correlation_df_full, args.station)
    create_deseasoned_scatter_plots(merged_df_clean, args.station)
    create_deseasoned_timeseries_overview_plot(merged_df_clean, args.station)
    
    # Print summary
    print_deseasoned_correlation_summary(correlation_df)
    
    # Save results
    correlation_df_full.to_csv(f'Graphs/station_{args.station}_deseasoned_correlation_matrix.csv')
    print(f"\nDeseasoned correlation matrix saved to: Graphs/station_{args.station}_deseasoned_correlation_matrix.csv")
    
    merged_df_clean.to_csv(f'Graphs/station_{args.station}_deseasoned_merged_data.csv', index=False)
    print(f"Deseasoned merged dataset saved to: Graphs/station_{args.station}_deseasoned_merged_data.csv")
    
    # Print final summary
    print("\n=== ENHANCED DESEASONED CORRELATION ANALYSIS SUMMARY ===")
    print(f"Station: {args.station}")
    print(f"Total records analyzed: {len(merged_df_clean)}")
    print(f"Time period: {merged_df_clean['datetime_utc'].min()} to {merged_df_clean['datetime_utc'].max()}")
    print("Seasonality removed: Daily (hourly) patterns + Weekly (day-of-week) patterns")
    print("\nPearson correlation matrix (deseasoned data):")
    print(correlation_df_full)
    
    # Perform linear regression analysis
    regression_df = perform_deseasoned_linear_regression_analysis(merged_df_clean)
    
    # Save regression results to CSV
    regression_df.to_csv(f'Graphs/station_{args.station}_deseasoned_regression_results.csv', index=False)
    print(f"Deseasoned regression results saved to: Graphs/station_{args.station}_deseasoned_regression_results.csv")
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 