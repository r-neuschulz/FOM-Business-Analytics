import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def count_files_by_year_and_type():
    """
    Count files by year and type for both OWM and BASt data from 2020 onwards.
    Returns a dictionary with counts for each year and type.
    """
    # Initialize counters
    file_counts = defaultdict(lambda: {'owm': 0, 'bast_csv': 0, 'bast_not_zip': 0})
    
    # Get current year for range
    current_year = 2024
    
    # Check OWM files
    owm_dir = "owm Hourly Data"
    if os.path.exists(owm_dir):
        for filename in os.listdir(owm_dir):
            if filename.startswith("owm") and filename.endswith(".csv"):
                # Extract year from filename (owm{station}_{year}.csv)
                match = re.match(r'owm\d+_(\d{4})\.csv', filename)
                if match:
                    year = int(match.group(1))
                    if year >= 2020:
                        file_counts[year]['owm'] += 1
    
    # Check BASt files
    bast_dir = "BASt Hourly Data"
    if os.path.exists(bast_dir):
        for filename in os.listdir(bast_dir):
            if filename.startswith("zst") and filename.endswith(".csv"):
                # Extract year from filename (zst{station}_{year}.csv)
                match = re.match(r'zst\d+_(\d{4})\.csv', filename)
                if match:
                    year = int(match.group(1))
                    if year >= 2020:
                        file_counts[year]['bast_csv'] += 1
            
            elif filename.startswith("zst") and filename.endswith("_not_zip"):
                # Extract year from filename (zst{station}_{year}_not_zip)
                match = re.match(r'zst\d+_(\d{4})_not_zip', filename)
                if match:
                    year = int(match.group(1))
                    if year >= 2020:
                        file_counts[year]['bast_not_zip'] += 1
    
    return file_counts

def create_download_quality_plot(file_counts):
    """
    Create a bar plot showing download quality by year.
    """
    if not file_counts:
        print("No files found for analysis.")
        return
    
    # Prepare data for plotting
    years = sorted(file_counts.keys())
    owm_counts = [file_counts[year]['owm'] for year in years]
    bast_csv_counts = [file_counts[year]['bast_csv'] for year in years]
    bast_not_zip_counts = [file_counts[year]['bast_not_zip'] for year in years]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars and positions of the bars
    bar_width = 0.25
    r1 = np.arange(len(years))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    bars1 = ax.bar(r1, owm_counts, bar_width, label='OWM Files', color='skyblue', alpha=0.8)
    bars2 = ax.bar(r2, bast_csv_counts, bar_width, label='BASt CSV Files', color='lightgreen', alpha=0.8)
    bars3 = ax.bar(r3, bast_not_zip_counts, bar_width, label='BASt Not-Zip Files', color='lightcoral', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Files', fontsize=12)
    ax.set_title('Download Quality Analysis: File Availability by Year (2020+)', fontsize=14, fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(years))])
    ax.set_xticklabels(years)
    ax.legend()
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = "Graphs/download_quality_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Download quality analysis plot saved to: {output_path}")
    
    return fig

def print_summary_statistics(file_counts):
    """
    Print summary statistics of the file counts.
    """
    if not file_counts:
        print("No files found for analysis.")
        return
    
    print("\n" + "="*60)
    print("DOWNLOAD QUALITY ANALYSIS SUMMARY")
    print("="*60)
    
    total_owm = sum(file_counts[year]['owm'] for year in file_counts)
    total_bast_csv = sum(file_counts[year]['bast_csv'] for year in file_counts)
    total_bast_not_zip = sum(file_counts[year]['bast_not_zip'] for year in file_counts)
    
    print(f"Total OWM files (2020+): {total_owm}")
    print(f"Total BASt CSV files (2020+): {total_bast_csv}")
    print(f"Total BASt Not-Zip files (2020+): {total_bast_not_zip}")
    print(f"Total files analyzed: {total_owm + total_bast_csv + total_bast_not_zip}")
    
    print("\nYear-by-Year Breakdown:")
    print("-" * 40)
    print(f"{'Year':<6} {'OWM':<8} {'BASt CSV':<10} {'BASt Not-Zip':<15}")
    print("-" * 40)
    
    for year in sorted(file_counts.keys()):
        counts = file_counts[year]
        print(f"{year:<6} {counts['owm']:<8} {counts['bast_csv']:<10} {counts['bast_not_zip']:<15}")

def main():
    """
    Main function to run the download quality analysis.
    """
    print("Starting Download Quality Analysis...")
    print("Checking files from 2020 onwards...")
    
    # Count files
    file_counts = count_files_by_year_and_type()
    
    if not file_counts:
        print("No files found for analysis. Please check if the data directories exist.")
        return
    
    # Print summary statistics
    print_summary_statistics(file_counts)
    
    # Create and save the plot
    create_download_quality_plot(file_counts)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
