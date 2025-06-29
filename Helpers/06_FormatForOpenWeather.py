#!/usr/bin/env python3
"""
06_FormatForOpenWeather.py

This script processes BASt hourly data CSV files to convert German time zone
dates and hours to Unix timestamps for OpenWeather API compatibility.

The script:
1. Reads all zst*.csv files in the BASt Hourly Data directory
2. Converts "Datum" (YYMMDD) and "Stunde" (HH) to Unix timestamps
3. Creates UnixStart (hour beginning - 1 hour) and UnixEnd (hour ending - 1 ms)
4. Handles German time zone (CET/CEST) conversion to UTC
5. Adds or overwrites UnixStart and UnixEnd columns

"""

import os
import pandas as pd
import glob
from datetime import datetime, timedelta
import pytz

def parse_german_date_time(datum, stunde):
    """
    Parse German date and hour to datetime object.
    
    Args:
        datum (str): Date in YYMMDD format (e.g., "230101")
        stunde (str): Hour in HH format (e.g., "01")
    
    Returns:
        datetime: Parsed datetime object in German timezone
    """
    # Parse date (YYMMDD format)
    year = 2000 + int(datum[:2])  # Convert YY to YYYY
    month = int(datum[2:4])
    day = int(datum[4:6])
    
    # Parse hour
    hour = int(stunde)
    
    # Create datetime object (assume German timezone)
    german_tz = pytz.timezone('Europe/Berlin')
    dt = german_tz.localize(datetime(year, month, day, hour, 0, 0))
    
    return dt

def convert_to_unix_timestamps(dt):
    """
    Convert German datetime to Unix start and end timestamps.
    
    Args:
        dt (datetime): German timezone datetime object
    
    Returns:
        tuple: (unix_start, unix_end) in seconds since epoch
    """
    # Convert to UTC
    utc_dt = dt.astimezone(pytz.UTC)
    
    # UnixStart: beginning of the hour (hour - 1 hour)
    unix_start = int((utc_dt - timedelta(hours=1)).timestamp())
    
    # UnixEnd: end of the hour (hour - 1 millisecond)
    unix_end = int((utc_dt - timedelta(milliseconds=1)).timestamp())
    
    return unix_start, unix_end

def process_csv_file(file_path, max_rows=None):
    """
    Process a single CSV file to add Unix timestamps.
    
    Args:
        file_path (str): Path to the CSV file
        max_rows (int, optional): Maximum number of rows to process (for testing)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Read CSV file
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        n_rows = len(df)
        
        # Check if required columns exist
        if 'Datum' not in df.columns or 'Stunde' not in df.columns:
            print(f"  Warning: Missing required columns 'Datum' or 'Stunde' in {file_path}")
            return False
        
        # Remove existing Unix columns if they exist
        if 'UnixStart' in df.columns:
            df = df.drop('UnixStart', axis=1)
            print("  Removed existing UnixStart column")
        if 'UnixEnd' in df.columns:
            df = df.drop('UnixEnd', axis=1)
            print("  Removed existing UnixEnd column")
        
        # Prepare new columns as pandas Series with nullable integer dtype
        unix_starts = pd.Series([pd.NA] * n_rows, dtype='Int64')
        unix_ends = pd.Series([pd.NA] * n_rows, dtype='Int64')
        
        # Only process the first max_rows rows (if specified)
        process_limit = max_rows if max_rows is not None else n_rows
        for idx in range(process_limit):
            try:
                row = df.iloc[idx]
                dt = parse_german_date_time(str(row['Datum']), str(row['Stunde']))
                unix_start, unix_end = convert_to_unix_timestamps(dt)
                unix_starts.iloc[idx] = unix_start
                unix_ends.iloc[idx] = unix_end
            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
        
        # Add new columns to DataFrame
        df['UnixStart'] = unix_starts
        df['UnixEnd'] = unix_ends
        
        # Save the modified file (all rows preserved)
        df.to_csv(file_path, sep=';', index=False, encoding='utf-8')
        
        print(f"  Successfully updated UnixStart and UnixEnd for first {process_limit} rows (total rows preserved: {n_rows})")
        return True
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False

def main():
    """
    Main function to process all BASt CSV files.
    """
    print("BASt Hourly Data to Unix Timestamp Converter")
    print("=" * 50)
    
    # Find all zst*.csv files in the BASt Hourly Data directory
    data_dir = "BASt Hourly Data"
    pattern = os.path.join(data_dir, "zst*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No zst*.csv files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print()
    
    # Process files
    successful = 0
    failed = 0
    
    for file_path in sorted(csv_files):
        # For testing, only process the first file (zst1104_2023.csv) with 5 rows
        if "zst1104_2023.csv" in file_path:
            success = process_csv_file(file_path, max_rows=5)
        else:
            # Skip other files for now (as requested)
            print(f"Skipping: {os.path.basename(file_path)} (not in test scope)")
            continue
            
        if success:
            successful += 1
        else:
            failed += 1
        
        print()
    
    print("Processing Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {successful + failed}")

if __name__ == "__main__":
    main() 