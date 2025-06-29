#!/usr/bin/env python3
"""
06_FormatForOpenWeather.py

This script processes BASt hourly data CSV files to convert German time zone
dates and hours to Unix timestamps for OpenWeather API compatibility.

The script:
1. Reads all zst*.csv files in the BASt Hourly Data directory
2. Extracts station number and year from filename (e.g., zst1101_2003.csv)
3. Adds latitude and longitude from bast_stations_by_city.csv
4. Converts "Datum" (YYMMDD) and "Stunde" (HH) to Unix timestamps
5. Creates UnixStart (hour beginning - 1 hour) and UnixEnd (hour ending - 1 ms)
6. Handles German time zone (CET/CEST) conversion to UTC
7. Adds or overwrites UnixStart, UnixEnd, latitude, and longitude columns

"""

import os
import pandas as pd
import glob
import re
from datetime import datetime, timedelta
import pytz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

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
    # Pad datum to 6 digits
    datum = datum.zfill(6)
    try:
        year = 2000 + int(datum[:2])  # Convert YY to YYYY
        month = int(datum[2:4])
        day = int(datum[4:6])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid date format '{datum}': {e}")
    
    # Parse hour
    try:
        hour = int(stunde)
    except ValueError:
        raise ValueError(f"Invalid hour format '{stunde}': Hour must be numeric")
    
    # Handle hour 24 (convert to hour 0 of next day)
    if hour == 24:
        hour = 0
        # Add one day to the date
        temp_dt = datetime(year, month, day)
        next_day = temp_dt + timedelta(days=1)
        year, month, day = next_day.year, next_day.month, next_day.day
    elif hour < 0 or hour > 23:
        raise ValueError(f"Invalid hour format '{stunde}': Hour must be in 0..23, got {hour}")
    
    # Create datetime object in German timezone
    try:
        dt = datetime(year, month, day, hour, 0, 0, tzinfo=pytz.timezone('Europe/Berlin'))
        return dt
    except Exception as e:
        print(f"  [DEBUG] Failed to create datetime with year={year}, month={month}, day={day}, hour={hour}, datum='{datum}', stunde='{stunde}'")
        raise ValueError(f"Error creating datetime object: {e}")

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

def extract_station_info_from_filename(filename):
    """
    Extract station number and year from filename.
    
    Args:
        filename (str): Filename like "zst1101_2003.csv"
    
    Returns:
        tuple: (station_number, year) or (None, None) if parsing fails
    """
    # Pattern: zst{station_number}_{year}.csv
    pattern = r'zst(\d+)_(\d{4})\.csv'
    match = re.match(pattern, filename)
    
    if match:
        station_number = int(match.group(1))
        year = int(match.group(2))
        return station_number, year
    else:
        return None, None

def load_stations_data():
    """
    Load the stations data from bast_stations_by_city.csv.
    
    Returns:
        dict: Dictionary with (year, station_number) as key and (lat, lon) as value
    """
    stations_file = "BASt Hourly Data/bast_stations_by_city.csv"
    
    try:
        df = pd.read_csv(stations_file)
        stations_dict = {}
        
        for _, row in df.iterrows():
            key = (row['year'], row['station_number'])
            stations_dict[key] = (row['latitude'], row['longitude'])
        
        print(f"Loaded {len(stations_dict)} station entries")
        return stations_dict
    
    except Exception as e:
        print(f"Error loading stations data: {e}")
        return {}

def process_csv_file(args):
    """
    Process a single CSV file to add Unix timestamps and location data.
    
    Args:
        args (tuple): (file_path, stations_dict, max_rows) tuple
    
    Returns:
        dict: Result dictionary with status and details
    """
    file_path, stations_dict, max_rows = args
    
    try:
        filename = os.path.basename(file_path)
        
        # Extract station info from filename
        station_number, year = extract_station_info_from_filename(filename)
        if station_number is None or year is None:
            return {
                'status': 'error',
                'file': filename,
                'message': f"Could not parse station number and year from filename"
            }
        
        # Get station coordinates
        station_key = (year, station_number)
        if station_key not in stations_dict:
            return {
                'status': 'skipped',
                'file': filename,
                'message': f"No station data found for year {year}, station {station_number}"
            }
        
        latitude, longitude = stations_dict[station_key]
        
        # Read CSV file
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        n_rows = len(df)
        
        # Check if required columns exist
        if 'Datum' not in df.columns or 'Stunde' not in df.columns:
            return {
                'status': 'error',
                'file': filename,
                'message': f"Missing required columns 'Datum' or 'Stunde'"
            }
        
        # Remove existing columns if they exist
        columns_to_remove = ['UnixStart', 'UnixEnd', 'latitude', 'longitude']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Prepare new columns as pandas Series with appropriate dtypes
        unix_starts = pd.Series([pd.NA] * n_rows, dtype='Int64')
        unix_ends = pd.Series([pd.NA] * n_rows, dtype='Int64')
        latitudes = pd.Series([latitude] * n_rows, dtype='float64')
        longitudes = pd.Series([longitude] * n_rows, dtype='float64')
        
        # Only process the first max_rows rows (if specified)
        process_limit = max_rows if max_rows is not None else n_rows
        processed_count = 0
        error_count = 0
        suppressed_count = 0
        
        for idx in range(process_limit):
            try:
                row = df.iloc[idx]
                dt = parse_german_date_time(str(row['Datum']), str(row['Stunde']))
                unix_start, unix_end = convert_to_unix_timestamps(dt)
                unix_starts.iloc[idx] = unix_start
                unix_ends.iloc[idx] = unix_end
                processed_count += 1
            except ValueError as e:
                error_msg = str(e)
                if "suppressed/missing" in error_msg:
                    suppressed_count += 1
                else:
                    error_count += 1
                    return {
                        'status': 'error',
                        'file': filename,
                        'message': f"Error at row {idx + 1}: {error_msg}",
                        'details': f"Datum: '{row['Datum']}', Stunde: '{row['Stunde']}'"
                    }
            except Exception as e:
                error_count += 1
                return {
                    'status': 'error',
                    'file': filename,
                    'message': f"Unexpected error at row {idx + 1}: {e}",
                    'details': f"Datum: '{row['Datum']}', Stunde: '{row['Stunde']}'"
                }
        
        # Add new columns to DataFrame
        df['UnixStart'] = unix_starts
        df['UnixEnd'] = unix_ends
        df['latitude'] = latitudes
        df['longitude'] = longitudes
        
        # Save the modified file (all rows preserved)
        df.to_csv(file_path, sep=';', index=False, encoding='utf-8')
        
        return {
            'status': 'success',
            'file': filename,
            'processed_count': processed_count,
            'suppressed_count': suppressed_count,
            'error_count': error_count,
            'total_rows': n_rows,
            'station': station_number,
            'year': year
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file': os.path.basename(file_path),
            'message': f"Processing error: {e}"
        }

def main():
    """
    Main function to process all BASt CSV files with parallel processing.
    """
    print("BASt Hourly Data to Unix Timestamp Converter with Location Data")
    print("=" * 60)
    
    # Load stations data
    print("Loading stations data...")
    stations_dict = load_stations_data()
    if not stations_dict:
        print("Failed to load stations data. Exiting.")
        return
    
    # Find all zst*.csv files in the BASt Hourly Data directory
    data_dir = "BASt Hourly Data"
    pattern = os.path.join(data_dir, "zst*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No zst*.csv files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Determine number of processes to use
    num_processes = min(cpu_count(), len(csv_files), 8)  # Cap at 8 processes
    print(f"Using {num_processes} parallel processes")
    print()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, stations_dict, None) for file_path in sorted(csv_files)]
    
    # Process files with progress bar
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm for progress tracking
        results = list(tqdm(
            pool.imap(process_csv_file, args_list),
            total=len(csv_files),
            desc="Processing files",
            unit=" files"
        ))
    
    # Process results
    for result in results:
        if result['status'] == 'success':
            successful += 1
        elif result['status'] == 'error':
            failed += 1
            print(f"ERROR: {result['file']} - {result['message']}")
            if 'details' in result:
                print(f"   Details: {result['details']}")
        elif result['status'] == 'skipped':
            skipped += 1
            print(f"SKIPPED: {result['file']} - {result['message']}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print()
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"SUCCESS: {successful}")
    print(f"FAILED: {failed}")
    print(f"SKIPPED: {skipped}")
    print(f"TOTAL: {successful + failed + skipped}")
    print(f"PROCESSING TIME: {processing_time:.2f} seconds")
    print(f"AVERAGE TIME PER FILE: {processing_time/len(csv_files):.3f} seconds")
    
    if failed > 0:
        print(f"\nWARNING: {failed} files failed to process. Check the error messages above.")
    else:
        print(f"\nSUCCESS: All files processed successfully!")

if __name__ == "__main__":
    main() 