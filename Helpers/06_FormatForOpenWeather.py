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
import requests
import json

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
    datum = str(datum).zfill(6)
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
    utc_dt = dt.astimezone(pytz.UTC)
    return int((utc_dt - timedelta(hours=1)).timestamp())

def convert_to_unix_end(dt):
    utc_dt = dt.astimezone(pytz.UTC)
    return int((utc_dt - timedelta(seconds=1)).timestamp())

def get_openweather_air_pollution_data(lat, lon, start_time, end_time, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        'lat': lat,
        'lon': lon,
        'start': start_time,
        'end': end_time,
        'appid': api_key
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()

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
                unix_start, unix_end = convert_to_unix_start(dt), convert_to_unix_end(dt)
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
    # Only process a single file for now
    data_dir = "BASt Hourly Data"
    pattern = os.path.join(data_dir, "zst*.csv")
    csv_files = [f for f in sorted(os.listdir(data_dir)) if f.startswith('zst') and f.endswith('.csv')]
    # Only keep files with year > 2020
    filtered_files = []
    for f in csv_files:
        m = re.match(r'zst\d+_(\d{4})\.csv', f)
        if m and int(m.group(1)) > 2020:
            filtered_files.append(f)
    if not filtered_files:
        print(f"No zst*.csv files with year > 2020 found in {data_dir}")
        return
    filename = filtered_files[0]
    file_path = os.path.join(data_dir, filename)
    print(f"Processing file: {filename}")

    # Extract station and year from filename
    m = re.match(r'zst(\d+)_(\d{4})\.csv', filename)
    if not m:
        print("Could not parse station number and year from filename")
        return
    station_number, year = m.group(1), m.group(2)

    # Get lat/lon from bast_stations_by_city.csv
    stations_file = os.path.join(data_dir, "bast_stations_by_city.csv")
    stations_df = pd.read_csv(stations_file)
    row = stations_df[(stations_df['station_number'] == int(station_number)) & (stations_df['year'] == int(year))]
    if row.empty:
        print(f"No station data found for year {year}, station {station_number}")
        return
    lat, lon = float(row.iloc[0]['latitude']), float(row.iloc[0]['longitude'])

    # Read the BASt CSV (read-only)
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    if df.empty:
        print("CSV is empty")
        return
    # Parse first and last row for start/end
    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    dt_start = parse_german_date_time(first_row['Datum'], first_row['Stunde'])
    dt_end = parse_german_date_time(last_row['Datum'], last_row['Stunde'])
    unix_start = convert_to_unix_start(dt_start)
    unix_end = convert_to_unix_end(dt_end)
    print(f"Unix start: {unix_start}, Unix end: {unix_end}")

    # OWM API call
    api_key = "489eb9ae90ccd3a36e081f88e281293f"
    try:
        response = get_openweather_air_pollution_data(lat, lon, unix_start, unix_end, api_key)
    except Exception as e:
        print(f"API error: {e}")
        return
    # Write aqi, components, dt to new CSV
    owm_list = response.get('list', [])
    if not owm_list:
        print("No data from OWM API")
        return
    out_rows = []
    for entry in owm_list:
        row = {'dt': entry.get('dt'), 'aqi': entry.get('main', {}).get('aqi')}
        row.update(entry.get('components', {}))
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)
    out_dir = "owm Hourly Data"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"owm{station_number}_{year}.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Wrote OWM data to {out_file}")

if __name__ == "__main__":
    main() 