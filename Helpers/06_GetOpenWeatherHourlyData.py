#!/usr/bin/env python3
"""
Batch process BASt hourly data CSV files (year > 2020) to extract the time range and location for each station,
query the OpenWeatherMap Air Pollution API for the corresponding period and location, and write the resulting
aqi, components, and dt values to new CSVs in 'owm Hourly Data/'.

- Does not modify the original BASt CSVs (read-only)
- Only processes files for years after 2020
- Outputs one OWM CSV per input file
"""

import requests
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm
import re

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download OpenWeatherMap hourly pollution data for BASt stations')
    parser.add_argument('--api-key', required=True, help='OpenWeatherMap API key (required)')
    args = parser.parse_args()
    
    # Set up directories and file lists
    data_dir = "BASt Hourly Data"
    csv_files = [f for f in sorted(os.listdir(data_dir)) if f.startswith('zst') and f.endswith('.csv')]
    filtered_files = []
    for f in csv_files:
        m = re.match(r'zst\d+_(\d{4})\.csv', f)
        if m and int(m.group(1)) >= 2020:
            filtered_files.append(f)
    if not filtered_files:
        print(f"No zst*.csv files with year >= 2020 found in {data_dir}")
        return
    stations_file = os.path.join(data_dir, "bast_stations_by_city.csv")
    stations_df = pd.read_csv(stations_file)
    api_key = args.api_key
    out_dir = "owm Hourly Data"
    os.makedirs(out_dir, exist_ok=True)
    success_count = 0
    fail_count = 0
    # Process each filtered BASt CSV file
    for filename in tqdm(filtered_files, desc="Processing BASt files", unit="file"):
        file_path = os.path.join(data_dir, filename)
        # Extract station number and year from filename
        m = re.match(r'zst(\d+)_(\d{4})\.csv', filename)
        if not m:
            print(f"Could not parse station number and year from filename {filename}")
            fail_count += 1
            continue
        station_number, year = m.group(1), m.group(2)
        # Get latitude and longitude for this station/year
        row = stations_df[(stations_df['station_number'] == int(station_number)) & (stations_df['year'] == int(year))]
        if row.empty:
            print(f"No station data found for year {year}, station {station_number}")
            fail_count += 1
            continue
        lat, lon = float(row.iloc[0]['latitude']), float(row.iloc[0]['longitude'])
        # Read the BASt CSV (read-only)
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        if df.empty:
            print(f"CSV {filename} is empty")
            fail_count += 1
            continue
        # Parse first and last row for start/end timestamps
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        dt_start = parse_german_date_time(first_row['Datum'], first_row['Stunde'])
        dt_end = parse_german_date_time(last_row['Datum'], last_row['Stunde'])
        unix_start = convert_to_unix_start(dt_start)
        unix_end = convert_to_unix_end(dt_end)
        # Determine output filename
        out_file = os.path.join(out_dir, f"owm{station_number}_{year}.csv")
        if os.path.exists(out_file):
            print(f"Output file {out_file} already exists, skipping download.")
            success_count += 1
            continue
        # Query OWM API for this station and time range
        try:
            response = get_openweather_air_pollution_data(lat, lon, unix_start, unix_end, api_key)
        except Exception as e:
            print(f"API error for {filename}: {e}")
            fail_count += 1
            continue
        owm_list = response.get('list', [])
        if not owm_list:
            print(f"No data from OWM API for {filename}")
            fail_count += 1
            continue
        out_rows = []
        for entry in owm_list:
            row = {'dt': entry.get('dt'), 'aqi': entry.get('main', {}).get('aqi')}
            row.update(entry.get('components', {}))
            out_rows.append(row)
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(out_file, index=False)
        success_count += 1
    print(f"\nProcessing complete. Success: {success_count}, Failed: {fail_count}, Total: {len(filtered_files)}")

if __name__ == "__main__":
    main() 