import pandas as pd
import requests
import os
import argparse
import time
import zipfile
from pathlib import Path
from pyproj import Transformer

def download_and_extract_zip(url, output_dir, station_number, year):
    """
    Simple download and extract function with basic error handling
    """
    try:
        # Download the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary zip file
        temp_zip_path = output_dir / f"temp_zst{station_number}_{year}.zip"
        
        # Save the downloaded content
        with open(temp_zip_path, 'wb') as f:
            f.write(response.content)
        
        # Try to extract as zip
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Successfully extracted: zst{station_number}_{year}")
            temp_zip_path.unlink()  # Remove temp file
            return True
            
        except zipfile.BadZipFile:
            # Not a zip file, save as-is
            non_zip_filename = f"zst{station_number}_{year}_not_zip"
            non_zip_path = output_dir / non_zip_filename
            temp_zip_path.rename(non_zip_path)
            print(f"Saved non-zip file: {non_zip_filename}")
            return "not_zip"
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download BASt station hourly data zip files')
    parser.add_argument('--city', choices=['cologne', 'berlin', 'duesseldorf'], nargs='+',
                       help='Filter stations by city coordinates')
    args = parser.parse_args()
    
    # City coordinate boundaries
    city_boundaries = {
        'cologne': {
            'min_lat': 50.83222498154889,
            'max_lat': 51.02000941119971,
            'min_lon': 6.772774889455155,
            'max_lon': 7.098966411977884
        },
        'berlin': {
            'min_lat': 52.3407949019091,
            'max_lat': 52.72708690928127,
            'min_lon': 13.11075075837016,
            'max_lon': 13.823930000417983
        },
        'duesseldorf': {
            'min_lat': 51.1219059201905,
            'max_lat': 51.346932771684,
            'min_lon': 6.689240012533243,
            'max_lon': 6.928627403474891
        }
    }
    
    print("Simple BASt Station Data Downloader")
    print("=" * 40)
    
    # Read the bast_locations.csv file
    csv_path = Path("BASt Station Files/bast_locations.csv")
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    required_columns = ['year', 'station_number', 'x_coordinate', 'y_coordinate']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        return
    
    # Convert coordinates to numeric, handling any non-numeric values
    df['x_coordinate'] = pd.to_numeric(df['x_coordinate'], errors='coerce')
    df['y_coordinate'] = pd.to_numeric(df['y_coordinate'], errors='coerce')
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['x_coordinate', 'y_coordinate'])
    print(f"Valid coordinates: {len(df)} locations")
    
    # Convert coordinates from EPSG:25832 to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(df['x_coordinate'].values, df['y_coordinate'].values)
    df['longitude'] = lons
    df['latitude'] = lats
    
    # Apply city filtering if specified
    if args.city:
        print(f"Filtering stations for {', '.join([city.title() for city in args.city])}...")
        
        # Initialize city column
        df['city'] = 'unknown'
        
        # Create a combined filter for all specified cities
        city_filters = []
        for city in args.city:
            boundaries = city_boundaries[city]
            print(f"  {city.title()}: Lat {boundaries['min_lat']:.6f}-{boundaries['max_lat']:.6f}, Lon {boundaries['min_lon']:.6f}-{boundaries['max_lon']:.6f}")
            
            city_filter = (
                (df['latitude'] >= boundaries['min_lat']) & 
                (df['latitude'] <= boundaries['max_lat']) & 
                (df['longitude'] >= boundaries['min_lon']) & 
                (df['longitude'] <= boundaries['max_lon'])
            )
            city_filters.append(city_filter)
            
            # Mark stations belonging to this city
            df.loc[city_filter, 'city'] = city
        
        # Combine all city filters with OR logic
        combined_filter = city_filters[0]
        for city_filter in city_filters[1:]:
            combined_filter = combined_filter | city_filter
        
        # Apply the combined filter
        df = df[combined_filter]
        
        # Print summary by city
        print(f"\nStations found by city:")
        for city in args.city:
            city_count = len(df[df['city'] == city])
            print(f"  {city.title()}: {city_count} stations")
        
        print(f"\nTotal stations in combined city areas: {len(df)}")
        
        if len(df) == 0:
            print(f"No stations found in any of the specified city areas. Exiting.")
            return
    
    # Create output directory if it doesn't exist
    output_dir = Path("BASt Hourly Data")
    output_dir.mkdir(exist_ok=True)
    
    # Process each station sequentially
    results = []
    total_stations = len(df)
    
    print(f"\nProcessing {total_stations} stations sequentially...")
    
    for i, (_, row) in enumerate(df.iterrows()):
        year = int(row['year'])
        station_number = int(row['station_number'])
        city = str(row.get('city', 'unknown'))
        
        # Generate URL
        url = f"https://www.bast.de/videos/{year}/zst{station_number}.zip"
        
        print(f"[{i + 1}/{total_stations}] Processing zst{station_number}_{year} ({city})")
        
        # Check if files already exist
        csv_filename = f"zst{station_number}_{year}.csv"
        not_zip_filename = f"zst{station_number}_{year}_not_zip"
        
        if (output_dir / csv_filename).exists():
            print(f"  CSV file already exists, skipping")
            results.append({
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': True,
                'downloaded': True,
                'not_zip_file': False,
                'city': city
            })
            continue
            
        if (output_dir / not_zip_filename).exists():
            print(f"  Non-zip file already exists, skipping")
            results.append({
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': True,
                'downloaded': True,
                'not_zip_file': True,
                'city': city
            })
            continue
        
        # Download and extract
        result = download_and_extract_zip(url, output_dir, station_number, year)
        
        if result == True:
            results.append({
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': True,
                'downloaded': True,
                'not_zip_file': False,
                'city': city
            })
        elif result == "not_zip":
            results.append({
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': True,
                'downloaded': True,
                'not_zip_file': True,
                'city': city
            })
        else:
            results.append({
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': False,
                'downloaded': False,
                'not_zip_file': False,
                'city': city
            })
        
        # Simple delay between requests
        time.sleep(0.5)
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = output_dir / "zip_file_existence_check.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Save city mapping if city filtering was used
    if args.city:
        city_mapping_file = output_dir / "bast_stations_by_city.csv"
        selected_columns = ['year', 'station_number', 'city', 'latitude', 'longitude', 'location_name']
        city_mapping_df = pd.DataFrame(df[selected_columns])
        city_mapping_df.to_csv(city_mapping_file, index=False, encoding='utf-8')
        print(f"\nStation-city assignments saved to: {city_mapping_file}")
    
    # Print summary
    existing_files = results_df['exists'].sum()
    not_zip_files = results_df['not_zip_file'].sum()
    final_downloaded = results_df['downloaded'].sum()
    final_checked = len(results_df)
    
    print(f"\nSummary:")
    print(f"Total BASt Station URLs checked: {final_checked}")
    print(f"Files found: {existing_files}")
    print(f"Files not found: {final_checked - existing_files}")
    print(f"Files downloaded and extracted: {final_downloaded}")
    print(f"Non-zip files saved: {not_zip_files}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
