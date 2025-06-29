import pandas as pd
import requests
import os
import argparse
import time
import zipfile
from pathlib import Path
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe file operations
file_lock = Lock()
print_lock = Lock()

def download_and_extract_zip(args):
    """
    Thread-safe download and extract function with improved error handling
    
    Args:
        args (tuple): (url, output_dir, station_number, year, city, index, total)
    
    Returns:
        dict: Result dictionary with status and details
    """
    url, output_dir, station_number, year, city, index, total = args
    
    try:
        # Add random delay to avoid overwhelming the server (0.1-0.3 seconds)
        time.sleep(random.uniform(0.1, 0.3))
        
        # Check if files already exist (thread-safe)
        csv_filename = f"zst{station_number}_{year}.csv"
        not_zip_filename = f"zst{station_number}_{year}_not_zip"
        
        with file_lock:
            if (output_dir / csv_filename).exists():
                with print_lock:
                    print(f"[{index}/{total}] zst{station_number}_{year} ({city}): CSV file already exists, skipping")
                return {
                    'year': year,
                    'station_number': station_number,
                    'url': url,
                    'exists': True,
                    'downloaded': True,
                    'not_zip_file': False,
                    'city': city,
                    'status': 'skipped_existing'
                }
                
            if (output_dir / not_zip_filename).exists():
                with print_lock:
                    print(f"[{index}/{total}] zst{station_number}_{year} ({city}): Non-zip file already exists, skipping")
                return {
                    'year': year,
                    'station_number': station_number,
                    'url': url,
                    'exists': True,
                    'downloaded': True,
                    'not_zip_file': True,
                    'city': city,
                    'status': 'skipped_existing'
                }
        
        # Download the file with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use session for better connection reuse
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                response = session.get(url, timeout=30)
                response.raise_for_status()
                break
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    with print_lock:
                        print(f"[{index}/{total}] zst{station_number}_{year} ({city}): Download failed after {max_retries} attempts: {e}")
                    return {
                        'year': year,
                        'station_number': station_number,
                        'url': url,
                        'exists': False,
                        'downloaded': False,
                        'not_zip_file': False,
                        'city': city,
                        'status': 'download_failed',
                        'error': str(e)
                    }
                else:
                    # Exponential backoff
                    time.sleep(2 ** attempt + random.uniform(0, 1))
        
        # Create temporary zip file with thread-safe naming
        temp_zip_path = output_dir / f"temp_zst{station_number}_{year}_{random.randint(1000, 9999)}.zip"
        
        # Save the downloaded content
        with open(temp_zip_path, 'wb') as f:
            f.write(response.content)
        
        # Try to extract as zip
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove temp file
            temp_zip_path.unlink()
            
            with print_lock:
                print(f"[{index}/{total}] zst{station_number}_{year} ({city}): Successfully extracted")
            
            return {
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': True,
                'downloaded': True,
                'not_zip_file': False,
                'city': city,
                'status': 'success'
            }
            
        except zipfile.BadZipFile:
            # Not a zip file, save as-is
            non_zip_path = output_dir / not_zip_filename
            temp_zip_path.rename(non_zip_path)
            
            with print_lock:
                print(f"[{index}/{total}] zst{station_number}_{year} ({city}): Saved non-zip file")
            
            return {
                'year': year,
                'station_number': station_number,
                'url': url,
                'exists': True,
                'downloaded': True,
                'not_zip_file': True,
                'city': city,
                'status': 'not_zip'
            }
            
    except Exception as e:
        with print_lock:
            print(f"[{index}/{total}] zst{station_number}_{year} ({city}): Unexpected error: {e}")
        return {
            'year': year,
            'station_number': station_number,
            'url': url,
            'exists': False,
            'downloaded': False,
            'not_zip_file': False,
            'city': city,
            'status': 'error',
            'error': str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download BASt station hourly data zip files')
    parser.add_argument('--city', choices=['cologne', 'berlin', 'duesseldorf'], nargs='+',
                       default=['cologne', 'berlin', 'duesseldorf'],
                       help='Filter stations by city coordinates (default: all three cities)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: skip actual downloads, only create city mapping')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retry attempts per download (default: 3)')
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

    print("Parallel BASt Station Data Downloader")
    print("=" * 50)
    print(f"Using {args.workers} parallel workers")
    print(f"Max retries per download: {args.max_retries}")

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

    # Filter to only include stations from 2003 onwards
    df = df[df['year'] >= 2003]
    print(f"Stations from 2003 onwards: {len(df)} locations")

    # Set df as a pandas DataFrame
    df = pd.DataFrame(df)

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
    else:
        # No city filtering specified, but still create city mapping for all stations
        print("No city filtering specified. Creating city mapping for all stations...")

        # Initialize city column
        df['city'] = 'unknown'

        # Apply city boundaries to categorize all stations
        for city, boundaries in city_boundaries.items():
            print(f"  {city.title()}: Lat {boundaries['min_lat']:.6f}-{boundaries['max_lat']:.6f}, Lon {boundaries['min_lon']:.6f}-{boundaries['max_lon']:.6f}")

            city_filter = (
                (df['latitude'] >= boundaries['min_lat']) & 
                (df['latitude'] <= boundaries['max_lat']) & 
                (df['longitude'] >= boundaries['min_lon']) & 
                (df['longitude'] <= boundaries['max_lon'])
            )

            # Mark stations belonging to this city
            df.loc[city_filter, 'city'] = city

        # Print summary by city
        print(f"\nStations found by city:")
        for city in city_boundaries.keys():
            city_count = len(df[df['city'] == city])
            print(f"  {city.title()}: {city_count} stations")

        unknown_count = len(df[df['city'] == 'unknown'])
        print(f"  Unknown/Other: {unknown_count} stations")

        print(f"\nTotal stations: {len(df)}")

    # Create output directory if it doesn't exist
    output_dir = Path("BASt Hourly Data")
    output_dir.mkdir(exist_ok=True)

    # Process stations
    results = []
    total_stations = len(df)

    if args.test:
        print(f"\nTest mode: Skipping actual downloads for {total_stations} stations")
        print("Creating city mapping file only...")

        # Create dummy results for test mode
        for i, (_, row) in enumerate(df.iterrows()):
            year = int(row['year'])
            station_number = int(row['station_number'])
            city = str(row.get('city', 'unknown'))

            results.append({
                'year': year,
                'station_number': station_number,
                'url': f"https://www.bast.de/videos/{year}/zst{station_number}.zip",
                'exists': False,
                'downloaded': False,
                'not_zip_file': False,
                'city': city,
                'status': 'test_mode'
            })
    else:
        print(f"\nProcessing {total_stations} stations with {args.workers} parallel workers...")
        print("Progress will be shown for each completed download.")

        # Prepare arguments for parallel processing
        download_args = []
        for i, (_, row) in enumerate(df.iterrows()):
            year = int(row['year'])
            station_number = int(row['station_number'])
            city = str(row.get('city', 'unknown'))
            url = f"https://www.bast.de/videos/{year}/zst{station_number}.zip"
            
            download_args.append((url, output_dir, station_number, year, city, i + 1, total_stations))

        # Process downloads in parallel with progress tracking
        successful = 0
        failed = 0
        skipped = 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(download_and_extract_zip, args): args for args in download_args}
            
            # Process completed tasks with progress bar
            with tqdm(total=total_stations, desc="Downloading files", unit="file") as pbar:
                for future in as_completed(future_to_args):
                    result = future.result()
                    results.append(result)
                    
                    # Update counters
                    if result['status'] == 'success':
                        successful += 1
                    elif result['status'] == 'skipped_existing':
                        skipped += 1
                    else:
                        failed += 1
                    
                    pbar.update(1)
                    
                    # Update progress bar description
                    pbar.set_postfix({
                        'Success': successful,
                        'Skipped': skipped,
                        'Failed': failed
                    })

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\nParallel processing completed in {processing_time:.2f} seconds")
        print(f"Average time per file: {processing_time/total_stations:.3f} seconds")

    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = output_dir / "zip_file_existence_check.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')

    # Save city mapping (always save, not just when city filtering was used)
    city_mapping_file = output_dir / "bast_stations_by_city.csv"
    selected_columns = ['year', 'station_number', 'city', 'latitude', 'longitude', 'location_name']

    # Ensure location_name column exists
    if 'location_name' not in df.columns:
        df['location_name'] = 'Unknown Location'

    # Filter to only include columns that exist in the DataFrame
    available_columns = [col for col in selected_columns if col in df.columns]
    city_mapping_df = pd.DataFrame(df[available_columns])
    city_mapping_df.to_csv(city_mapping_file, index=False, encoding='utf-8')
    print(f"\nStation-city assignments saved to: {city_mapping_file}")

    # Print summary
    if not args.test:
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
    else:
        print(f"\nTest mode summary:")
        print(f"Total stations processed: {len(results_df)}")
        print(f"City mapping saved to: {city_mapping_file}")

if __name__ == "__main__":
    main()
