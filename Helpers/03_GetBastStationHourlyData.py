import pandas as pd
import requests
import os
import argparse
import time
import random
import zipfile
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# User agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
]

# Global variable to track if termination was requested
termination_requested = False
current_process = None

# Thread-safe counters
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self):
        with self._lock:
            return self._value

def create_session():
    """
    Create a requests session with retry strategy and random user agent.
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set random user agent
    session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    
    return session

def random_delay(min_delay=1, max_delay=3):
    """
    Add random delay between requests to avoid detection.
    """
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)

def check_zip_file_exists(url, max_retries=6, base_delay=1):
    """
    Check if a zip file exists at the given URL without downloading it.
    Uses immediate first attempt, then exponential backoff strategy for retries.
    Returns True if the file exists, False otherwise.
    """
    session = create_session()
    
    # First attempt - immediate, no delay
    try:
        response = session.head(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"First attempt failed for {url}: {e}")
    
    # Subsequent attempts with backoff
    for attempt in range(max_retries):
        try:
            response = session.head(url, timeout=10)
            return response.status_code == 200
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to check {url} after {max_retries + 1} attempts: {e}")
                return False
            
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Add jitter
            print(f"Attempt {attempt + 2} failed for {url}, retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    return False

def download_and_extract_zip(url, output_dir, station_number, year, max_retries=6, base_delay=1):
    """
    Download a zip file from the given URL and immediately extract it to the output directory.
    Uses immediate first attempt, then exponential backoff strategy for retries.
    Returns True if download and extraction successful, False otherwise.
    """
    session = create_session()
    
    # First attempt - immediate, no delay
    try:
        response = session.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Create temporary zip file path
        temp_zip_path = output_dir / f"temp_zst{station_number}_{year}.zip"
        
        # Download zip file to temporary location
        with open(temp_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file immediately
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # Handle encoding issues in zip file names
                for file_info in zip_ref.infolist():
                    try:
                        # Try to decode filename with UTF-8, fallback to cp437
                        filename = file_info.filename
                        if isinstance(filename, bytes):
                            try:
                                filename = filename.decode('utf-8')
                            except UnicodeDecodeError:
                                filename = filename.decode('cp437', errors='replace')
                        
                        # Extract with proper encoding
                        zip_ref.extract(file_info, output_dir)
                        
                    except Exception as e:
                        print(f"Warning: Could not extract {file_info.filename} from zst{station_number}_{year}: {e}")
                        continue
                
                print(f"Successfully extracted: zst{station_number}_{year}")
                
                # Remove temporary zip file
                temp_zip_path.unlink()
                return True
                
        except zipfile.BadZipFile as e:
            error_msg = str(e)
            if "File is not a zip file" in error_msg:
                # Save the non-zip file as-is
                non_zip_filename = f"zst{station_number}_{year}_not_zip"
                non_zip_path = output_dir / non_zip_filename
                temp_zip_path.rename(non_zip_path)
                print(f"Saved non-zip file: {non_zip_filename}")
                return "not_zip"
            else:
                print(f"Error extracting zip file for zst{station_number}_{year}: {e}")
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                return False
        except Exception as e:
            print(f"Unexpected error extracting zst{station_number}_{year}: {e}")
            if temp_zip_path.exists():
                temp_zip_path.unlink()
            return False
            
    except requests.RequestException as e:
        print(f"First download attempt failed for {url}: {e}")
    
    # Subsequent attempts with backoff
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create temporary zip file path
            temp_zip_path = output_dir / f"temp_zst{station_number}_{year}.zip"
            
            # Download zip file to temporary location
            with open(temp_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file immediately
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    # Handle encoding issues in zip file names
                    for file_info in zip_ref.infolist():
                        try:
                            # Try to decode filename with UTF-8, fallback to cp437
                            filename = file_info.filename
                            if isinstance(filename, bytes):
                                try:
                                    filename = filename.decode('utf-8')
                                except UnicodeDecodeError:
                                    filename = filename.decode('cp437', errors='replace')
                            
                            # Extract with proper encoding
                            zip_ref.extract(file_info, output_dir)
                            
                        except Exception as e:
                            print(f"Warning: Could not extract {file_info.filename} from zst{station_number}_{year}: {e}")
                            continue
                    
                    print(f"Successfully extracted: zst{station_number}_{year}")
                    
                    # Remove temporary zip file
                    temp_zip_path.unlink()
                    return True
                    
            except zipfile.BadZipFile as e:
                error_msg = str(e)
                if "File is not a zip file" in error_msg:
                    # Save the non-zip file as-is
                    non_zip_filename = f"zst{station_number}_{year}_not_zip"
                    non_zip_path = output_dir / non_zip_filename
                    temp_zip_path.rename(non_zip_path)
                    print(f"Saved non-zip file: {non_zip_filename}")
                    return "not_zip"
                else:
                    print(f"Error extracting zip file for zst{station_number}_{year}: {e}")
                    if temp_zip_path.exists():
                        temp_zip_path.unlink()
                    return False
            except Exception as e:
                print(f"Unexpected error extracting zst{station_number}_{year}: {e}")
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                return False
                
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Error downloading {url} after {max_retries + 1} attempts: {e}")
                return False
            
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Add jitter
            print(f"Download attempt {attempt + 2} failed for {url}, retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    return False

def process_station_download(row_data, output_dir, max_files_to_download, downloaded_counter, checked_counter, batch_size, batch_delay):
    """
    Process a single station download - designed for parallel execution
    """
    year = row_data['year']
    station_number = row_data['station_number']
    city = row_data.get('city', 'unknown')
    
    # Generate URL
    url = f"https://www.bast.de/videos/{year}/zst{station_number}.zip"
    
    # Check if file exists
    exists = check_zip_file_exists(url)
    checked_counter.increment()
    
    # Download if exists and within download limit
    downloaded = False
    not_zip_file = False
    if exists:
        if max_files_to_download and downloaded_counter.get() >= max_files_to_download:
            print(f"Skipping download (limit reached): zst{station_number}_{year} ({city})")
        else:
            # Check if extracted files already exist for this specific year
            csv_filename = f"zst{station_number}_{year}.csv"
            not_zip_filename = f"zst{station_number}_{year}_not_zip"

            # If already attempted to download and either the csv file or the not_zip file exists, skip download
            extracted_files_exist = (output_dir / csv_filename).exists() or (output_dir / not_zip_filename).exists()
            
            if not extracted_files_exist:
                print(f"Downloading and extracting: zst{station_number}_{year} ({city})")
                result = download_and_extract_zip(url, output_dir, station_number, year)
                if result == True:
                    downloaded = True
                    downloaded_counter.increment()
                elif result == "not_zip":
                    not_zip_file = True
                    downloaded_counter.increment()
            else:
                print(f"Extracted files already exist locally: zst{station_number}_{year} ({city})")
                downloaded = True
                downloaded_counter.increment()
    
    return {
        'year': year,
        'station_number': station_number,
        'url': url,
        'exists': exists,
        'downloaded': downloaded,
        'not_zip_file': not_zip_file,
        'city': city
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check and download BASt station hourly data zip files')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode with limited URLs checked and downloads')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Number of URLs to process before taking a longer break (default: 20)')
    parser.add_argument('--batch-delay', type=int, default=10,
                       help='Seconds to wait between batches (default: 10)')
    parser.add_argument('--city', choices=['cologne', 'berlin', 'duesseldorf'], nargs='+',
                       help='Filter stations by city coordinates (can specify multiple: cologne, berlin, duesseldorf)')
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of parallel download workers (default: 5)')
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
    
    # Configuration based on test mode
    if args.test:
        MAX_URLS_TO_CHECK = 20
        MAX_FILES_TO_DOWNLOAD = 5
        print("Running in TEST mode:")
        print(f"  - Maximum BASt Station URLs to check: {MAX_URLS_TO_CHECK}")
        print(f"  - Maximum files to download: {MAX_FILES_TO_DOWNLOAD}")
    else:
        MAX_URLS_TO_CHECK = None  # No limit
        MAX_FILES_TO_DOWNLOAD = None  # No limit
        print("Running in FULL mode (no limits)")
    
    print(f"Download strategy:")
    print(f"  - After first download attempt, delays between requests: 1-3 seconds")
    print(f"  - User agent rotation: {len(USER_AGENTS)} different agents")
    print(f"  - Parallel processing: {args.workers} workers")
    print(f"  - Batch processing: {args.batch_size} URLs per batch")
    print(f"  - Batch delays: {args.batch_delay} seconds between batches")
    print(f"  - Immediate extraction: Zip files are extracted and deleted immediately")
    
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
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Convert coordinates to numeric, handling any non-numeric values
    df['x_coordinate'] = pd.to_numeric(df['x_coordinate'], errors='coerce')
    df['y_coordinate'] = pd.to_numeric(df['y_coordinate'], errors='coerce')
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['x_coordinate', 'y_coordinate'])
    print(f"Valid coordinates: {len(df)} locations")
    
    # Convert coordinates from EPSG:25832 (ETRS89 / UTM zone 32N) to EPSG:4326 (WGS84 decimal degrees)
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    
    # Convert coordinates to decimal degrees
    lons, lats = transformer.transform(df['x_coordinate'].values, df['y_coordinate'].values)
    df['longitude'] = lons
    df['latitude'] = lats
    
    print(f"Coordinate conversion completed. Sample coordinates:")
    print(f"  Original (EPSG:25832): X={df['x_coordinate'].iloc[0]:.0f}, Y={df['y_coordinate'].iloc[0]:.0f}")
    print(f"  Converted (WGS84): Lon={df['longitude'].iloc[0]:.6f}, Lat={df['latitude'].iloc[0]:.6f}")
    
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
    
    # Apply limit if in test mode
    if MAX_URLS_TO_CHECK:
        df = df.head(MAX_URLS_TO_CHECK)
    
    # Reverse the DataFrame to process from most recent to oldest data
    df = df.iloc[::-1].reset_index(drop=True)
    
    total_urls = len(df)
    print(f"Processing {total_urls} BASt Station URLs with {args.workers} parallel workers...")
    
    # Initialize thread-safe counters
    downloaded_counter = ThreadSafeCounter()
    checked_counter = ThreadSafeCounter()
    
    # Prepare data for parallel processing
    station_data = df.to_dict('records')
    
    # Process downloads in parallel
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_station = {
                executor.submit(
                    process_station_download, 
                    row_data, 
                    output_dir, 
                    MAX_FILES_TO_DOWNLOAD, 
                    downloaded_counter, 
                    checked_counter, 
                    args.batch_size, 
                    args.batch_delay
                ): row_data for row_data in station_data
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_station):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Progress indicator
                    if completed % 50 == 0:
                        print(f"Completed {completed}/{total_urls} downloads...")
                        
                except Exception as e:
                    station = future_to_station[future]
                    print(f"Error processing station {station['station_number']}_{station['year']}: {e}")
                    # Add error result to continue processing
                    results.append({
                        'year': station['year'],
                        'station_number': station['station_number'],
                        'url': f"https://www.bast.de/videos/{station['year']}/zst{station['station_number']}.zip",
                        'exists': False,
                        'downloaded': False,
                        'not_zip_file': False,
                        'city': station.get('city', 'unknown'),
                        'error': str(e)
                    })
                    completed += 1
    except Exception as e:
        print(f"Critical error in parallel processing: {e}")
        print("Attempting to save partial results...")
        # If we have some results, try to save them
        if results:
            print(f"Saving {len(results)} completed results...")
        else:
            print("No results to save.")
            return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / "zip_file_existence_check.csv"
    try:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
    except UnicodeEncodeError:
        # Fallback to a more compatible encoding
        results_df.to_csv(output_file, index=False, encoding='latin-1')
    
    # Save city mapping if city filtering was used
    if args.city:
        city_mapping_file = output_dir / "bast_stations_by_city.csv"
        city_mapping_df = df[['year', 'station_number', 'city', 'latitude', 'longitude', 'location_name']].copy()
        try:
            city_mapping_df.to_csv(city_mapping_file, index=False, encoding='utf-8')
        except UnicodeEncodeError:
            # Fallback to a more compatible encoding
            city_mapping_df.to_csv(city_mapping_file, index=False, encoding='latin-1')
        print(f"Station-city assignments saved to: {city_mapping_file}")
    
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
    if args.test:
        print(f"Test mode limits: {MAX_URLS_TO_CHECK} URLs checked, {MAX_FILES_TO_DOWNLOAD} downloads max")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
