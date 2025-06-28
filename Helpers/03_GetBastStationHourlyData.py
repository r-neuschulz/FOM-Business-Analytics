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

# User agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
]

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
                zip_ref.extractall(output_dir)
            print(f"Successfully extracted: zst{station_number}_{year}")
            
            # Remove temporary zip file
            temp_zip_path.unlink()
            return True
            
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip file for zst{station_number}_{year}: {e}")
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
                    zip_ref.extractall(output_dir)
                print(f"Successfully extracted: zst{station_number}_{year}")
                
                # Remove temporary zip file
                temp_zip_path.unlink()
                return True
                
            except zipfile.BadZipFile as e:
                print(f"Error extracting zip file for zst{station_number}_{year}: {e}")
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check and download BASt station hourly data zip files')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode with limited URLs checked and downloads')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Number of URLs to process before taking a longer break (default: 20)')
    parser.add_argument('--batch-delay', type=int, default=10,
                       help='Seconds to wait between batches (default: 10)')
    args = parser.parse_args()
    
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
    required_columns = ['year', 'station_number']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path("BASt Hourly Data")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare results list
    results = []
    total_urls = len(df)
    downloaded_count = 0
    checked_count = 0
    
    # Apply limit if in test mode
    if MAX_URLS_TO_CHECK:
        total_urls = min(total_urls, MAX_URLS_TO_CHECK)
        df = df.head(MAX_URLS_TO_CHECK)
    
    # Reverse the DataFrame to process from most recent to oldest data
    df = df.iloc[::-1].reset_index(drop=True)
    
    print(f"Checking {total_urls} BASt Station URLs (most recent data first)...")
    
    for counter, (index, row) in enumerate(df.iterrows(), 1):
        year = row['year']
        station_number = row['station_number']
        
        # Generate URL
        url = f"https://www.bast.de/videos/{year}/zst{station_number}.zip"
        
        # Check if file exists
        exists = check_zip_file_exists(url)
        checked_count += 1
        
        # Download if exists and within download limit
        downloaded = False
        if exists:
            if MAX_FILES_TO_DOWNLOAD and downloaded_count >= MAX_FILES_TO_DOWNLOAD:
                print(f"Skipping download (limit reached): zst{station_number}_{year}")
            else:
                # Check if extracted files already exist
                extracted_files_exist = any(output_dir.glob(f"*zst{station_number}*"))
                
                if not extracted_files_exist:
                    print(f"Downloading and extracting: zst{station_number}_{year}")
                    downloaded = download_and_extract_zip(url, output_dir, station_number, year)
                    if downloaded:
                        downloaded_count += 1
                else:
                    print(f"Extracted files already exist locally: zst{station_number}_{year}")
                    downloaded = True
                    downloaded_count += 1
        
        results.append({
            'year': year,
            'station_number': station_number,
            'url': url,
            'exists': exists,
            'downloaded': downloaded
        })
        
        # Add random delay between requests
        random_delay()
        
        # Progress indicator and batch processing
        if counter % 100 == 0:
            print(f"Processed {counter}/{total_urls} BASt Station URLs...")
        
        # Take longer break between batches
        if counter % args.batch_size == 0 and counter < total_urls:
            print(f"Completed batch of {args.batch_size} URLs. Taking {args.batch_delay}s break...")
            time.sleep(args.batch_delay)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / "zip_file_existence_check.csv"
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    existing_files = results_df['exists'].sum()
    print(f"\nSummary:")
    print(f"Total BASt Station URLs checked: {checked_count}")
    print(f"Files found: {existing_files}")
    print(f"Files not found: {checked_count - existing_files}")
    print(f"Files downloaded and extracted: {downloaded_count}")
    if args.test:
        print(f"Test mode limits: {MAX_URLS_TO_CHECK} URLs checked, {MAX_FILES_TO_DOWNLOAD} downloads max")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
