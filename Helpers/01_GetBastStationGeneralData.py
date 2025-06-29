import requests
import re
import csv
import os
import sys
from urllib.parse import urljoin, urlparse, parse_qs


def parse_add_bkg_poempel(content, year):
    """
    Parse addBKGPoempel function calls to extract coordinates and location information
    """
    # Pattern to match addBKGPoempel function calls
    pattern = r'addBKGPoempel\(results,\s*"([^"]+)"'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    locations = []
    
    for match in matches:
        try:
            # Split by comma, but be careful with commas inside quotes
            # The structure is: x_coord,y_coord,color,location_name,html_content
            parts = match.split(',')
            
            if len(parts) >= 4:
                # First two parts are coordinates
                x_coord = float(parts[0].strip())
                y_coord = float(parts[1].strip())
                
                # Third part is color
                color = parts[2].strip()
                
                # Fourth part and beyond is the location name (until we hit HTML content)
                location_parts = []
                for i in range(3, len(parts)):
                    part = parts[i].strip()
                    if '<div' in part:
                        # Stop when we hit HTML content
                        break
                    location_parts.append(part)
                
                location_name = ','.join(location_parts)
            
            # Parse road classification (A/B followed by road number, possibly with letters like 'n', 'a', etc.)
            road_class = ""
            road_number = ""
            road_class_match = re.search(r'^([AB])\s*(\d{1,3}[a-zA-Z]*):', location_name)
            if road_class_match:
                road_class = road_class_match.group(1)
                road_number = road_class_match.group(2)
            
            # Extract the number in parentheses at the end
            station_number = ""
            station_number_match = re.search(r'\((\d+)\)\s*$', location_name)
            if station_number_match:
                station_number = station_number_match.group(1)
            
            # Extract Kfz-Verkehr and Schwerverkehr if present
            kfz_verkehr = "---"
            kfz_match = re.search(r'Kfz-Verkehr/Tag:\s*([^<]+)', match)
            if kfz_match:
                kfz_verkehr = kfz_match.group(1).strip()
            
            schwerverkehr = "---"
            schwer_match = re.search(r'Schwerverkehr/Tag:\s*([^<]+)', match)
            if schwer_match:
                schwerverkehr = schwer_match.group(1).strip()
            
            # Extract href link if present
            href = ""
            href_match = re.search(r'href=\'([^\']+)\'', match)
            if href_match:
                href = href_match.group(1)
            
            location_data = {
                'year': year,
                'x_coordinate': x_coord,
                'y_coordinate': y_coord,
                'color': color,
                'road_class': road_class,
                'road_number': road_number,
                'location_name': location_name,
                'station_number': station_number,
                'kfz_verkehr': kfz_verkehr,
                'schwerverkehr': schwerverkehr,
                'href': href
            }
            
            locations.append(location_data)
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing match for year {year}: {e}")
            print(f"Problematic match: {match[:100]}...")
            continue
    
    return locations

def get_bast_locations():
    """
    Retrieve BASt locations from all years and save to CSV
    """
    # Check for --fresh parameter
    fresh_download = '--fresh' in sys.argv
    
    # Create BASt Station Files directory if it doesn't exist
    bast_files_dir = "BASt Station Files"
    if not os.path.exists(bast_files_dir):
        os.makedirs(bast_files_dir)
        print(f"Created directory: {bast_files_dir}")
    
    # Generate URLs for years 2003 to 2023
    base_url = "https://www.bast.de/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Daten/{year}_1/Jawe{year}.html"
    urls = []
    for year in range(2020, 2024):
        url = base_url.format(year=year)
        urls.append((year, url))
    
    all_locations = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for year, url in urls:
        print(f"Processing year: {year}")
        
        # Check if HTML file already exists
        html_filename = os.path.join(bast_files_dir, f"Jawe{year}.html")
        if os.path.exists(html_filename) and not fresh_download:
            print(f"  Using existing file: {html_filename}")
            with open(html_filename, 'r', encoding='utf-8') as html_file:
                content = html_file.read()
        else:
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                content = response.text
                print(f"  Retrieved {len(content)} characters")
                
                # Save raw HTML content to file
                with open(html_filename, 'w', encoding='utf-8') as html_file:
                    html_file.write(content)
                print(f"  Saved HTML content to: {html_filename}")
                
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching content for {year}: {e}")
                continue
            except Exception as e:
                print(f"  Unexpected error for {year}: {e}")
                continue
        
        # Parse addBKGPoempel function calls
        year_locations = parse_add_bkg_poempel(content, year)
        all_locations.extend(year_locations)
        
        print(f"  Found {len(year_locations)} locations")
    
    # Write to CSV file
    if all_locations:
        csv_filename = os.path.join(bast_files_dir, 'bast_locations.csv')
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['year', 'x_coordinate', 'y_coordinate', 'color', 'road_class', 'road_number', 'location_name', 'station_number', 'kfz_verkehr', 'schwerverkehr', 'href']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for location in all_locations:
                writer.writerow(location)
        
        print(f"\nAll {len(all_locations)} locations saved to '{csv_filename}'")
    
    return all_locations

# def get_bast_content_dynamic():
#     ... (dynamic code commented out) ...

if __name__ == "__main__":
    print("BASt Location Extraction")
    print("=" * 40)
    
    if '--fresh' in sys.argv:
        print("Fresh download mode: Will re-download all HTML files")
    else:
        print("Normal mode: Will use existing HTML files if available")
    
    locations = get_bast_locations()
    
    if locations:
        print(f"\nTotal locations found: {len(locations)}")
        
        # Show summary by year
        by_year = {}
        for location in locations:
            year = location['year']
            if year not in by_year:
                by_year[year] = 0
            by_year[year] += 1
        
        print("\nLocations by year:")
        for year in sorted(by_year.keys()):
            print(f"  {year}: {by_year[year]} locations")

