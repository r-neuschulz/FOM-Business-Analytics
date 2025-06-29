#!/usr/bin/env python3
"""
Helper script to query OpenWeatherMap API for historical air pollution data.
Takes a CSV entry and makes an API call to get air pollution data for the given location and time period.
"""

import requests
import json
import sys
from typing import Dict, Any


def get_openweather_air_pollution_data(lat: float, lon: float, start_time: int, end_time: int, api_key: str) -> Dict[Any, Any]:
    """
    Query OpenWeatherMap API for historical air pollution data.
    
    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate  
        start_time (int): Start time in Unix timestamp (UTC)
        end_time (int): End time in Unix timestamp (UTC)
        api_key (str): OpenWeatherMap API key
        
    Returns:
        Dict: JSON response from the API
    """
    
    # API endpoint for historical air pollution data
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    
    # Parameters for the API call
    params = {
        'lat': lat,
        'lon': lon,
        'start': start_time,
        'end': end_time,
        'appid': api_key
    }
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Return the JSON response
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return {}


def parse_csv_entry(csv_line: str) -> Dict[str, Any]:
    """
    Parse a CSV entry from the BASt data format.
    
    Args:
        csv_line (str): A single line from the CSV file
        
    Returns:
        Dict: Parsed data with relevant fields
    """
    # Split by semicolon (German CSV format)
    fields = csv_line.strip().split(';')
    
    # Extract relevant fields based on the CSV structure
    # Fields are: TKNR;Zst;Land;Strklas;Strnum;Datum;Wotag;Fahrtzw;Stunde;...;UnixStart;UnixEnd;latitude;longitude
    try:
        return {
            'tknr': fields[0],
            'zst': fields[1], 
            'land': fields[2],
            'strklas': fields[3],
            'strnum': fields[4],
            'datum': fields[5],
            'wotag': fields[6],
            'fahrtzw': fields[7],
            'stunde': fields[8],
            'unix_start': int(fields[-4]),  # UnixStart
            'unix_end': int(fields[-3]),    # UnixEnd
            'latitude': float(fields[-2]),  # latitude
            'longitude': float(fields[-1])  # longitude
        }
    except (IndexError, ValueError) as e:
        print(f"Error parsing CSV line: {e}")
        return {}


def main():
    """
    Main function to demonstrate the API call with a sample CSV entry.
    """
    
    # Sample CSV entry with a more recent date (2020) for testing
    # Using a timestamp from 2020 to see if we get actual air pollution data
    sample_csv_entry = "4806;5014;5;A;57;210101;5;s;1;246;-;274;-;4;-;2;-;242;-;232;-;10;-;0;-;0;-;0;-;1;-;3;-;2;-;0;-;270;-;258;-;12;-;0;-;0;-;0;-;1;-;1;-;1;-;2;-;1609456020;1609459619;51.17425651198109;6.715488991346401"
    
    # Parse the CSV entry
    parsed_data = parse_csv_entry(sample_csv_entry)
    
    if not parsed_data:
        print("Failed to parse CSV entry")
        return
    
    print("Parsed CSV data:")
    print(f"Station ID: {parsed_data['tknr']}")
    print(f"Date: {parsed_data['datum']}")
    print(f"Hour: {parsed_data['stunde']}")
    print(f"Latitude: {parsed_data['latitude']}")
    print(f"Longitude: {parsed_data['longitude']}")
    print(f"Unix Start: {parsed_data['unix_start']}")
    print(f"Unix End: {parsed_data['unix_end']}")
    print()
    
    # OpenWeatherMap API key (you'll need to replace this with your actual API key)
    api_key = "489eb9ae90ccd3a36e081f88e281293f"  # This appears to be the key from the example
    
    print("Making API request to OpenWeatherMap...")
    print(f"URL: http://api.openweathermap.org/data/2.5/air_pollution/history")
    print(f"Parameters: lat={parsed_data['latitude']}, lon={parsed_data['longitude']}, start={parsed_data['unix_start']}, end={parsed_data['unix_end']}")
    print()
    
    # Make the API call
    response_data = get_openweather_air_pollution_data(
        lat=parsed_data['latitude'],
        lon=parsed_data['longitude'],
        start_time=parsed_data['unix_start'],
        end_time=parsed_data['unix_end'],
        api_key=api_key
    )
    
    if response_data:
        print("API Response:")
        print(json.dumps(response_data, indent=2))
    else:
        print("No response data received")


if __name__ == "__main__":
    main() 