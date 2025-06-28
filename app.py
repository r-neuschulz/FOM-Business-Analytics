#!/usr/bin/env python3
"""
Main application for BASt Business Analytics
Handles coordinate retrieval, visualization, and hourly data download of BASt traffic counting stations
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

def run_script(script_path, args=None):
    """
    Run a Python script with optional arguments and forward stdout in real-time
    
    Args:
        script_path (str): Path to the Python script
        args (list): Optional list of arguments to pass to the script
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    try:
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        
        # Run script with real-time stdout forwarding
        result = subprocess.run(cmd, check=True, text=True, bufsize=1, universal_newlines=True)
        
        print("-" * 60)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return False

def main():
    """
    Main function that orchestrates the BASt data processing pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BASt Business Analytics Pipeline')
    parser.add_argument('--fresh', action='store_true', 
                       help='Force fresh data retrieval for coordinates')
    parser.add_argument('--test', action='store_true', 
                       help='Run hourly data download in test mode (limited URLs)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Number of URLs to process before taking a longer break (default: 20)')
    parser.add_argument('--batch-delay', type=int, default=10,
                       help='Seconds to wait between batches (default: 10)')
    parser.add_argument('--skip-hourly', action='store_true',
                       help='Skip hourly data download step')
    
    args = parser.parse_args()
    
    print("BASt Business Analytics Pipeline")
    print("=" * 40)
    
    # Step 1: Get BASt coordinates
    print("\nStep 1: Retrieving BASt coordinates...")
    bast_coords_script = os.path.join("Helpers", "01_GetBastStationGeneralData.py")
    
    if not os.path.exists(bast_coords_script):
        print(f"Error: {bast_coords_script} not found.")
        return False
    
    # Prepare arguments for coordinate retrieval
    coords_args = ["--fresh"] if args.fresh else []
    
    if not run_script(bast_coords_script, coords_args):
        print("Failed to retrieve BASt coordinates. Exiting.")
        return False
    
    # Step 2: Create visualizations
    print("\nStep 2: Creating visualizations...")
    viz_script = os.path.join("Helpers", "02_DrawBastLocations.py")
    
    if not os.path.exists(viz_script):
        print(f"Error: {viz_script} not found.")
        return False
    
    if not run_script(viz_script):
        print("Failed to create visualizations. Exiting.")
        return False
    
    # Step 3: Download BASt station hourly data
    if not args.skip_hourly:
        print("\nStep 3: Downloading BASt station hourly data...")
        hourly_script = os.path.join("Helpers", "03_GetBastStationHourlyData.py")
        
        if not os.path.exists(hourly_script):
            print(f"Error: {hourly_script} not found.")
            return False
        
        # Prepare arguments for hourly data download
        hourly_args = []
        if args.test:
            hourly_args.append("--test")
        if args.batch_size != 20:
            hourly_args.extend(["--batch-size", str(args.batch_size)])
        if args.batch_delay != 10:
            hourly_args.extend(["--batch-delay", str(args.batch_delay)])
        
        if not run_script(hourly_script, hourly_args):
            print("Failed to download BASt station hourly data. Exiting.")
            return False
    else:
        print("\nStep 3: Skipping hourly data download (--skip-hourly specified)")
    
    print("\n" + "=" * 40)
    print("BASt Business Analytics Pipeline completed successfully!")
    print("\nGenerated files:")
    print("- BASt Station Files/bast_locations.csv (coordinate data)")
    print("- Graphs/bast_locations_heatmap.png (heatmap visualization)")
    print("- Graphs/bast_stations_by_year.png (year comparison)")
    if not args.skip_hourly:
        print("- BASt Hourly Data/ (extracted hourly traffic data)")
        print("- BASt Hourly Data/zip_file_existence_check.csv (download status report)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
