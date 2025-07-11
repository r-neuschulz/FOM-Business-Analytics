#!/usr/bin/env python3
"""
Main application for BASt Business Analytics
Handles coordinate retrieval, visualization, and hourly data download of BASt traffic counting stations
"""

import sys
import subprocess
import os
import argparse
import signal
import time
import logging
from datetime import datetime
from pathlib import Path

# Global variable to track if termination was requested
termination_requested = False
current_process = None

def setup_logging():
    """
    Set up logging to both console and file with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"output_run_{timestamp}.txt"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename

def log_print(message):
    """
    Print message to both console and log file
    """
    logging.info(message)

def signal_handler(signum, frame):
    """
    Handle termination signals gracefully
    """
    global termination_requested, current_process
    log_print(f"\n\nReceived termination signal ({signum}). Gracefully shutting down...")
    termination_requested = True
    
    # Terminate current subprocess if running
    if current_process and current_process.poll() is None:
        log_print("Terminating current subprocess...")
        current_process.terminate()
        try:
            current_process.wait(timeout=5)  # Wait up to 5 seconds
        except subprocess.TimeoutExpired:
            log_print("Force killing subprocess...")
            current_process.kill()
    
    log_print("Cleanup completed. Exiting.")
    sys.exit(1)

def run_script(script_path, args=None):
    """
    Run a Python script with optional arguments and forward stdout in real-time
    
    Args:
        script_path (str): Path to the Python script
        args (list): Optional list of arguments to pass to the script
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    global current_process, termination_requested
    
    try:
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        log_print(f"Running: {' '.join(cmd)}")
        log_print("-" * 60)
        
        # Run script with real-time stdout forwarding
        current_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Forward output in real-time
        try:
            if current_process.stdout:
                for line in iter(current_process.stdout.readline, ''):
                    if termination_requested:
                        break
                    log_print(line.rstrip())
                    sys.stdout.flush()  # Ensure immediate output
        except KeyboardInterrupt:
            pass  # Handle Ctrl+C during output reading
        
        # Wait for process to complete
        if not termination_requested:
            current_process.wait()
        
        log_print("-" * 60)
        
        # Check if process was terminated
        if termination_requested:
            return False
        
        return current_process.returncode == 0
        
    except subprocess.CalledProcessError as e:
        log_print(f"Error running {script_path}: {e}")
        if e.stdout:
            log_print("STDOUT:" + e.stdout)
        if e.stderr:
            log_print("STDERR:" + e.stderr)
        return False
    except FileNotFoundError:
        log_print(f"Script not found: {script_path}")
        return False
    except KeyboardInterrupt:
        log_print(f"\nScript interrupted by user: {script_path}")
        return False
    finally:
        current_process = None

def main():
    """
    Main function that orchestrates the BASt data processing pipeline
    """
    global termination_requested
    
    # Set up logging
    log_filename = setup_logging()
    
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # SIGTERM (Unix)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BASt Business Analytics Pipeline')
    parser.add_argument('--api-key', required=True, help='OpenWeatherMap API key (required)')
    parser.add_argument('--fresh', action='store_true', 
                       help='Force fresh data retrieval for coordinates')
    parser.add_argument('--test', action='store_true', 
                       help='Run hourly data download in test mode (limited URLs and downloads)')
    parser.add_argument('--skip-hourly', action='store_true',
                       help='Skip hourly data download step')
    parser.add_argument('--city', choices=['cologne', 'berlin', 'duesseldorf'], nargs='+',
                       default=['cologne', 'berlin', 'duesseldorf'],
                       help='Filter stations by city coordinates (can specify multiple: cologne, berlin, duesseldorf)')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel download workers (default: 10)')
    parser.add_argument('--skip-steps', nargs='+', type=int, choices=range(0, 11),
                       help='Skip specific steps (0-10) in the pipeline')
    
    args = parser.parse_args()
    
    # Define the steps to skip
    skip_steps = set(args.skip_steps) if args.skip_steps else set()
    
    log_print("BASt Business Analytics Pipeline")
    log_print("=" * 40)
    log_print(f"Logging output to: {log_filename}")
    log_print("Press Ctrl+C to gracefully terminate the pipeline at any time")
    log_print("=" * 40)
    
    # Define all helper scripts in sequential order
    helper_scripts = [
        ("00_DrawBastOpenWeatherMapOverlap.py", "Step 0: Creating BASt-OpenWeatherMap overlap visualization", []),
        ("01_GetBastStationGeneralData.py", "Step 1: Retrieving BASt coordinates", ["--fresh"] if args.fresh else []),
        ("02_DrawBastLocations.py", "Step 2: Creating BASt locations visualization", []),
        ("03_GetBastStationHourlyData.py", "Step 3: Downloading BASt station hourly data", 
         (["--test"] if args.test else []) + (["--city"] + args.city) + (["--workers", str(args.workers)] if args.workers != 10 else [])),
        ("04_DrawBastLocationsByCity.py", "Step 4: Creating city-by-year stacked bar visualization", []),
        ("05_DrawBastLocationsByCityHeatmap.py", "Step 5: Creating city heatmap visualization", []),
        ("06_GetOpenWeatherHourlyData.py", "Step 6: Downloading OpenWeather hourly data", ["--api-key", args.api_key]),
        ("07_DrawDownloadQuality.py", "Step 7: Creating download quality visualization", []),
        ("08_DrawTrafficVsPollution.py", "Step 8: Creating traffic vs pollution analysis", []),
        ("09_PerformAssociationCorrelationAnalysis.py", "Step 9: Performing association and correlation analysis", []),
        ("10_PerformDeseasonedCorrelationAnalysis.py", "Step 10: Performing deseasoned correlation analysis", [])
    ]
    
    try:
        # Execute each helper script sequentially
        for step_num, (script_name, description, script_args) in enumerate(helper_scripts):
            # Check if this step should be skipped
            if step_num in skip_steps:
                log_print(f"\nSkipping {description} (step {step_num})")
                continue
            
            # Special handling for Step 3 (hourly data download)
            if step_num == 3 and args.skip_hourly:
                log_print(f"\n{description} (skipping actual downloads, creating city mapping only)...")
                # Run just to create city mapping, with --test to skip actual downloads
                script_args = ["--test"] + (["--city"] + args.city)
            elif step_num == 3:
                log_print(f"\n{description}...")
            else:
                log_print(f"\n{description}...")
            
            script_path = os.path.join("Helpers", script_name)
            
            if not os.path.exists(script_path):
                log_print(f"Error: {script_path} not found.")
                if step_num in [0, 1, 2, 3]:  # Critical steps
                    return False
                else:
                    log_print(f"Warning: Skipping step {step_num} due to missing script.")
                    continue
            
            if not run_script(script_path, script_args):
                if termination_requested:
                    log_print(f"Pipeline terminated by user during {description.lower()}.")
                    return False
                log_print(f"Failed to complete {description.lower()}. Exiting.")
                if step_num in [0, 1, 2, 3]:  # Critical steps
                    return False
                else:
                    log_print(f"Warning: Continuing pipeline despite failure in step {step_num}.")
                    continue
            
            # Check for termination request
            if termination_requested:
                log_print(f"Pipeline terminated by user after {description.lower()}.")
                return False
        
        log_print("\n" + "=" * 40)
        log_print("BASt Business Analytics Pipeline completed successfully!")
        if args.city:
            log_print(f"City filter applied: {', '.join(args.city)}")
        if skip_steps:
            log_print(f"Steps skipped: {', '.join(map(str, sorted(skip_steps)))}")
        log_print("\nGenerated files:")
        log_print("- BASt Station Files/bast_locations.csv (coordinate data)")
        log_print("- Graphs/bast_locations_heatmap.png (heatmap visualization)")
        log_print("- Graphs/bast_stations_by_year.png (year comparison)")
        log_print("- Graphs/bast_locations_by_year_by_city.png (city-by-year stacked bar)")
        log_print("- Graphs/bast_locations_by_city_heatmap.png (city heatmap visualization)")
        log_print("- Graphs/bast_openweather_overlap.png (BASt-OpenWeatherMap overlap)")
        log_print("- Graphs/download_quality_analysis.png (download quality visualization)")
        log_print("- Graphs/traffic_vs_pollution_analysis.png (traffic vs pollution analysis)")
        log_print("- Graphs/correlation_analysis_results.png (correlation analysis)")
        log_print("- Graphs/deseasoned_correlation_analysis.png (deseasoned correlation analysis)")
        if not args.skip_hourly:
            log_print("- BASt Hourly Data/zst*.csv (extracted hourly traffic data)")
            log_print("- BASt Hourly Data/zip_file_existence_check.csv (download status report)")
            log_print("- owm Hourly Data/owm*.csv (OpenWeather API downloaded hourly weather data)")

        
        return True
        
    except KeyboardInterrupt:
        log_print("\n\nPipeline interrupted by user.")
        return False
    except Exception as e:
        log_print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
