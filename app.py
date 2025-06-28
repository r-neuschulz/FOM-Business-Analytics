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
from pathlib import Path

# Global variable to track if termination was requested
termination_requested = False
current_process = None

def signal_handler(signum, frame):
    """
    Handle termination signals gracefully
    """
    global termination_requested, current_process
    print(f"\n\nReceived termination signal ({signum}). Gracefully shutting down...")
    termination_requested = True
    
    # Terminate current subprocess if running
    if current_process and current_process.poll() is None:
        print("Terminating current subprocess...")
        current_process.terminate()
        try:
            current_process.wait(timeout=5)  # Wait up to 5 seconds
        except subprocess.TimeoutExpired:
            print("Force killing subprocess...")
            current_process.kill()
    
    print("Cleanup completed. Exiting.")
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
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        
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
                    print(line.rstrip())
                    sys.stdout.flush()  # Ensure immediate output
        except KeyboardInterrupt:
            pass  # Handle Ctrl+C during output reading
        
        # Wait for process to complete
        if not termination_requested:
            current_process.wait()
        
        print("-" * 60)
        
        # Check if process was terminated
        if termination_requested:
            return False
        
        return current_process.returncode == 0
        
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
    except KeyboardInterrupt:
        print(f"\nScript interrupted by user: {script_path}")
        return False
    finally:
        current_process = None

def main():
    """
    Main function that orchestrates the BASt data processing pipeline
    """
    global termination_requested
    
    # Set up signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # SIGTERM (Unix)
    
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
    parser.add_argument('--city', choices=['cologne', 'berlin', 'duesseldorf'],
                       help='Filter stations by city coordinates (cologne, berlin, or duesseldorf)')
    
    args = parser.parse_args()
    
    print("BASt Business Analytics Pipeline")
    print("=" * 40)
    print("Press Ctrl+C to gracefully terminate the pipeline at any time")
    print("=" * 40)
    
    try:
        # Step 1: Get BASt coordinates
        print("\nStep 1: Retrieving BASt coordinates...")
        bast_coords_script = os.path.join("Helpers", "01_GetBastStationGeneralData.py")
        
        if not os.path.exists(bast_coords_script):
            print(f"Error: {bast_coords_script} not found.")
            return False
        
        # Prepare arguments for coordinate retrieval
        coords_args = ["--fresh"] if args.fresh else []
        
        if not run_script(bast_coords_script, coords_args):
            if termination_requested:
                print("Pipeline terminated by user during coordinate retrieval.")
                return False
            print("Failed to retrieve BASt coordinates. Exiting.")
            return False
        
        # Check for termination request
        if termination_requested:
            print("Pipeline terminated by user after coordinate retrieval.")
            return False
        
        # Step 2: Create visualizations
        print("\nStep 2: Creating visualizations...")
        viz_script = os.path.join("Helpers", "02_DrawBastLocations.py")
        
        if not os.path.exists(viz_script):
            print(f"Error: {viz_script} not found.")
            return False
        
        if not run_script(viz_script):
            if termination_requested:
                print("Pipeline terminated by user during visualization creation.")
                return False
            print("Failed to create visualizations. Exiting.")
            return False
        
        # Check for termination request
        if termination_requested:
            print("Pipeline terminated by user after visualization creation.")
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
            if args.city:
                hourly_args.extend(["--city", args.city])
            
            if not run_script(hourly_script, hourly_args):
                if termination_requested:
                    print("Pipeline terminated by user during hourly data download.")
                    return False
                print("Failed to download BASt station hourly data. Exiting.")
                return False
        else:
            print("\nStep 3: Skipping hourly data download (--skip-hourly specified)")
        
        # Check for termination request
        if termination_requested:
            print("Pipeline terminated by user after hourly data download.")
            return False
        
        print("\n" + "=" * 40)
        print("BASt Business Analytics Pipeline completed successfully!")
        if args.city:
            print(f"City filter applied: {args.city.title()}")
        print("\nGenerated files:")
        print("- BASt Station Files/bast_locations.csv (coordinate data)")
        print("- Graphs/bast_locations_heatmap.png (heatmap visualization)")
        print("- Graphs/bast_stations_by_year.png (year comparison)")
        if not args.skip_hourly:
            print("- BASt Hourly Data/ (extracted hourly traffic data)")
            print("- BASt Hourly Data/zip_file_existence_check.csv (download status report)")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
