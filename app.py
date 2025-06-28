#!/usr/bin/env python3
"""
Main application for BASt Business Analytics
Handles coordinate retrieval and visualization of BASt traffic counting stations
"""

import sys
import subprocess
import os
from pathlib import Path

def run_script(script_path, args=None):
    """
    Run a Python script with optional arguments
    
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
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
    print("BASt Business Analytics Pipeline")
    print("=" * 40)
    
    # Check if --fresh parameter is provided
    fresh_mode = "--fresh" in sys.argv
    
    # Step 1: Get BASt coordinates
    print("\nStep 1: Retrieving BASt coordinates...")
    bast_coords_script = "01_GetBastCoords.py"
    
    if not os.path.exists(bast_coords_script):
        print(f"Error: {bast_coords_script} not found in current directory")
        return False
    
    # Prepare arguments for coordinate retrieval
    coords_args = ["--fresh"] if fresh_mode else []
    
    if not run_script(bast_coords_script, coords_args):
        print("Failed to retrieve BASt coordinates. Exiting.")
        return False
    
    # Step 2: Create visualizations
    print("\nStep 2: Creating visualizations...")
    viz_script = "02_DrawBastLocations.py"
    
    if not os.path.exists(viz_script):
        print(f"Error: {viz_script} not found in current directory")
        return False
    
    if not run_script(viz_script):
        print("Failed to create visualizations. Exiting.")
        return False
    
    print("\n" + "=" * 40)
    print("BASt Business Analytics Pipeline completed successfully!")
    print("\nGenerated files:")
    print("- BASt Files/bast_locations.csv (coordinate data)")
    print("- Graphs/bast_locations_heatmap.png (heatmap visualization)")
    print("- Graphs/bast_stations_by_year.png (year comparison)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
