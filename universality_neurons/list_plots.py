#!/usr/bin/env python3

import os
import glob
from datetime import datetime

def list_analysis_files():
    """List all analysis files in the plots directory"""
    plots_dir = "plots"
    
    if not os.path.exists(plots_dir):
        print("No plots directory found.")
        return
    
    # Get all files in plots directory
    files = glob.glob(os.path.join(plots_dir, "*"))
    
    if not files:
        print("No analysis files found in plots directory.")
        return
    
    # Group files by timestamp
    timestamps = set()
    for file in files:
        filename = os.path.basename(file)
        if '_' in filename:
            # Extract timestamp from filename
            parts = filename.split('_')
            if len(parts) >= 2:
                # Look for timestamp pattern YYYYMMDD_HHMMSS
                for i, part in enumerate(parts[:-1]):
                    if len(part) == 8 and len(parts[i+1]) >= 6:  # YYYYMMDD_HHMMSS
                        timestamps.add(f"{part}_{parts[i+1][:6]}")
                        break
    
    print("Analysis runs found:")
    print("=" * 50)
    
    for timestamp in sorted(timestamps, reverse=True):
        print(f"\nTimestamp: {timestamp}")
        print("-" * 30)
        
        # Find all files for this timestamp
        timestamp_files = [f for f in files if timestamp in os.path.basename(f)]
        
        for file in sorted(timestamp_files):
            filename = os.path.basename(file)
            file_size = os.path.getsize(file)
            file_size_mb = file_size / (1024 * 1024)
            
            if filename.endswith('.png'):
                print(f"{filename} ({file_size_mb:.2f} MB)")
            elif filename.endswith('.txt'):
                print(f"{filename}")
            else:
                print(f"{filename}")

if __name__ == "__main__":
    list_analysis_files() 