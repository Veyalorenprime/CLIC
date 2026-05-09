"""
Prepare PV datasets for CLIC training
"""

import sys
import os
from pathlib import Path

# ✅ FIX: Always use project root regardless of where script is run from
script_path = Path(__file__).resolve()  # Get absolute path of this script
project_root = script_path.parent.parent  # Go up two levels: scripts/ -> CLIC/

# Change to project root directory
os.chdir(project_root)
print(f"Changed working directory to: {project_root}\n")

# Add project root to path
sys.path.insert(0, str(project_root))

from src.data.preprocessing import prepare_all_data


if __name__ == "__main__":
    # All paths relative to project root
    data_dir = Path("data")  # Now relative to project_root
    normal_path = data_dir / "normal_data_engineered_new.csv"
    output_dir = data_dir / "processed" / "splits"
    
    print(f"Project root: {project_root}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Normal file: {normal_path.absolute()}")
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Check if normal data exists
    if not normal_path.exists():
        print(f"❌ ERROR: Normal data file not found!")
        print(f"   Expected at: {normal_path.absolute()}")
        print(f"\nFiles in data directory:")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                for f in csv_files:
                    print(f"   ✓ {f.name}")
            else:
                print(f"   (no CSV files found)")
                print(f"\nSearching entire data/ tree:")
                for f in data_dir.rglob("*.csv"):
                    print(f"   Found: {f}")
        else:
            print(f"   Directory does not exist: {data_dir.absolute()}")
        sys.exit(1)
    
    print(f"✓ Found normal data\n")
    
    # Check for anomaly files
    anomaly_files = [
        "anomaly_data_no_soiling_engineered_new.csv",
        "anomaly_data_soiling_100_no_degradation_engineered_new.csv", 
        "anomaly_data_soiling_100_engineered_new.csv"
    ]
    
    print(f"Checking for anomaly files:")
    all_found = True
    for fname in anomaly_files:
        fpath = data_dir / fname
        if fpath.exists():
            print(f"   ✓ {fname}")
        else:
            print(f"   ✗ {fname} NOT FOUND")
            all_found = False
    
    if not all_found:
        print(f"\n⚠️  WARNING: Some anomaly files missing!")
        print(f"Please ensure all 4 CSV files are in: {data_dir.absolute()}")
    
    print("\n" + "="*70)
    print("Starting data preparation...")
    print("="*70 + "\n")
    
    prepare_all_data(
        normal_path=str(normal_path),
        data_dir=str(data_dir),
        train_ratio=0.85,
        val_ratio=0.15,
        output_dir=str(output_dir),
        random_seed=42
    )