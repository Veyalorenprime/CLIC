"""CLIC: Causal Latent Circuits for Interpretable OOD Detection"""

__version__ = "0.1.0"
"""
Check which features are normalized in the dataset.

Usage:
    python scripts/check_normalization.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def check_normalization(csv_path):
    """
    Analyze normalization state of all columns in a CSV file.
    
    Normalized features should have:
    - Mean ≈ 0
    - Std ≈ 1
    
    Min-max normalized features have:
    - Min = 0
    - Max = 1
    """
    print("\n" + "="*80)
    print(f"Checking: {csv_path}")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Analyze each column
    results = []
    
    for col in df.columns:
        if col == 'datetime':
            continue
            
        data = df[col].values
        
        stats = {
            'column': col,
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'std': data.std(),
            'has_zeros': (data == 0).sum() / len(data) * 100,  # % zeros
        }
        
        # Determine normalization type
        if abs(stats['mean']) < 0.1 and abs(stats['std'] - 1.0) < 0.2:
            stats['type'] = 'STANDARDIZED (mean≈0, std≈1)'
        elif stats['min'] >= -0.01 and stats['max'] <= 1.01:
            stats['type'] = 'MIN-MAX (0-1)'
        else:
            stats['type'] = 'RAW (not normalized)'
        
        results.append(stats)
    
    # Print results
    print("\n" + "-"*80)
    print(f"{'Column':<40} {'Type':<30} {'Mean':<10} {'Std':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['column']:<40} {r['type']:<30} {r['mean']:>9.4f} {r['std']:>9.4f}")
    
    print("-"*80)
    
    # Detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    print(f"\n{'Column':<40} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12} {'%Zeros':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['column']:<40} "
              f"{r['min']:>11.6f} "
              f"{r['max']:>11.6f} "
              f"{r['mean']:>11.6f} "
              f"{r['std']:>11.6f} "
              f"{r['has_zeros']:>9.1f}%")
    
    print("="*80)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    standardized = [r for r in results if 'STANDARDIZED' in r['type']]
    minmax = [r for r in results if 'MIN-MAX' in r['type']]
    raw = [r for r in results if 'RAW' in r['type']]
    
    print(f"\nStandardized (mean≈0, std≈1): {len(standardized)}")
    for r in standardized:
        print(f"  - {r['column']}")
    
    print(f"\nMin-Max Normalized (0-1):     {len(minmax)}")
    for r in minmax:
        print(f"  - {r['column']}")
    
    print(f"\nRaw (not normalized):         {len(raw)}")
    for r in raw:
        print(f"  - {r['column']}")
    
    print("="*80 + "\n")


def main():
    # Check all your data files
    data_dir = Path('data')
    
    files_to_check = [
        'processed/splits/normal_train.csv',
        'processed/splits/normal_val.csv',
        'anomaly_data_no_soiling_engineered_new.csv',
    ]
    
    for file in files_to_check:
        filepath = data_dir / file
        if filepath.exists():
            check_normalization(filepath)
        else:
            print(f"\n⚠️  File not found: {filepath}")


if __name__ == "__main__":
    main()