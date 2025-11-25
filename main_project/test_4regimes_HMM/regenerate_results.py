#!/usr/bin/env python3
"""
Script to regenerate regime detection results with corrected classification.
This will delete old results and regenerate everything.
"""

from pathlib import Path
import shutil

# Set up paths
project_root = Path(__file__).parent.parent
results_dir = Path(__file__).parent / 'results'

print("=" * 80)
print("REGENERATING REGIME DETECTION RESULTS")
print("=" * 80)

# Delete old results (except keep the directory)
if results_dir.exists():
    print(f"\nCleaning old results from: {results_dir}")
    for file in results_dir.glob('*'):
        if file.is_file():
            print(f"  Deleting: {file.name}")
            file.unlink()
        elif file.is_dir():
            print(f"  Deleting directory: {file.name}")
            shutil.rmtree(file)
    print("  Old results deleted.")
else:
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated results directory: {results_dir}")

# Now run the analysis
print("\n" + "=" * 80)
print("RUNNING REGIME DETECTION ANALYSIS")
print("=" * 80)

from regime_detection_hmm import RegimeDetectionHMM

macro_dir = project_root / 'data' / 'macro_processed' / 'selection'
sentiment_path = project_root / 'data' / 'news_data' / 'sentiment_scores.csv'

# Initialize detector
detector = RegimeDetectionHMM(
    macro_dir=macro_dir,
    sentiment_path=sentiment_path,
    macro_weight=0.4,
    sentiment_weight=0.6,
    n_regimes=4
)

# Run analysis (this will regenerate everything)
results = detector.run_full_analysis(output_dir=results_dir)

# Print summary with verification
print("\n" + "=" * 80)
print("VERIFICATION: REGIME CLASSIFICATIONS")
print("=" * 80)
for regime, chars in results['regime_characteristics'].items():
    print(f"\n{chars.get('regime_id', f'Regime {regime}')}: {chars['name']}")
    print(f"  avg_growth: {chars['avg_growth']:.4f} → {chars['growth_level']} Growth")
    print(f"  avg_inflation: {chars['avg_inflation']:.4f} → {chars['inflation_level']} Inflation")
    print(f"  Classification: {chars['growth_level']} Growth / {chars['inflation_level']} Inflation")
    
    # Verify classification is correct
    expected_growth = "High" if chars['avg_growth'] >= 0 else "Low"
    expected_inflation = "High" if chars['avg_inflation'] >= 0 else "Low"
    
    if chars['growth_level'] != expected_growth or chars['inflation_level'] != expected_inflation:
        print(f"  ⚠️  WARNING: Classification mismatch!")
        print(f"     Expected: {expected_growth} Growth / {expected_inflation} Inflation")
    else:
        print(f"  ✓ Classification correct")

print(f"\n{'=' * 80}")
print(f"Results regenerated and saved to: {results_dir}")
print(f"{'=' * 80}")



