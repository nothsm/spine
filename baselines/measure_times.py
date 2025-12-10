"""
Script to measure execution times for all baseline methods
Returns a pandas DataFrame with timing results

Usage:
    In notebook: from measure_times import measure_and_return_times
    Then: df = measure_and_return_times()
"""

import sys
sys.path.append('..')

import pandas as pd
from baselines import (
    load_tsp_data,
    evaluate_baselines,
    NETWORKX_AVAILABLE,
    ORTOOLS_AVAILABLE,
    CONCORDE_AVAILABLE
)

def measure_and_return_times():
    """Measure times for all TSP sizes and return as DataFrame"""

    # Define test files
    test_files = {
        5: '../data/data/tsp_5_train/tsp5_test.txt',
        10: '../data/data/tsp_10_train/tsp10_test.txt',
        20: '../data/data/tsp_20_test.txt',
        50: '../data/data/tsp50_test.txt'
    }

    # Methods to evaluate
    methods = []
    if CONCORDE_AVAILABLE:
        methods.append('concorde')
    if NETWORKX_AVAILABLE:
        methods.append('christofides')
    if ORTOOLS_AVAILABLE:
        methods.append('ortools')
    methods.append('dataset_output')  # for simulated annealing if you have it

    print("\n" + "="*70)
    print("Measuring execution times for baseline methods")
    print("="*70 + "\n")

    # Store results
    all_results = {}

    for size, file_path in test_files.items():
        print(f"\n--- Processing TSP-{size} ---")
        try:
            # Load test data (use first 100 instances for timing to be reasonable)
            data = load_tsp_data(file_path)[:100]
            print(f"Loaded {len(data)} test instances")

            # Evaluate with timing
            results = evaluate_baselines(
                data,
                methods=methods,
                progress=True,
                measure_time=True,
                ortools_time_limit=5,
                ortools_fast_mode=True
            )

            all_results[size] = results

            # Print results for this size
            print(f"\nTSP-{size} Average Times (seconds per instance):")
            for method, metrics in results.items():
                if 'avg_time' in metrics:
                    print(f"  {method:20s}: {metrics['avg_time']:.6f}s")

        except FileNotFoundError:
            print(f"  File not found: {file_path}")
            all_results[size] = {}
        except Exception as e:
            print(f"  Error: {e}")
            all_results[size] = {}

    # Build DataFrame
    print("\n" + "="*70)
    print("Building timing results DataFrame")
    print("="*70 + "\n")

    # Method name mapping for display
    method_display = {
        'concorde': 'Concorde',
        'christofides': 'Christofides',
        'ortools': 'OR-Tools',
        'supervised': 'Supervised Learning',
        'dataset_output': 'Simulated Annealing'
    }

    # Build data for DataFrame
    data_rows = []
    for method in methods:
        display_name = method_display.get(method, method.title())
        row = {'Method': display_name}

        for size in [5, 10, 20, 50]:
            col_name = f'TSP-{size}'
            if size in all_results and method in all_results[size]:
                if 'avg_time' in all_results[size][method]:
                    row[col_name] = all_results[size][method]['avg_time']
                else:
                    row[col_name] = None
            else:
                row[col_name] = None

        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    print("Timing measurements complete!\n")
    return df


if __name__ == "__main__":
    df = measure_and_return_times()
    print("\nTiming Results (seconds per instance):")
    print("="*70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else '---'))
    print("="*70)
