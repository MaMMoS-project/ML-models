import pandas as pd
import argparse
import yaml
import csv
import os
import pdb

def read_results_from_yml(RESULT_YML_PATH):
    
    # read results
    try:
        with open(RESULT_YML_PATH,'r') as f:
            results = yaml.safe_load(f)
            return results
    except FileNotFoundError:
        print(f"Error: File not found -> {RESULT_YML_PATH}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    return {}, print(f"Returning empty YML file.")

def get_subdirectories(path):
    """
    Return a list of subdirectory names inside the given path.
    """
    return [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]

def generate_metrics_table(results, csv_filename='metrics_table.csv'):
    
    print(results)
    
    rows = []
    header = ['model_type', 'dataset', 'split']

    # First pass to collect all possible metric names
    metric_names = set()

    for model_type, model_res in results['results'].items():

        for dataset, dataset_res in model_res['datasets'].items():
            for split in ['train_metrics', 'test_metrics']:
                metrics = dataset_res.get(split, {})
                metric_names.update(metrics.keys())

    metric_names = sorted(metric_names)
    header.extend(metric_names)

    # Second pass to build rows
    for model_type, model_res in results['results'].items():
        
        for dataset, dataset_res in model_res['datasets'].items():
            
            for split in ['train_metrics', 'test_metrics']:
                metrics = dataset_res.get(split, {})
                row = {
                    'model_type': model_type,
                    'dataset': dataset,
                    'split': split.replace('_metrics', '')  # 'train' or 'test'
                }
                # Fill in metric values, default to None if missing
                for name in metric_names:
                    row[name] = metrics.get(name)
                rows.append(row)
                
    # pdb.set_trace()
    rows.sort(key=lambda r: (r['MAE'] is not None, r['MAE']))
    rows.sort(key=lambda r: (r['R2'] is not None, r['R2']), reverse=True)


    # Write to CSV
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"CSV written to {csv_filename}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process results in each subdirectory.")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the parent directory containing experiment result folders."
    )

    args = parser.parse_args()
    base_dir = args.dir
    
    # python src/post_process_results.py --dir "artifacts/"

    subdirs = get_subdirectories(base_dir)

    print(f"Found {len(subdirs)} subdirectories in '{base_dir}'")

    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        yml_path = os.path.join(full_path, "result_logs.yml")  

        if not os.path.exists(yml_path):
            print(f"  [SKIP] No result_logs.yml found in '{full_path}'")
            continue

        try:
            print(f"  [OK] Reading results from: {yml_path}")
            results = read_results_from_yml(RESULT_YML_PATH=yml_path)

            output_csv = os.path.join(full_path, "metrics_table.csv")
            generate_metrics_table(results=results, csv_filename=output_csv)

            print(f"  [DONE] Saved metrics to {output_csv}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process '{full_path}': {e}")