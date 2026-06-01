"""
Script to visualize metrics from overall_results.json files.
Creates tables for each dataset and model, highlighting the best metrics.
"""

import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import re

def load_results(results_dir):
    """Load the overall_results.json file from the specified directory."""
    results_path = Path(results_dir) / "overall_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def get_datasets_and_models(results):
    """Extract all datasets and models from the results."""
    models = list(results.keys())
    
    # Get all unique datasets across all models and cluster types
    datasets = set()
    cluster_types = set()
    
    for model in models:
        for cluster_type in results[model]:
            cluster_types.add(cluster_type)
            for dataset in results[model][cluster_type]:
                if dataset != "rf_errors" and "metrics" in results[model][cluster_type][dataset]:
                    datasets.add(dataset)
    
    return list(models), list(datasets), list(cluster_types)


def create_metric_tables(results, output_dir):
    """Create tables for each dataset and model, highlighting the best metrics."""
    models, datasets, cluster_types = get_datasets_and_models(results)
    
    # Define which metrics are better when higher or lower
    higher_better = {'r2', 'adj_r2'}
    lower_better = {'mse', 'mae', 'mape', 'gini'}  # Note: gini coefficient is typically better when closer to 0
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each cluster type
    for cluster_type in cluster_types:
        # print(f"Processing {cluster_type} data...")
        
        # Get datasets for this cluster type
        cluster_datasets = []
        for dataset in datasets:
            if cluster_type in dataset or (cluster_type == "all" and "_cluster" not in dataset):
                cluster_datasets.append(dataset)
        
        if not cluster_datasets:
            continue
        
        # Process each dataset in this cluster
        for dataset in cluster_datasets:
            # print(f"  Creating tables for dataset: {dataset}")
            
            # Create tables for overall metrics
            create_overall_metrics_table(results, models, dataset, cluster_type, 
                                        higher_better, lower_better, output_path)
            
            # Create tables for per-variable metrics
            create_per_variable_metrics_tables(results, models, dataset, cluster_type,
                                             higher_better, lower_better, output_path)


def create_overall_metrics_table(results, models, dataset, cluster_type, higher_better, lower_better, output_path):
    """Create a table for overall metrics across all models for a specific dataset."""
    # Collect metrics for all models
    metrics_data = {
        'Model': [],
        'Split': []
    }
    
    # Initialize with empty metrics
    all_metrics = set()
    
    # First pass: collect all metrics
    for model in models:
        if cluster_type in results[model] and dataset in results[model][cluster_type]:
            if "metrics" in results[model][cluster_type][dataset]:
                model_metrics = results[model][cluster_type][dataset]["metrics"]
                for split in ['train', 'test']:
                    if split in model_metrics:
                        all_metrics.update(model_metrics[split].keys())
    
    # Initialize metrics columns
    for metric in all_metrics:
        metrics_data[metric] = []
    
    # Second pass: fill in the data
    for model in models:
        if cluster_type in results[model] and dataset in results[model][cluster_type]:
            if "metrics" in results[model][cluster_type][dataset]:
                model_metrics = results[model][cluster_type][dataset]["metrics"]
                for split in ['train', 'test']:
                    if split in model_metrics:
                        metrics_data['Model'].append(model)
                        metrics_data['Split'].append(split)
                        
                        # Add metrics values
                        for metric in all_metrics:
                            value = model_metrics[split].get(metric, np.nan)
                            metrics_data[metric].append(value)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)
    
    if df.empty:
        print(f"    No data available for {dataset}")
        return
    
    # Create a styled DataFrame for display
    styled_df = style_dataframe(df, higher_better, lower_better)
    
    # Create safe filename
    safe_filename = sanitize_filename(dataset)
    
    # Save to HTML
    html_path = output_path / f"{safe_filename}_overall_metrics.html"
    styled_df.to_html(html_path)
    # print(f"    Saved overall metrics to {html_path}")
    
    # Save to CSV for further analysis
    csv_path = output_path / f"{safe_filename}_overall_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    # Create and save PNG visualization
    create_metrics_plot(df, dataset, "Overall", output_path, higher_better, lower_better)


def create_per_variable_metrics_tables(results, models, dataset, cluster_type, higher_better, lower_better, output_path):
    """Create tables for per-variable metrics across all models for a specific dataset."""
    # Check if any model has per-variable metrics
    has_per_variable = False
    variables = set()
    
    for model in models:
        if (cluster_type in results[model] and 
            dataset in results[model][cluster_type] and 
            "metrics" in results[model][cluster_type][dataset] and
            "per_variable" in results[model][cluster_type][dataset]["metrics"]):
            
            has_per_variable = True
            variables.update(results[model][cluster_type][dataset]["metrics"]["per_variable"].keys())
    
    if not has_per_variable:
        print(f"    No per-variable metrics available for {dataset}")
        return
    
    # Process each variable
    for variable in variables:
        # print(f"    Creating table for variable: {variable}")
        
        # Collect metrics for all models
        metrics_data = {
            'Model': [],
            'Split': []
        }
        
        # Initialize with empty metrics
        all_metrics = set()
        
        # First pass: collect all metrics
        for model in models:
            if (cluster_type in results[model] and 
                dataset in results[model][cluster_type] and 
                "metrics" in results[model][cluster_type][dataset] and
                "per_variable" in results[model][cluster_type][dataset]["metrics"] and
                variable in results[model][cluster_type][dataset]["metrics"]["per_variable"]):
                
                var_metrics = results[model][cluster_type][dataset]["metrics"]["per_variable"][variable]
                for split in ['train', 'test']:
                    if split in var_metrics:
                        all_metrics.update(var_metrics[split].keys())
        
        # Initialize metrics columns
        for metric in all_metrics:
            metrics_data[metric] = []
        
        # Second pass: fill in the data
        for model in models:
            if (cluster_type in results[model] and 
                dataset in results[model][cluster_type] and 
                "metrics" in results[model][cluster_type][dataset] and
                "per_variable" in results[model][cluster_type][dataset]["metrics"] and
                variable in results[model][cluster_type][dataset]["metrics"]["per_variable"]):
                
                var_metrics = results[model][cluster_type][dataset]["metrics"]["per_variable"][variable]
                for split in ['train', 'test']:
                    if split in var_metrics:
                        metrics_data['Model'].append(model)
                        metrics_data['Split'].append(split)
                        
                        # Add metrics values
                        for metric in all_metrics:
                            value = var_metrics[split].get(metric, np.nan)
                            metrics_data[metric].append(value)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data)
        
        if df.empty:
            print(f"    No data available for {variable} in {dataset}")
            continue
        
        # Create a styled DataFrame for display
        styled_df = style_dataframe(df, higher_better, lower_better)
        
        # Create safe filenames
        safe_dataset = sanitize_filename(dataset)
        safe_variable = sanitize_filename(variable)
        
        # Save to HTML
        html_path = output_path / f"{safe_dataset}_{safe_variable}_metrics.html"
        styled_df.to_html(html_path)
        # print(f"    Saved {variable} metrics to {html_path}")
        
        # Save to CSV for further analysis
        csv_path = output_path / f"{safe_dataset}_{safe_variable}_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Create and save PNG visualization
        create_metrics_plot(df, dataset, variable, output_path, higher_better, lower_better)


def style_dataframe(df, higher_better, lower_better):
    """Style the DataFrame to highlight the best metrics."""
    # Create a copy to avoid modifying the original
    styled_df = df.copy()
    
    # Define a function to highlight the best values
    def highlight_best(s, better_higher=True):
        is_numeric = pd.to_numeric(s, errors='coerce').notna()
        
        if is_numeric.all():
            if better_higher:
                return ['font-weight: bold' if v == s.max() else '' for v in s]
            else:
                return ['font-weight: bold' if v == s.min() else '' for v in s]
        return ['' for _ in range(len(s))]
    
    # Apply styling
    styled = styled_df.style
    
    # Apply conditional formatting for each metric
    for col in df.columns:
        if col not in ['Model', 'Split']:
            if col in higher_better:
                styled = styled.apply(highlight_best, axis=0, subset=[col], better_higher=True)
            elif col in lower_better:
                styled = styled.apply(highlight_best, axis=0, subset=[col], better_higher=False)
    
    # Format numbers to be more readable
    format_dict = {}
    for col in df.columns:
        if col not in ['Model', 'Split']:
            format_dict[col] = '{:.4f}'
    
    styled = styled.format(format_dict)
    
    # Add background color alternating by model
    def highlight_models(s):
        return ['background-color: #f2f2f2' if i % 2 == 0 else '' 
                for i, _ in enumerate(s)]
    
    styled = styled.apply(highlight_models, axis=0, subset=['Model'])
    
    # Add caption
    styled = styled.set_caption("Metrics Evaluation")
    
    return styled


def sanitize_filename(name):
    """Convert a string to a safe filename by removing special characters."""
    # Replace special characters with underscores
    safe_name = re.sub(r'[\\/*?:\"<>|\(\)\s\^\-\+\=\!\@\#\$\%\&]', '_', name)
    return safe_name


def create_metrics_plot(df, dataset, variable_name, output_path, higher_better, lower_better):
    """Create a visual plot of metrics as a table with both train and test metrics."""
    # Filter out non-metric columns
    metric_cols = [col for col in df.columns if col not in ['Model', 'Split']]
    if not metric_cols:
        return
    
    # Get unique models
    models = df['Model'].unique()
    if len(models) == 0:
        return
    
    # Create a figure with appropriate size
    fig_width = max(10, len(metric_cols) * 1.5)
    fig_height = max(8, len(models) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Prepare data for the table
    table_data = []
    row_labels = []
    
    # For each model, add train and test rows
    for model in models:
        model_df = df[df['Model'] == model]
        
        # Add train row if available
        train_df = model_df[model_df['Split'] == 'train']
        if not train_df.empty:
            train_values = [train_df[metric].values[0] for metric in metric_cols]
            table_data.append(train_values)
            row_labels.append(f"{model} (Train)")
        
        # Add test row if available
        test_df = model_df[model_df['Split'] == 'test']
        if not test_df.empty:
            test_values = [test_df[metric].values[0] for metric in metric_cols]
            table_data.append(test_values)
            row_labels.append(f"{model} (Test)")
    
    # Create a table
    table = ax.table(cellText=[[f"{val:.4f}" for val in row] for row in table_data],
                     rowLabels=row_labels,
                     colLabels=metric_cols,
                     loc='center',
                     cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust row height for better readability
    
    # Highlight best values
    for (row_idx, col_idx), cell in table._cells.items():
        if row_idx == 0:  # Skip header row
            cell.set_text_props(weight='bold')
            continue
            
        # Get the actual row index in our data (subtract 1 for header)
        data_row_idx = row_idx - 1
        
        if col_idx >= 0 and data_row_idx < len(table_data) and col_idx < len(metric_cols):
            metric = metric_cols[col_idx]
            
            # Get all values for this metric and split
            split_type = 'train' if '(Train)' in row_labels[data_row_idx] else 'test'
            split_rows = [i for i, label in enumerate(row_labels) if f"({split_type.capitalize()})" in label]
            split_values = [table_data[i][col_idx] for i in split_rows]
            
            value = table_data[data_row_idx][col_idx]
            
            # Check if this is the best value
            is_best = False
            if metric in higher_better and value == max(split_values):
                is_best = True
            elif metric in lower_better and value == min(split_values):
                is_best = True
                
            if is_best:
                cell.set_text_props(weight='bold')
    
    # Remove axis
    ax.axis('off')
    
    # Add title
    title = f"{dataset} - {variable_name} Metrics"
    ax.set_title(title, pad=20, fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    safe_dataset = sanitize_filename(dataset)
    safe_variable = sanitize_filename(variable_name)
    fig_path = output_path / f"{safe_dataset}_{safe_variable}_metrics.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics plot to {fig_path}")


_KNOWN_SCALERS = ['standard', 'minmax', 'robust']

# Models that cannot be exported to ONNX and are therefore skipped when
# selecting the best model for linking.
_ONNX_UNSUPPORTED_MODELS = {'gaussian_process', 'neural_network'}


def find_best_model_per_cluster(results):
    """Return all models per cluster label, sorted by test R² (best first).

    Returns a dict keyed by cluster label (e.g. 'cluster0', 'cluster1') where
    each value is a list of candidate dicts with fields: model, result_key,
    dataset_name, r2 — sorted descending by r2.
    """
    candidates = {}

    for model_name, clusters in results.items():
        for cluster_label, datasets in clusters.items():
            if 'cluster' not in cluster_label:
                continue  # skip the 'all' aggregation
            for result_key, result_data in datasets.items():
                if 'metrics' not in result_data:
                    continue  # skip auxiliary entries like rf_errors
                r2 = result_data['metrics'].get('test', {}).get('r2')
                if r2 is None:
                    continue

                # Strip the trailing scaler suffix to recover the dataset_name
                dataset_name = result_key
                for scaler in _KNOWN_SCALERS:
                    if result_key.endswith(f'_{scaler}'):
                        dataset_name = result_key[: -len(f'_{scaler}')]
                        break

                candidates.setdefault(cluster_label, []).append({
                    'model': model_name,
                    'result_key': result_key,
                    'dataset_name': dataset_name,
                    'r2': r2,
                })

    return {
        label: sorted(cands, key=lambda c: c['r2'], reverse=True)
        for label, cands in candidates.items()
    }


def link_best_models(results, results_dir):
    """Create best_model_<cluster> directories with symlinks to the winning model.

    For each cluster (cluster0, cluster1) the model with the highest test R² is
    selected, skipping models that have no ONNX export (gaussian_process,
    neural_network).  Symlinks are created for the .onnx, .pkl, _scaler.pkl and
    _metrics.json files as well as the plots subdirectory.  A
    best_model_summary.json is written alongside them.

    Existing symlinks are refreshed so the directory always reflects the latest
    run.
    """
    results_dir = Path(results_dir).resolve()
    all_candidates = find_best_model_per_cluster(results)

    if not all_candidates:
        print("No per-cluster results found; skipping best-model linking.")
        return

    for cluster_label, candidates in sorted(all_candidates.items()):
        # Pick the highest-ranked model that supports ONNX export.
        skipped = []
        info = None
        for candidate in candidates:
            if candidate['model'] in _ONNX_UNSUPPORTED_MODELS:
                skipped.append(candidate)
            else:
                info = candidate
                break

        if info is None:
            print(f"\nNo ONNX-compatible model found for {cluster_label}; skipping.")
            continue

        model_name = info['model']
        dataset_name = info['dataset_name']
        r2 = info['r2']

        if skipped:
            skipped_desc = ', '.join(
                f"{c['model']} (R²={c['r2']:.4f})" for c in skipped
            )
            print(
                f"\nNote: best model(s) for {cluster_label} lack ONNX export and "
                f"were skipped: {skipped_desc}"
            )
            print(
                f"Using next best model for {cluster_label}: "
                f"{model_name}  (test R²={r2:.4f})"
            )
        else:
            print(f"\nBest model for {cluster_label}: {model_name}  (test R²={r2:.4f})")

        best_dir = results_dir / f"best_model_{cluster_label}"
        best_dir.mkdir(exist_ok=True)

        # --- model artefacts ---
        model_src_dir = results_dir / 'models' / dataset_name
        for suffix in ['.onnx', '.pkl', '_scaler.pkl', '_metrics.json']:
            src = model_src_dir / f"{model_name}{suffix}"
            if not src.exists():
                continue
            dst = best_dir / f"{model_name}{suffix}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(os.path.relpath(src, best_dir))
            print(f"  linked {dst.name} -> {os.path.relpath(src, best_dir)}")

        # --- plots directory ---
        plots_src = results_dir / 'plots' / dataset_name / model_name
        if plots_src.exists():
            dst = best_dir / 'plots'
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)
            dst.symlink_to(os.path.relpath(plots_src, best_dir))
            print(f"  linked plots/ -> {os.path.relpath(plots_src, best_dir)}")

        # --- human-readable summary ---
        summary = {
            'cluster': cluster_label,
            'best_model': model_name,
            'dataset': dataset_name,
            'test_r2': r2,
            'result_key': info['result_key'],
        }
        with open(best_dir / 'best_model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  summary -> {best_dir / 'best_model_summary.json'}")


def main():

    parser = argparse.ArgumentParser(description="Generate metric tables from overall_results.json")
    parser.add_argument("results_dir", help="Directory containing overall_results.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for tables (default: <results_dir>/metric_tables)")

    args = parser.parse_args()

    try:
        # Set matplotlib style
        plt.style.use('ggplot')

        # Set default output directory as a subdirectory of the input directory
        results_path = Path(args.results_dir)
        if args.output is None:
            output_path = results_path / "metric_tables"
        else:
            output_path = Path(args.output)

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Load results
        results = load_results(args.results_dir)

        # Create tables
        create_metric_tables(results, output_path)

        print(f"Tables and plots generated successfully in {output_path}")
        plt.close()

        # Determine and link best models per cluster
        link_best_models(results, args.results_dir)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
