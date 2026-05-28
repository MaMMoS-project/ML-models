#!/usr/bin/env python
"""
Script for model training and evaluation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import yaml
import argparse
import importlib
import json
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
import pickle
import torch
import joblib
# Add the src directory to the Python path
# src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# sys.path.append(src_dir)

from src.utils.data_preprocessing import preprocess_data
from src.utils.clustering_hardsoft import get_hard_soft_clusters, threshold_clustering, kmeans_clustering
from src.utils.supervised_clustering import apply_inverse_supervised_hardsoft_clustering
from src.utils.labels_preprocessing import add_magnetic_properties
from src.utils.log_to_file import log_output
from src.models.evaluator import Evaluator
from src.models.scalers import scale_data
from src.models.train_rf import calculate_jackknife_variance

import mammos_entity as me
import mammos_units as u

# Create log directory 
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

class MLPipeline:
    """Main class for running the ML pipeline with consistent data handling."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file path."""

        self.config = self._load_config(config_path)
        self.evaluator = Evaluator(self.config)
        
        # Initialize results directory from config
        results_dir = self.config.get('data', {}).get('results_dir', 'results')
        self.results_dir = Path(results_dir)
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    @log_output('logs/run_ml_pipeline.txt')
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("\nLoaded configurations:")
            print(json.dumps(config['preprocessing']['configurations'], indent=2))
            return config
    
    def process_dataset(self, df: pd.DataFrame, cluster_name: str = "") -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a dataset with different preprocessing configurations."""
        all_datasets = {}
        
        for config_name, config_params in self.config['preprocessing']['configurations'].items():
            # Process dataset with current configuration
            X_processed, y_processed = preprocess_data(
                df,
                input_columns=self.config['preprocessing']['input_columns'],
                output_columns=self.config['preprocessing']['output_columns'],
                **config_params
            )
            
            dataset_name = f"{config_name}{cluster_name}"
            all_datasets[dataset_name] = (X_processed, y_processed)
            
            print(f"Processed dataset: {dataset_name}")
        
        return all_datasets
    
    def train_and_evaluate_model(self, model_name: str, model_config: dict, datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]):
        """Train and evaluate a specific model."""
        try:
            # Import model module
            module = importlib.import_module(f"src.models.{model_config['module']}")
            train_func = getattr(module, model_config['function'])
            
            results = {}
            for dataset_name, (X, y) in datasets.items():
                print(f"\nTraining {model_name} on {dataset_name}...")
                
                # Create train/test split 
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                for scaler_type in self.config['scaling']['enabled_scalers']:
                    print(f"Using scaler: {scaler_type}")
                    
                    # Scale data for both training and evaluation
                    X_train_np = X_train_raw.to_numpy() if hasattr(X_train_raw, 'to_numpy') else X_train_raw
                    X_test_np = X_test_raw.to_numpy() if hasattr(X_test_raw, 'to_numpy') else X_test_raw
                    X_train_scaled, X_test_scaled, scaler = scale_data(X_train_np, X_test_np, scaler_type)
                    
                    # Always preserve feature names if they existed in original data
                    feature_names = None
                    if hasattr(X_train_raw, 'columns'):
                        feature_names = X_train_raw.columns
                        X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train_raw.index)
                        X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test_raw.index)
                    else:
                        X_train = X_train_scaled
                        X_test = X_test_scaled
                    
                    # Add cross-validation config to model parameters
                    model_params = model_config.get('params', {})
                    model_params['cv_config'] = self.config['cross_validation']
                    
                    # Add additional parameters for feature importance plotting
                    if model_config['module'] == 'train_rf' and model_params.get('show_feature_importance', False):
                        model_params['results_dir'] = self.config['data']['results_dir']
                        model_params['dataset_name'] = dataset_name
                        model_params['scaler_type'] = scaler_type
                    
                    # Train on the scaled training data
                    model, best_params = train_func(X_train, y_train, **model_params)
                    print(f"Best parameters: {best_params}")
                    
                    # Calculate jackknife variance for Random Forest models
                    errors = None
                    if model_config['module'] == 'train_rf':
                        errors = calculate_jackknife_variance(model, X_train, X_test, y_train)
                        
                        # Store errors in results for later use
                        if errors is not None:
                            if 'rf_errors' not in results:
                                results['rf_errors'] = {}
                            results['rf_errors'][f"{dataset_name}_{scaler_type}"] = errors
                    
                    # Evaluate the model using our evaluator with the scaled data
                    metrics = self.evaluator.evaluate_model(
                        model, X_train, y_train, X_test, y_test, dataset_name, model_name, errors=errors, scaler=scaler
                    )
                    
                    # Store results
                    results[f"{dataset_name}_{scaler_type}"] = {
                        'model': model,
                        'best_params': best_params,
                        'scaler': scaler,
                        'metrics': metrics
                    }
                    
                    # Add errors if available
                    if errors is not None:
                        results[f"{dataset_name}_{scaler_type}"]['errors'] = errors
            
            return results
        
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
#    def _train_inverted_hardsoft_classifier(self, df_clusters, input_cols, cluster_col, save_path):
#        """Train and save a hard/soft ONNX classifier using the new input features for inverted inference."""
#        from sklearn.pipeline import Pipeline
#        from sklearn.ensemble import RandomForestClassifier
#        from sklearn.calibration import CalibratedClassifierCV
#        from sklearn.model_selection import GridSearchCV, train_test_split
#        from sklearn.metrics import accuracy_score
#        from skl2onnx import convert_sklearn
#        from skl2onnx.common.data_types import FloatTensorType
#
#        print(f"\nTraining inverted hard/soft classifier on features: {input_cols}")
#
#        df_valid = df_clusters[input_cols + [cluster_col]].dropna()
#        y = df_valid[cluster_col].values.astype(int)
#        X = df_valid[input_cols].to_numpy()
#
#        X_train, X_test, y_train, y_test = train_test_split(
#            X, y, test_size=0.2, random_state=42, stratify=y
#        )
#        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test, 'standard')
#
#        calibrated = CalibratedClassifierCV(RandomForestClassifier(random_state=24))
#        grid_search = GridSearchCV(
#            calibrated, {'estimator__max_depth': [2, 4, 6, 8, 10]}, cv=3, n_jobs=-1, verbose=0
#        )
#        grid_search.fit(X_train_scaled, y_train)
#        best_model = grid_search.best_estimator_
#
#        y_test_pred = best_model.predict(X_test_scaled)
#        print(f"Inverted classifier test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
#
#        pipeline = Pipeline([('scaler', scaler), ('clf', best_model)])
#
#        os.makedirs(save_path, exist_ok=True)
#
#        bundle_path = os.path.join(save_path, 'supervised_hardsoft_clustering_pipeline_inverted.joblib')
#        joblib.dump({'pipeline': pipeline, 'feature_cols': input_cols, 'label_names': ['Soft', 'Hard']}, bundle_path)
#        print(f"Inverted classifier saved to {bundle_path}")
#
#        initial_type = [('float_input', FloatTensorType([None, len(input_cols)]))]
#        onnx_model = convert_sklearn(
#            pipeline, initial_types=initial_type, target_opset=11,
#            options={id(pipeline): {'zipmap': False}}
#        )
#        onnx_path = os.path.join(save_path, 'supervised_hardsoft_clustering_pipeline_inverted.onnx')
#        with open(onnx_path, 'wb') as f:
#            f.write(onnx_model.SerializeToString())
#        print(f"Inverted classifier ONNX saved to {onnx_path}")

    #@log_output('logs/run_ml_pipeline.txt')
    def run(self):
        """Run the complete pipeline."""
        
        print(f"Using config file: {self.config}")
        
        # Create results directory
        results_dir = Path(self.config['data']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        # Read the dataset
        try:
            #df = pd.read_csv(self.config['data']['input_file'])
            content = me.from_csv(self.config['data']['input_file'])
            df = content.to_dataframe(include_units=False)
            df = df.dropna()
            df = df.drop('D', axis=1)

            df = df.rename(columns={"Ms": "Ms (A/m)", "A": "A (J/m)", "K1": "K (J/m^3)", "Hc": "Hc (A/m)", "Mr": "Mr (A/m)", "BHmax": "BHmax (J/m^3)"})

            print("Datasize before droping: \n",df.shape[0])
            df = df[df['Mr (A/m)'] >= 1e4]
            df = df[df['Hc (A/m)'] >= 1e4]
            print("Datasize after droping: \n",df.shape[0])

            print("Dataset loaded successfully")
            print(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            print(f"Error: Could not find the dataset at {self.config['data']['input_file']}")
            return
        
        # Check if computation of analytical magnetic properties and differences is needed
        compute_difference = self.config.get('data', {}).get('compute_difference', False)
        
        if compute_difference:
            print("\nAdding computed magnetic properties and calculating differences...")
            output_columns = self.config['preprocessing']['output_columns']
            df = add_magnetic_properties(df, output_columns=output_columns, compute_differences=True)
            
            # Get the list of difference columns added
            diff_columns = [f"{col.split(' ')[0]}_diff {col.split(' ')[1]}" for col in output_columns 
                           if col in df.columns and f"{col.split(' ')[0]}_diff {col.split(' ')[1]}" in df.columns]
            
            print(f"Added difference columns: {diff_columns}")
            
            # Get the list of analytical columns added
            analyt_columns = [f"{col.split(' ')[0]}_analyt {col.split(' ')[1]}" for col in output_columns 
                             if f"{col.split(' ')[0]}_analyt {col.split(' ')[1]}" in df.columns]
            
            print(f"Added analytical columns: {analyt_columns}")
            
            # Change the output columns to use the difference columns for training and evaluation
            print(f"\nChanging output columns from original values to differences...")
            print(f"Original output columns: {self.config['preprocessing']['output_columns']}")
            self.config['preprocessing']['output_columns'] = diff_columns
            print(f"New output columns: {self.config['preprocessing']['output_columns']}")
            
            # Save the original and analytical columns for reference
            self.config['preprocessing']['original_output_columns'] = output_columns
            self.config['preprocessing']['analytical_output_columns'] = analyt_columns
        
        # Get clusters based on the specified clustering method
        clustering_method = self.config.get('clustering', {}).get('method', 'supervised')
        clustering_model_path = self.config.get('clustering', {}).get('model_path', None)
        
        print(f"\nUsing clustering method: {clustering_method}")
        
        if clustering_method == 'threshold':
            # Use threshold-based clustering
            df_clusters = threshold_clustering(df)
            df_cluster0 = df_clusters[df_clusters['Clusters'] == 0]
            df_cluster1 = df_clusters[df_clusters['Clusters'] == 1]
            cluster_col = 'Clusters'
        elif clustering_method == 'kmeans':
            # Use k-means clustering
            df_clusters = kmeans_clustering(df)
            df_cluster0 = df_clusters[df_clusters['Clusters_KMeans'] == 0]
            df_cluster1 = df_clusters[df_clusters['Clusters_KMeans'] == 1]
            cluster_col = 'Clusters_KMeans'
        else:  # Default to supervised
            # Use supervised clustering (either train new model or apply existing one)
            try:
                # Try to apply an existing model
                df_clusters = apply_inverse_supervised_hardsoft_clustering(df, model_path=clustering_model_path)
                print("Applied pre-trained supervised clustering model")
            except Exception as e:
                print(f"Could not apply pre-trained model: {str(e)}")
                print("Falling back to get_hard_soft_clusters with supervised method")
                # Fall back to get_hard_soft_clusters with supervised method
                cluster_dict = get_hard_soft_clusters(df, method='supervised')
                df_clusters = cluster_dict['all']
            
            df_cluster0 = df_clusters[df_clusters['pred_clusters'] == 0]
            df_cluster1 = df_clusters[df_clusters['pred_clusters'] == 1]
            cluster_col = 'Clusters_Supervised'
        
        print(f"Total samples: {len(df)}")
        print(f"Cluster 0 samples: {len(df_cluster0)}")
        print(f"Cluster 1 samples: {len(df_cluster1)}")

        # Determine the actual cluster label column in df_clusters
        if clustering_method == 'threshold':
            actual_cluster_col = 'Clusters'
        elif clustering_method == 'kmeans':
            actual_cluster_col = 'Clusters_KMeans'
        else:
            actual_cluster_col = 'pred_clusters'

        ## Train a hard/soft classifier on the new input features so that
        ## inverted-model inference can route samples to the correct cluster model
        #plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
        #self._train_inverted_hardsoft_classifier(
        #    df_clusters,
        #    self.config['preprocessing']['input_columns'],
        #    actual_cluster_col,
        #    plots_dir
        #)

        # Process datasets
        datasets = {
            'all': self.process_dataset(df, "_all"),
            'cluster0': self.process_dataset(df_cluster0, "_cluster0"),
            'cluster1': self.process_dataset(df_cluster1, "_cluster1")
        }
        
        # Train and evaluate enabled models
        all_results = {}
        for model_name, model_config in self.config['models'].items():
            if model_config['enabled']:
                print(f"\nTraining and evaluating {model_name}...")
                model_results = {}
                
                for dataset_type, dataset_dict in datasets.items():
                    print(f"\nProcessing {dataset_type} dataset...")
                    results = self.train_and_evaluate_model(
                        model_name, model_config, dataset_dict
                    )
                    if results:
                        model_results[dataset_type] = results
                
                all_results[model_name] = model_results
        
        # Save overall results
        results_file = self.results_dir / 'overall_results.json'
        
        # Convert results to a serializable format
        serializable_results = {}
        for model_name, model_data in all_results.items():
            serializable_results[model_name] = {}
            for dataset_type, dataset_results in model_data.items():
                serializable_results[model_name][dataset_type] = {}
                for result_key, result_data in dataset_results.items():
                    result_dict = {'metrics': result_data.get('metrics', {})}  
                    
                    # Only add best_params if they exist
                    if 'best_params' in result_data:
                        result_dict['best_params'] = result_data['best_params']
                    
                    serializable_results[model_name][dataset_type][result_key] = result_dict
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        print("Pipeline execution completed.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train and evaluate ML models.')
    parser.add_argument('--config', type=str, default='config/ml_config_test.yaml',
                        help='Path to the configuration file')
    
    args = parser.parse_args()
    
    pipeline = MLPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
