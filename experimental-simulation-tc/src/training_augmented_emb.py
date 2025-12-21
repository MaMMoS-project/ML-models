# -*- coding: utf-8 -*-
"""Train models on augmented data WITH EMBEDDINGS.

This script is intended to be run from the project root:

    python src/training_augmented_emb.py

It trains models on the augmented All-Augm dataset WITH embeddings:

1. Raw 200D embeddings (compound_embedding)
2. Compressed versions (comp_emb_pca_*_components)

The script requires the outputs from the embedding and compression pipeline:

    src/out/Augm_all_emb_w_embeddings.pkl

Model families:
1. Linear models (best of LASSO/Ridge/Linear)
2. Random Forest
3. FCNN/MLP (neural network)

For each family, it records R², RMSE, and MAE on the test split and
creates a comparison table.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    # Ensure we can import training modules
    script_dir = Path(__file__).parent
    training_dir = script_dir / "training"
    sys.path.insert(0, str(training_dir))
    
    # Get project root directory for storing results
    project_root = script_dir.parent
    results_dir = project_root / "results"

    # Import trainers after adjusting sys.path
    from linear_models import LinearModelsTrainer
    from random_forest import RandomForestTrainer
    from fcnn_mlp import FCNNTrainer

    print("=" * 80)
    print("AUGMENTED DATA TRAINING WITH EMBEDDINGS")
    print("=" * 80)

    # Define the dataset configurations to train on
    dataset_configs = [
        {"name": "All-Augm", "type": "all"},
        {"name": "RE-Augm", "type": "re"},
        {"name": "RE-Free-Augm", "type": "re-free"}
    ]
    
    is_augmented = True
    # Prioritize PCA16 and PCA32 as requested
    embedding_types = [
        None,       # Use raw 200D embedding
        "pca_16",   # Prioritize these two
        "pca_32",
        "pca_8", 
        "pca_64"
    ]
    
    all_results = []
    
    # The file must be a pickle containing the embeddings
    # First check if the PCA version exists, otherwise fall back to the original
    project_root = script_dir.parent  # Get the project root directory
    output_dir = project_root / "outputs"  # Standardized outputs directory
    
    # Dictionary to map dataset types to their corresponding file prefixes
    dataset_file_map = {
        "all": "Augm_all_emb_w_embeddings",
        "re": "Augm_RE_emb_w_embeddings",
        "re-free": "Augm_RE_Free_emb_w_embeddings"
    }
    
    # Dictionary to store the loaded DataFrames for each dataset type
    loaded_dataframes = {}
    
    # Check for all dataset files upfront
    for data_type, file_prefix in dataset_file_map.items():
        emb_file_pca = output_dir / f"{file_prefix}_PCA.pkl"
        emb_file_orig = output_dir / f"{file_prefix}.pkl"
        
        if emb_file_pca.exists():
            print(f"Found PCA-enriched embedding file for {data_type}: {emb_file_pca}")
            loaded_dataframes[data_type] = {"file": emb_file_pca, "loaded": False}
        elif emb_file_orig.exists():
            print(f"Found original embedding file for {data_type}: {emb_file_orig}")
            print(f"Note: PCA-enriched file not found for {data_type}. For better results, run compress_embedding_PCA.py first.")
            loaded_dataframes[data_type] = {"file": emb_file_orig, "loaded": False}
        else:
            print(f"WARNING: Required embedding files for {data_type} not found")
            print(f"Checked: {emb_file_pca} and {emb_file_orig}")
            print(f"Dataset type {data_type} will be skipped.")
    
    # Check if we have at least one dataset available
    if not loaded_dataframes:
        print("ERROR: No valid embedding files found for any dataset")
        print("Please run augment_data.py, create_embeddings.py, and compress_embedding_PCA.py first.")
        sys.exit(1)

    # Function to load the DataFrame for a specific dataset type when needed
    def load_dataset_for_type(dataset_type):
        if dataset_type not in loaded_dataframes:
            print(f"WARNING: No data file found for dataset type '{dataset_type}'")
            return None
            
        if not loaded_dataframes[dataset_type]["loaded"]:
            file_path = loaded_dataframes[dataset_type]["file"]
            print(f"Loading embeddings file for '{dataset_type}': {file_path}")
            try:
                df = pd.read_pickle(file_path)
                print(f"Successfully loaded {len(df)} samples for '{dataset_type}'")
                # Cache the loaded DataFrame
                loaded_dataframes[dataset_type]["data"] = df
                loaded_dataframes[dataset_type]["loaded"] = True
                
                # Check what embedding columns are available
                emb_cols = [col for col in df.columns if 'emb' in col.lower()]
                print(f"Available embedding columns for {dataset_type}: {emb_cols}")
                
            except Exception as e:
                print(f"ERROR loading file {file_path}: {e}")
                return None
        
        return loaded_dataframes[dataset_type]["data"].copy()
        
    # Try to preload "all" dataset to check available embeddings
    try:
        if "all" in loaded_dataframes:
            sample_df = load_dataset_for_type("all")
            if sample_df is not None:
                # Check what embedding columns are available
                emb_cols = [col for col in sample_df.columns if 'emb' in col.lower()]
                print(f"Available embedding columns: {emb_cols}")
            else:
                print("Could not load sample DataFrame to check embedding columns")
    except Exception as e:
        print(f"Error checking embedding columns: {e}")
        
        # Adjust embedding_types based on what's actually available
        available_types = []
        for emb_type in embedding_types:
            if emb_type is None:
                if 'compound_embedding' in df_data.columns:
                    available_types.append(emb_type)
                    print(f"Found raw embeddings: compound_embedding")
            else:
                # Try different naming patterns for PCA embeddings
                possible_names = [
                    f'comp_emb_pca_{emb_type}_components',  # Pattern from compress_embeddings.py
                    f'comp_emb_{emb_type}_components',      # Alternate pattern
                    emb_type                                # Direct name
                ]
                
                for name in possible_names:
                    if name in df_data.columns:
                        available_types.append(emb_type)
                        print(f"Found compressed embedding: {name} for type {emb_type}")
                        break
        
        if not available_types:
            print("ERROR: No valid embedding columns found in the DataFrame.")
            print("Available columns with 'emb' in name:")
            for col in emb_cols:
                print(f"  - {col}")
            sys.exit(1)
            
        print(f"Using embedding types: {available_types}")
        embedding_types = available_types
        
    except Exception as e:
        print(f"ERROR loading embeddings file: {e}")
        sys.exit(1)

    # First loop through datasets, then embeddings (dataset is primary grouping)
    for config in dataset_configs:
        dataset_name = config["name"]
        dataset_type = config["type"]
        
        # Skip datasets we don't have files for
        if dataset_type not in loaded_dataframes:
            print(f"\n{'='*60}")
            print(f"SKIPPING DATASET: {dataset_name} (type: {dataset_type}) - No data file available")
            print(f"{'='*60}")
            continue
            
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name} (type: {dataset_type})")
        print(f"{'='*60}")
        
        # Load the specific DataFrame for this dataset type
        df_data = load_dataset_for_type(dataset_type)
        if df_data is None:
            print(f"Error loading data for dataset type '{dataset_type}'. Skipping.")
            continue
        
        # Train models for each embedding type within this dataset
        for embedding_type in embedding_types:
            emb_name = "raw_200D" if embedding_type is None else embedding_type
            print(f"\n{'-'*50}")
            print(f"Training with embedding: {emb_name}")
            print(f"{'-'*50}")
            
            dataset_emb_results = []

            # 1. Linear models (LASSO, Ridge, Linear)
            try:
                lin_trainer = LinearModelsTrainer(
                    output_dir=str(results_dir / "augmented_emb_linear")
                )
                # Monkey patch load_augmented_data to return our DataFrame
                original_load_augmented = lin_trainer.loader.load_augmented_data
                lin_trainer.loader.load_augmented_data = lambda dataset_type=None: df_data.copy()
                
                # Modify prepare_dataset.use_embedding to check for our compressed column names
                original_prepare = lin_trainer.loader.prepare_dataset
                def patched_prepare_dataset(df, dataset_type, use_embedding=False, embedding_type=None):
                    if use_embedding and embedding_type is not None:
                        # Get X, y arrays using correct column names
                        #target_col = 'Tc_exp' if dataset_type == 'all' else f"Tc_exp_{dataset_type.replace('-', '_')}"
                        if 'Tc_exp' in df.columns:
                            target_col = 'Tc_exp'
                        
                        y = df[target_col].values
                        
                        # Try different column name patterns
                        if embedding_type is None:
                            X = np.vstack(df['compound_embedding'].values)
                        else:
                            possible_names = [
                                f'comp_emb_pca_{embedding_type}_components',  
                                f'comp_emb_{embedding_type}_components',
                                embedding_type
                            ]
                            
                            col_name = None
                            for name in possible_names:
                                if name in df.columns:
                                    col_name = name
                                    print(f"Using column {col_name} for embedding type {embedding_type}")
                                    break
                            
                            if col_name:
                                X = np.vstack(df[col_name].values)
                            else:
                                raise ValueError(f"Embedding column for type {embedding_type} not found")
                        
                        # Add Tc_sim
                        Tc_sim = df['Tc_sim'].values.reshape(-1, 1)
                        X = np.hstack([X, Tc_sim])
                        
                        return X, y
                    else:
                        # Fall back to original prepare_dataset for non-embedding cases
                        return original_prepare(df, dataset_type, use_embedding, embedding_type)
                        
                # Apply the monkey patch
                lin_trainer.loader.prepare_dataset = patched_prepare_dataset
                
                lin_results = lin_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=True,
                    embedding_type=embedding_type,
                    model_types=["lasso", "ridge"],
                )
                
                # Restore original methods
                lin_trainer.loader.load_augmented_data = original_load_augmented
                lin_trainer.loader.prepare_dataset = original_prepare
                
                # Find best linear model
                best_model = None
                best_r2 = -np.inf
                for model_name, metrics in lin_results.items():
                    if metrics["R2"] > best_r2:
                        best_r2 = metrics["R2"]
                        best_model = model_name
                        
                if best_model:
                    best_metrics = lin_results[best_model]
                    dataset_emb_results.append({
                        "Model_Family": "Linear",
                        "Model": best_model.upper(),
                        "Dataset": dataset_name,
                        "Embedding": emb_name,
                        "R2": best_metrics["R2"],
                        "RMSE": best_metrics["RMSE"],
                        "MAE": best_metrics["MAE"],
                    })
                    print(f"  Linear - Best model: {best_model.upper()}, R²: {best_metrics['R2']:.4f}")
            except Exception as e:
                print(f"Error running Linear models: {e}")

            # 2. Random Forest
            try:
                rf_trainer = RandomForestTrainer(
                    output_dir=str(results_dir / "augmented_emb_rf")
                )
                # Monkey patch load_augmented_data to return our DataFrame
                original_load_augmented = rf_trainer.loader.load_augmented_data
                rf_trainer.loader.load_augmented_data = lambda dataset_type=None: df_data.copy()
                
                # Apply the same patched prepare_dataset
                original_prepare = rf_trainer.loader.prepare_dataset
                def patched_prepare_dataset(df, dataset_type, use_embedding=False, embedding_type=None):
                    if use_embedding and embedding_type is not None:
                        # Get X, y arrays using correct column names
                        target_col = 'Tc_exp' if dataset_type == 'all' else f"Tc_exp_{dataset_type.replace('-', '_')}"
                        if 'Tc_exp' in df.columns:
                            target_col = 'Tc_exp'
                        
                        y = df[target_col].values
                        
                        # Try different column name patterns
                        if embedding_type is None:
                            X = np.vstack(df['compound_embedding'].values)
                        else:
                            possible_names = [
                                f'comp_emb_pca_{embedding_type}_components',  
                                f'comp_emb_{embedding_type}_components',
                                embedding_type
                            ]
                            
                            col_name = None
                            for name in possible_names:
                                if name in df.columns:
                                    col_name = name
                                    print(f"Using column {col_name} for embedding type {embedding_type}")
                                    break
                            
                            if col_name:
                                X = np.vstack(df[col_name].values)
                            else:
                                raise ValueError(f"Embedding column for type {embedding_type} not found")
                        
                        # Add Tc_sim
                        Tc_sim = df['Tc_sim'].values.reshape(-1, 1)
                        X = np.hstack([X, Tc_sim])
                        
                        return X, y
                    else:
                        # Fall back to original prepare_dataset for non-embedding cases
                        return original_prepare(df, dataset_type, use_embedding, embedding_type)
                        
                # Apply the monkey patch
                rf_trainer.loader.prepare_dataset = patched_prepare_dataset
                
                rf_metrics = rf_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=True,
                    embedding_type=embedding_type,
                )
                
                # Restore original methods
                rf_trainer.loader.load_augmented_data = original_load_augmented
                rf_trainer.loader.prepare_dataset = original_prepare
                
                dataset_emb_results.append({
                    "Model_Family": "RandomForest",
                    "Model": "RF",
                    "Dataset": dataset_name,
                    "Embedding": emb_name,
                    "R2": rf_metrics["R2"],
                    "RMSE": rf_metrics["RMSE"],
                    "MAE": rf_metrics["MAE"],
                })
                print(f"  Random Forest - R²: {rf_metrics['R2']:.4f}")
            except Exception as e:
                print(f"Error running Random Forest: {e}")

            # 3. FCNN/MLP
            try:
                mlp_trainer = FCNNTrainer(
                    output_dir=str(results_dir / "augmented_emb_fcnn")
                )
                # Monkey patch load_augmented_data to return our DataFrame
                original_load_augmented = mlp_trainer.loader.load_augmented_data
                mlp_trainer.loader.load_augmented_data = lambda dataset_type=None: df_data.copy()
                
                # Apply the same patched prepare_dataset
                original_prepare = mlp_trainer.loader.prepare_dataset
                def patched_prepare_dataset(df, dataset_type, use_embedding=False, embedding_type=None):
                    if use_embedding and embedding_type is not None:
                        # Get X, y arrays using correct column names
                        target_col = 'Tc_exp' if dataset_type == 'all' else f"Tc_exp_{dataset_type.replace('-', '_')}"
                        if 'Tc_exp' in df.columns:
                            target_col = 'Tc_exp'
                        
                        y = df[target_col].values
                        
                        # Try different column name patterns
                        if embedding_type is None:
                            X = np.vstack(df['compound_embedding'].values)
                        else:
                            possible_names = [
                                f'comp_emb_pca_{embedding_type}_components',  
                                f'comp_emb_{embedding_type}_components',
                                embedding_type
                            ]
                            
                            col_name = None
                            for name in possible_names:
                                if name in df.columns:
                                    col_name = name
                                    print(f"Using column {col_name} for embedding type {embedding_type}")
                                    break
                            
                            if col_name:
                                X = np.vstack(df[col_name].values)
                            else:
                                raise ValueError(f"Embedding column for type {embedding_type} not found")
                        
                        # Add Tc_sim
                        Tc_sim = df['Tc_sim'].values.reshape(-1, 1)
                        X = np.hstack([X, Tc_sim])
                        
                        return X, y
                    else:
                        # Fall back to original prepare_dataset for non-embedding cases
                        return original_prepare(df, dataset_type, use_embedding, embedding_type)
                        
                # Apply the monkey patch
                mlp_trainer.loader.prepare_dataset = patched_prepare_dataset
                
                mlp_metrics = mlp_trainer.train_and_evaluate(
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    is_augmented=is_augmented,
                    use_embedding=True,
                    embedding_type=embedding_type,
                )
                
                # Restore original methods
                mlp_trainer.loader.load_augmented_data = original_load_augmented
                mlp_trainer.loader.prepare_dataset = original_prepare
                
                dataset_emb_results.append({
                    "Model_Family": "MLP",
                    "Model": "FCNN",
                    "Dataset": dataset_name,
                    "Embedding": emb_name,
                    "R2": mlp_metrics["R2"],
                    "RMSE": mlp_metrics["RMSE"],
                    "MAE": mlp_metrics["MAE"],
                })
                print(f"  FCNN/MLP - R²: {mlp_metrics['R2']:.4f}")
            except Exception as e:
                print(f"Error running FCNN/MLP: {e}")
                
            # Add this embedding's results to overall results
            all_results.extend(dataset_emb_results)

    # Build comparison table
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by=["Dataset", "Embedding", "R2"], ascending=[True, True, False]).reset_index(drop=True)

        out_dir = results_dir / "augmented_emb_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "augmented_emb_models_comparison.csv"
        df.to_csv(out_csv, index=False)

        print("\n" + "=" * 80)
        print("AUGMENTED DATA WITH EMBEDDINGS - MODEL COMPARISON")
        print("=" * 80)
        pd.set_option('display.max_rows', None)
        print(df.to_string(index=False))
        print("=" * 80)
        print(f"Results saved to: {out_csv}")
        
        # Create a table with the best model for each dataset-embedding combination sorted by R2
        best_by_dataset_emb = df.loc[df.groupby(['Dataset', 'Embedding'])['R2'].idxmax()]
        best_by_dataset_emb = best_by_dataset_emb.sort_values(['Dataset', 'R2'], ascending=[True, False])
        out_best_dataset_emb_csv = out_dir / "augmented_emb_best_by_dataset_embedding.csv"
        best_by_dataset_emb.to_csv(out_best_dataset_emb_csv, index=False)
        
        # Create a table with the single best model for each dataset (across all embeddings)
        best_by_dataset = df.loc[df.groupby('Dataset')['R2'].idxmax()].sort_values('R2', ascending=False)
        out_best_csv = out_dir / "augmented_emb_best_by_dataset.csv"
        best_by_dataset.to_csv(out_best_csv, index=False)
        
        print("\n" + "=" * 80)
        print("BEST MODEL BY DATASET (SORTED BY R2)")
        print("=" * 80)
        print(best_by_dataset.to_string(index=False))
        print("=" * 80)
        print(f"Best models by dataset saved to: {out_best_csv}")
        
        # Also create a pivot table for clearer comparison
        if len(df) > 3:
            try:
                # First pivot by dataset and embedding
                pivot_df = df.pivot_table(
                    index=["Dataset", "Embedding"], 
                    columns="Model_Family", 
                    values="R2",
                    aggfunc='max'
                ).reset_index()
                
                pivot_csv = out_dir / "augmented_emb_comparison_pivot.csv"
                pivot_df.to_csv(pivot_csv, index=False)
                print("\nR² BY EMBEDDING AND MODEL FAMILY:")
                print(pivot_df.to_string(index=False))
                print(f"Pivot table saved to: {pivot_csv}")
            except Exception as e:
                print(f"Could not create pivot table: {e}")
    else:
        print("No results were produced; please check error messages above.")


if __name__ == "__main__":
    main()
