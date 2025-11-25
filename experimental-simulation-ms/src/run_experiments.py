import os
import sys
import yaml
import shutil
import joblib
import math
import argparse
import matplotlib.pyplot as plt


import os
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes' # avoid segmentation fault error
from pysr import PySRRegressor
from pysr import pysr, best

import torch

import sklearn
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import numpy as np
import pandas as pd
from datetime import datetime

from pre_process_data import add_mat200_embeddings_to_df, scatter_plot_raw_data
from scikit_models import optimize_scikit_model
from pysr_models import optimize_sr_model
from pytorch_mlp import optimize_mlp
from utils import get_git_commit_hash 

import pdb

scaler_dict = {'StandardScaler': StandardScaler,
               'MinMaxScaler': MinMaxScaler,
               'RobustScaler': RobustScaler}

def run_all_experiments(config):

    # === General Config ===
    experiment = config['experiment']
    artifact_paths = config['artifact_paths']
    ARTIFACT_PATH = artifact_paths['root']
    DATA_PATH = config['data_path'] 
    input_col = config['input_col']
    emb_col = config['emb_col']
    target_col = config['target_col']
    sim_col = config['sim_col']
    exp_col = config['exp_col']
    transform = config['transform']
    modeltypes = config['modeltypes']
    threshold = config['threshold']
    transform = config['transform']
    augment = config['augment']
    material_group = config['material_group']

    # # === Make sure directories exist ===
    # for path in artifact_paths.values():
    #     os.makedirs(path, exist_ok=True)

    # === Load datasets ===
    df = pd.read_csv(DATA_PATH, index_col=0)

    mask_re = df['has_rare_earth']
    mask_re_free = mask_re != 1
    
    # === Create Embeddings ===
    if emb_col != 'None':
        EMBEDDING_PATH = config['embedding_path']
        df = add_mat200_embeddings_to_df(df, 
                                         emb_col,
                                         EMBEDDING_PATH, 
                                         task_dict={'Ms (ampere/meter)_s':'regression'})
    if not augment:
        datasets = {
            "all": df,
            "re": df[mask_re], 
            "re_free": df[mask_re_free]
        }
        
    else:
        datasets = {f"{material_group}": df}
        
    # === Timestamp for logging ===
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    experiment_dir = f"experiment-{experiment}-{modeltypes}-{timestamp}/"
    
    os.makedirs(ARTIFACT_PATH + experiment_dir, exist_ok=True)
    log_file_path = os.path.join(ARTIFACT_PATH + experiment_dir, f"result_logs.yml")

    # Plot raw data
    os.makedirs(ARTIFACT_PATH + experiment_dir + 'data_plots/raw_data/', exist_ok=True)
    scatter_plot_raw_data(df, sim_col, exp_col, mask_re, mask_re_free, 
                          ARTIFACT_PATH + experiment_dir + 'data_plots/raw_data/')

    best_models = {}
    log_data = {
        'experiment': experiment,
        'timestamp': timestamp,
        'data_path': DATA_PATH,
        'results': {},
        'modeltype':'',
        'threshold': threshold,
        # 'max_diff_ratio': max_diff_ratio,
        'system_info': {
            'git_commit': get_git_commit_hash(),
            'python_version': sys.version,
            'scikit_learn_version': sklearn.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
        }
    }

    # === Setup models ===
    if 'scikit' in modeltypes:

        model_registry = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "RandomForestRegressor": RandomForestRegressor,
        }
        
        models_with_grids = {
            model_name: (
                model_registry[model_cfg['model']](**model_cfg.get('init_args', {})),
                model_cfg['parameters']
            )
            for model_name, model_cfg in config['models'].items()
        }

        optimize_model = optimize_scikit_model
    
    elif modeltypes == 'pysr_linear':
        models_with_grids = {
            'symbolic_regression': (PySRRegressor(niterations=100, binary_operators=["+", "*"],
                                                  batching=True, batch_size=500), None           
            )
            
        }
        
        optimize_model = optimize_sr_model
    
    elif modeltypes == 'pysr_polyn':
        models_with_grids = {'symbolic_regression': (PySRRegressor(
                                                        niterations=100,
                                                        binary_operators=["+", "*", "^"],
                                                        unary_operators=["square", "cube"],
                                                        maxsize=10,
                                                        constraints={'^': (-1,1)} # base can be arbitrarily complex, exponent avoids nested expressions, constants or variables only
                                                    ), None)}
        optimize_model = optimize_sr_model
    
    elif modeltypes == 'pytorch':
        models_with_grids = {'pytorch_mlp': ("","")} # is populated in optimize_mlp function
        optimize_model = optimize_mlp

    # Copy config file to experiment dir 
    shutil.copy(config['config_path'], os.path.join(ARTIFACT_PATH + experiment_dir, f"config.yml"))
    
    # === Run experiments ===
    for model_name, (model, param_grid) in models_with_grids.items():

        print(f"Running model: {model_name}")

        log_data['results'][model_name] = {
            'param_grid': param_grid,
            'datasets': {}
        }
        
        for dsetname, dset in datasets.items():

            print(f"➤ Dataset: {dsetname}")

            run_plot_dir = os.path.join(ARTIFACT_PATH + experiment_dir + '/images/', f"{experiment}-{model_name}")
            os.makedirs(run_plot_dir, exist_ok=True)
            plot_path = os.path.join(run_plot_dir, dsetname)

            dset = dset.sort_index()

            inputs = dset[input_col].values
            target = dset[target_col].values

            if transform == 'log':
                inputs = np.log(inputs)
                target = np.log(target)               

            if emb_col != 'None':
                embeddings = dset[emb_col].values
                embeddings = np.vstack(embeddings) 

                if inputs.ndim == 1:
                    inputs = inputs.reshape(-1, 1)
                inputs = np.concatenate([inputs, embeddings], axis=1)

            # Apply scaler
            scaler = config.get('scaler')
            if scaler is not None:  

                if inputs.ndim <= 1:
                    inputs = inputs.reshape(-1,1)
                    
                scaler_x = scaler_dict[scaler]()
                inputs = scaler_x.fit_transform(inputs)
                
                if target.ndim <= 1:
                    target = target.reshape(-1,1)
                    
                scaler_y = scaler_dict[scaler]() 
                target = scaler_y.fit_transform(target)

            best_model, train_metrics, test_metrics, grid = optimize_model(
                X=inputs,  
                y=target,
                xlabel='Ms_sim (A/m)',
                ylabel='Ms_exp (A/m)',
                model=model,
                param_grid=param_grid,
                plot_name=plot_path
            )
            
            # Save results
            result_entry = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'plot_path': plot_path,
                'config_path': config['config_path']
            }
            
            if 'pysr' in modeltypes:
                result_entry['best_model'] = best_model

            log_data['results'][model_name]['datasets'][dsetname] = result_entry
            
            os.makedirs(f'{ARTIFACT_PATH}{experiment_dir}models/', exist_ok=True)
            
            if modeltypes == 'pytorch':
                result_entry["best_architecture"] = grid["best_architecture"]
                result_entry["best_val_loss"] = grid["best_val_loss"]   
                result_entry["trained_for_epochs"] = grid["trained_for_epochs"]
                                                                  
                torch.save(best_model.state_dict(), ARTIFACT_PATH + experiment_dir + f'models/{model_name}_{experiment}_{dsetname}_thresh_{threshold}_weights.pth')
                
            else:    
                with open(f'{ARTIFACT_PATH}{experiment_dir}models/{model_name}_{experiment}_{dsetname}_thresh_{threshold}.pkl', 'wb') as f:
                    joblib.dump(best_model, f)                 

    # === Write YAML log ===
    with open(log_file_path, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=True, sort_keys=False)

    print(f"\n All experiments completed. Log saved to:\n {log_file_path}")
    return best_models, log_file_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Path to experiment YML config file.")
    parser.add_argument(
        "--configdir",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    # python src/run_experiments.py --config "configs/scikit_models_config.yml"

    args = parser.parse_args()

    with open(args.configdir, "r") as f:
        config = yaml.safe_load(f)

    best_models, log_file_path = run_all_experiments(config)
    # run_all_experiments(config)
    # post-processing
    # results = read_results_from_yml(RESULT_YML_PATH=log_file_path) #  python src/post_process_results.py --results_yml "artifacts/result_logs/experiment-sim2exp-2025-10-07T10-21-02.yml"
    # generate_metrics_table(results=results, csv_filename=f'{log_file_path}_metric_table.csv')   