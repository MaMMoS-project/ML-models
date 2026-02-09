import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import joblib
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from .clustering_hardsoft import threshold_clustering
from src.models.scalers import scale_data

def train_and_tune(X_train, y_train):
    """Performs Grid Search for hyperparameter tuning."""
    model = RandomForestClassifier(random_state=24)

    param_grid = {
             'estimator__max_depth': [2, 4, 6, 8, 10, 12, 14],
    }
    
    calibrated_forest = CalibratedClassifierCV(model)
  
    grid_search = GridSearchCV(calibrated_forest, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def supervised_clustering(df, Ms_col='Ms (A/m)', Mr_col='Mr (A/m)',
                          input_cols=['Ms (A/m)', 'A (J/m)', 'K (J/m^3)'], 
                          save_path=None,
                          results_dir=None):
    """
    Cluster magnetic materials into hard and soft using supervised classification based on threshold clustering labels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the magnetic data
    Ms_col : str
        Name of the column containing saturation magnetization values
    Mr_col : str
        Name of the column containing remanent magnetization values
    input_cols : list of str
        List of column names to use as input features
    save_path : str, optional
        Path to save the plot and model
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Clusters_Supervised' column (0: soft, 1: hard)
    """
    # First get threshold clustering labels
    df_threshold = threshold_clustering(df, Ms_col=Ms_col, Mr_col=Mr_col, save_path=save_path)
    
    y = df_threshold['Clusters'].values
    
    # Compute Mr/Ms ratio
    ratio = np.divide(df[Mr_col], df[Ms_col], 
                     out=np.zeros_like(df[Mr_col], dtype=float), 
                     where=df[Ms_col]!=0)
    
    # Prepare features for supervised learning
    X = np.column_stack([df[input_cols]])

    
    # Split data for training and testing
    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Scale (reuse your helper; it returns a *fitted* scaler)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test, 'standard')

    # Train and tune the model
    print("Training supervised classification model...", file=open(results_dir + "/clustering_analysis.txt", "a"))
    best_model, best_params = train_and_tune(X_train_scaled, y_train)
    print(f"Best parameters: {best_params}", file=open(results_dir + "/clustering_analysis.txt", "a"))
    
    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_scaled)

    # Calculate metrics on test set
    print("\nModel Performance on Test Set:", file=open(results_dir + "/clustering_analysis.txt", "a"))
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}", file=open(results_dir + "/clustering_analysis.txt", "a"))
    print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}", file=open(results_dir + "/clustering_analysis.txt", "a"))
    print(f"Recall: {recall_score(y_test, y_test_pred):.4f}", file=open(results_dir + "/clustering_analysis.txt", "a"))
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}", file=open(results_dir + "/clustering_analysis.txt", "a"))

    # Print classification report
    print("\nClassification Report:", file=open(results_dir + "/clustering_analysis.txt", "a"))
    print(classification_report(y_test, y_test_pred, target_names=['Soft', 'Hard']), file=open(results_dir + "/clustering_analysis.txt", "a"))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Soft', 'Hard'], 
                yticklabels=['Soft', 'Hard'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if (save_path is not None):
        plt.savefig(f"{save_path}/supervised_confusion_matrix.png", bbox_inches='tight', dpi=300)
        plt.ioff()
        
    else:
        plt.show()
        
    plt.close()
    
    # Make predictions on the entire dataset
    # Build a single fitted Pipeline = scaler + classifier
    pipeline = Pipeline([('scaler', scaler), ('clf', best_model)])

    # 9) Predict on full data
    y_pred = pipeline.predict(X)

    # 10) Attach predictions
    df_clustered = df.copy()
    if 'Clusters_Supervised' in df_clustered.columns:
        df_clustered['Clusters_Supervised'] = y_pred
    else:
        df_clustered.insert(len(df_clustered.columns), "Clusters_Supervised", y_pred)    
 
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(df[Ms_col][y_pred == 0], ratio[y_pred == 0], 
               c='blue', label='Soft', alpha=0.6)
    plt.scatter(df[Ms_col][y_pred == 1], ratio[y_pred == 1], 
               c='red', label='Hard', alpha=0.6)
    
    plt.xlabel('Ms (A/m)')
    plt.ylabel('Mr/Ms ratio')
    plt.title('Supervised Classification of Magnetic Materials')
    plt.legend()
    

    # 11) Save artifacts
    if (save_path is not None):
        
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(f"{save_path}/supervised_clustering.png", bbox_inches='tight', dpi=300)
        plt.ioff()

        # Save pipeline (recommended with joblib)
        pipe_path = f"{save_path}/supervised_clustering_pipeline.joblib"
        joblib.dump({
            "pipeline": pipeline,
            "feature_cols": input_cols,
            "label_names": ['Soft', 'Hard'],
            #"meta": {"Ms_col": Ms_col, "Mr_col": Mr_col}
        }, pipe_path)
        print(f"Pipeline saved to {pipe_path}", file=open(results_dir + "/clustering_analysis.txt", "a"))

        # Save the trained model
        model_path = f"{save_path}/supervised_clustering_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model saved to {model_path}", file=open(results_dir + "/clustering_analysis.txt", "a"))
        
        # Save metrics to a text file
        metrics_path = f"{save_path}/supervised_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Model Performance on Test Set:\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")
            f.write(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}\n")
            f.write(f"Recall: {recall_score(y_test, y_test_pred):.4f}\n")
            f.write(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_test_pred, target_names=['Soft', 'Hard']))
        print(f"Metrics saved to {metrics_path}", file=open(results_dir + "/clustering_analysis.txt", "a"))
    
    else:
        plt.show()
    
    plt.close()
    return df_clustered


def apply_supervised_clustering(df, model_path=None, 
                                input_cols=['Ms (A/m)', 'A (J/m)', 'K (J/m^3)'], 
                                save_path=None):
    """
    Apply a pre-trained supervised clustering model to classify magnetic materials.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the magnetic data
    model_path : str
        Path to the saved supervised clustering model pickle file
    input_cols : list of str
        List of column names to use as input features
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'pred_clusters' column (0: soft, 1: hard)
        and 'ratio'
    """


    # 1) Resolve default path if needed
    if model_path is None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'plots', 'supervised_clustering_pipeline.joblib'
        )
        if os.path.exists(default_path):
            model_path = default_path
        else:
            raise FileNotFoundError(
                f"No model path provided and could not find default pipeline at {default_path}"
            )

    # 2) Load pipeline bundle
    try:
        bundle = joblib.load(model_path)
        pipeline = bundle["pipeline"]
        feature_cols = bundle.get("feature_cols", input_cols)
        print(f"Loaded pipeline from {model_path}")
    except Exception as e:
        raise Exception(f"Error loading pipeline from {model_path}: {str(e)}")

    # 3) Prepare raw inputs in the SAME order as training
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in input df: {missing}")

    X = df[feature_cols].to_numpy()

    # 4) Predict directly with the pipeline
    y_pred = pipeline.predict(X)

    
    # Create a copy of the dataframe to avoid modifying the original
    df_clustered = df.copy()
    
    # Add clusters column
    if 'pred_clusters' in df_clustered.columns:
        df_clustered['pred_clusters'] = y_pred
    else:
        df_clustered.insert(len(df_clustered.columns), "pred_clusters", y_pred)
    

    # Plot the results if save_path is provided and Mr column is there
    if (save_path is not None):
        
        # Compute Mr/Ms ratio for plotting
        ratio = np.divide(df['Mr (A/m)'], df['Ms (A/m)'], 
                          out=np.zeros_like(df['Mr (A/m)'], dtype=float), 
                          where=df['Ms (A/m)']!=0)

        plt.figure(figsize=(10, 6))
        plt.scatter(df[input_cols[0]][y_pred == 0], ratio[y_pred == 0], 
                   c='blue', label='Soft', alpha=0.6)
        plt.scatter(df[input_cols[0]][y_pred == 1], ratio[y_pred == 1], 
                   c='red', label='Hard', alpha=0.6)
        
        plt.xlabel(input_cols[0])
        plt.ylabel(input_cols[1])
        plt.title('Supervised Classification of Magnetic Materials (Inference)')
        plt.legend()
        
        plt.savefig(f"{save_path}/supervised_clustering_inference.png", bbox_inches='tight', dpi=300)
        plt.ioff()
        
    else:
        plt.show()
        
    plt.close()
    
    return df_clustered
