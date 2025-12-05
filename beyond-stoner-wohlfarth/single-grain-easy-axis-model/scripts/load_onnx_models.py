import os
import numpy as np
import onnxruntime as ort

MODELS = {
    0: "../results/models/LogTransformation_cluster0/random_forest.onnx",
    1: "../results/models/LogTransformation_cluster1/random_forest.onnx",
}
_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3

def classify_magnetic_material(Ms, A, K):
    """Classify material as 'soft' or 'hard'."""
    session = ort.InferenceSession("../plots/supervised_clustering_pipeline.onnx", _SESSION_OPTIONS)
    X = np.array([[Ms, A, K]], dtype=np.float32)
    return "soft" if session.run(None, {session.get_inputs()[0].name: X})[0][0] == 0 else "hard"

def calculate_extrinsic_properties(Ms, A, K):
    Ms, A, K = np.atleast_1d(Ms), np.atleast_1d(A), np.atleast_1d(K)
    
    # 1. Determine class
    mat_class = classify_magnetic_material(Ms, A, K)
    
    # 2. Load regression model
    session = ort.InferenceSession(MODELS[mat_class], _SESSION_OPTIONS)
    
    # 3. Preprocess
    X_log = np.log1p(np.array([[Ms, A, K]], dtype=np.float32))
    
    # 4. Inference
    y_log = session.run(None, {session.get_inputs()[0].name: X_log})[0]
    
    # 5. Postprocess
    y = np.expm1(y_log)[0]
    
    return {'Hc': y[0], 'Mr': y[1], 'BHmax': y[2], 'class': mat_class}

if __name__ == "__main__":
    # Example: Hard Magnet
    Ms, A, K = 1.0e6, 1.0e-11, 4.5e6
    print(f"Input: Ms={Ms:.1e}, A={A:.1e}, K={K:.1e}")
    
    try:
        res = calculate_extrinsic_properties(Ms, A, K)
        print(f"Result: Class={res['class']}, Hc={res['Hc']:.2e}, Mr={res['Mr']:.2e}, BHmax={res['BHmax']:.2e}")
    except Exception as e:
        print(f"Error: {e}")

