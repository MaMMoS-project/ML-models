

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct, ConstantKernel as C, WhiteKernel, RBF,
    RationalQuadratic, ExpSineSquared, Matern
)

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

from typing import Dict, Tuple, Any
import numpy as np

class GPRWrapper(GaussianProcessRegressor):
    """Wrapper around GaussianProcessRegressor to provide predict_with_std method."""
    
    def predict_with_std(self, X):
        """Predict with standard deviation."""
        return self.predict(X, return_std=True)

def create_kernel(kernel_config: Dict[str, Any]):
    """Create a kernel based on configuration.
    
    Args:
        kernel_config: Dictionary containing kernel parameters:
            - type: Type of kernel (RationalQuadratic, RBF, Matern, etc.)
            - alpha: Kernel parameter
            - length_scale: Length scale parameter
            
    Returns:
        sklearn kernel object
    """
    kernel_type = kernel_config.get('type', 'RationalQuadratic')
    alpha = kernel_config.get('alpha', 100.0)
    length_scale = kernel_config.get('length_scale', 1.5)
    
    if kernel_type == 'RationalQuadratic':
        kernel = C(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=alpha, length_scale=length_scale)
    elif kernel_type == 'RBF':
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale)
    elif kernel_type == 'Matern':
        kernel = C(1.0) * Matern(length_scale=length_scale, nu=1.5)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    return kernel

def train_gaussian_process(X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[GPRWrapper, Dict[str, Any]]:
    """Train a Gaussian Process model.
    
    Args:
        X: Input features
        y: Target values
        **kwargs: Additional parameters from config, including:
            - kernel: Dictionary with kernel configuration
            - n_restarts_optimizer: Number of restarts for optimizer
            - alpha: Noise parameter
            
    Returns:
        Tuple of (trained model, parameters used)
    """
    kernel_config = kwargs.get('kernel', {
        'type': 'RationalQuadratic',
        'alpha': 100.0,
        'length_scale': 1.5
    })
    
    n_restarts = kwargs.get('n_restarts_optimizer', 2)

    # Honor the known ~1% simulation error on the targets as the GP observation noise.
    # Targets are log1p-transformed, so a relative error r is ~r in log space. Because the
    # GP standardises the targets internally (normalize_y=True), express that noise in the
    # normalised space: sigma_norm = r / std(y_target), and use its variance as the GP's
    # diagonal noise `alpha`. Setting the noise to the *known* label variance (instead of the
    # previous ~0 default of 1e-6) stops the GP from interpolating the noisy training points
    # -- which fixes the documented GP over-fitting -- and yields a calibrated variance.
    LABEL_REL_ERR = float(kwargs.get('label_rel_err', 0.01))
    y_arr = np.asarray(y, dtype=float)
    y_std = np.std(y_arr, axis=0) if y_arr.ndim > 1 else np.std(y_arr)
    alpha_from_noise = float(np.mean((LABEL_REL_ERR / np.maximum(y_std, 1e-12)) ** 2))
    alpha = float(kwargs.get('alpha', alpha_from_noise))   # explicit `alpha` still overrides

    kernel = create_kernel(kernel_config)

    model = GPRWrapper(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=alpha,
        normalize_y=True,
        random_state=42
    )
    
    print("Training Gaussian Process...")
    print("Input shape:", X.shape)
    model.fit(X, y)
    print("Final kernel:", model.kernel_)
    
    params = {
        'kernel': kernel_config,
        'n_restarts_optimizer': n_restarts,
        'alpha': alpha,
        'label_rel_err': LABEL_REL_ERR
    }
    
    return model, params
