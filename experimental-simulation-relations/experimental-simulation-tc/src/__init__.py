"""experimental-simulation-tc source package.

Registers a project-wide suppression of the benign scikit-learn / LightGBM UserWarning
    "X does not have valid feature names, but LGBMRegressor was fitted with feature names"
which fires when a model fitted with feature names is predicted on a plain numpy array (the
prediction is unaffected). Doing it here -- run on the very first ``from src... import`` --
makes the suppression robust to entry point and import order, e.g. notebooks that import from
src before the trainer module (which also sets this filter) is imported.
"""
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
