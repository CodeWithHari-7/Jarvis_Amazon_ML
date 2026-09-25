"""
Model training and inference for JARVIS_CHECKER.
"""
import pickle
import numpy as np
import pandas as pd


def load_model(model_path: str):
    """Load a pickled model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def save_model(model, model_path: str):
    """Save a pickled model."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def predict(model, features_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Returns probability of match (class 1) for the given feature matrix."""
    X = features_df[feature_cols].fillna(0).values
    return model.predict_proba(X)[:, 1]


def predict_array(model, X: np.ndarray) -> np.ndarray:
    """Returns match probability for a numpy array."""
    return model.predict_proba(X)[:, 1]
