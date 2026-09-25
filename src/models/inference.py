import pickle
import pandas as pd
import numpy as np

def load_model(model_path: str):
    """Loads the frozen LightGBM model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict(model, features_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Returns probability of match (class 1) for the given feature set."""
    X = features_df[feature_cols].values
    return model.predict_proba(X)[:, 1]
