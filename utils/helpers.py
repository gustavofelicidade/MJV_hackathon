from joblib import load

def load_model(model_path):
    return load(model_path)
