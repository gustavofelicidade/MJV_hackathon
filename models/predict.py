import pandas as pd
from joblib import load

def predict_critical_cases(model, data):
    predictions = model.predict(data)
    data['predictions'] = predictions
    return data
