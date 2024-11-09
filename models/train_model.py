import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


# Função para treinar o modelo
def train_model(data_path):
    data = pd.read_csv(data_path)
    X = data.drop('critical', axis=1)
    y = data['critical']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    dump(model, 'data/model/trained_model.pkl')


if __name__ == "__main__":
    train_model('data/historical_data.csv')
