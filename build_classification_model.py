import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def load_data():

    data = pd.read_csv('classification_model_data.csv')
    return data


def train_model(data):

    X = data.drop(labels=['density'], axis=1)
    y = data['density']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(class_weight={0: 1, 1: 2, 2: 4})
    model.fit(X_train, y_train)

    return model

def save_model(model, filename = 'classification_model.pkl'):

    with open(filename, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":

    data = load_data()
    model = train_model(data)
    save_model(model)