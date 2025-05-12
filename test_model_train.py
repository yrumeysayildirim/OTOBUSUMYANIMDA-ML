from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import xgboost as xgb
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, call
from sklearn.datasets import make_classification, make_regression

# original functions
def train_classification(data):

    X = data.drop(labels=['density'], axis=1)
    y = data['density']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(class_weight={0: 1, 1: 2, 2: 4})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred

def train_noon_regression(data):

    X = data.drop(labels=['bus_stop_count'], axis=1)
    y = data['bus_stop_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    noon_model = GradientBoostingRegressor(random_state=9,n_estimators=300, learning_rate=0.3, max_depth=5, subsample=0.8)
    noon_model.fit(X_train, y_train)
    noon_y_predict = noon_model.predict(X_test)

    return noon_y_predict

def train_evening_regression(data):

    X = data.drop(labels=['bus_stop_count'], axis=1)
    y = data['bus_stop_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evening_model = xgb.XGBRegressor(n_estimators=24)
    evening_model.fit(X_train, y_train)
    evening_y_predict = evening_model.predict(X_test)

    return evening_y_predict


# tests

def test_random_forest_classifier():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    data['density'] = y

    preds = train_classification(data)
    assert len(preds) == int(0.2 * len(data))
    assert np.issubdtype(np.array(preds).dtype, np.integer)

def test_gradient_boosting_regressor():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    data['bus_stop_count'] = y

    preds = train_noon_regression(data)
    assert len(preds) == int(0.2 * len(data))
    assert np.issubdtype(np.array(preds).dtype, np.floating)

def test_xgboost_regressor():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    data['bus_stop_count'] = y

    preds = train_evening_regression(data)
    assert len(preds) == int(0.2 * len(data))
    assert np.issubdtype(np.array(preds).dtype, np.floating)