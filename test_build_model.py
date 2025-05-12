import os
import pickle
import pytest
from sklearn.linear_model import LinearRegression

# original function
def save_model(model, filename = 'regression_model.pkl'):

    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# test functions

@pytest.fixture
def dummy_model():
    model = LinearRegression()
    model.fit([[0], [1]], [0, 1]) 
    return model

def test_save_model_creates_file(dummy_model, tmp_path):
    filepath = tmp_path / "test_model.pkl"

    save_model(dummy_model, filename=str(filepath))

    assert filepath.exists(), "Model file was not created."

    
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)

    assert isinstance(loaded_model, LinearRegression)



