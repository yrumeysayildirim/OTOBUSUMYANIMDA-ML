from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pytest
import pandas as pd


# original function
def encode_density(data):

    density_col = data['density']
    le = LabelEncoder()
    d_ = le.fit_transform(density_col)
    data['density'] = d_

    return data

def encode_day(data):

    le = LabelEncoder()
    data['day_encoded'] = le.fit_transform(data['day'])
    data = data.drop(columns=['day'])
    return data

# test functions


def test_encode_density_assigns_numerical_labels():
    """
    Tests that encode_density replaces the 'density' column with numerical labels.
    """
    data = {
        'other_data': [1, 2, 3, 4, 5, 6],
        'density': [
            'LOW',
            'MEDIUM',
            'HIGH',
            'LOW',
            'HIGH',
            'MEDIUM'
        ]
    }
    df_input = pd.DataFrame(data)

    expected_labels = [
        1, 
        2, 
        0, 
        1,
        0,
        2 
    ]

    df_result = encode_density(df_input)

    assert 'density' in df_result.columns


    assert df_result['density'].dtype in ['int64', 'int32'] 
    assert list(df_result['density']) == expected_labels


    assert df_result is df_input


def test_encode_density_handles_empty_dataframe():
    """
    Tests that encode_density correctly handles an empty input DataFrame.
    """

    df_empty_input = pd.DataFrame({'density': [], 'other_data': []})

    df_empty_result = encode_density(df_empty_input)


    assert 'density' in df_empty_result.columns

    assert df_empty_result['density'].dtype in ['int64', 'int32', 'float66', 'float64']

    assert list(df_empty_result['density']) == []

    assert df_empty_result.shape == (0, 2) 

    assert df_empty_result is df_empty_input

