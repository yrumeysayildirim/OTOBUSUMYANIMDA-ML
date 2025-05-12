import pytest
import pandas as pd

# original function taken from classification_model.ipynb
def density_estimation(df):

    density = ['LOW', 'MEDIUM', 'HIGH']
    student_density = []

    for _, row in df.iterrows():

        if row['bus_stop_count'] <= 60:
            student_density.append(density[0]) 
        elif row['bus_stop_count'] > 60 and row['bus_stop_count'] <= 100:
            student_density.append(density[1]) 
        elif row['bus_stop_count'] > 100:
            student_density.append(density[2])

    df['density'] = student_density 
    return df


# test functions

def test_density_estimation_assigns_correct_categories():

 
    data = {
        'some_other_column': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        'bus_stop_count': [
            0,   
            40,  
            60,  
            61,  
            80,  
            100, 
            101, 
            150  
        ]
    }
    df_input = pd.DataFrame(data)

    # Expected density values corresponding to the input, based on the code's logic:
    # <= 60 -> LOW
    # > 60 and <= 100 -> MEDIUM
    # > 100 -> HIGH
    expected_density = [
        'LOW',    # 0
        'LOW',    # 40
        'LOW',    # 60
        'MEDIUM', # 61
        'MEDIUM', # 80
        'MEDIUM', # 100
        'HIGH',   # 101
        'HIGH'    # 150
    ]

    df_result = density_estimation(df_input.copy()) 

    df_input_after = density_estimation(df_input)

    assert 'density' in df_input_after.columns

    assert list(df_input_after['density']) == expected_density

    assert df_input_after is df_input 


def test_density_estimation_handles_empty_dataframe():


    df_empty_input = pd.DataFrame({'bus_stop_count': []})

    df_empty_result = density_estimation(df_empty_input)

    assert 'density' in df_empty_result.columns

    assert list(df_empty_result['density']) == []

    assert df_empty_result.shape == (0, 2)

    assert df_empty_result is df_empty_input 