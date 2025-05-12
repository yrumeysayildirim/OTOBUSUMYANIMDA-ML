import pandas as pd
from datetime import datetime, timedelta

# original function taken from classification_model.ipynb
def calculate_bus_stop_counts(df):
    """
    Takes a class schedule DataFrame and returns a new DataFrame with time slots and bus_stop_count.
    Works using minutes-from-midnight as time representation.
    
    Required columns in df:
    - 'day' (e.g., Monday)
    - 'end_time' (string in 'HH:MM' format)
    - 'student_nums' (integer)
    """
    
    # Convert end_time to datetime for parsing, then extract minutes-from-midnight
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M')
    df['end_time_minutes'] = df['end_time'].dt.hour * 60 + df['end_time'].dt.minute
    
    time_slots = []

    for _, row in df.iterrows():
        end_time_min = row['end_time_minutes']
        student_count = row['student_nums']
        day = row['day']
        
        # Define departure distribution
        departures = [
            (0, 0.20),  # 0-5 minutes
            (5, 0.50),  # 5-10 minutes
            (10, 0.30)  # 10-15 minutes
        ]

        for offset, fraction in departures:
            slot_min = end_time_min + offset  
            time_slots.append({
                'day': day,
                'time_slot_minutes': slot_min,
                'bus_stop_count': int(student_count * fraction)
            })


    result_df = pd.DataFrame(time_slots)

    result_df = result_df.groupby(['day', 'time_slot_minutes'], as_index=False).sum()


    return result_df

# test functions

def test_calculate_bus_stop_counts_basic():
    df = pd.DataFrame({
        'day': ['Monday'],
        'end_time': ['10:00'],
        'student_nums': [100]
    })

    result = calculate_bus_stop_counts(df)

    expected = pd.DataFrame({
        'day': ['Monday', 'Monday', 'Monday'],
        'time_slot_minutes': [600, 605, 610],  # 10:00, 10:05, 10:10
        'bus_stop_count': [20, 50, 30]
    })

    pd.testing.assert_frame_equal(result.sort_values(by='time_slot_minutes').reset_index(drop=True),
                                  expected.sort_values(by='time_slot_minutes').reset_index(drop=True))


def test_calculate_bus_stop_counts_multiple_rows():
    df = pd.DataFrame({
        'day': ['Monday', 'Monday'],
        'end_time': ['10:00', '10:05'],
        'student_nums': [100, 50]
    })

    result = calculate_bus_stop_counts(df)

    assert len(result) == 4

    assert 'bus_stop_count' in result.columns
    assert result['bus_stop_count'].sum() == 150


def test_zero_students():
    df = pd.DataFrame({
        'day': ['Tuesday'],
        'end_time': ['09:00'],
        'student_nums': [0]
    })

    result = calculate_bus_stop_counts(df)
    assert result['bus_stop_count'].sum() == 0


def test_multiple_days():
    df = pd.DataFrame({
        'day': ['Monday', 'Tuesday'],
        'end_time': ['10:00', '11:00'],
        'student_nums': [100, 200]
    })

    result = calculate_bus_stop_counts(df)

    assert set(result['day']) == {'Monday', 'Tuesday'}
    assert result['bus_stop_count'].sum() == 300
