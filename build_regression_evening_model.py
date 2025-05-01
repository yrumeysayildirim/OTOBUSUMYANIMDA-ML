import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import xgboost as xgb

def load_data():

    data = pd.read_csv('regression_evening_model_data.csv')
    return data

def train_model(data):

    evening_X = data.drop(labels=['bus_stop_count'], axis=1)
    evening_y = data['bus_stop_count']

    evening_X_train, evening_X_test, evening_y_train, evening_y_test = train_test_split(evening_X, evening_y, test_size=0.2, random_state=42)



    evening_model = xgb.XGBRegressor(random_state=42, n_estimators=24)
    evening_model.fit(evening_X_train, evening_y_train)
    
    return evening_model, evening_X_test, evening_y_test
    

def test_model(evening_model, evening_X_test, evening_y_test):

    evening_y_predict = evening_model.predict(evening_X_test)

    print('XGBRegressor:\n')
    print(f'mean absolute error = {mean_absolute_error(evening_y_test, evening_y_predict)}')
    print(f'mean absolute percentage error = {mean_absolute_percentage_error(evening_y_test, evening_y_predict)}')
    print(f'mean squared error = {mean_squared_error(evening_y_test, evening_y_predict)}')
    print(f'root mean squared error = {root_mean_squared_error(evening_y_test, evening_y_predict)}')
    print(f'r2 score = {r2_score(evening_y_test, evening_y_predict)}')

    df_results = evening_X_test.copy()
    df_results["actual"] = evening_y_test
    df_results["predicted"] = evening_y_predict
    df_results["error"] = df_results["actual"] - df_results["predicted"]

    # Group by time slot and calculate mean error
    error_by_time = df_results.groupby("time_slot_minutes")["error"].mean()

    # Plot using Matplotlib
    plt.figure(figsize=(14, 7))
    bars = plt.bar(error_by_time.index, error_by_time.values, color='skyblue')
    plt.axhline(0, color='red', linestyle='--')  # horizontal line at 0
    plt.xlabel('Time Slot (minutes)')
    plt.ylabel('Average Prediction Error')
    plt.title('Average Error by Time Slot (Annotated)')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add annotation text on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),  # different offset
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8, color='black')


    plt.tight_layout()

    # Optional: Save the figure
    # plt.savefig("plots/error_by_time_slot_annotated.png")

    plt.show()

def save_model(model, filename = 'regression_evening_model.pkl'):

    with open(filename, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":

    data = load_data()
    model, X_test, y_test  = train_model(data)
    test_model(model, X_test, y_test)
    save_model(model)

